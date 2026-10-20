"""iterative LQR solver"""

import math
import warnings
from typing import Callable, Tuple, Any, Optional
from functools import partial
from jax import Array
import jax
from jax import lax
import jax.numpy as jnp

from diffilqrax import lqr
from diffilqrax.lqr import bmm
from diffilqrax.typs import (
    iLQRParams,
    System,
    LQR,
    Gains,
    CostToGo,
    LQRParams,
    ParallelSystem,
)
from diffilqrax.utils import time_map

# from chex import assert_scalar_negative, assert_scalar_positive

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_disable_jit", False)  # uncomment for debugging purposes


MU_MIN = 1e-6
LOOPS = ("while", "scan")
LINESEARCHES = ("backtrack", "parallel")


def bounded_loop(cond_fun: Callable, body_fun: Callable, init: Any, max_steps: int, loop: str = "while") -> Any:
    """
    Apply ``body_fun`` while ``cond_fun(carry)`` holds, at most ``max_steps`` times.

    Parameters
    ----------
    cond_fun : Callable
        Maps the carry to a boolean scalar; the loop continues while it is True.
    body_fun : Callable
        Maps the carry to the next carry.
    init : Any
        Initial carry.
    max_steps : int
        Maximum number of applications of ``body_fun``.
    loop : str, optional
        "while" (default) runs a ``lax.while_loop``, which executes only the steps needed (under ``vmap``,
        until the last batch element stops). It supports forward-mode but not reverse-mode differentiation.
        "scan" runs a fixed-length ``lax.scan`` that skips finished steps with ``lax.cond``; it is reverse-mode
        differentiable, but stores every step and under ``vmap`` executes all ``max_steps`` steps.

    Returns
    -------
    Any
        The final carry.
    """
    assert loop in LOOPS, f"loop must be one of {LOOPS}, got {loop}"
    if loop == "scan":

        def scan_step(carry, _):
            return lax.cond(cond_fun(carry), body_fun, lambda c: c, carry), None

        return lax.scan(scan_step, init, None, length=max_steps)[0]

    def while_cond(step_carry):
        step, carry = step_carry
        return jnp.logical_and(step < max_steps, cond_fun(carry))

    def while_body(step_carry):
        step, carry = step_carry
        return step + 1, body_fun(carry)

    return lax.while_loop(while_cond, while_body, (jnp.zeros((), dtype=jnp.int32), init))[1]


def backtracking_search(
    trial: Callable,
    init: Tuple[Array, Array, Array],
    alpha_first: float,
    max_steps: int,
    tau: float = 0.5,
    alpha_min: float = 1e-8,
    loop: str = "while",
    n_parallel: int = 0,
) -> Tuple[Array, Tuple[Array, Array], Array]:
    """
    Try step sizes α = alpha_first, tau * alpha_first, ... until ``trial`` accepts one.

    The search stops at the first accepted α, after the first rejected α below ``alpha_min``, or after
    ``max_steps`` trials. With ``n_parallel > 0`` the first ``n_parallel`` step sizes are tried at once
    (``vmap`` over α) and the largest accepted one is kept, which gives the same result as trying them in turn;
    only if none is accepted does the search continue one α at a time.

    Parameters
    ----------
    trial : Callable
        Maps α to (accepted, Xs, Us, cost) of the trial step.
    init : Tuple[Array, Array, Array]
        (Xs, Us, cost) returned if no step is accepted.
    alpha_first : float
        First step size tried.
    max_steps : int
        Maximum number of step sizes tried.
    tau : float, optional
        Reduction factor for α, by default 0.5.
    alpha_min : float, optional
        Minimum α, by default 1e-8.
    loop : str, optional
        "while" or "scan", see :func:`bounded_loop`, by default "while".
    n_parallel : int, optional
        Number of step sizes tried at once, by default 0 (one at a time).

    Returns
    -------
    Tuple[Array, Tuple[Array, Array], Array]
        Whether a step was accepted, the (accepted or initial) state and input trajectories, and the
        corresponding cost.
    """
    Xs_init, Us_init, cost_init = init
    n_parallel = min(n_parallel, max_steps)
    if n_parallel > 0:
        alphas = alpha_first * tau ** jnp.arange(n_parallel)
        accepted, Xs_try, Us_try, cost_try = jax.vmap(trial)(alphas)
        # α_k is only tried in turn if every previous α was at least alpha_min
        tried = jnp.concatenate([jnp.array([True]), jnp.cumprod(alphas[:-1] >= alpha_min).astype(bool)])
        accepted = jnp.logical_and(accepted, tried)
        idx = jnp.argmax(accepted)
        any_accepted = jnp.any(accepted)
        carry = (
            jnp.where(any_accepted, Xs_try[idx], Xs_init),
            jnp.where(any_accepted, Us_try[idx], Us_init),
            jnp.where(any_accepted, cost_try[idx], cost_init),
            alphas[-1] * tau,
            any_accepted,
            jnp.logical_and(jnp.logical_not(any_accepted), jnp.all(alphas >= alpha_min)),
        )
    else:
        alpha = jnp.asarray(alpha_first, dtype=jnp.result_type(cost_init))
        carry = (Xs_init, Us_init, cost_init, alpha, jnp.array(False), jnp.array(True))

    def backtrack_iter(carry):
        """Trial step with the current α; keep it if accepted, else shrink α"""
        Xs, Us, cost, alpha, _, _ = carry
        accepted, new_Xs, new_Us, new_cost = trial(alpha)
        carry_on = jnp.logical_and(jnp.logical_not(accepted), alpha >= alpha_min)
        return (
            jnp.where(accepted, new_Xs, Xs),
            jnp.where(accepted, new_Us, Us),
            jnp.where(accepted, new_cost, cost),
            alpha * tau,
            accepted,
            carry_on,
        )

    if max_steps > n_parallel:
        carry = bounded_loop(lambda c: c[-1], backtrack_iter, carry, max_steps - n_parallel, loop)
    Xs_star, Us_star, cost_star, _, accepted, _ = carry
    return accepted, (Xs_star, Us_star), cost_star


def increase_regularisation(delta: Array, mu: Array) -> Tuple[Array, Array]:
    """
    Increase Levenberg-Marquardt regularisation: δ ← max(2, 2δ), μ ← max(μ_min, μδ).

    Parameters
    ----------
    delta : Array
        Regularisation scaling factor.
    mu : Array
        Regularisation strength.

    Returns
    -------
    Tuple[Array, Array]
        Updated (delta, mu).
    """
    delta = jnp.maximum(2.0, 2.0 * delta)
    mu = jnp.maximum(MU_MIN, mu * delta)
    return delta, mu


def decrease_regularisation(delta: Array, mu: Array) -> Tuple[Array, Array]:
    """
    Decrease Levenberg-Marquardt regularisation: δ ← min(1/2, δ/2), μ ← μδ.

    Parameters
    ----------
    delta : Array
        Regularisation scaling factor.
    mu : Array
        Regularisation strength.

    Returns
    -------
    Tuple[Array, Array]
        Updated (delta, mu), with mu set to zero once it falls below ``MU_MIN``.
    """
    delta = jnp.minimum(0.5, delta / 2.0)
    mu = mu * delta
    mu = jnp.where(mu > MU_MIN, mu, 0.0)
    return delta, mu


def stationary(hu_max: Array, dLdU_max: Array, tol: float) -> Array:
    """
    Convergence test of the iLQR solvers: max|dL/dU| and max|hu| below ``tol``, or max|hu| alone where dL/dU
    is not finite.

    Both vanish exactly at a stationary point, but away from it dL/dU = hu + B^T (λ - v) carries hu back
    through the open-loop dynamics: with unstable dynamics a small max|hu| leaves a much larger max|dL/dU|, so
    both are required. Over long unstable horizons the open-loop adjoint λ overflows and dL/dU is never finite;
    there the bounded max|hu| decides alone.

    Parameters
    ----------
    hu_max : Array
        max_t |hu_t| of the unregularised backward pass.
    dLdU_max : Array
        max_t |dL/dU_t| from :func:`kkt_input_residual`.
    tol : float
        Tolerance.

    Returns
    -------
    Array
        Boolean convergence flag.
    """
    return jnp.logical_and(hu_max < tol, jnp.logical_or(dLdU_max < tol, jnp.logical_not(jnp.isfinite(dLdU_max))))


def kkt_input_residual(lqr_lin: LQR) -> Array:
    """
    Gradient of the total non-linear cost wrt the inputs, i.e. the dL/dU KKT residual.

    The adjoint is propagated through the linearised dynamics so that dL/dX = 0 and dL/dλ = 0
    hold by construction along a rollout, leaving dL/dU as the only non-zero KKT condition.

    Parameters
    ----------
    lqr_lin : LQR
        LQR approximation without offsets (from :func:`approx_lqr`), i.e. q=Cx, r=Cu, qf=Cfx.

    Returns
    -------
    Array
        dL/dU residual [Txm].
    """

    def adjoint_step(lamb, inputs):
        A, q = inputs
        return A.T @ lamb + q, lamb

    _, lambs_next = lax.scan(adjoint_step, lqr_lin.qf, (lqr_lin.A, lqr_lin.q), reverse=True)
    return lqr_lin.r + bmm(lqr_lin.B.transpose(0, 2, 1), lambs_next)


def sum_cost_to_go(x: CostToGo) -> Array:
    """
    Sum linear and quadratic components of cost-to-go tuple.

    Parameters
    ----------
    x : CostToGo
        The cost-to-go tuple containing linear and quadratic components.

    Returns
    -------
    Array
        The sum of the linear and quadratic components.
    """
    return x.V + x.v


def compute_lqr_params(
    model: Any,
    Xs: Array,
    Us: Array,
    params: iLQRParams,
    include_dyn_offset: bool = False,
    include_cost_offset: bool = False,
) -> LQR:
    """
    Compute LQR parameters with optional offset for dynamics and cost.

    Parameters
    ----------
    model : System | ParallelSystem
        The system model.
    Xs : Array
        The state trajectory.
    Us : Array
        The control trajectory.
    params : iLQRParams
        The iLQR parameters.
    include_dyn_offset : bool, optional
        Whether to include dynamic offset, by default False.
    include_cost_offset : bool, optional
        Whether to include cost offset, by default False.

    Returns
    -------
    LQR
        The LQR parameters.
    """
    theta = params.theta
    tps = jnp.arange(model.dims.horizon)

    Fx, Fu = time_map(model.lin_dyn)(tps, Xs[:-1], Us, theta)
    Cx, Cu = time_map(model.lin_cost)(tps, Xs[:-1], Us, theta)
    (Cxx, Cxu), (_, Cuu) = time_map(model.quad_cost)(tps, Xs[:-1], Us, theta)
    fCx = jax.jacrev(model.costf)(Xs[-1], theta)
    fCxx = jax.jacfwd(jax.jacrev(model.costf))(Xs[-1], theta)

    def _get_diff_dyn(t, x, u, theta):
        Fx, Fu = model.lin_dyn(t, x, u, theta)
        return model.dynamics(t, x, u, theta) - Fx @ x - Fu @ u

    # the flags are static, so only the selected branch is traced
    if include_dyn_offset:
        f = jax.vmap(_get_diff_dyn, in_axes=(0, 0, 0, None))(tps, Xs[:-1], Us, theta)
    else:
        f = jnp.zeros((model.dims.horizon, model.dims.n), dtype=Xs.dtype)
    if include_cost_offset:
        r = Cu - bmm(Cuu, Us) - bmm(Cxu.transpose(0, 2, 1), Xs[:-1])
        q = Cx - bmm(Cxx, Xs[:-1]) - bmm(Cxu, Us)
        qf = fCx - jnp.matmul(fCxx, Xs[-1])
    else:
        r, q, qf = Cu, Cx, fCx

    return LQR(
        A=Fx,
        B=Fu,
        a=f,
        Q=Cxx,
        q=q,
        R=Cuu,
        r=r,
        S=Cxu,
        Qf=fCxx,
        qf=qf,
    )()


def approx_lqr_offset(model: System | ParallelSystem, Xs: Array, Us: Array, params: iLQRParams) -> LQR:
    """
    Approximate non-linear model as LQR with offset.

    Parameters
    ----------
    model : System | ParallelSystem
        The system model.
    Xs : Array
        The state trajectory.
    Us : Array
        The control trajectory.
    params : iLQRParams
        The iLQR parameters.

    Returns
    -------
    LQR
        The LQR parameters.
    """
    return compute_lqr_params(model, Xs, Us, params, include_dyn_offset=True, include_cost_offset=True)


def approx_lqr(model: System | ParallelSystem, Xs: Array, Us: Array, params: iLQRParams) -> LQR:
    """
    Approximate non-linear model as LQR without offset.

    Parameters
    ----------
    model : System | ParallelSystem
        The system model.
    Xs : Array
        The state trajectory.
    Us : Array
        The control trajectory.
    params : iLQRParams
        The iLQR parameters.

    Returns
    -------
    LQR
        The LQR parameters.
    """
    return compute_lqr_params(model, Xs, Us, params, include_dyn_offset=False, include_cost_offset=False)


def approx_lqr_dyn(model: System | ParallelSystem, Xs: Array, Us: Array, params: iLQRParams) -> LQR:
    """
    Approximate non-linear model as LQR with dynamic offset.

    Parameters
    ----------
    model : System | ParallelSystem
        The system model.
    Xs : Array
        The state trajectory.
    Us : Array
        The control trajectory.
    params : iLQRParams
        The iLQR parameters.

    Returns
    -------
    LQR
        The LQR parameters.
    """
    return compute_lqr_params(model, Xs, Us, params, include_dyn_offset=True, include_cost_offset=False)


def ilqr_simulate(model: System, Us: Array, params: iLQRParams) -> Tuple[Tuple[Array, Array], float]:
    """
    Simulate forward trajectory and cost with nonlinear parameters.

    Parameters
    ----------
    model : System
        The system model.
    Us : Array
        The control trajectory.
    params : iLQRParams
        The iLQR parameters.

    Returns
    -------
    Tuple[Tuple[Array, Array], float]
        A tuple containing the updated state and control trajectory, and the total cost of the
        trajectory.
    """
    x0, theta = params.x0, params.theta
    tps = jnp.arange(model.dims.horizon)

    def fwd_step(state, inputs):
        t, u = inputs
        x, nx_cost = state
        nx = model.dynamics(t, x, u, theta)
        nx_cost = nx_cost + model.cost(t, x, u, theta)
        return (nx, nx_cost), (nx, u)

    (xf, nx_cost), (new_Xs, new_Us) = lax.scan(fwd_step, init=(x0, jnp.zeros((), dtype=x0.dtype)), xs=(tps, Us))
    total_cost = nx_cost + model.costf(xf, theta)
    new_Xs = jnp.vstack([x0[None], new_Xs])
    return (new_Xs, new_Us), total_cost


def ilqr_forward_pass(
    model: System,
    params: iLQRParams,
    Ks: Gains,
    Xs: Array,
    Us: Array,
    alpha: float = 1.0,
) -> Tuple[Tuple[Array, Array], float]:
    """
    Perform a forward pass of the iterative Linear Quadratic Regulator (iLQR) algorithm.

    Parameters
    ----------
    model : System
        The nonlinear system model.
    params : iLQRParams
        The parameters of the system.
    Ks : Gains
        The gains obtained from the LQR controller.
    Xs : Array
        The target state trajectory.
    Us : Array
        The control trajectory.
    alpha : float, optional
        The linesearch parameter, by default 1.0.

    Returns
    -------
    Tuple[Tuple[Array, Array], float]
        A tuple containing the updated state trajectory and control trajectory, and the total cost
        of the trajectory.
    """
    x0, theta = params.x0, params.theta
    tps = jnp.arange(model.dims.horizon)
    x_hat0 = x0

    def fwd_step(state, inputs):
        x_hat, nx_cost = state
        t, x, u, K, k = inputs

        delta_x = x_hat - x
        delta_u = K @ delta_x + alpha * k
        u_hat = u + delta_u
        nx_hat = model.dynamics(t, x_hat, u_hat, theta)
        nx_cost = nx_cost + model.cost(t, x_hat, u_hat, theta)
        return (nx_hat, nx_cost), (nx_hat, u_hat)

    (xf, nx_cost), (new_Xs, new_Us) = lax.scan(
        fwd_step, init=(x_hat0, jnp.zeros((), dtype=x0.dtype)), xs=(tps, Xs[:-1], Us, Ks.K, Ks.k)
    )
    total_cost = nx_cost + model.costf(xf, theta)
    new_Xs = jnp.vstack([x0[None], new_Xs])
    return (new_Xs, new_Us), total_cost


def ilqr_solver(
    model: System,
    params: iLQRParams,
    Us_init: Array,
    max_iter: int = 100,
    convergence_thresh: Optional[float] = None,
    alpha_init: float = 1.0,
    verbose: bool = False,
    use_linesearch: bool = True,
    kkt_tol: float = 1e-10,
    mu_max: float = 1e10,
    max_reg_iter: int = 20,
    loop: str = "while",
    linesearch: str = "backtrack",  # pylint: disable=redefined-outer-name
    return_converged: bool = False,
    return_info: bool = False,
    **linesearch_kwargs,
) -> Tuple[Tuple[Array, Array, Array], float, Array]:
    """
    Solve the iterative Linear Quadratic Regulator (iLQR) problem.

    The non-linear system is approximated to be a local LQR problem. The optimal control gains are
    iteratively added to the x and u perturbations until the KKT conditions are satisfied. The
    input Hessian is regularised with an adaptive Levenberg-Marquardt term until positive definite, and the step size is
    chosen by an Armijo backtracking line search (see :func:`armijo_linesearch`). A rejected
    step increases the regularisation and retries; an accepted step decreases it.

    Parameters
    ----------
    model : System
        The system model.
    params : iLQRParams
        The parameters of the system.
    Us_init : Array
        The initial control trajectory.
    max_iter : int, optional
        The maximum number of iterations, by default 100. The default step schedule (first trial
        α = tau * alpha_init = 0.5) converges linearly at first, so harder problems may need more.
    convergence_thresh : float, optional
        Deprecated and ignored; convergence is determined by ``kkt_tol``.
    alpha_init : float, optional
        Line search scale; the first step tried is ``tau * alpha_init``, by default 1.0.
    verbose : bool, optional
        Whether to print debug information, by default False.
    use_linesearch : bool, optional
        Whether to use line search for the forward pass, by default True. If False, every step
        is taken with α = 1 and a step that increases the cost is rejected, increasing the
        regularisation instead.
    kkt_tol : float, optional
        Converged when max|dL/dU| (:func:`kkt_input_residual`) and max|hu| both fall below this tolerance,
        by default 1e-10. Use ~1e-4 for float32. hu is the gradient of the cost-to-go wrt the inputs from the
        unregularised backward pass (:func:`diffilqrax.lqr.regularised_backward_pass`); it vanishes if and only
        if dL/dU does, and stays finite when the open-loop adjoint of dL/dU overflows (unstable dynamics, long
        horizons), where the test falls back to max|hu| alone (:func:`stationary`).
    mu_max : float, optional
        Stop when the regularisation exceeds this value without an accepted step, by default 1e10.
    max_reg_iter : int, optional
        Maximum number of regularisation increases per backward pass, by default 20.
    loop : str, optional
        How the iterations, regularisation retries and line search are looped (see :func:`bounded_loop`).
        "while" (default) runs only the iterations needed, also under ``vmap``, but is not reverse-mode
        differentiable; forward mode and the implicit gradients of :func:`diffilqrax.diff_ilqr.dilqr` work.
        "scan" runs ``max_iter`` iterations that skip finished work with ``lax.cond``, so reverse mode
        differentiates the unrolled iterations and stores every one of them.
    linesearch : str, optional
        "backtrack" (default) tries the step sizes one at a time. "parallel" rolls out ``n_parallel``
        (default 8) step sizes at once and keeps the largest accepted one, which takes the same step but
        needs fewer sequential rollouts (useful on GPU). Ignored if ``use_linesearch`` is False.
    return_converged : bool, optional
        If True, also return whether the solution passes the convergence test, by default False.
    return_info : bool, optional
        If True, also return the convergence flag (as ``return_converged``) and the number of iterations run,
        by default False. Takes precedence over ``return_converged``.
    **linesearch_kwargs
        Additional keyword arguments for :func:`armijo_linesearch` (tau, beta, alpha_min,
        max_iter_linesearch, n_parallel).

    Returns
    -------
    Tuple[Tuple[Array, Array, Array], float, Array]
        A tuple containing the final state trajectory, control trajectory, and the adjoint
        variables. Also returns the total cost of the trajectory and the cost history [max_iter + 1],
        which repeats the final cost after the last iteration. With ``return_converged=True`` a fourth
        element, the boolean convergence flag, is appended; with ``return_info=True`` the flag and the
        integer number of iterations are appended.

    Notes
    -----
    To differentiate the solution with ``loop="while"``, use :func:`diffilqrax.diff_ilqr.dilqr`. The
    parameters must then enter the problem through ``params`` (e.g. ``params.theta``), not through arrays
    the model closes over, so that no tangents reach the loop.
    """
    assert loop in LOOPS, f"loop must be one of {LOOPS}, got {loop}"
    assert linesearch in LINESEARCHES, f"linesearch must be one of {LINESEARCHES}, got {linesearch}"
    if linesearch == "parallel":
        linesearch_kwargs.setdefault("n_parallel", 8)
    if convergence_thresh is not None:
        warnings.warn(
            "convergence_thresh is deprecated and ignored; use kkt_tol instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    # simulate initial cost
    (Xs_init, _), c_init = ilqr_simulate(model, Us_init, params)
    delta_init = jnp.ones_like(c_init)
    mu_init = jnp.zeros_like(c_init)
    # define initial carry tuple: (Xs, Us, total cost, delta, mu, iteration, cost history, carry_on)
    costs_init = jnp.zeros(max_iter, dtype=jnp.result_type(c_init))
    initial_carry = (Xs_init, Us_init, c_init, delta_init, mu_init, 0, costs_init, jnp.array(True))

    rollout = partial(ilqr_forward_pass, model, params)

    def stationarity(Xs: Array, Us: Array):
        """local LQR, its unregularised backward pass, max|hu|, max|dL/dU| and the convergence test"""
        lqr_params = approx_lqr(model, Xs, Us, params)
        *unregularised, hu_max = lqr.regularised_backward_pass(lqr_params, 0.0, return_grad=True)
        dLdU_max = jnp.max(jnp.abs(kkt_input_residual(lqr_params)))
        return lqr_params, tuple(unregularised), hu_max, dLdU_max, stationary(hu_max, dLdU_max, kkt_tol)

    def backward_with_reg(lqr_params: LQR, delta: Array, mu: Array, unregularised: Tuple):
        """backward pass, increasing mu until Huu_reg is positive definite; reuses the mu = 0 pass"""

        def needs_reg(carry):
            return jnp.logical_and(jnp.logical_not(carry[2]), carry[4] < mu_max)

        def retry(carry):
            _, _, _, delta, mu = carry
            delta, mu = increase_regularisation(delta, mu)
            return (*lqr.regularised_backward_pass(lqr_params, mu), delta, mu)

        first = lax.cond(mu == 0.0, lambda: unregularised, lambda: lqr.regularised_backward_pass(lqr_params, mu))
        return bounded_loop(needs_reg, retry, (*first, delta, mu), max_reg_iter, loop)

    def lqr_iter(carry_tuple):
        """lqr iteration update function"""
        old_Xs, old_Us, old_cost, delta, mu, n_iter, costs, _ = carry_tuple
        # approximate dyn and loss to LQR around current trajectory, and test for convergence
        lqr_params, unregularised, hu_max, dLdU_max, converged = stationarity(old_Xs, old_Us)

        def take_step(_):
            exp_cost_red, gains, _, delta_new, mu_new = backward_with_reg(lqr_params, delta, mu, unregularised)
            if use_linesearch:
                accepted, (new_Xs, new_Us), new_cost = armijo_linesearch(
                    rollout,
                    gains,
                    old_Xs,
                    old_Us,
                    alpha_init,
                    cost_init=old_cost,
                    expected_dJ=exp_cost_red,
                    loop=loop,
                    **linesearch_kwargs,
                )
            else:
                # full step; reject cost increases beyond round-off so mu grows (Levenberg-Marquardt)
                (new_Xs, new_Us), new_cost = rollout(gains, old_Xs, old_Us, alpha=1.0)
                noise = 100.0 * jnp.finfo(jnp.result_type(old_cost)).eps * jnp.maximum(jnp.abs(old_cost), 1.0)
                accepted = jnp.logical_and(jnp.isfinite(new_cost), old_cost - new_cost > -noise)
            delta_new, mu_new = lax.cond(accepted, decrease_regularisation, increase_regularisation, delta_new, mu_new)
            new_Xs = jnp.where(accepted, new_Xs, old_Xs)
            new_Us = jnp.where(accepted, new_Us, old_Us)
            new_cost = jnp.where(accepted, new_cost, old_cost)
            # give up once no step is accepted even with maximal regularisation
            stuck = jnp.logical_and(jnp.logical_not(accepted), mu_new >= mu_max)
            if verbose:
                jax.debug.print(
                    "it={i} J={j:.12e} max|dLdU|={g:.3e} max|hu|={h:.3e} mu={m:.1e} accepted={a}",
                    i=n_iter + 1,
                    j=new_cost,
                    g=dLdU_max,
                    h=hu_max,
                    m=mu_new,
                    a=accepted,
                )
            costs_new = costs.at[n_iter].set(new_cost)
            return (new_Xs, new_Us, new_cost, delta_new, mu_new, n_iter + 1, costs_new, jnp.logical_not(stuck))

        def stop(_):
            return (old_Xs, old_Us, old_cost, delta, mu, n_iter, costs, jnp.array(False))

        return lax.cond(converged, stop, take_step, None)

    Xs_star, Us_star, total_cost, _, _, n_iters, costs, _ = bounded_loop(
        lambda c: c[-1], lqr_iter, initial_carry, max_iter, loop
    )
    # the history repeats the final cost after the last iteration
    costs = jnp.where(jnp.arange(max_iter) < n_iters, costs, total_cost)
    if verbose:
        jax.debug.print("Ran {n}/{N} iterations, cost: {c}", n=n_iters, N=max_iter, c=total_cost)
    lqr_params_stars = approx_lqr_offset(model, Xs_star, Us_star, params)
    Lambs_star = lqr.lqr_adjoint_pass(Xs_star, Us_star, LQRParams(Xs_star[0], lqr_params_stars))
    out = ((Xs_star, Us_star, Lambs_star), total_cost, jnp.concatenate([jnp.array([c_init]), costs]))
    if not (return_converged or return_info):
        return out
    converged = stationarity(Xs_star, Us_star)[-1]
    return (*out, converged, n_iters) if return_info else (*out, converged)


def armijo_linesearch(
    update: Callable,
    Ks: Gains,
    Xs_init: Array,
    Us_init: Array,
    alpha_init: float,
    cost_init: float,
    expected_dJ: CostToGo,
    tau: float = 0.5,
    beta: float = 0.1,
    alpha_min: float = 1e-8,
    max_iter_linesearch: Optional[int] = None,
    loop: str = "while",
    n_parallel: int = 0,
) -> Tuple[Array, Tuple[Array, Array], Array]:
    """
    Armijo backtracking line search.

    Step sizes α = tau * alpha_init, tau^2 * alpha_init, ... are tried until the cost reduction
    exceeds ``beta`` times the expected reduction, or α drops below ``alpha_min``. The expected
    reduction is the textbook quadratic model ΔJ(α) = -(α k^T hu + ½ α² k^T Huu k)
    (:func:`diffilqrax.lqr.calc_expected_change`). When the expected reduction is below the
    floating point resolution of the cost the comparison is meaningless, so the step is accepted
    provided the cost does not increase beyond that resolution.

    References
    ----------
    Tassa, Y., Erez, T., & Todorov, E. (2012).
    Synthesis and stabilization of complex behaviors through online trajectory optimization.
    IEEE. https://doi.org/10.1109/iros.2012.6386025

    Parameters
    ----------
    update : Callable
        Rollout function which returns new Xs, Us and cost.
    Ks : Gains
        Gains obtained from the LQR controller.
    Xs_init : Array
        State trajectory.
    Us_init : Array
        Input trajectory.
    alpha_init : float
        Line search scale; the first step tried is ``tau * alpha_init``.
    cost_init : float
        Cost of initial trajectory.
    expected_dJ : CostToGo
        Expected change in cost terms from the LQR backward pass.
    tau : float, optional
        Reduction factor for alpha, by default 0.5.
    beta : float, optional
        Armijo sufficient decrease constant, by default 0.1.
    alpha_min : float, optional
        Minimum alpha value, by default 1e-8.
    max_iter_linesearch : int, optional
        Maximum number of rollouts. Defaults to the number needed to reach ``alpha_min``.
    loop : str, optional
        "while" (default) or "scan", see :func:`bounded_loop`.
    n_parallel : int, optional
        Number of step sizes rolled out at once, by default 0 (one at a time). The accepted step is the same
        either way (see :func:`backtracking_search`); parallel rollouts suit GPUs, at the price of up to
        ``n_parallel`` rollouts per call.

    Returns
    -------
    Tuple[Array, Tuple[Array, Array], Array]
        Whether a step was accepted, the (accepted or initial) state and input trajectories, and
        the corresponding cost.
    """
    if max_iter_linesearch is None:
        max_iter_linesearch = int(math.floor(math.log(alpha_min / (tau * alpha_init)) / math.log(tau))) + 2
    noise = 100.0 * jnp.finfo(jnp.result_type(cost_init)).eps * jnp.maximum(jnp.abs(cost_init), 1.0)

    def trial(alpha):
        """Rollout with step size alpha; accept on sufficient decrease"""
        (new_Xs, new_Us), new_cost = update(Ks, Xs_init, Us_init, alpha=alpha)
        expected_red = lqr.calc_expected_change(expected_dJ, alpha=alpha)
        actual_red = cost_init - new_cost
        below_noise = jnp.logical_and(expected_red < noise, actual_red > -noise)
        accepted = jnp.logical_and(jnp.isfinite(new_cost), jnp.logical_or(actual_red > beta * expected_red, below_noise))
        return accepted, new_Xs, new_Us, new_cost

    return backtracking_search(
        trial, (Xs_init, Us_init, cost_init), tau * alpha_init, max_iter_linesearch, tau, alpha_min, loop, n_parallel
    )


def linesearch(
    update: Callable,
    Ks: Gains,
    Xs_init: Array,
    Us_init: Array,
    alpha_init: float,
    cost_init: float,
    expected_dJ: CostToGo,
    beta: float = 0.5,
    max_iter_linesearch: int = 12,
    tol: float = 0.1,
    alpha_min: float = 1e-6,
    verbose: bool = False,
) -> Tuple[Tuple[Array, Array], float, float, Array]:
    """
    Implementation of the line search backtracking algorithm for iLQR algorithm.

    Each iteration of the line search algorithm performs a forward pass with the new control inputs
    defined by the gains and an alpha value. The change in cost is compared to the expected change
    in cost and the alpha is selected based on the ratio of the two values above a given tolerance.
    Otherwise, the alpha value is reduced by a factor of beta.

    References
    ----------
    Tassa, Y., Erez, T., & Todorov, E. (2012).
    Synthesis and stabilization of complex behaviors through online trajectory optimization.
    IEEE. https://doi.org/10.1109/iros.2012.6386025

    Parameters
    ----------
    update : Callable
        Rollout function which returns new Xs, Us and cost.
    Ks : Gains
        Gains obtained from the LQR controller.
    Xs_init : Array
        State trajectory.
    Us_init : Array
        Input trajectory.
    alpha_init : float
        Initial alpha value.
    cost_init : float
        Cost of initial trajectory.
    expected_dJ : CostToGo
        Expected change in cost from LQR controller.
    beta : float, optional
        Reduction factor for alpha, by default 0.5.
    max_iter_linesearch : int, optional
        Maximum iterations of linesearch, by default 12.
    tol : float, optional
        Tolerance of ratio of actual to expected cost change to accept alpha value, by default 0.1.
    alpha_min : float, optional
        Minimum alpha value, by default 1e-6.
    verbose : bool, optional
        Whether to print debug information, by default False.

    Returns
    -------
    Tuple[Tuple[Array, Array], float, float, Array]
        Returns the updated state and control trajectories, the alpha value, the total cost, and
        the cost history.
    """
    # initialise carry: Xs, Us, old ilqr cost, alpha, n_iter, carry_on
    initial_carry = (Xs_init, Us_init, 0.0, cost_init, alpha_init, 0, 10.0, 0.0, True)
    # jax.debug.print(f"\nLinesearch: J0={cost_init:.03f} α={alpha_init:.03f} β={beta:.03f}")

    def backtrack_iter(carry):
        """Rollout with new alpha and update alpha if z-value is above threshold"""
        # parse out carry
        Xs, Us, new_cost, old_cost, alpha, n_iter, _, _, carry_on = carry
        # print(alpha)
        # rollout with alpha
        (new_Xs, new_Us), new_cost = update(Ks, Xs, Us, alpha=alpha)

        # calc expected cost reduction
        expected_delta_j = lqr.calc_expected_change(expected_dJ, alpha=alpha)
        # calc z-value
        z = (old_cost - new_cost) / jnp.abs(expected_delta_j)
        ## Note : so here I think we want the absolute value of the expected dJ
        # (because we are doing old - new, and
        # so that will be positive hopefully, and we want to check that the magnitude
        # of the change is larger than some scaled version of the expected change

        if verbose:
            jax.debug.print(
                f"it={1+n_iter:02} α={alpha:.03f} z:{z:.03f} pJ:{old_cost:.03f}",
                f"nJ:{new_cost:.03f} ΔJ:{old_cost-new_cost:.03f} <ΔJ>:{expected_delta_j:.03f}",
            )
            jax.debug.print(f"{alpha} < {alpha_min} {jnp.float64(alpha-alpha_min)}")

        # ensure to keep Xs and Us that reduce z-value
        new_cost = jnp.where(jnp.isnan(new_cost), cost_init, new_cost)
        # add control flow to carry on or not
        above_threshold = z > tol
        carry_on = lax.bitwise_not(jnp.logical_or(alpha < alpha_min, above_threshold))

        # print(f"{alpha} < {alpha_min} {jnp.float64(alpha-alpha_min)}")
        # assert_scalar_positive(alpha-alpha_min)
        # assert_scalar_positive(max_iter_linesearch-1-n_iter)
        # Only return new trajs if leads to a strict cost decrease
        new_Xs = jnp.where(above_threshold, new_Xs, Xs)
        new_Us = jnp.where(above_threshold, new_Us, Us)
        new_cost = jnp.where(above_threshold, new_cost, old_cost)
        # update alpha
        alpha *= beta

        return (
            new_Xs,
            new_Us,
            new_cost,
            old_cost,
            alpha,
            n_iter + 1,
            z,
            expected_delta_j,
            carry_on,
        )

    def loop_fun(carry_tuple: Tuple[Array, Array, float, float, int, bool], _):
        """if cond false return existing carry else run another rollout with new alpha"""
        # assign function given carry_on condition
        updated_carry = lax.cond(carry_tuple[-1], backtrack_iter, lambda x: x, carry_tuple)
        return updated_carry, (updated_carry[2], updated_carry[3])

    # scan through with max iterations
    (Xs_star, Us_star, cost_opt, old_cost, alpha, its, z, exp_dj, *_), costs = lax.scan(
        loop_fun, initial_carry, None, length=max_iter_linesearch
    )
    # if verbose:
    # jax.debug.print(
    # f"Nit:{its:02} α:{alpha/beta:.03f} z:{z:.03f} J*:{cost_opt:.03f}",
    # f"ΔJ:{cost_init-cost_opt:.03f} <ΔJ>:{exp_dj:.03f}"
    # )
    # assert old_cost < cost_opt
    # assert jax.device_get(old_cost) < jax.device_get(cost_opt)
    # lax.cond(old_cost > cost_opt, lambda : True, lambda : False)
    # chex.assert_scalar_negative(jax.device_get(old_cost) - jax.device_get(cost_opt))
    # #, "Cost did not decrease"
    return (Xs_star, Us_star), cost_opt
