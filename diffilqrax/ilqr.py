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


def increase_regularisation(delta: Array, mu: Array) -> Tuple[Array, Array]:
    """
    Increase Levenberg-Marquardt regularisation, as in OCaml ``Regularisation.increase``.

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
    Decrease Levenberg-Marquardt regularisation, as in OCaml ``Regularisation.decrease``.

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

    (Fx, Fu) = time_map(model.lin_dyn)(tps, Xs[:-1], Us, theta)
    (Cx, Cu) = time_map(model.lin_cost)(tps, Xs[:-1], Us, theta)
    (Cxx, Cxu), (_, Cuu) = time_map(model.quad_cost)(tps, Xs[:-1], Us, theta)
    fCx = jax.jacrev(model.costf)(Xs[-1], theta)
    fCxx = jax.jacfwd(jax.jacrev(model.costf))(Xs[-1], theta)

    def _get_diff_dyn(t, x, u, theta):
        (Fx, Fu) = model.lin_dyn(t, x, u, theta)
        return model.dynamics(t, x, u, theta) - Fx @ x - Fu @ u

    def _compute_f_with_offset(tps, Xs, Us, theta):
        return jax.vmap(_get_diff_dyn, in_axes=(0, 0, 0, None))(tps, Xs[:-1], Us, theta)

    def _compute_f_without_offset(tps, Xs, Us, theta):
        return jnp.zeros((model.dims.horizon, model.dims.n), dtype=Xs.dtype)

    def _compute_q_with_offset(c_lin, c_quad, c_mixed, Xs, Us):
        return c_lin - bmm(c_quad, Xs) - bmm(c_mixed, Us)

    def _compute_qf_with_offset(c_lin, c_quad, Xs):
        return c_lin - jnp.matmul(c_quad, Xs)

    f = lax.cond(
        include_dyn_offset,
        _compute_f_with_offset,
        _compute_f_without_offset,
        tps,
        Xs,
        Us,
        theta,
    )
    r = lax.cond(
        include_cost_offset,
        lambda x: _compute_q_with_offset(*x),
        lambda x: x[0],
        (Cu, Cuu, Cxu.transpose(0, 2, 1), Us, Xs[:-1]),
    )
    q = lax.cond(
        include_cost_offset,
        lambda x: _compute_q_with_offset(*x),
        lambda x: x[0],
        (Cx, Cxx, Cxu, Xs[:-1], Us),
    )
    qf = lax.cond(
        include_cost_offset,
        lambda x: _compute_qf_with_offset(*x),
        lambda x: x[0],
        (fCx, fCxx, Xs[-1]),
    )

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


def approx_lqr_offset(
    model: System | ParallelSystem, Xs: Array, Us: Array, params: iLQRParams
) -> LQR:
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
    return compute_lqr_params(
        model, Xs, Us, params, include_dyn_offset=True, include_cost_offset=True
    )


def approx_lqr(
    model: System | ParallelSystem, Xs: Array, Us: Array, params: iLQRParams
) -> LQR:
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
    return compute_lqr_params(
        model, Xs, Us, params, include_dyn_offset=False, include_cost_offset=False
    )


def approx_lqr_dyn(
    model: System | ParallelSystem, Xs: Array, Us: Array, params: iLQRParams
) -> LQR:
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
    return compute_lqr_params(
        model, Xs, Us, params, include_dyn_offset=True, include_cost_offset=False
    )


def ilqr_simulate(
    model: System, Us: Array, params: iLQRParams
) -> Tuple[Tuple[Array, Array], float]:
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
    max_iter: int = 40,
    convergence_thresh: Optional[float] = None,
    alpha_init: float = 1.0,
    verbose: bool = False,
    use_linesearch: bool = True,
    kkt_tol: float = 1e-10,
    mu_max: float = 1e10,
    max_reg_iter: int = 20,
    **linesearch_kwargs,
) -> Tuple[Tuple[Array, Array, Array], float, Array]:
    """
    Solve the iterative Linear Quadratic Regulator (iLQR) problem.

    The non-linear system is approximated to be a local LQR problem. The optimal control gains are
    iteratively added to the x and u perturbations until the KKT conditions are satisfied. The
    control flow follows the OCaml ``dilqr`` implementation: the input Hessian is regularised
    with an adaptive Levenberg-Marquardt term until positive definite, and the step size is
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
        The maximum number of iterations, by default 40.
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
        Converged when max|dL/dU| falls below this tolerance, by default 1e-10. Use ~1e-4 for
        float32.
    mu_max : float, optional
        Stop when the regularisation exceeds this value without an accepted step, by default 1e10.
    max_reg_iter : int, optional
        Maximum number of regularisation increases per backward pass, by default 20.
    **linesearch_kwargs
        Additional keyword arguments for :func:`armijo_linesearch` (tau, beta, alpha_min,
        max_iter_linesearch).

    Returns
    -------
    Tuple[Tuple[Array, Array, Array], float, Array]
        A tuple containing the final state trajectory, control trajectory, and the adjoint
        variables. Also returns the total cost of the trajectory and the cost history.
    """
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
    # define initial carry tuple: (Xs, Us, total cost, delta, mu, iteration, carry_on)
    initial_carry = (Xs_init, Us_init, c_init, delta_init, mu_init, 0, True)

    rollout = partial(ilqr_forward_pass, model, params)

    def backward_with_reg(lqr_params: LQR, delta: Array, mu: Array):
        """backward pass, increasing mu until Huu_reg is positive definite (OCaml lqr.ml)"""
        dJ, gains, is_pd = lqr.regularised_backward_pass(lqr_params, mu)

        def reg_step(carry, _):
            def retry(carry):
                _, _, _, delta, mu = carry
                delta, mu = increase_regularisation(delta, mu)
                return (*lqr.regularised_backward_pass(lqr_params, mu), delta, mu)

            needs_reg = jnp.logical_and(jnp.logical_not(carry[2]), carry[4] < mu_max)
            return lax.cond(needs_reg, retry, lambda c: c, carry), None

        return lax.scan(reg_step, (dJ, gains, is_pd, delta, mu), None, length=max_reg_iter)[0]

    def lqr_iter(carry_tuple):
        """lqr iteration update function"""
        old_Xs, old_Us, old_cost, delta, mu, n_iter, _ = carry_tuple
        # approximate dyn and loss to LQR around current trajectory
        lqr_params = approx_lqr(model, old_Xs, old_Us, params)
        kkt_res = jnp.max(jnp.abs(kkt_input_residual(lqr_params)))
        converged = kkt_res < kkt_tol

        def take_step(_):
            exp_cost_red, gains, _, delta_new, mu_new = backward_with_reg(lqr_params, delta, mu)
            if use_linesearch:
                accepted, (new_Xs, new_Us), new_cost = armijo_linesearch(
                    rollout,
                    gains,
                    old_Xs,
                    old_Us,
                    alpha_init,
                    cost_init=old_cost,
                    expected_dJ=exp_cost_red,
                    **linesearch_kwargs,
                )
            else:
                # full step; reject cost increases beyond round-off so mu grows (Levenberg-Marquardt)
                (new_Xs, new_Us), new_cost = rollout(gains, old_Xs, old_Us, alpha=1.0)
                noise = 100.0 * jnp.finfo(jnp.result_type(old_cost)).eps * jnp.maximum(jnp.abs(old_cost), 1.0)
                accepted = jnp.logical_and(jnp.isfinite(new_cost), old_cost - new_cost > -noise)
            delta_new, mu_new = lax.cond(
                accepted, decrease_regularisation, increase_regularisation, delta_new, mu_new
            )
            new_Xs = jnp.where(accepted, new_Xs, old_Xs)
            new_Us = jnp.where(accepted, new_Us, old_Us)
            new_cost = jnp.where(accepted, new_cost, old_cost)
            # give up once no step is accepted even with maximal regularisation
            stuck = jnp.logical_and(jnp.logical_not(accepted), mu_new >= mu_max)
            if verbose:
                jax.debug.print(
                    "it={i} J={j:.12e} max|dLdU|={g:.3e} mu={m:.1e} accepted={a}",
                    i=n_iter + 1, j=new_cost, g=kkt_res, m=mu_new, a=accepted,
                )
            return (new_Xs, new_Us, new_cost, delta_new, mu_new, n_iter + 1, jnp.logical_not(stuck))

        def stop(_):
            return (old_Xs, old_Us, old_cost, delta, mu, n_iter, jnp.array(False))

        return lax.cond(converged, stop, take_step, None)

    def loop_fun(carry_tuple, _):
        """if cond false return existing carry else run another iteration of lqr_iter"""
        updated_carry = lax.cond(carry_tuple[-1], lqr_iter, lambda x: x, carry_tuple)
        return updated_carry, updated_carry[2]

    # scan through with max iterations
    (Xs_star, Us_star, total_cost, *_, n_iters, _), costs = lax.scan(
        loop_fun, initial_carry, None, length=max_iter
    )
    if verbose:
        jax.debug.print("Ran {n}/{N} iterations, cost: {c}", n=n_iters, N=max_iter, c=total_cost)
    lqr_params_stars = approx_lqr_offset(model, Xs_star, Us_star, params)
    Lambs_star = lqr.lqr_adjoint_pass(Xs_star, Us_star, LQRParams(Xs_star[0], lqr_params_stars))
    return (
        (Xs_star, Us_star, Lambs_star),
        total_cost,
        jnp.concatenate([jnp.array([c_init]), costs]),
    )


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
) -> Tuple[Array, Tuple[Array, Array], Array]:
    """
    Armijo backtracking line search, following OCaml ``Linesearch.backtrack``.

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

    Returns
    -------
    Tuple[Array, Tuple[Array, Array], Array]
        Whether a step was accepted, the (accepted or initial) state and input trajectories, and
        the corresponding cost.
    """
    if max_iter_linesearch is None:
        max_iter_linesearch = int(math.floor(math.log(alpha_min / (tau * alpha_init)) / math.log(tau))) + 2
    noise = 100.0 * jnp.finfo(jnp.result_type(cost_init)).eps * jnp.maximum(jnp.abs(cost_init), 1.0)
    initial_carry = (Xs_init, Us_init, cost_init, tau * alpha_init, jnp.array(False), jnp.array(True))

    def backtrack_iter(carry):
        """Rollout with current alpha; accept on sufficient decrease else shrink alpha"""
        Xs, Us, cost, alpha, _, _ = carry
        (new_Xs, new_Us), new_cost = update(Ks, Xs_init, Us_init, alpha=alpha)
        expected_red = lqr.calc_expected_change(expected_dJ, alpha=alpha)
        actual_red = cost_init - new_cost
        below_noise = jnp.logical_and(expected_red < noise, actual_red > -noise)
        accepted = jnp.logical_and(
            jnp.isfinite(new_cost), jnp.logical_or(actual_red > beta * expected_red, below_noise)
        )
        carry_on = jnp.logical_and(jnp.logical_not(accepted), alpha >= alpha_min)
        return (
            jnp.where(accepted, new_Xs, Xs),
            jnp.where(accepted, new_Us, Us),
            jnp.where(accepted, new_cost, cost),
            alpha * tau,
            accepted,
            carry_on,
        )

    def loop_fun(carry, _):
        """if cond false return existing carry else run another rollout with new alpha"""
        return lax.cond(carry[-1], backtrack_iter, lambda x: x, carry), None

    (Xs_star, Us_star, cost_star, _, accepted, _), _ = lax.scan(
        loop_fun, initial_carry, None, length=max_iter_linesearch
    )
    return accepted, (Xs_star, Us_star), cost_star


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
        updated_carry = lax.cond(
            carry_tuple[-1], backtrack_iter, lambda x: x, carry_tuple
        )
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
