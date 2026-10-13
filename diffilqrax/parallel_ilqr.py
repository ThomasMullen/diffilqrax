"""Solve iLQR with parallel dynamics using associative scans"""

import warnings
from typing import Tuple, Any, Optional
from functools import partial
import jax
from jax import Array
from jax import lax
import jax.numpy as jnp

from diffilqrax.lqr import lqr_adjoint_pass
from diffilqrax.lqr import bmm
from diffilqrax.plqr import (
    build_fwd_lin_dyn_elements,
    closed_loop_scan,
    dynamic_operator,
    plqr_backward_pass,
)
from diffilqrax.ilqr import (
    approx_lqr,
    armijo_linesearch,
    approx_lqr_dyn,
    kkt_input_residual,
    increase_regularisation,
    decrease_regularisation,
)
from diffilqrax.typs import (
    iLQRParams,
    System,
    Gains,
    LQRParams,
    ParallelSystem,
)

jax.config.update("jax_enable_x64", True)  # double precision


def parallel_forward_lin_integration_ilqr(model: System, params: iLQRParams, Us_init: Array, a_term: Array) -> Array:
    """
    Perform associative scan for forward linear dynamics.

    Parameters
    ----------
    model : System
        The system model.
    params : iLQRParams
        The iLQR parameters.
    Us_init : Array
        The initial control sequence.
    a_term : Array
        The affine term for the dynamics.

    Returns
    -------
    Array
        The state trajectory.
    """
    x0 = params.x0
    lqr_params = approx_lqr(
        model,
        jnp.r_[x0[None, ...], jnp.zeros((Us_init.shape[0], x0.shape[0]))],
        Us_init,
        params,
    )
    dyn_elements = build_fwd_lin_dyn_elements(LQRParams(x0, lqr_params), Us_init, a_term)
    c_as, c_bs = jax.lax.associative_scan(dynamic_operator, dyn_elements)
    return c_bs


def parallel_feedback_lin_dyn_ilqr(model: System, params: iLQRParams, Us_init: Array, a_term: Array, Kx: Array) -> Array:
    """
    Perform associative scan for forward linear dynamics with feedback.

    Parameters
    ----------
    model : System
        The system model.
    params : iLQRParams
        The iLQR parameters.
    Us_init : Array
        The initial control sequence.
    a_term : Array
        The affine term for the dynamics.
    Kx : Array
        The feedback gain matrix.

    Returns
    -------
    Array
        The state trajectory.
    """
    lqr_params = approx_lqr(
        model,
        jnp.r_[params.x0[None], jnp.zeros((model.dims.horizon, model.dims.n))],
        Us_init,
        params,
    )
    lqr_params = lqr_params._replace(A=lqr_params.A - bmm(lqr_params.B, Kx))
    dyn_elements = build_fwd_lin_dyn_elements(LQRParams(params.x0, lqr_params), Us_init, a_term)
    c_as, c_bs = jax.lax.associative_scan(dynamic_operator, dyn_elements)
    return c_bs


def pilqr_forward_pass(
    parallel_model: ParallelSystem | Any,
    params: iLQRParams,
    gains: Gains,
    Xs: Array,
    Us: Array,
    alpha: float = 1.0,
) -> Tuple[Tuple[Array, Array], float]:
    """
    Perform a forward pass of the parallel iLQR algorithm.

    Parameters
    ----------
    parallel_model : ParallelSystem | Any
        The parallel system model.
    params : iLQRParams
        The iLQR parameters.
    gains : Gains
        Feedback gains (u = Kx + k) of the local LQR problem, from :func:`plqr_backward_pass`.
    Xs : Array
        The state trajectory.
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
    model = parallel_model.model

    lqr_mats = approx_lqr_dyn(model, Xs, Us, params)
    dyn_bias = lqr_mats.a
    # this is the model from delta_x, so delta_x0 = 0
    lqr_model = LQRParams(x0=jnp.zeros_like(params.x0), lqr=lqr_mats._replace(a=jnp.zeros_like(dyn_bias)))
    cs = closed_loop_scan(lqr_model, gains, alpha)
    # feedback convention of parallel_dynamics_feedback is u - Ks @ x
    Ks, ks = -gains.K, alpha * gains.k
    delta_Xs = jnp.r_[jnp.zeros_like(params.x0)[None], cs]
    delta_Us = ks - bmm(Ks, delta_Xs[:-1])
    Kxxs = bmm(Ks, Xs[:-1]) + ks
    # NOTE: in the case of a linear system, equivalent to `parallel_feedback_lin_dyn_ilqr`
    new_Xs = parallel_model.parallel_dynamics_feedback(model, params, Us + Kxxs, dyn_bias, Ks)
    # this should define the dynamics incorporating the feedback term that says how to handle
    # delta_X (current state - initial traj) we assume Ks@initial_traj is already passed as
    # input so only care about the current state and the parallel_dynamics_feedback function
    # should define how to handle that
    new_Us = Us + delta_Us
    # new_Xs = parallel_model.parallel_dynamics(model, params,  new_Us,  lqr_model_with_a.lqr.a)
    total_cost = jnp.sum(
        jax.vmap(model.cost, in_axes=(0, 0, 0, None))(jnp.arange(model.dims.horizon), new_Xs[:-1], new_Us, params.theta)
    )
    total_cost += model.costf(new_Xs[-1], params.theta)
    return (new_Xs, new_Us), total_cost


def pilqr_solver(
    parallel_model: ParallelSystem,
    params: iLQRParams,
    Us_init: Array,
    max_iter: int = 40,
    convergence_thresh: Optional[float] = None,
    alpha_init: float = 1.0,
    verbose: bool = False,
    use_linesearch: bool = True,
    regularise: bool = True,
    kkt_tol: float = 1e-10,
    mu_max: float = 1e10,
    max_reg_iter: int = 20,
    **linesearch_kwargs,
) -> Tuple[Tuple[Array, Array, Array], float, Array]:
    """
    Solve the parallel iterative Linear Quadratic Regulator (iLQR) problem.

    The backward pass is an associative Riccati scan of the local LQR problem regularised with
    R -> R + μI (:func:`diffilqrax.plqr.plqr_backward_pass`), which keeps it an exact LQR so the scan
    applies. μ is adapted as in :func:`diffilqrax.ilqr.ilqr_solver`: it is increased until every Huu is
    positive definite, increased after a rejected step and decreased after an accepted one.

    Parameters
    ----------
    parallel_model : ParallelSystem
        The parallel system model.
    params : iLQRParams
        The iLQR parameters.
    Us_init : Array
        The initial control trajectory.
    max_iter : int, optional
        The maximum number of iterations, by default 40.
    convergence_thresh : float, optional
        If given, also stop once an accepted step reduces the cost by less than this relative amount.
        By default None, so convergence is determined by ``kkt_tol``.
    alpha_init : float, optional
        The initial step size for the forward pass, by default 1.0.
    verbose : bool, optional
        Whether to print debug information, by default False.
    use_linesearch : bool, optional
        Whether to use the Armijo line search (:func:`diffilqrax.ilqr.armijo_linesearch`) for the
        forward pass, by default True. If False, every step is taken with α = 1 and a step that
        increases the cost beyond round-off is rejected.
    regularise : bool, optional
        Whether to adapt the regularisation μ, by default True. If False, μ = 0 and the solver stops
        at the first step that does not decrease the cost.
    kkt_tol : float, optional
        Converged when max|dL/dU| falls below this tolerance, by default 1e-10.
    mu_max : float, optional
        Stop when the regularisation exceeds this value without an accepted step, by default 1e10.
    max_reg_iter : int, optional
        Maximum number of regularisation increases per backward pass, by default 20.
    **linesearch_kwargs
        Additional keyword arguments for :func:`diffilqrax.ilqr.armijo_linesearch` (tau, beta,
        alpha_min, max_iter_linesearch).

    Returns
    -------
    Tuple[Tuple[Array, Array, Array], float, Array]
        A tuple containing the final state trajectory, control trajectory, and the adjoint
        variables. Also returns the total cost of the trajectory and the cost history.
    """
    if "tol" in linesearch_kwargs:
        warnings.warn(
            "pilqr_solver now uses armijo_linesearch: the ratio-test kwargs (beta: step shrink factor, tol: "
            "accepted fraction of the expected decrease) are mapped to tau and beta; pass those instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy = {"beta": "tau", "tol": "beta"}
        linesearch_kwargs = {legacy.get(k, k): v for k, v in linesearch_kwargs.items()}
    model = parallel_model.model
    Xs_init = parallel_model.parallel_dynamics(model, params, Us_init, jnp.zeros_like(Us_init[..., 0]))

    a_term = approx_lqr_dyn(parallel_model.model, Xs_init, Us_init, params).a
    Xs_init = parallel_model.parallel_dynamics(model, params, Us_init, a_term)
    c_init = jnp.sum(
        jax.vmap(model.cost, in_axes=(0, 0, 0, None))(jnp.arange(model.dims.horizon), Xs_init[:-1], Us_init, params.theta)
    )
    c_init += model.costf(Xs_init[-1], params.theta)
    # carry: (Xs, Us, total cost, delta, mu, iteration, carry_on)
    initial_carry = (Xs_init, Us_init, c_init, jnp.ones_like(c_init), jnp.zeros_like(c_init), 0, True)
    prollout = partial(pilqr_forward_pass, parallel_model, params)

    def backward_with_reg(lqr_params: LQRParams, delta: Array, mu: Array):
        """parallel backward pass, increasing mu until Huu_reg is positive definite"""
        dJ, gains, is_pd = plqr_backward_pass(lqr_params, mu)
        if not regularise:
            return dJ, gains, is_pd, delta, mu

        def reg_step(carry, _):
            def retry(carry):
                _, _, _, delta, mu = carry
                delta, mu = increase_regularisation(delta, mu)
                return (*plqr_backward_pass(lqr_params, mu), delta, mu)

            needs_reg = jnp.logical_and(jnp.logical_not(carry[2]), carry[4] < mu_max)
            return lax.cond(needs_reg, retry, lambda c: c, carry), None

        return lax.scan(reg_step, (dJ, gains, is_pd, delta, mu), None, length=max_reg_iter)[0]

    def plqr_iter(carry_tuple):
        """One parallel iLQR iteration: backward scan, line search, regularisation update"""
        old_Xs, old_Us, old_cost, delta, mu, n_iter, _ = carry_tuple
        lqr = approx_lqr(model, old_Xs, old_Us, params)
        kkt_res = jnp.max(jnp.abs(kkt_input_residual(lqr)))
        converged = kkt_res < kkt_tol

        def take_step(_):
            exp_dJ, gains, _, delta_new, mu_new = backward_with_reg(LQRParams(params.x0, lqr), delta, mu)

            if use_linesearch:
                accepted, (new_Xs, new_Us), new_cost = armijo_linesearch(
                    prollout,
                    gains,
                    old_Xs,
                    old_Us,
                    alpha_init,
                    cost_init=old_cost,
                    expected_dJ=exp_dJ,
                    **linesearch_kwargs,
                )
            else:
                # full step; reject cost increases beyond round-off
                (new_Xs, new_Us), new_cost = prollout(gains, old_Xs, old_Us, alpha=1.0)
                noise = 100.0 * jnp.finfo(jnp.result_type(old_cost)).eps * jnp.maximum(jnp.abs(old_cost), 1.0)
                accepted = jnp.logical_and(jnp.isfinite(new_cost), old_cost - new_cost > -noise)
            if regularise:
                delta_new, mu_new = lax.cond(accepted, decrease_regularisation, increase_regularisation, delta_new, mu_new)
                stuck = jnp.logical_and(jnp.logical_not(accepted), mu_new >= mu_max)
            else:
                stuck = jnp.logical_not(accepted)
            new_Xs = jnp.where(accepted, new_Xs, old_Xs)
            new_Us = jnp.where(accepted, new_Us, old_Us)
            new_cost = jnp.where(accepted, new_cost, old_cost)
            stop = stuck
            if convergence_thresh is not None:
                small = (old_cost - new_cost) / jnp.abs(old_cost) < convergence_thresh
                stop = jnp.logical_or(stop, jnp.logical_and(accepted, small))
            if verbose:
                jax.debug.print(
                    "it={i} J={j:.12e} max|dLdU|={g:.3e} mu={m:.1e} accepted={a}",
                    i=n_iter + 1,
                    j=new_cost,
                    g=kkt_res,
                    m=mu_new,
                    a=accepted,
                )
            return (new_Xs, new_Us, new_cost, delta_new, mu_new, n_iter + 1, jnp.logical_not(stop))

        def stop(_):
            return (old_Xs, old_Us, old_cost, delta, mu, n_iter, jnp.array(False))

        return lax.cond(converged, stop, take_step, None)

    def loop_fun(carry_tuple, _):
        """if cond false return existing carry else run another iteration of plqr_iter"""
        updated_carry = lax.cond(carry_tuple[-1], plqr_iter, lambda x: x, carry_tuple)
        return updated_carry, updated_carry[2]

    # scan through with max iterations
    (Xs_star, Us_star, total_cost, *_, n_iters, _), costs = lax.scan(loop_fun, initial_carry, None, length=max_iter)
    if verbose:
        jax.debug.print("Ran {n}/{N} iterations, cost: {c}", n=n_iters, N=max_iter, c=total_cost)
    lqr_params_stars = approx_lqr(model, Xs_star, Us_star, params)
    # TODO : Not sure if this is needed - otherwise should solve with Vs and vs
    Lambs_star = lqr_adjoint_pass(Xs_star, Us_star, LQRParams(Xs_star[0], lqr_params_stars))
    return (Xs_star, Us_star, Lambs_star), total_cost, costs
