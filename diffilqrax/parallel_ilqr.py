"""Solve iLQR with parallel (associative-scan) LQR subproblems"""

import math
import warnings
from typing import Tuple, Optional
from functools import partial
import jax
from jax import Array
from jax import lax
import jax.numpy as jnp

from diffilqrax.lqr import bmm
from diffilqrax.plqr import (
    closed_loop_scan,
    parallel_reverse_lin_integration,
    plqr_backward_pass,
    project_psd_lqr,
)
from diffilqrax.ilqr import (
    approx_lqr,
    approx_lqr_offset,
    armijo_linesearch,
    ilqr_forward_pass,
    ilqr_simulate,
    kkt_input_residual,
    increase_regularisation,
    decrease_regularisation,
)
from diffilqrax.typs import iLQRParams, System, Gains, LQR, LQRParams, ParallelSystem

jax.config.update("jax_enable_x64", True)  # double precision

ROLLOUTS = ("feedback", "parallel")


def total_cost(model: System, params: iLQRParams, Xs: Array, Us: Array) -> Array:
    """Total cost of a state-input trajectory, evaluated in parallel over time"""
    tps = jnp.arange(model.dims.horizon)
    costs = jax.vmap(model.cost, in_axes=(0, 0, 0, None))(tps, Xs[:-1], Us, params.theta)
    return jnp.sum(costs) + model.costf(Xs[-1], params.theta)


def dynamics_defects(model: System, params: iLQRParams, Xs: Array, Us: Array) -> Array:
    """Dynamics defects d_t = f(x_t, u_t) - x_{t+1} [Txn], evaluated in parallel over time"""
    tps = jnp.arange(model.dims.horizon)
    return jax.vmap(model.dynamics, in_axes=(0, 0, 0, None))(tps, Xs[:-1], Us, params.theta) - Xs[1:]


def multiple_shooting_step(delta_lqr: LQR, gains: Gains, values: Tuple[Array, Array]) -> Tuple[Array, Array, Array]:
    """
    Full step of the local LQR in deviation coordinates (δx_0 = 0, dynamics offsets a = defects), by an
    associative scan, and its multipliers λ_t = J_t δx_t - η_t from the value functions. Unlike the open-loop
    adjoint recursion λ_t = A_tᵀ λ_{t+1} + ..., this does not overflow when the linearised dynamics are
    unstable over a long horizon.

    Parameters
    ----------
    delta_lqr : LQR
        Local LQR around the current trajectory with the dynamics defects as offsets ``a``.
    gains : Gains
        Gains of ``delta_lqr`` (δu = K δx + k), from :func:`diffilqrax.plqr.plqr_backward_pass`.
    values : Tuple[Array, Array]
        Value functions (η, J) of ``delta_lqr`` (regularised as for ``gains``), from
        :func:`diffilqrax.plqr.plqr_backward_pass` with ``return_values=True``.

    Returns
    -------
    Tuple[Array, Array, Array]
        State step [(T+1)xn], input step [Txm] and multipliers λ [(T+1)xn] of the linearised dynamics
        (λ_{t+1} belongs to the constraint f(x_t, u_t) - x_{t+1} = 0).
    """
    model = LQRParams(jnp.zeros_like(delta_lqr.a[0]), delta_lqr)
    dXs = jnp.r_[model.x0[None], closed_loop_scan(model, gains, 1.0)]
    dUs = bmm(gains.K, dXs[:-1]) + gains.k
    etas, Js = values
    return dXs, dUs, bmm(Js, dXs) - etas


def merit_linesearch(
    model: System,
    params: iLQRParams,
    Xs: Array,
    Us: Array,
    step: Tuple[Array, Array, Array],
    lqr: LQR,
    cost_init: Array,
    defects: Array,
    rho: float,
    alpha_init: float,
    tau: float = 0.5,
    beta: float = 0.1,
    alpha_min: float = 1e-8,
    max_iter_linesearch: Optional[int] = None,
) -> Tuple[Array, Tuple[Array, Array], Array]:
    """
    Armijo backtracking on the augmented Lagrangian merit φ = J + λᵀd + ρ/2 ‖d‖² along (X + α dX, U + α dU),
    with the multipliers λ of the local LQR. Its slope at α = 0 is -pᵀHp - ρ‖d‖², negative whenever the
    (regularised) local LQR is strictly convex, for any ρ ≥ 0. Step sizes follow
    :func:`diffilqrax.ilqr.armijo_linesearch`: tau * alpha_init, tau² * alpha_init, ...

    Parameters
    ----------
    model : System
        The system model.
    params : iLQRParams
        The iLQR parameters.
    Xs, Us : Array
        Current (not necessarily feasible) state and input trajectories.
    step : Tuple[Array, Array, Array]
        (dX, dU, λ) from :func:`multiple_shooting_step`.
    lqr : LQR
        Local LQR without offsets (gradients q = Cx, r = Cu, qf = Cfx), from :func:`approx_lqr`.
    cost_init : Array
        Cost of the current trajectory.
    defects : Array
        Dynamics defects of the current trajectory.
    rho : float
        Penalty on the squared defects.
    alpha_init, tau, beta, alpha_min, max_iter_linesearch
        As in :func:`diffilqrax.ilqr.armijo_linesearch`.

    Returns
    -------
    Tuple[Array, Tuple[Array, Array], Array]
        Whether a step was accepted, the (accepted or initial) state and input trajectories, and their cost.
    """
    dXs, dUs, lambs = step
    if max_iter_linesearch is None:
        max_iter_linesearch = int(math.floor(math.log(alpha_min / (tau * alpha_init)) / math.log(tau))) + 2

    def merit(cost, d):
        return cost + jnp.sum(lambs[1:] * d) + 0.5 * rho * jnp.sum(d**2)

    merit_init = merit(cost_init, defects)
    grad_dot_step = jnp.sum(lqr.q * dXs[:-1]) + jnp.sum(lqr.r * dUs) + jnp.dot(lqr.qf, dXs[-1])
    slope = grad_dot_step - jnp.sum(lambs[1:] * defects) - rho * jnp.sum(defects**2)
    noise = 100.0 * jnp.finfo(jnp.result_type(merit_init)).eps * jnp.maximum(jnp.abs(merit_init), 1.0)

    def backtrack_iter(carry):
        X, U, cost, alpha, _, _ = carry
        new_Xs, new_Us = Xs + alpha * dXs, Us + alpha * dUs
        new_cost = total_cost(model, params, new_Xs, new_Us)
        decrease = merit(new_cost, dynamics_defects(model, params, new_Xs, new_Us)) - merit_init
        below_noise = jnp.logical_and(-alpha * slope < noise, decrease < noise)
        accepted = jnp.logical_and(jnp.isfinite(decrease), jnp.logical_or(decrease <= beta * alpha * slope, below_noise))
        accepted = jnp.logical_and(accepted, slope < noise)
        carry_on = jnp.logical_and(jnp.logical_not(accepted), alpha >= alpha_min)
        return (
            jnp.where(accepted, new_Xs, X),
            jnp.where(accepted, new_Us, U),
            jnp.where(accepted, new_cost, cost),
            alpha * tau,
            accepted,
            carry_on,
        )

    def loop_fun(carry, _):
        return lax.cond(carry[-1], backtrack_iter, lambda x: x, carry), None

    initial_carry = (Xs, Us, cost_init, tau * alpha_init, jnp.array(False), jnp.array(True))
    (Xs_star, Us_star, cost_star, _, accepted, _), _ = lax.scan(loop_fun, initial_carry, None, length=max_iter_linesearch)
    return accepted, (Xs_star, Us_star), cost_star


def pilqr_solver(
    parallel_model: ParallelSystem | System,
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
    make_psd: Optional[str] = None,
    psd_delta: float = 1e-6,
    rollout: str = "feedback",
    defect_tol: Optional[float] = None,
    merit_rho: float = 1.0,
    **linesearch_kwargs,
) -> Tuple[Tuple[Array, Array, Array], float, Array]:
    """
    Solve the iterative Linear Quadratic Regulator (iLQR) problem with parallel LQR subproblems.

    The backward pass is an associative Riccati scan of the local LQR problem regularised with
    R -> R + μI (:func:`diffilqrax.plqr.plqr_backward_pass`), which keeps it an exact LQR so the scan
    applies. μ is adapted as in :func:`diffilqrax.ilqr.ilqr_solver`: it is increased until every Huu is
    positive definite, increased after a rejected step and decreased after an accepted one.

    The new trajectory is found in one of two ways (``rollout``):

    - "feedback": sequential non-linear closed-loop rollout u = ū + K(x - x̄) + αk, as in
      :func:`diffilqrax.ilqr.ilqr_solver`, with an Armijo line search on the cost. Iterates are feasible.
    - "parallel": multiple shooting, with no sequential rollout. The states are free variables, the dynamics
      defects d_t = f(x_t, u_t) - x_{t+1} enter the local LQR as offsets, the step (X + α dX, U + α dU) is
      found by associative scans and accepted by an Armijo line search on the augmented Lagrangian merit
      J + λᵀd + ρ/2 ‖d‖² (:func:`merit_linesearch`). Iterates are feasible only at convergence.

    Parameters
    ----------
    parallel_model : ParallelSystem | System
        The system model. A :class:`ParallelSystem` is accepted for backward compatibility; its integrators
        are not used.
    params : iLQRParams
        The iLQR parameters.
    Us_init : Array
        The initial control trajectory. The initial states are its non-linear rollout.
    max_iter : int, optional
        The maximum number of iterations, by default 40.
    convergence_thresh : float, optional
        If given, also stop once an accepted step reduces the cost by less than this relative amount.
        By default None, so convergence is determined by ``kkt_tol``.
    alpha_init : float, optional
        Line search scale; the first step tried is ``tau * alpha_init``, by default 1.0.
    verbose : bool, optional
        Whether to print debug information, by default False.
    use_linesearch : bool, optional
        Whether to use the line search, by default True. If False, every step is taken with α = 1 and a step
        that increases the cost (or merit) beyond round-off is rejected.
    regularise : bool, optional
        Whether to adapt the regularisation μ, by default True. If False, μ = 0 and the solver stops
        at the first step that does not decrease the cost.
    kkt_tol : float, optional
        Converged when max|dL/dU| falls below this tolerance (and the defects below ``defect_tol``), by
        default 1e-10.
    mu_max : float, optional
        Stop when the regularisation exceeds this value without an accepted step, by default 1e10.
    max_reg_iter : int, optional
        Maximum number of regularisation increases per backward pass, by default 20.
    make_psd : str, optional
        Convexify the local LQR before the backward pass by projection onto the PSD cone
        (:func:`diffilqrax.plqr.project_psd_lqr`), by default None (no projection). "R" projects the
        input cost only: it removes negative input curvature without retries, which helps far from a
        solution, and is inactive where R ≻ δI. If R is indefinite at the solution (with Huu ≻ 0 through
        BᵀJB) it stays active and final convergence is slower. "full" also projects Q - S R⁻¹Sᵀ and Qf, which
        makes every Huu positive definite but turns the step into a Gauss-Newton-like step where the
        state cost is locally non-convex, which can converge only linearly. Composes with ``regularise``.
    psd_delta : float, optional
        Minimum eigenvalue after the PSD projection, by default 1e-6.
    rollout : str, optional
        "feedback" (default) or "parallel", see above.
    defect_tol : float, optional
        Converged only when max|d_t| is below this tolerance, by default ``kkt_tol``. Only binding for
        ``rollout="parallel"``.
    merit_rho : float, optional
        Penalty ρ on the squared defects in the merit function of ``rollout="parallel"``, by default 1.0.
    **linesearch_kwargs
        Additional keyword arguments for the line search (tau, beta, alpha_min, max_iter_linesearch), see
        :func:`diffilqrax.ilqr.armijo_linesearch`.

    Returns
    -------
    Tuple[Tuple[Array, Array, Array], float, Array]
        A tuple containing the final state trajectory, control trajectory, and the adjoint
        variables. Also returns the total cost of the trajectory and the cost history.
    """
    assert rollout in ROLLOUTS, f"rollout must be one of {ROLLOUTS}, got {rollout}"
    if "tol" in linesearch_kwargs:
        warnings.warn(
            "pilqr_solver now uses armijo_linesearch: the ratio-test kwargs (beta: step shrink factor, tol: "
            "accepted fraction of the expected decrease) are mapped to tau and beta; pass those instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy = {"beta": "tau", "tol": "beta"}
        linesearch_kwargs = {legacy.get(k, k): v for k, v in linesearch_kwargs.items()}
    model = parallel_model.model if isinstance(parallel_model, ParallelSystem) else parallel_model
    defect_tol = kkt_tol if defect_tol is None else defect_tol
    (Xs_init, _), c_init = ilqr_simulate(model, Us_init, params)
    # carry: (Xs, Us, total cost, delta, mu, iteration, carry_on)
    initial_carry = (Xs_init, Us_init, c_init, jnp.ones_like(c_init), jnp.zeros_like(c_init), 0, True)
    prollout = partial(ilqr_forward_pass, model, params)

    def backward_with_reg(lqr_params: LQRParams, delta: Array, mu: Array):
        """parallel backward pass, increasing mu until Huu_reg is positive definite"""
        dJ, gains, is_pd, values = plqr_backward_pass(lqr_params, mu, return_values=True)
        if not regularise:
            return dJ, gains, is_pd, values, delta, mu

        def reg_step(carry, _):
            def retry(carry):
                *_, delta, mu = carry
                delta, mu = increase_regularisation(delta, mu)
                return (*plqr_backward_pass(lqr_params, mu, return_values=True), delta, mu)

            needs_reg = jnp.logical_and(jnp.logical_not(carry[2]), carry[5] < mu_max)
            return lax.cond(needs_reg, retry, lambda c: c, carry), None

        return lax.scan(reg_step, (dJ, gains, is_pd, values, delta, mu), None, length=max_reg_iter)[0]

    def feedback_step(gains, exp_dJ, old_Xs, old_Us, old_cost):
        """Sequential non-linear closed-loop rollout"""
        if use_linesearch:
            return armijo_linesearch(
                prollout, gains, old_Xs, old_Us, alpha_init, cost_init=old_cost, expected_dJ=exp_dJ, **linesearch_kwargs
            )
        (new_Xs, new_Us), new_cost = prollout(gains, old_Xs, old_Us, alpha=1.0)
        noise = 100.0 * jnp.finfo(jnp.result_type(old_cost)).eps * jnp.maximum(jnp.abs(old_cost), 1.0)
        accepted = jnp.logical_and(jnp.isfinite(new_cost), old_cost - new_cost > -noise)
        return accepted, (new_Xs, new_Us), new_cost

    def parallel_step(lqr, delta_lqr, gains, values, defects, old_Xs, old_Us, old_cost):
        """Multiple-shooting step with a merit line search"""
        step = multiple_shooting_step(delta_lqr, gains, values)
        ls_kwargs = linesearch_kwargs if use_linesearch else {"tau": 1.0, "max_iter_linesearch": 1}
        alpha = alpha_init if use_linesearch else 1.0
        return merit_linesearch(model, params, old_Xs, old_Us, step, lqr, old_cost, defects, merit_rho, alpha, **ls_kwargs)

    def plqr_iter(carry_tuple):
        """One parallel iLQR iteration: backward scan, line search, regularisation update"""
        old_Xs, old_Us, old_cost, delta, mu, n_iter, _ = carry_tuple
        lqr = approx_lqr(model, old_Xs, old_Us, params)
        defects = dynamics_defects(model, params, old_Xs, old_Us)
        kkt_res = jnp.max(jnp.abs(kkt_input_residual(lqr)))
        max_defect = jnp.max(jnp.abs(defects))
        converged = jnp.logical_and(kkt_res < kkt_tol, max_defect < defect_tol)

        def take_step(_):
            # deviation coordinates: the defects are the dynamics offsets (zero along a feedback rollout)
            local_lqr = lqr._replace(a=defects)
            local_lqr = local_lqr if make_psd is None else project_psd_lqr(local_lqr, psd_delta, make_psd)
            delta_params = LQRParams(jnp.zeros_like(params.x0), local_lqr)
            exp_dJ, gains, _, values, delta_new, mu_new = backward_with_reg(delta_params, delta, mu)
            if rollout == "feedback":
                accepted, (new_Xs, new_Us), new_cost = feedback_step(gains, exp_dJ, old_Xs, old_Us, old_cost)
            else:
                accepted, (new_Xs, new_Us), new_cost = parallel_step(
                    lqr, local_lqr, gains, values, defects, old_Xs, old_Us, old_cost
                )
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
                    "it={i} J={j:.12e} max|dLdU|={g:.3e} max|d|={d:.3e} mu={m:.1e} accepted={a}",
                    i=n_iter + 1,
                    j=new_cost,
                    g=kkt_res,
                    d=max_defect,
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
    (Xs_star, Us_star, total_cost_star, *_, n_iters, _), costs = lax.scan(loop_fun, initial_carry, None, length=max_iter)
    if verbose:
        jax.debug.print("Ran {n}/{N} iterations, cost: {c}", n=n_iters, N=max_iter, c=total_cost_star)
    # adjoints λ_T = ∇Cf, λ_t = A_tᵀ λ_{t+1} + ∇_x C_t of the non-linear problem, by an associative scan
    lqr_stars = approx_lqr_offset(model, Xs_star, Us_star, params)
    Lambs_star = parallel_reverse_lin_integration(LQRParams(Xs_star[0], lqr_stars), Xs_star, Us_star)
    return (Xs_star, Us_star, Lambs_star), total_cost_star, costs
