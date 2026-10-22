"""
Module solves the differential iterative linear quadratic regulator (DiLQR) problem with the parallel
(associative-scan) iLQR solver.
"""

from typing import Tuple

from jax import Array, lax
import jax.numpy as jnp

from diffilqrax.typs import iLQRParams, LQRParams, ParallelSystem, System
from diffilqrax.ilqr import approx_lqr_offset
from diffilqrax.diff_ilqr import add_dynamics_curvature
from diffilqrax.parallel_ilqr import pilqr_solver
from diffilqrax.parallel_dlqr import pdllqr


def pdilqr(
    parallel_model: ParallelSystem | System,
    params: iLQRParams,
    Us_init: Array,
    exact_hessian: bool = True,
    adjoint: str = "parallel",
    **kwargs,
) -> Array:
    """
    Solves the differential iLQR problem with :func:`diffilqrax.parallel_ilqr.pilqr_solver`. Parallel
    counterpart of :func:`diffilqrax.diff_ilqr.dilqr`.

    Parameters
    ----------
    parallel_model : ParallelSystem | System
        The system model.
    params : iLQRParams
        The iLQR parameters.
    Us_init : Array
        The initial control sequence.
    exact_hessian : bool, optional
        If True (default), the implicit gradients use the Hessian of the Lagrangian and match direct
        differentiation; if False, only the cost Hessians (Gauss-Newton). See :func:`dilqr`.
    adjoint : str, optional
        Solver of the adjoint LQR problem in the reverse pass: "parallel" (associative scan, default)
        or "sequential" (Riccati recursion). See :func:`diffilqrax.parallel_dlqr.pdllqr`.
    **kwargs
        Additional keyword arguments for :func:`diffilqrax.parallel_ilqr.pilqr_solver`, e.g. ``rollout``.

    Returns
    -------
    Array
        The optimized state-control trajectory [(T+1)x(n+m)].

    Notes
    -----
    The gradients are implicit: they are exact only if the solver has converged (see ``kkt_tol`` and
    ``max_iter`` of :func:`pilqr_solver`). ``Us_init`` receives no gradient and the solver is never
    differentiated, so it can run as a ``lax.while_loop`` (``loop="while"``, the default).
    """
    sol = pilqr_solver(parallel_model, lax.stop_gradient(params), lax.stop_gradient(Us_init), **kwargs)
    return implicit_pdilqr(parallel_model, params, sol[0], exact_hessian, adjoint)


def implicit_pdilqr(
    parallel_model: ParallelSystem | System,
    params: iLQRParams,
    solution: Tuple[Array, Array, Array],
    exact_hessian: bool = True,
    adjoint: str = "parallel",
) -> Array:
    """
    The differentiable part of :func:`pdilqr`: the optimal trajectory, with implicit gradients wrt ``params``.

    Parameters
    ----------
    parallel_model : ParallelSystem | System
        The system model.
    params : iLQRParams
        The iLQR parameters.
    solution : Tuple[Array, Array, Array]
        Optimal (Xs, Us, Lambs) of the solver for ``params``; they receive no gradient.
    exact_hessian, adjoint
        See :func:`pdilqr`.

    Returns
    -------
    Array
        The optimized state-control trajectory [(T+1)x(n+m)].
    """
    model = parallel_model.model if isinstance(parallel_model, ParallelSystem) else parallel_model
    Xs_star, Us_star, Lambs_star = lax.stop_gradient(solution)
    tau_star = jnp.c_[Xs_star, jnp.r_[Us_star, jnp.zeros(shape=(1, model.dims.m))]]
    local_lqr = approx_lqr_offset(model, Xs_star, Us_star, params)
    if exact_hessian:
        local_lqr = add_dynamics_curvature(model, local_lqr, Xs_star, Us_star, Lambs_star, params)
    return pdllqr(model.dims, LQRParams(lqr=local_lqr, x0=params.x0), tau_star, adjoint=adjoint)


parallel_dilqr = pdilqr
