"""
Differentiable parallel (associative-scan) LQR solver.

The forward problem is solved with :func:`diffilqrax.plqr.solve_plqr`. Gradients are implicit, as in
:func:`diffilqrax.diff_lqr.dlqr`: the reverse pass solves an adjoint LQR problem, either with the
associative scan (``adjoint="parallel"``) or with the sequential Riccati recursion
(``adjoint="sequential"``). Both give the same gradients up to round-off.
"""

from functools import partial
from typing import Optional, Tuple
from jax import Array, custom_vjp
import jax.numpy as jnp

from diffilqrax.plqr import solve_plqr, solve_plqr_swap_x0, parallel_reverse_lin_integration
from diffilqrax.lqr import solve_lqr_swap_x0, lqr_adjoint_pass
from diffilqrax.diff_lqr import build_ajoint_lqr, offset_lqr
from diffilqrax.typs import LQRParams, ModelDims

ADJOINT_SOLVERS = {"parallel": solve_plqr_swap_x0, "sequential": solve_lqr_swap_x0}


def _check_adjoint(adjoint: str) -> None:
    assert adjoint in ADJOINT_SOLVERS, f"adjoint must be one of {list(ADJOINT_SOLVERS)}, got {adjoint}"


def _tau(dims: ModelDims, Xs: Array, Us: Array) -> Array:
    """Concatenate states and inputs into [(T+1)x(n+m)], padding the inputs with a final zero row"""
    return jnp.c_[Xs, jnp.r_[Us, jnp.zeros(shape=(1, dims.m))]]


def _rev(dims: ModelDims, adjoint: str, res, tau_bar: Array) -> LQRParams:
    """Implicit gradients wrt the LQR parameters, from the adjoint LQR problem"""
    params, (Xs_star, Us_star, Lambs) = res
    tau_star = _tau(dims, Xs_star, Us_star)
    return build_ajoint_lqr(dims, params, tau_star, Lambs, tau_bar, ADJOINT_SOLVERS[adjoint])


@partial(custom_vjp, nondiff_argnums=(0, 1))
def _pdlqr(dims: ModelDims, adjoint: str, params: LQRParams) -> Array:
    Xs_star, Us_star, _ = solve_plqr(params)
    return _tau(dims, Xs_star, Us_star)


def _fwd_pdlqr(dims: ModelDims, adjoint: str, params: LQRParams):
    sol = solve_plqr(params)
    Xs_star, Us_star, _ = sol
    new_params = LQRParams(params.x0, offset_lqr(params.lqr, Xs_star, Us_star))
    return _tau(dims, Xs_star, Us_star), (new_params, sol)


def _rev_pdlqr(dims: ModelDims, adjoint: str, res, tau_bar: Array) -> Tuple[LQRParams]:
    return (_rev(dims, adjoint, res, tau_bar),)


_pdlqr.defvjp(_fwd_pdlqr, _rev_pdlqr)


def pdlqr(dims: ModelDims, params: LQRParams, tau_guess: Optional[Array] = None, adjoint: str = "parallel") -> Array:
    """
    Solves the LQR problem with the associative scan, with implicit gradients wrt the LQR parameters.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the model.
    tau_guess : Array, optional
        Unused; kept for API compatibility with :func:`diffilqrax.diff_lqr.dlqr`.
    adjoint : str, optional
        Solver of the adjoint LQR problem in the reverse pass: "parallel" (associative scan, default)
        or "sequential" (Riccati recursion).

    Returns
    -------
    Array
        Concatenated optimal state and control sequence along axis=1 [(T+1)x(n+m)].
    """
    del tau_guess
    _check_adjoint(adjoint)
    return _pdlqr(dims, adjoint, params)


@partial(custom_vjp, nondiff_argnums=(0, 1))
def _pdllqr(dims: ModelDims, adjoint: str, params: LQRParams, tau_star: Array) -> Array:
    return tau_star


def _fwd_pdllqr(dims: ModelDims, adjoint: str, params: LQRParams, tau_star: Array):
    Xs_star, Us_star = tau_star[:, : dims.n], tau_star[:-1, dims.n :]
    if adjoint == "parallel":
        Lambs = parallel_reverse_lin_integration(params, Xs_star, Us_star)
    else:
        Lambs = lqr_adjoint_pass(Xs_star, Us_star, params)
    new_params = LQRParams(params.x0, offset_lqr(params.lqr, Xs_star, Us_star))
    return tau_star, (new_params, (Xs_star, Us_star, Lambs))


def _rev_pdllqr(dims: ModelDims, adjoint: str, res, tau_bar: Array) -> Tuple[LQRParams, None]:
    return _rev(dims, adjoint, res, tau_bar), None


_pdllqr.defvjp(_fwd_pdllqr, _rev_pdllqr)


def pdllqr(dims: ModelDims, params: LQRParams, tau_star: Array, adjoint: str = "parallel") -> Array:
    """
    Identity on a precomputed optimal trajectory, with implicit gradients wrt the LQR parameters.
    Parallel counterpart of :func:`diffilqrax.diff_lqr.dllqr`: the adjoints at ``tau_star`` and the
    adjoint LQR problem of the reverse pass are computed with associative scans (``adjoint="parallel"``)
    or sequentially (``adjoint="sequential"``).

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        LQR whose optimum is ``tau_star`` (e.g. the local LQR of an iLQR solution).
    tau_star : Array
        The optimal state-control trajectory. Returned unchanged; it receives no gradient.
    adjoint : str, optional
        "parallel" (default) or "sequential".

    Returns
    -------
    Array
        Concatenated optimal state and control sequence along axis=1 [(T+1)x(n+m)].
    """
    _check_adjoint(adjoint)
    return _pdllqr(dims, adjoint, params, tau_star)
