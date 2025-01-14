"""
Module contains functions for solving the differential linear quadratic regulator (DLQR) problem.
"""

from functools import partial
from typing import Tuple
from jax import Array, custom_vjp
import jax.numpy as jnp
from diffilqrax.plqr import (
    # kkt,
    solve_plqr,
)
from diffilqrax.diff_lqr import build_ajoint_lqr, offset_lqr
from diffilqrax.typs import LQRParams, ModelDims

# v_outer = jax.vmap(jnp.outer) # vectorized outer product through time i.e. 'ij,ik->ijk'


@partial(custom_vjp, nondiff_argnums=(0,))
def pdllqr(dims: ModelDims, params: LQRParams, tau_star: Array) -> Array:
    """
    Solves the differential linear quadratic regulator (DLQR) problem. Custom VJP function for DLQR.
    Reverse mode uses an LQR solver to solve the reverse LQR problem of the gradients on state and
    input trajectory gradients.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the model.
    tau_star : Array
        The optimal state-control trajectory.

    Returns
    -------
    Array
        Concatenated optimal state and control sequence along axis=1.
    """
    return tau_star


def fwd_pdllqr(
    dims: ModelDims, params: LQRParams, tau_star: Array
) -> Tuple[Array, Tuple[LQRParams, Tuple[Array, Array, Array, Array]]]:
    """
    Solves the forward differential linear quadratic regulator (DLQR) problem.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the DLQR problem.
    tau_star : Array
        The optimal state-control trajectory.

    Returns
    -------
    Tuple[Array, Tuple[LQRParams, Tuple[Array, Array, Array, Array]]]
        A tuple containing the optimal state-control trajectory and the updated parameters
        and solution.
    """
    lqr = params.lqr
    sol = solve_plqr(params)
    Xs_star, Us_star, Lambs = sol
    tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    new_lqr = offset_lqr(lqr, Xs_star, Us_star)
    new_params = LQRParams(params.x0, new_lqr)
    return tau_star, (new_params, sol)  # check whether params or new_params


def rev_pdllqr(
    dims: ModelDims, res: Tuple[LQRParams, Tuple[Array, Array, Array]], tau_bar: Array
) -> LQRParams:
    """
    Reverse mode for DLQR.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    res : Tuple[LQRParams, Tuple[Array, Array, Array]]
        The result from the forward pass.
    tau_bar : Array
        The gradients with respect to tau.

    Returns
    -------
    LQRParams
        The LQR parameters with reverse gradients.
    """
    params, sol = res
    (Xs_star, Us_star, Lambs) = sol
    tau_star = jnp.c_[Xs_star, jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    return build_ajoint_lqr(dims, params, tau_star, Lambs, tau_bar), None


pdllqr.defvjp(fwd_pdllqr, rev_pdllqr)


@partial(custom_vjp, nondiff_argnums=(0,))
def pdlqr(dims: ModelDims, params: LQRParams, tau_guess: Array) -> Array:
    """
    Solves the differential linear quadratic regulator (DLQR) problem. Custom VJP function for DLQR.
    Reverse mode uses an LQR solver to solve the reverse LQR problem of the gradients on state and
    input trajectory gradients.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the model.
    tau_guess : Array
        The initial guess for the optimal control sequence.

    Returns
    -------
    Array
        Concatenated optimal state and control sequence along axis=1.
    """
    sol = solve_plqr(params)  #  tau_guess)
    Xs_star, Us_star, _ = sol
    tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    return tau_star


def fwd_pdlqr(
    dims: ModelDims, params: LQRParams, tau_guess: Array
) -> Tuple[Array, Tuple[LQRParams, Tuple[Array, Array, Array, Array]]]:
    """
    Solves the forward differential linear quadratic regulator (DLQR) problem.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the DLQR problem.
    tau_guess : Array
        The initial guess for the state-control trajectory.

    Returns
    -------
    Tuple[Array, Tuple[LQRParams, Tuple[Array, Array, Array, Array]]]
        A tuple containing the optimal state-control trajectory and the updated parameters
        and solution.
    """
    lqr = params.lqr
    sol = solve_plqr(params)
    Xs_star, Us_star, _ = sol
    tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    new_lqr = offset_lqr(lqr, Xs_star, Us_star)
    new_params = LQRParams(params.x0, new_lqr)
    return tau_star, (new_params, sol)  # check whether params or new_params


def rev_pdlqr(
    dims: ModelDims, res: Tuple[LQRParams, Tuple[Array, Array, Array]], tau_bar: Array
) -> LQRParams:
    """
    Reverse mode for DLQR.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    res : Tuple[LQRParams, Tuple[Array, Array, Array]]
        The result from the forward pass.
    tau_bar : Array
        The gradients with respect to tau.

    Returns
    -------
    LQRParams
        The LQR parameters with reverse gradients.
    """
    params, sol = res
    (Xs_star, Us_star, Lambs) = sol
    tau_star = jnp.c_[Xs_star, jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    return build_ajoint_lqr(dims, params, tau_star, Lambs, tau_bar), None


pdlqr.defvjp(fwd_pdlqr, rev_pdlqr)
