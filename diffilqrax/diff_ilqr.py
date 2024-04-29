"""
Module contains functions for solving the differential iterative linear quadratic regulator (DiLQR) problem.
"""
from functools import partial
from typing import Tuple
from jax.numpy import matmul as mm
import jax.numpy as jnp
from jax import Array, custom_vjp
from diffilqrax.lqr import (
    # kkt,
    solve_lqr,
    solve_lqr_swap_x0,
    symmetrise_tensor,
    bmm,
)
from diffilqrax.ilqr import solve_ilqr
import diffilqrax.diff_lqr as dlqr
from diffilqrax.typs import iLQRParams, ModelDims, System, LQR


@partial(custom_vjp, nondiff_argnums=(0,))
def dilqr(model: System, params: iLQRParams, tau_guess: Array) -> Array:
    """
    Solves the differential linear quadratic regulator (DLQR) problem. Custom VJP function for DLQR. 
    Reverse mode uses an LQR solver to solve the reverse LQR problem of the gradients on state and 
    input trajectory gradients.

    Args:
        dims (ModelDims): The dimensions of the model.
        params (LQRParams): The parameters of the model.
        tau_guess (Array): The initial guess for the optimal control sequence.

    Returns:
        Array: Concatenated optimal state and control sequence along axis=1.
    """
    # sol = solve_ilqr(model, params,)  #  tau_guess)
    # (Xs_star, Us_star), *_ = sol
    # tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, model.dims.m))]]
    # return tau_star


    # model: System,
    # params: iLQRParams,
    # U_inits: Array,
    # max_iter: int = 40,
    # convergence_thresh: float = 1e-6,
    # alpha_init: float = 1.0,
    # verbose: bool = False,
    # use_linesearch: bool = True,
    # **linesearch_kwargs,