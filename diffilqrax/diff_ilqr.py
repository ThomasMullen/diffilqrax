"""
Module solves the differential iterative linear quadratic regulator (DiLQR) problem.
"""

from jax import Array, lax
import jax.numpy as jnp

from diffilqrax.typs import iLQRParams, System, LQR, LQRParams
from diffilqrax.ilqr import ilqr_solver, approx_lqr_offset
from diffilqrax.diff_lqr import dllqr


def dilqr(model: System, params: iLQRParams, Us_init: Array, **kwargs) -> Array:
    """
    Solves the differential iLQR problem.

    Parameters
    ----------
    model : System
        The system model.
    params : iLQRParams
        The iLQR parameters.
    Us_init : Array
        The initial control sequence.
    **kwargs
        Additional keyword arguments for the iLQR solver.

    Returns
    -------
    Array
        The optimized control sequence.
    """
    sol = ilqr_solver(
        model, lax.stop_gradient(params), Us_init, **kwargs
    )  #  tau_guess)
    (Xs_star, Us_star, Lambs_star), _, costs = sol
    tau_star = jnp.c_[
        Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, model.dims.m))]
    ]
    local_lqr = approx_lqr_offset(model, Xs_star, Us_star, params)  ##equiv of g1
    params = LQRParams(lqr=local_lqr, x0=params.x0)
    tau_star = dllqr(model.dims, params, tau_star)
    return tau_star
