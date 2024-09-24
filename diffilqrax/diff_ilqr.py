"""
Module solves the differential iterative linear quadratic regulator (DiLQR) problem.
"""

from jax import Array, lax
import jax.numpy as jnp
from jax.numpy import matmul as mm

from diffilqrax.typs import iLQRParams, System, LQR, LQRParams
from diffilqrax.ilqr import ilqr_solver, approx_lqr_dyn
from diffilqrax.diff_lqr import dllqr, offset_lqr
from diffilqrax.lqr import bmm


def make_local_lqr(model, Xs_star, Us_star, params):
    """Approximate the local LQR around the given trajectory"""
    lqr = approx_lqr_dyn(model, Xs_star, Us_star, params)
    new_lqr = offset_lqr(lqr, Xs_star, Us_star)
    # get the local LQR like that, and then gradients wrt to that from the function,
    return new_lqr


# so do need the custom ilqr
def dilqr(model: System, params: iLQRParams, Us_init: Array, **kwargs) -> Array:
    """Solves the differential iLQR problem.

    Args:
        model (System): The system model.
        params (iLQRParams): The iLQR parameters.
        Us_init (Array): The initial control sequence.

    Returns:
        Array: The optimized control sequence.
    """
    sol = ilqr_solver(
        model, lax.stop_gradient(params), Us_init, **kwargs
    )  #  tau_guess)
    (Xs_star, Us_star, Lambs_star), _, costs = sol
    tau_star = jnp.c_[
        Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, model.dims.m))]
    ]
    local_lqr = make_local_lqr(model, Xs_star, Us_star, params)  ##equiv of g1
    params = LQRParams(lqr=local_lqr, x0=params.x0)
    tau_star = dllqr(model.dims, params, tau_star)
    return tau_star  # jnp.nan_to_num(tau_star)*(1 - jnp.isnan(jnp.sum(tau_star)))
    # def f_success():
    #     return jnp.nan_to_num(tau_star)*(1 - jnp.isnan(jnp.sum(tau_star)))
    # def f_failed():
    #     return jnp.zeros_like(tau_star)#*jnp.nan
    # return lax.cond(costs[0] > costs[-1], f_success, f_failed)
    # def f_success():
    #     return jnp.nan_to_num(tau_star)*(1 - jnp.isnan(jnp.sum(tau_star)))
    # def f_failed():
    #     return jnp.zeros_like(tau_star)#*jnp.nan
    # return jnp.nan_to_num(tau_star)*(1 - jnp.isnan(jnp.sum(tau_star))) #lax.cond(costs[0] > costs[-1], f_success, f_failed)
    # assert costs[0] > costs[-1]
    # return tau_star  # might make sense to return the full solution instead of tau_star
