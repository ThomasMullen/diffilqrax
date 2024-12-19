"""
Module solves the differential iterative linear quadratic regulator (DiLQR) problem.
"""

from jax import Array, lax
import jax.numpy as jnp

from diffilqrax.typs import iLQRParams, LQRParams, ParallelSystem
from diffilqrax.ilqr import approx_lqr_offset
from diffilqrax.parallel_ilqr import pilqr_solver

# from diffilqrax.diff_lqr import dllqr
from diffilqrax.parallel_dlqr import pdlqr


def parallel_dilqr(
    parallel_model: ParallelSystem, params: iLQRParams, Us_init: Array, **kwargs
) -> Array:
    """Solves the differential iLQR problem.

    Args:
        model (System): The system model.
        params (iLQRParams): The iLQR parameters.
        Us_init (Array): The initial control sequence.

    Returns:
        Array: The optimized control sequence.
    """
    model = parallel_model.model
    sol = pilqr_solver(
        parallel_model, lax.stop_gradient(params), Us_init, **kwargs
    )  #  tau_guess)
    (Xs_star, Us_star, Lambs_star), *_ = sol
    tau_star = jnp.c_[
        Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, model.dims.m))]
    ]
    local_lqr = approx_lqr_offset(model, Xs_star, Us_star, params)  ##equiv of g1
    params = LQRParams(lqr=local_lqr, x0=params.x0)
    tau_star = pdlqr(model.dims, params, tau_star)
    return tau_star  # might make sense to return the full solution instead of tau_star
