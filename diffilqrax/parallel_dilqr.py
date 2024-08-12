"""
Module solves the differential iterative linear quadratic regulator (DiLQR) problem.
"""

from jax import Array, lax
import jax.numpy as jnp
from jax.numpy import matmul as mm

from diffilqrax.typs import iLQRParams, System, LQR, LQRParams, ParallelSystem
from diffilqrax.ilqr import ilqr_solver, approx_lqr_dyn
from diffilqrax.parallel_ilqr import pilqr_solver
from diffilqrax.diff_lqr import dllqr
from diffilqrax.lqr import bmm
from diffilqrax.parallel_dlqr import pdlqr


def make_local_lqr(model, Xs_star, Us_star, params):
    """Approximate the local LQR around the given trajectory"""
    lqr = approx_lqr_dyn(model, Xs_star, Us_star, params)
    new_lqr = LQR(
        A=lqr.A,
        B=lqr.B,
        a=lqr.a,
        Q=lqr.Q,
        q=lqr.q - bmm(lqr.Q, Xs_star[:-1]) - bmm(lqr.S, Us_star),
        Qf=lqr.Qf,
        qf=lqr.qf - mm(lqr.Qf, Xs_star[-1]),
        R=lqr.R,
        r=lqr.r - bmm(lqr.R, Us_star) - bmm(lqr.S.transpose(0, 2, 1), Xs_star[:-1]),
        S=lqr.S,
    )
    #get the local LQR like that, and then gradients wrt to that from the function,
    # but still outputting the right Us_star
    return new_lqr


# so do need the custom ilqr
def parallel_dilqr(parallel_model: ParallelSystem, params: iLQRParams, Us_init: Array, **kwargs) -> Array:
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
    local_lqr = make_local_lqr(model, Xs_star, Us_star, params)  ##equiv of g1
    params = LQRParams(lqr=local_lqr, x0=params.x0)
    tau_star = pdlqr(model.dims, params, tau_star)
    return tau_star  # might make sense to return the full solution instead of tau_star
