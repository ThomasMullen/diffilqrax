"""
This module contains functions for solving the differential linear quadratic regulator (DLQR) problem.
"""
from functools import partial
from jax.numpy import matmul as mm
import jax.numpy as jnp
from typing import Tuple
from jax import Array, custom_vjp, lax
from src.lqr import (
    LQR,
    LQRParams,
    ModelDims,
    kkt,
    solve_lqr,
    solve_lqr_swap_x0,
    symmetrise_tensor,
    bmm,
)
from src.ilqr import iLQRParams, System, ilQR_solver, approx_lqr, approx_lqr_dyn
from src.diff_lqr import get_qra_bar, dlqr, dllqr


def make_local_lqr(model, Xs_star, Us_star, params):
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
    return new_lqr ##get the local LQR like that, and then gradients wrt to that from the function, but still outputting the right Us_star
#so do need the custom iqlr
    
def dilqr(model: System, params: iLQRParams, U_inits: Array, **kwargs) -> Array:
    sol = ilQR_solver(model, lax.stop_gradient(params), U_inits, **kwargs)  #  tau_guess)
    (Xs_star, Us_star, Lambs_stars), total_cost, costs = sol
    tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, model.dims.m))]]
    local_lqr = make_local_lqr(model, Xs_star, Us_star, params) ##equiv of g1
    params = LQRParams(lqr = local_lqr, x0 = params.x0)
    tau_star =  dllqr(model.dims, params, tau_star)
    return tau_star 

