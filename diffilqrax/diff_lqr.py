"""
Module contains functions for solving the differential linear quadratic regulator (DLQR) problem.
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
from diffilqrax.typs import LQRParams, ModelDims, LQR

# v_outer = jax.vmap(jnp.outer) # vectorized outer product through time i.e. 'ij,ik->ijk'


def get_qra_bar(
    dims: ModelDims, params: LQRParams, tau_bar: Array, tau_bar_f: Array
) -> Tuple[Array, Array, Array]:
    """
    Helper function to get gradients wrt to q, r, a. Variables q_bar, r_bar, a_bar from solving the 
    rev LQR problem where q_rev = x_bar, r_rev = u_bar, a_rev = lambda_bar (set to 0 here)
    Args:
        dims (ModelDims): The dimensions of the model.
        params (LQRParams): The parameters of the model.
        tau_bar (Array): The tau_bar array.
        tau_bar_f (Array): The tau_bar_f array.

    Returns:
        Tuple[Array, Array, Array]: The q_bar, r_bar, and a_bar arrays.
    """
    lqr = params.lqr
    n = dims.n
    x_bar, u_bar = tau_bar[:, :n], tau_bar[:, n:]
    swapped_lqr = LQR(
        A=lqr.A,
        B=lqr.B,
        a=jnp.zeros_like(lqr.a),
        Q=lqr.Q,
        q=x_bar,
        Qf=lqr.Qf,
        qf=tau_bar_f[:n],
        R=lqr.R,
        r=u_bar,
        S=lqr.S,
    )
    swapped_params = LQRParams(params.x0, swapped_lqr)
    _, q_bar, r_bar, a_bar = solve_lqr_swap_x0(swapped_params, dims)
    return q_bar, jnp.r_[r_bar, jnp.zeros((1,dims.m,))], a_bar


@partial(custom_vjp, nondiff_argnums=(0,))
def dlqr(dims: ModelDims, params: LQRParams, tau_guess: Array) -> Array:
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
    sol = solve_lqr(params, dims)  #  tau_guess)
    _, Xs_star, Us_star, _ = sol
    tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    return tau_star


def fwd_dlqr(
    dims: ModelDims, params: LQRParams, tau_guess: Array
) -> Tuple[Array, Tuple[LQRParams, Tuple[Array, Array, Array, Array]]]:
    """Solves the forward differential linear quadratic regulator (DLQR) problem.

    Args:
        dims (ModelDims): The dimensions of the model.
        params (LQRParams): The parameters of the DLQR problem.
        tau_guess (Array): The initial guess for the state-control trajectory.

    Returns:
        Tuple: A tuple containing the optimal state-control trajectory and the updated parameters 
        and solution.
    """
    lqr = params.lqr
    sol = solve_lqr(params, dims)
    gains, Xs_star, Us_star, Lambs = sol
    tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    new_lqr = LQR(
        A=lqr.A,
        B=lqr.B,
        a=jnp.zeros_like(lqr.a),
        Q=lqr.Q,
        q=lqr.q - bmm(lqr.Q, Xs_star[:-1]) - bmm(lqr.S, Us_star),
        Qf=lqr.Qf,
        qf=lqr.qf - mm(lqr.Qf, Xs_star[-1]),
        R=lqr.R,
        r=lqr.r - bmm(lqr.R, Us_star) - bmm(lqr.S.transpose(0, 2, 1), Xs_star[:-1]),
        S=lqr.S,
    )
    new_params = LQRParams(params.x0, new_lqr)
    _, new_Xs_star, new_Us_star, new_Lambs = solve_lqr(new_params, dims)
    # new_sol = gains, new_Xs_star, new_Us_star, new_Lambs
    return tau_star, (new_params, sol)  # check whether params or new_params


def rev_dlqr(dims: ModelDims, res, tau_bar) -> LQRParams:
    """
    Inputs : params (contains lqr parameters, x0), tau_star_bar (gradients wrt to tau at tau_star)
    params : LQR(A, B, a, Q, q, Qf, qf, R, r, S)
    A : T x N x N
    B : T x N x M
    a : T x N x 1
    q : T x N x 1
    Q : T x N x N
    R : T x M x M
    r : T x M x 1
    S : T x N x M

    - q_bar, r_bar, a_bar from solving the rev LQR problem where q_rev = x_bar, r_rev = u_bar, 
      a_rev = lambda_bar (set to 0 here)
    - define c_bar = [q_bar, r_bar]
    - define F_bar (where F = [A, B] and C_bar (where C = [Q,R]) as 
      C_bar 0.5*(c_bar tau_star.T + tau_star c_bar.T))
    - F_bar_t = lambda_star_{t+1}c_bar.T + f_{t+1} tau_star_t.T

    Returns : params_bar, i.e tuple with gradients wrt to x0, LQR params, and horizon
    """
    params, sol = res
    (_, Xs_star, Us_star, Lambs) = sol
    M = dims.m
    tau_bar, tau_bar_f = tau_bar[:-1], tau_bar[-1]
    tau_star = jnp.c_[Xs_star, jnp.r_[Us_star, jnp.zeros(shape=(1, M))]]
    n = dims.n
    q_bar, r_bar, a_bar = get_qra_bar(dims, params, tau_bar, tau_bar_f)
    c_bar = jnp.concatenate([q_bar, r_bar], axis=1)
    F_bar = jnp.einsum("ij,ik->ijk", a_bar[1:], tau_star[:-1]) + jnp.einsum(
        "ij,ik->ijk", Lambs[1:], c_bar[:-1]
    )
    C_bar = symmetrise_tensor(
        jnp.einsum("ij,ik->ijk", c_bar, tau_star)
    )  # factor of 2 included in symmetrization
    Q_bar, R_bar = C_bar[:, :n, :n], C_bar[:, n:, n:]
    S_bar = 2 * C_bar[:, :n, n:]
    A_bar, B_bar = F_bar[..., :n], F_bar[..., n:]
    LQR_bar = LQR(
        A=A_bar,
        B=B_bar,
        a=a_bar[1:],
        Q=Q_bar[:-1],
        q=q_bar[:-1],
        Qf=Q_bar[-1],
        qf=q_bar[-1],
        R=R_bar[:-1],
        r=r_bar[:-1],
        S=S_bar[:-1],
    )
    x0_bar = a_bar[0]  # Lambs[0] #jnp.zeros_like(params.x0) #Lambs[0]#
    return LQRParams(x0=x0_bar, lqr=LQR_bar), None


dlqr.defvjp(fwd_dlqr, rev_dlqr)
