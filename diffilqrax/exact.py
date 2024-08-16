"""
This module contains the exact solution to the LQR problem by converting sequential time evolution
into closed matrix form then inverting the matrix to get the optimal control sequence. The function
quad_solve uses the conjugate gradient method to solve the linear system Ax = b where A = big_G, 
b = -big_g. The function exact_solve uses numpy's linear solver to solve the same linear system.
"""

from typing import Tuple
import numpy as np
import jax
from jax import Array
import jax.numpy as jnp
from jax.numpy.linalg import matrix_power
from jax.scipy.linalg import block_diag
import jaxopt

from diffilqrax.typs import ModelDims, LQRParams

#jax.config.update("jax_enable_x64", True)  # sets float to 64 precision by default


def t_span_mpartial(arr: Array, dims: ModelDims) -> Array:
    """Span matrix along time dimension."""
    return jnp.tile(arr, (dims.horizon, 1, 1))


def t_span_vpartial(arr: Array, dims: ModelDims) -> Array:
    """Span vector along time dimension."""
    return jnp.tile(arr, (dims.horizon,))


def quad_solve(params: LQRParams, dims: ModelDims, x0: Array) -> Tuple[Array, Array]:
    """Solves a quadratic optimization problem.

    Args:
        params (Params): The parameters for the optimization problem.
        dims (ModelDims): The dimensions of the model.
        x0 (Array): The initial state.

    Returns:
        Tuple[Array, Array]: A tuple containing the optimal state trajectory and control inputs.
    """

    A = params.lqr.A[0]
    B = params.lqr.B[0]
    Q = params.lqr.Q[0]
    R = params.lqr.R[0]
    q = params.lqr.q[0]
    r = params.lqr.r[0]

    F0 = block_diag(*[matrix_power(A, j) for j in range(dims.horizon)])
    F = np.block(
        [
            [
                (
                    np.linalg.matrix_power(A, i - j - 1) @ B
                    if j < i
                    else np.zeros((dims.n, dims.m))
                )
                for j in range(dims.horizon)
            ]
            for i in range(dims.horizon)
        ]
    )
    # C(U) = U^T@big_R@U + big_r^T@U + X^T@big_Q@X + big_q^T@X
    # and  X = F0x0 + FU so
    # C(U) = U^T@big_R@U + big_r^T@U + (F0x0 + FU)^T@big_Q*(F0x0 + FU) +
    #       big_q^T@(F0x0 + FU) = U^T@big_G@U + big_g^T@U + cg
    # where big_G = @*(F^T@big_Q@F + big_R) and big_g = 2*F^T@big_Q@F0 + big_r
    # and cg = x0^T@F0^T@big_Q@F0@x0 + big_q^T@F0@x0

    # this is minimized by solving Ax = b where A = big_G, b = -big_g
    big_Q = block_diag(*t_span_mpartial(Q, dims))
    big_q = t_span_vpartial(q, dims)
    big_R = block_diag(*t_span_mpartial(R, dims))
    big_r = t_span_vpartial(r, dims)
    big_x0 = t_span_vpartial(x0, dims)

    big_G = 2 * (F.T @ big_Q @ F + big_R)
    big_g = (
        F.T @ big_q
        + (F.T @ big_Q @ F0 @ big_x0)
        + (big_x0.T @ F0.T @ big_Q.T @ F)
        + big_r
    )

    def matvec(x):
        return big_G @ x

    us_star = jaxopt.linear_solve.solve_cg(matvec, -big_g)
    xs_star = F0 @ big_x0 + F @ us_star
    return xs_star.reshape((dims.horizon,dims.n,)), us_star.reshape((dims.horizon, dims.m,))[:-1]


def exact_solve(params: LQRParams, dims: ModelDims, x0: Array) -> Tuple[Array, Array]:
    """Solves the optimal control problem using the linear solver method.

    Args:
        params (Params): The parameters for the LQR problem.
        dims (ModelDims): The dimensions of the model.
        x0 (jnp.ndarray): The initial state.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The optimal trajectory and control inputs.
    """

    A = params.lqr.A[0]
    B = params.lqr.B[0]
    Q = params.lqr.Q[0]
    R = params.lqr.R[0]
    q = params.lqr.q[0]
    r = params.lqr.r[0]
    F0 = block_diag(*[matrix_power(A, j) for j in range(dims.horizon)])
    F = np.block(
        [
            [
                (
                    np.linalg.matrix_power(A, i - j - 1) @ B
                    if j < i
                    else np.zeros((dims.n, dims.m))
                )
                for j in range(dims.horizon)
            ]
            for i in range(dims.horizon)
        ]
    )

    big_Q = block_diag(*t_span_mpartial(Q, dims))
    big_q = t_span_vpartial(q, dims)
    big_R = block_diag(*t_span_mpartial(R, dims))
    big_r = t_span_vpartial(r, dims)
    big_x0 = t_span_vpartial(x0, dims)

    big_G = 2 * (F.T @ big_Q @ F + big_R)
    big_G = 0.5 * (big_G + big_G.T)
    big_g = (
        F.T @ big_q
        + (F.T @ big_Q @ F0 @ big_x0)
        + (big_x0.T @ F0.T @ big_Q.T @ F)
        + big_r
    )
    us_star = np.linalg.solve(big_G, -big_g)
    xs_star = F0 @ big_x0 + F @ us_star
    return xs_star.reshape((dims.horizon,dims.n,)), us_star.reshape((dims.horizon, dims.m,))[:-1]
