
from jaxopt import EqualityConstrainedQP as qp
import jax.numpy as jnp
import numpy as np
import jax, scipy
import jaxopt
from jax.scipy.linalg import block_diag
from jax.numpy.linalg import matrix_power

from src import ModelDims, Params

jax.config.update("jax_enable_x64", True)  # sets float to 64 precision by default

"""The original problem is \sum_t x_t^T Q x_t + u_t^T R u_t + 2 x_t^T S u_t + c^T x_t subject to 
x_{t+1} = A x_t + B u_t + a, for all t, and x_0 = x0.
This translates into a dynamics constraint of the form x = F_0 x_0 + F u where F0 
is an upper diagonal matrix with blocks F0_{ij} = A^{j - i} if i>=j and 0 otherwise, and F is a block matrix with blocks F_{ij} = A^{j - i} B if i>j and 0 otherwise.


We can build the matrices F0 and F as follows:  
F0 = np.block([[np.linalg.matrix_power(A, j - i) for j in range(T)] for i in range(T)]) -> Follows Toeplitz structure!"""

def quad_solve(params:Params, dims:ModelDims, x0:jnp.ndarray):
    t_span_mpartial = lambda arr: jnp.tile(arr, (dims.horizon,1,1))
    t_span_vpartial = lambda arr: jnp.tile(arr, (dims.horizon,))

    A = params.lqr.A[0]
    B = params.lqr.B[0]
    Q = params.lqr.Q[0]
    R = params.lqr.R[0]
    q = params.lqr.q[0]
    r = params.lqr.r[0]
    #F0 = np.block([[np.linalg.matrix_power(A, i-j) if (j <= i) else np.zeros((n, n)) for j in range(T)] for i in range(T)])
    F0 = block_diag(*[matrix_power(A, j) for j in range(dims.horizon)])
    # F = jnp.block([[matrix_power(A, i-j-1) @ B if j < i else jnp.zeros((dims.n, dims.m)) for j in range(dims.horizon)] for i in range(dims.horizon)])
    F = np.block([[np.linalg.matrix_power(A, i-j-1) @ B if j < i else np.zeros((dims.n, dims.m)) for j in range(dims.horizon)] for i in range(dims.horizon)])
    #C(U) = U^T@big_R@U + big_r^T@U + X^T@big_Q@X + big_q^T@X and  X = F0x0 + FU so 
    #C(U) = U^T@big_R@U + big_r^T@U + (F0x0 + FU)^T@big_Q*(F0x0 + FU) + big_q^T@(F0x0 + FU) = U^T@big_G@U + big_g^T@U + cg
    #where big_G = @*(F^T@big_Q@F + big_R) and big_g = 2*F^T@big_Q@F0 + big_r and cg = x0^T@F0^T@big_Q@F0@x0 + big_q^T@F0@x0
    #this is minimized by solving Ax = b where A = big_G, b = -big_g
    big_Q = block_diag(*t_span_mpartial(Q))
    big_q = t_span_vpartial(q)
    big_R =  block_diag(*t_span_mpartial(R))
    big_r = t_span_vpartial(r)
    big_x0 = t_span_vpartial(x0)
    # big_Q = jax.scipy.linalg.block_diag(*[Q for t in range(dims.horizon)])
    # big_q = np.concatenate([q for t in range(dims.horizon)])
    # big_r = np.concatenate([r for t in range(dims.horizon)])
    # big_x0 = np.concatenate([x0 for t in range(dims.horizon)])

    big_G = 2*(F.T @ big_Q @ F + big_R)
    big_g =  F.T@big_q + (F.T @ big_Q @ F0 @ big_x0) + (big_x0.T @ F0.T @ big_Q.T @ F) + big_r
    def matvec(x):
        return big_G @ x
    us_star = jaxopt.linear_solve.solve_cg(matvec, -big_g)
    xs_star = F0 @ big_x0 + F @ us_star
    #c = 0.5*us_star[...,None]^T@big_R@us_star[...,None] + big_r[...,None]^T@us_star[...,None] + 0.5*xs_star[...,None]^T@big_Q@xs_star[...,None] + big_q[...,None]^T@xs_star[...,None]
    u = np.reshape(us_star, (dims.horizon, dims.m,))[:-1]
    x = np.reshape(xs_star, (dims.horizon, dims.n,))
    return x, u
     
def exact_solve(params:Params, dims:ModelDims, x0:jnp.ndarray):
    t_span_mpartial = lambda arr: jnp.tile(arr, (dims.horizon,1,1))
    t_span_vpartial = lambda arr: jnp.tile(arr, (dims.horizon,))

    A = params.lqr.A[0]
    B = params.lqr.B[0]
    Q = params.lqr.Q[0]
    R = params.lqr.R[0]
    q = params.lqr.q[0]
    r = params.lqr.r[0]
    #F0 = np.block([[np.linalg.matrix_power(A, i-j) if j <= i else np.zeros((n, n)) for j in range(T)] for i in range(T)])
    F0 = block_diag(*[matrix_power(A, j) for j in range(dims.horizon)])
    F = np.block([[np.linalg.matrix_power(A, i-j-1) @ B if j < i else np.zeros((dims.n, dims.m)) for j in range(dims.horizon)] for i in range(dims.horizon)])
    #C(U) = U^T@big_R@U + big_r^T@U + X^T@big_Q@X + big_q^T@X and  X = F0x0 + FU so 
    #C(U) = U^T@big_R@U + big_r^T@U + (F0x0 + FU)^T@big_Q*(F0x0 + FU) + big_q^T@(F0x0 + FU) = U^T@big_G@U + big_g^T@U + cg
    #where big_G = @*(F^T@big_Q@F + big_R) and big_g = 2*F^T@big_Q@F0 + big_r and cg = x0^T@F0^T@big_Q@F0@x0 + big_q^T@F0@x0
    #this is minimized by solving Ax = b where A = big_G, b = -big_g
    big_Q = block_diag(*t_span_mpartial(Q))
    big_q = t_span_vpartial(q)
    big_R =  block_diag(*t_span_mpartial(R))
    big_r = t_span_vpartial(r)
    big_x0 = t_span_vpartial(x0)

    big_G = 2*(F.T @ big_Q @ F + big_R)
    big_G = 0.5*(big_G + big_G.T)
    big_g =  F.T@big_q + (F.T @ big_Q @ F0 @ big_x0) + (big_x0.T @ F0.T @ big_Q.T @ F) + big_r
    us_star = np.linalg.solve(big_G, -big_g)
    xs_star = F0 @ big_x0 + F @ us_star
    #c = 0.5*us_star[...,None]^T@big_R@us_star[...,None] + big_r[...,None]^T@us_star[...,None] + 0.5*xs_star[...,None]^T@big_Q@xs_star[...,None] + big_q[...,None]^T@xs_star[...,None]
    u = np.reshape(us_star, (dims.horizon, dims.m,))[:-1]
    x = np.reshape(xs_star, (dims.horizon, dims.n,))
    return x, u