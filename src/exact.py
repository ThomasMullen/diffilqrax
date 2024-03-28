
from jaxopt import EqualityConstrainedQP as qp
import jax.numpy as jnp
import numpy as np
import jax, scipy
import jaxopt

"""The original problem is \sum_t x_t^T Q x_t + u_t^T R u_t + 2 x_t^T S u_t + c^T x_t subject to 
x_{t+1} = A x_t + B u_t + a, for all t, and x_0 = x0.
This translates into a dynamics constraint of the form x = F_0 x_0 + F u where F0 
is an upper diagonal matrix with blocks F0_{ij} = A^{j - i} if i>=j and 0 otherwise, and F is a block matrix with blocks F_{ij} = A^{j - i} B if i>j and 0 otherwise.


We can build the matrices F0 and F as follows:  
F0 = np.block([[np.linalg.matrix_power(A, j - i) for j in range(T)] for i in range(T)])"""

def quad_solve(params, n, m, T, x0):
    T = T 
    A = params.lqr.A[0].astype(jnp.float64)
    B = params.lqr.B[0].astype(jnp.float64)
    Q = params.lqr.Q[0].astype(jnp.float64)
    R = params.lqr.R[0].astype(jnp.float64)
    q = params.lqr.q[0].astype(jnp.float64)
    r = params.lqr.r[0].astype(jnp.float64)
    #F0 = np.block([[np.linalg.matrix_power(A, i-j) if (j <= i) else np.zeros((n, n)) for j in range(T)] for i in range(T)])
    F0 = jax.scipy.linalg.block_diag(*[np.linalg.matrix_power(A, j) for j in range(T)])
    F = np.block([[np.linalg.matrix_power(A, i-j-1) @ B if j < i else np.zeros((n, m)) for j in range(T)] for i in range(T)])
    #C(U) = U^T@big_R@U + big_r^T@U + X^T@big_Q@X + big_q^T@X and  X = F0x0 + FU so 
    #C(U) = U^T@big_R@U + big_r^T@U + (F0x0 + FU)^T@big_Q*(F0x0 + FU) + big_q^T@(F0x0 + FU) = U^T@big_G@U + big_g^T@U + cg
    #where big_G = @*(F^T@big_Q@F + big_R) and big_g = 2*F^T@big_Q@F0 + big_r and cg = x0^T@F0^T@big_Q@F0@x0 + big_q^T@F0@x0
    #this is minimized by solving Ax = b where A = big_G, b = -big_g
    big_Q = jax.scipy.linalg.block_diag(*[Q for t in range(T)])
    big_q = np.concatenate([q for t in range(T)])
    big_R =  jax.scipy.linalg.block_diag(*[R for t in range(T)])
    big_r = np.concatenate([r for t in range(T)])
    big_x0 = np.concatenate([x0 for t in range(T)])

    big_G = 2*(F.T @ big_Q @ F + big_R)
    big_g =  F.T@big_q + (F.T @ big_Q @ F0 @ big_x0).squeeze()[...,None] + (big_x0.T @ F0.T @ big_Q.T @ F).squeeze()[...,None] + big_r.squeeze()[...,None]
    def matvec(x):
        return big_G @ x
    us_star = jaxopt.linear_solve.solve_cg(matvec, -big_g)
    xs_star = F0 @ big_x0 + F @ us_star
    #c = 0.5*us_star[...,None]^T@big_R@us_star[...,None] + big_r[...,None]^T@us_star[...,None] + 0.5*xs_star[...,None]^T@big_Q@xs_star[...,None] + big_q[...,None]^T@xs_star[...,None]
    u = np.reshape(us_star, (T, m, 1))[:-1]
    x = np.reshape(xs_star, (T, n, 1))
    return x, u
     
def exact_solve(params, n, m, T, x0):
    T = T 
    A = params.lqr.A[0].astype(jnp.float64)
    B = params.lqr.B[0].astype(jnp.float64)
    Q = params.lqr.Q[0].astype(jnp.float64)
    R = params.lqr.R[0].astype(jnp.float64)
    q = params.lqr.q[0].astype(jnp.float64)
    r = params.lqr.r[0].astype(jnp.float64)
    #F0 = np.block([[np.linalg.matrix_power(A, i-j) if j <= i else np.zeros((n, n)) for j in range(T)] for i in range(T)])
    F0 = jax.scipy.linalg.block_diag(*[np.linalg.matrix_power(A, j) for j in range(T)])
    F = np.block([[np.linalg.matrix_power(A, i-j-1) @ B if j < i else np.zeros((n, m)) for j in range(T)] for i in range(T)])
    #C(U) = U^T@big_R@U + big_r^T@U + X^T@big_Q@X + big_q^T@X and  X = F0x0 + FU so 
    #C(U) = U^T@big_R@U + big_r^T@U + (F0x0 + FU)^T@big_Q*(F0x0 + FU) + big_q^T@(F0x0 + FU) = U^T@big_G@U + big_g^T@U + cg
    #where big_G = @*(F^T@big_Q@F + big_R) and big_g = 2*F^T@big_Q@F0 + big_r and cg = x0^T@F0^T@big_Q@F0@x0 + big_q^T@F0@x0
    #this is minimized by solving Ax = b where A = big_G, b = -big_g
    big_Q = jax.scipy.linalg.block_diag(*[Q for t in range(T)])
    big_q = np.concatenate([q for t in range(T)])
    big_R =  jax.scipy.linalg.block_diag(*[R for t in range(T)])
    big_r = np.concatenate([r for t in range(T)])
    big_x0 = np.concatenate([x0 for t in range(T)])

    big_G = 2*(F.T @ big_Q @ F + big_R)
    big_G = 0.5*(big_G + big_G.T)
    big_g =  F.T@big_q + (F.T @ big_Q @ F0 @ big_x0).squeeze()[...,None] + (big_x0.T @ F0.T @ big_Q.T @ F).squeeze()[...,None] + big_r.squeeze()[...,None]
    us_star = np.linalg.solve(big_G, -big_g)
    xs_star = F0 @ big_x0 + F @ us_star
    #c = 0.5*us_star[...,None]^T@big_R@us_star[...,None] + big_r[...,None]^T@us_star[...,None] + 0.5*xs_star[...,None]^T@big_Q@xs_star[...,None] + big_q[...,None]^T@xs_star[...,None]
    u = np.reshape(us_star, (T, m, 1))[:-1]
    x = np.reshape(xs_star, (T, n, 1))
    return x, u