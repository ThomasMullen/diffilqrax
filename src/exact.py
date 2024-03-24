
from jaxopt import EqualityConstrainedQP as qp
import jax.numpy as jnp
import numpy as np
import jax, scipy
import jaxopt

T = 10
Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
q = 2 * jnp.array([[2.0, 0.5]])
R = 2 * np.array([1.0])
r = 2 * np.array([[1.0]])
c = np.array([1.0, 1.0])
A = np.array([[1.0, 1.0], [1.0, 2.0]])
B = np.array([[1.0], [1.0]])
print(B.shape)
b = np.array([1.0])
x0 = np.array([[0.,0.]]).T
n, m = 2,1

# def quad_solve(params):
#     T = 10 #params.dims.T
#     Q = params.LQR.Q
#     c = 
# Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
# c = jnp.array([1.0, 1.0])
# A = jnp.array([[1.0, 1.0]])
# b = jnp.array([1.0])

#params_obj = #Q and c in min (x^TQx/2 + c^Tx)
#params_eq = #A,b : should be for the constraint Ax = b
"""The original problem is \sum_t x_t^T Q x_t + u_t^T R u_t + 2 x_t^T S u_t + c^T x_t subject to 
x_{t+1} = A x_t + B u_t + a, for all t, and x_0 = x0.
This translates into a dynamics constraint of the form x = F_0 x_0 + F u where F0 
is an upper diagonal matrix with blocks F0_{ij} = A^{j - i} if i>=j and 0 otherwise, and F is a block matrix with blocks F_{ij} = A^{j - i} B if i>j and 0 otherwise.


We can build the matrices F0 and F as follows:  
F0 = np.block([[np.linalg.matrix_power(A, j - i) for j in range(T)] for i in range(T)])"""

# sol = qp.run(params_obj=params_obj, params_eq=params_eq).params

# print(sol.primal)
# print(sol.dual_eq)


#  implicit_diff.custom_root(
#             kkt, linear_solve.solve_cg
#         )(solve_lqr)(self.params)


def quad_solve(params, n, m, T, x0):
    T = T 
    A = params.lqr.A[0]
    B = params.lqr.B[0]
    Q = params.lqr.Q[0]
    R = params.lqr.R[0]
    q = params.lqr.q[0]
    r = params.lqr.r[0]
    F0 = np.block([[np.linalg.matrix_power(A, i-j) if j <= i else np.zeros((n, n)) for j in range(T)] for i in range(T)])
    F = np.block([[np.linalg.matrix_power(A, i-j-1) @ B if j < i else np.zeros((n, m)) for j in range(T)] for i in range(T)])
    print(F, F.shape)
    #we have X = F0x0 + FU, but want to write it as Atau = b where tau = [x,u]
    #C(U) = 0.5*U^T@big_R@U + big_r^T@U + 0.5*X^T@big_Q@X + big_q^T@X and  X = F0x0 + FU so 
    #C(U) = 0.5*U^T@big_R@U + big_r^T@U + 0.5*(F0x0 + FU)^T@big_Q*(F0x0 + FU) + big_q^T@(F0x0 + FU) = 0.5*U^T@big_G@U + big_g^T@U + cg
    #where big_G = F^T@big_Q@F + big_R and big_g = F^T@big_Q@F0 + big_r and cg = 0.5*x0^T@F0^T@big_Q@F0@x0 + big_q^T@F0@x0
    #this is minimized by solving Ax = b where A = big_G, b = -big_g
    big_Q = jax.scipy.linalg.block_diag(*[Q for t in range(T)])
    big_q = np.concatenate([q for t in range(T)])
    big_R =  jax.scipy.linalg.block_diag(*[R for t in range(T)])
    big_r = np.concatenate([r for t in range(T)])
    big_x0 = np.concatenate([x0 for t in range(T)])

    big_G = F.T @ big_Q @ F + big_R
    print(F0.shape, big_r.shape, x0.shape)
    big_g = (F.T @ big_Q @ F0 @ big_x0).squeeze()[...,None] + big_r.squeeze()[...,None]
    print(big_g.shape)
    def matvec(x):
        return big_G @ x
    us_star = jaxopt.linear_solve.solve_cg(matvec, -big_g)
    xs_star = F0 @ big_x0 + F @ us_star
    #c = 0.5*us_star[...,None]^T@big_R@us_star[...,None] + big_r[...,None]^T@us_star[...,None] + 0.5*xs_star[...,None]^T@big_Q@xs_star[...,None] + big_q[...,None]^T@xs_star[...,None]
    u = np.reshape(us_star, (T, m, 1))
    x = np.reshape(xs_star, (T, n, 1))
    return x, u
     
    