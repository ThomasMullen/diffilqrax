"""
Unit test for the LQR covariance module
"""

from pathlib import Path
from typing import NamedTuple
import unittest
from os import getcwd
import chex
import h5py
import jax
from jax import Array, vmap
from jax.typing import ArrayLike
from jax.lax import scan
import jax.random as jr
import jax.numpy as jnp
import jax.scipy as jsc
import numpy as onp
from matplotlib.pyplot import subplots, close, style
import matplotlib.pyplot as plt

from diffilqrax.utils import keygen, initialise_stable_dynamics
from diffilqrax import ilqr
from diffilqrax import lqr
from diffilqrax.typs import (
    iLQRParams,
    LQR,
    LQRParams,
    System,
    ModelDims,
)
from diffilqrax.lqr import (
    simulate_trajectory,
    lqr_adjoint_pass,
    lin_dyn_step,
    lqr_forward_pass,
    lqr_backward_pass,
    # solve_lqr,
    kkt,
)
from diffilqrax.lqr_cov import solve_lqr
from diffilqrax.exact import quad_solve, exact_solve

jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_enable_x64", True)  # double precision

PLOT_URL = ("https://gist.githubusercontent.com/"
       "ThomasMullen/e4a6a0abd54ba430adc4ffb8b8675520/"
       "raw/1189fbee1d3335284ec5cd7b5d071c3da49ad0f4/"
       "figure_style.mplstyle")
style.use(PLOT_URL)

# LOG2PI = jnp.log(2*jnp.pi)

# class Theta(NamedTuple):
#     """RNN parameters"""

#     A: Array
#     B: Array
#     C: Array
#     Q: Array
#     R: Array


def convert_tracking_to_lqr(target:Array, *args)->LQR:
    A, C, Q, R = args
    
    T=100
    span_time_m=(T-1,1,1)
    span_time_v=(T-1,1)
    
    # absorb target in cost
    q= -2.*jnp.einsum("ij,kj,lk->li", Q, C, target)
    # q= -1.*jnp.einsum("ij,kj->ki", Q, target)
    
    return LQR(
        A=jnp.tile(A[None],span_time_m),
        B=jnp.tile(.5*C.T[None],span_time_m),
        a=jnp.zeros((T-1,A.shape[0],)),
        Q=jnp.tile(Q[None],span_time_m),
        q=q[:-1],
        R=jnp.tile(R[None],span_time_m),
        r=jnp.zeros(span_time_v),
        S=jnp.zeros((T-1,)+C.T.shape),
        Qf=Q,
        qf=q[-1]        
    )


# def setup_model(obs_sample:Array):
    
#         def cost(t: int, x: Array, u: Array, theta: Theta):
#             o = obs_sample[t]
#             c_xx = 0.5 * ((theta.C@x - o).T @ theta.A @ (theta.C@x - o))
#             c_uu = 0.5 * (u.T @ theta.R @ u)
#             return c_xx+c_uu

#         def costf(x: Array, theta: Theta):
#             o = obs_sample[-1]
#             c_xx = jnp.linalg.norm(theta.C@x - o) * theta.std
#             return c_xx

#         def dynamics(t: int, x: Array, u: Array, theta: Theta):
#             return theta.A @ x + theta.B @ u

#         model = System(
#             cost, costf, dynamics, ModelDims(horizon=100, n=4, m=2, dt=0.1)
#         )
#         return model
    
# dims=ModelDims(horizon=100, n=4, m=2, dt=0.1)

# # define LQR problem
# q=1.
# r=0.5
# A = jnp.eye(dims.n) + dims.dt * jnp.eye(dims.n, k=2)
# B = jnp.eye(dims.n, dims.m)
# Q = q * jnp.kron(jnp.array([[dims.dt**3/3, dims.dt**2/2],
#                         [dims.dt**2/2, dims.dt]]), 
#             jnp.eye(dims.m))
# R = r ** 2 * jnp.eye(dims.m)
# ilqr_thetas=Theta(A,B,Q,R)

exp_dir="/Users/thomasmullen/VSCodeProjects/ilqr_vae_jax/tests/fixtures"
with h5py.File(f'{exp_dir}/covariance_fixtures.h5', 'r') as f:
    # observation
    obs = f['data']['true_x'][:100]
    partial_obs = f['data']['partial_obs'][:100]
    # kalman res
    sms = f['ks']['ms'][:100]
    sps = f['ks']['Ps'][:100]
    fms = f['kf']['ms'][:100]
    fps = f['kf']['Ps'][:100]
    # params
    A = f['theta']['A'][()]
    C = f['theta']['C'][()]
    Q = f['theta']['Q'][()]
    R = f['theta']['R'][()]
    m0 = f['theta']['m0'][()]
    P0 = f['theta']['P0'][()]
    
# load lqr_theta
lqr_theta = convert_tracking_to_lqr(partial_obs, A, C, Q, R)
# lqr_theta = convert_tracking_to_lqr(obs, A, C, Q, R)
    
# initial state distribution
key = jr.PRNGKey(seed=234)
key, skeys = keygen(key, 5)
m0=jnp.array([0., 0., 1., -1.])
P0=jnp.eye(4)
# x0 = jr.multivariate_normal(next(skeys), m0, P0)
x0 = obs[0]
u_init = jr.normal(next(skeys), (100, 2,))

lqr_params=LQRParams(x0=x0, lqr=lqr_theta)
gains, Xs, Us, Lambs, (xcvs, ucvs, ps) = solve_lqr(lqr_params)

fig, ax = plt.subplots(1,3,sharey=True)
ax[0].plot(obs)
ax[0].set(title="True")
ax[1].plot(sms)
ax[1].set(title="Kalman smoother")
ax[2].plot(Xs)
ax[2].set(title="LQR")
fig.suptitle("State trajectory")

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(obs[:100, 0], obs[:100, 1], label="True State", color="b")
ax.plot(Xs[:100, 0], Xs[:100, 1], label="lqr", color="g", linestyle="--")
ax.plot(sms[:100, 0], sms[:100, 1], label="Smoothed", color="k", linestyle="--")
# ax.scatter(*ys[:100].T, label="Observations", color="r")
_ = plt.legend()


# ilqr_params = iLQRParams(x0=x0, theta=ilqr_thetas)

# ilqr_model = setup_model(obs_sample=None)

# # forward pass

# # ilqr solve
# (Xs_init, Us_init), cost_init = ilqr.ilqr_simulate(model, Us, params)

# # exercise ilqr solver
# (Xs_stars, Us_stars, Lambs_stars), total_cost, _ = ilqr.ilqr_solver(
#     model,
#     params,
#     Us,
#     max_iter=70,
#     tol=1e-8,
#     alpha_init=0.8,
#     verbose=True,
#     use_linesearch=False,
# )

