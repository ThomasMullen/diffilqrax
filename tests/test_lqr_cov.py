"""
Unit test for the LQR covariance module
"""

from pathlib import Path
from typing import NamedTuple, Tuple
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

dt=0.1
# define state, obs, time DIMS
DIMS=ModelDims(n=4,m=2,horizon=10**4,dt=dt)

def make_tracking_model(q: float, dt: float, r: float, m0: jnp.ndarray, P0: jnp.ndarray)->LQR:
    # define state, obs, time params
    A = jnp.eye(DIMS.n) + dt * jnp.eye(DIMS.n, k=2)
    H = jnp.eye(DIMS.m, DIMS.n)
    Q = jnp.kron(jnp.array([[dt**3/3, dt**2/2], [dt**2/2, dt]]), jnp.eye(DIMS.m))
    R = r ** 2 * jnp.eye(DIMS.m)
    
    # In Kalman problem: swap B, Qf, qf with H, P0, m0
    return LQR(A=A, 
               B=H,
               a= jnp.zeros(DIMS.n),
               Q= q * Q,
               q= jnp.zeros(DIMS.n),
               R= R,
               r= jnp.zeros(DIMS.m),
               S= jnp.zeros_like(H),
               Qf= P0,
               qf= m0)
    

def convert_kalman_to_lqr_problem(target:Array, *args)->LQR:
    A, C, Q, R = args
    
    T=10000
    span_time_m=(T-1,1,1)
    span_time_v=(T-1,1)
    
    # absorb target in cost
    q= -1.*jnp.einsum("ij,kj,lk->li", Q, C, target)
    
    return LQR(
        A=jnp.tile(A[None],span_time_m),
        # B=jnp.tile(C.T[None],span_time_m),
        B=jnp.tile(jnp.array([[0., 0.], [0., 0.], [1.*DIMS.dt, 0.], [0., 1.*DIMS.dt]]),span_time_m),
        a=jnp.zeros((T-1,A.shape[0],)),
        Q=jnp.tile(Q[None],span_time_m),
        q=q[:-1],
        R=jnp.tile(R[None],span_time_m),
        r=jnp.zeros(span_time_v),
        S=jnp.zeros((T-1,)+C.T.shape),
        Qf=Q,
        qf=q[-1]        
    )

def generate_data(model: LQR, model_dims:ModelDims=DIMS, T:float=10**4, seed:int=0)->Tuple[Array, Array]:
    # We first generate the normals we will be using to simulate the SSM:
    key = jr.PRNGKey(seed=seed)
    key, skeys = keygen(key, 3)
    normals = jr.normal(next(skeys), (1 + T, model_dims.n + model_dims.m))
    
    # Then we allocate the arrays where the simulated path and observations will
    # be stored:
    xs = jnp.empty((T, model_dims.n))
    ys = jnp.empty((T, model_dims.m))

    # So that we can now run the sampling routine:
    Q_chol = jsc.linalg.cholesky(model.Q, lower=True)
    R_chol = jsc.linalg.cholesky(model.R, lower=True)
    P0_chol = jsc.linalg.cholesky(model.Qf, lower=True)
    x = model.qf + P0_chol @ normals[0, :model_dims.n]
    for i, norm in enumerate(normals[1:]):
        x = model.A @ x + Q_chol @ norm[:model_dims.n]
        y = model.B @ x + R_chol @ norm[model_dims.n:]
        xs = xs.at[i].set(x)
        ys = ys.at[i].set(y)
    return xs, ys    


def kalman_filter(model:LQR, observations:Array)->Tuple[Array, Array]:
    def body(carry, y):
        m, P = carry
        m = model.A @ m
        P = model.A @ P @ model.A.T + model.Q

        obs_mean = model.B @ m
        S = model.B @ P @ model.B.T + model.R

        K = jsc.linalg.solve(S, model.B @ P, assume_a='pos').T  # notice the jsc here
        m = m + K @ (y - model.B @ m)
        P = P - K @ S @ K.T
        return (m, P), (m, P)

    _, (fms, fPs) = scan(body, (model.qf, model.Qf), observations)
    return fms, fPs


def kalman_smoother(model:LQR, ms:Array, Ps:Array)->Tuple[Array, Array]:
    def body(carry, inp):
        m, P = inp
        sm, sP = carry

        pm = model.A @ m
        pP = model.A @ P @ model.A.T + model.Q

        C = jsc.linalg.solve(pP, model.A @ P, assume_a='pos').T  # notice the jsc here
        
        sm = m + C @ (sm - pm)
        sP = P + C @ (sP - pP) @ C.T
        return (sm, sP), (sm, sP)

    _, (sms, sPs) = scan(body, (ms[-1], Ps[-1]), (ms[:-1], Ps[:-1]), reverse=True)
    sms = jnp.append(sms, jnp.expand_dims(ms[-1], 0), 0)
    sPs = jnp.append(sPs, jnp.expand_dims(Ps[-1], 0), 0)
    return sms, sPs


def save_kalman_solution(exp_dir="/Users/thomasmullen/VSCodeProjects/ilqr_vae_jax/tests/fixtures"):
    # make tracking model
    tracking_model = make_tracking_model(q=1., dt=0.1, r=0.5, 
                                             m0=jnp.array([0., 0., 1., -1.]), 
                                             P0=jnp.eye(4))
    # Generate true and partial trajectory
    true_xs, ys = generate_data(tracking_model, DIMS, DIMS.horizon, seed=0)
    # kalman reverse smoother
    sms, sPs = kalman_smoother(tracking_model, *kalman_filter(tracking_model, ys))
    # save results
    with h5py.File(f'{exp_dir}/covariance_fixtures_1.h5', 'w') as f:
        gen_data = f.create_group('data')
        dset = gen_data.create_dataset(f'true_x', data=true_xs)
        dset = gen_data.create_dataset(f'partial_obs', data=ys)

        theta_params = f.create_group('theta')
        dset = theta_params.create_dataset(f'A', data=tracking_model.A)
        dset = theta_params.create_dataset(f'C', data=tracking_model.B)
        dset = theta_params.create_dataset(f'Q', data=tracking_model.Q)
        dset = theta_params.create_dataset(f'R', data=tracking_model.R)
        dset = theta_params.create_dataset(f'm0', data=tracking_model.qf)
        dset = theta_params.create_dataset(f'P0', data=tracking_model.Qf)

        ks_res = f.create_group('ks')
        dset = ks_res.create_dataset(f'ms', data=sms)
        dset = ks_res.create_dataset(f'Ps', data=sPs)

        f.attrs['T']=10**4
        f.attrs['dt']=0.1
        f.attrs['r']=0.5
        f.attrs['q']=1.
    return 


if __name__ == "__main__":

    # save kalman solution
    save_kalman_solution()

    exp_dir="/Users/thomasmullen/VSCodeProjects/ilqr_vae_jax/tests/fixtures"
    with h5py.File(f'{exp_dir}/covariance_fixtures_1.h5', 'r') as f:
        # observation
        obs = f['data']['true_x'][:10000]
        partial_obs = f['data']['partial_obs'][:10000]
        # kalman res
        sms = f['ks']['ms'][:10000]
        sps = f['ks']['Ps'][:10000]
        # params
        A = f['theta']['A'][()]
        C = f['theta']['C'][()]
        Q = f['theta']['Q'][()]
        R = f['theta']['R'][()]
        m0 = f['theta']['m0'][()]
        P0 = f['theta']['P0'][()]


    # load lqr_theta from fixtures
    lqr_theta = convert_kalman_to_lqr_problem(partial_obs, A, C, Q, R)
    # lqr_theta = convert_kalman_to_lqr_problem(obs, A, C.T, Q, R)
    
    # initial state distribution
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 5)
    m0=jnp.array([0., 0., 1., -1.])
    P0=jnp.eye(4)
    # x0 = jr.multivariate_normal(next(skeys), m0, P0)
    x0 = obs[0]

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
    ax.plot(obs[:10000, 0], obs[:10000, 1], label="True State", color="b")
    ax.plot(sms[:10000, 0], sms[:10000, 1], label="Kalman Smoother", color="k", linestyle="--")
    ax.plot(Xs[:10000, 0], Xs[:10000, 1], label="LQR", color="g", linestyle="--")
    # ax.set(
    #     xlim=[0,4000],
    #     ylim=[-4000,0],
    # )
    ax.legend()
    
    
    
    