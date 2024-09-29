"""
Testing KKT optimality conditions for LQR problem under different parameterizations
"""
from pathlib import Path
from functools import partial
from typing import Tuple
import unittest
from os import getcwd
import chex
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import numpy as np
from matplotlib.pyplot import subplots, close, style

from diffilqrax.typs import (
    LQR,
    LQRParams,
    ModelDims,
)
from diffilqrax.lqr import (
    simulate_trajectory,
    lqr_adjoint_pass,
    lin_dyn_step,
    lqr_forward_pass,
    lqr_backward_pass,
    solve_lqr,
    kkt,
)
from diffilqrax.exact import quad_solve, exact_solve
from diffilqrax.utils import keygen, initialise_stable_dynamics

jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_enable_x64", True)  # double precision


# helper functions
def _trim_horizon(lqr_params: LQRParams, horizon: int) -> LQRParams:
    _pop_horizon = partial(jax.tree_map, lambda x: x[:horizon])
    return LQRParams(lqr_params.x0, LQR(_pop_horizon(lqr_params.lqr[:-2]), lqr_params.lqr[-2:]))

def _tile_lqr_params(lqr_params: LQRParams, T: int) -> LQRParams:
    
    _tile_mat = lambda x: jnp.tile(x, (T, 1, 1))
    _tile_vec = lambda x: jnp.tile(x, (T, 1))
    return LQRParams(lqr_params.x0, 
                     LQR(
                        _tile_mat(lqr_params.lqr.A), 
                        _tile_mat(lqr_params.lqr.B), 
                        _tile_vec(lqr_params.lqr.a), 
                        _tile_mat(lqr_params.lqr.Q), 
                        _tile_vec(lqr_params.lqr.q), 
                        _tile_mat(lqr_params.lqr.R), 
                        _tile_vec(lqr_params.lqr.r), 
                        _tile_mat(lqr_params.lqr.S), 
                        lqr_params.lqr.Qf,
                        lqr_params.lqr.qf
                        )
                     )

def _init_lqr_params(n: int, m: int, seed:int=234) -> LQRParams:
    key = jr.PRNGKey(seed=seed)
    key, skeys = keygen(key, 4)
    A=initialise_stable_dynamics(next(skeys),n,1,radii=0.1)
    B = jr.normal(next(skeys), (n,m))
    a = jr.normal(next(skeys), (n,))
    
    Q = 0.8 * jnp.eye(n)
    q = 1e-1 * jnp.ones(n)
    R = 5e-1 *jnp.eye(m)
    r = 1e-1 * jnp.ones(m)
    S = 0.2 * jnp.ones((n,m))
    Qf = 0.8 * jnp.eye(n)
    qf = 1e-1 * jnp.ones(n)
    
    x0 = jr.normal(next(skeys), (n,))
    lqr = LQR(A, B, a, Q, q, R, r, S, Qf, qf)
    return LQRParams(x0, lqr)


def _abs_sum_val(x: Array) -> float:
    return np.float64(round(jnp.sum(jnp.abs(x)),3))

def _abs_mean_val(x: Array) -> float:
    return np.float64(round(jnp.mean(jnp.abs(x)),3))


def compute_kkt_val(lqr_params: LQRParams, horizon: int) -> Tuple[[Array, Array, Array]]:
    lqr_params = _tile_lqr_params(lqr_params, horizon)
    state = solve_lqr(lqr_params)
    return kkt(lqr_params, *state)


def plot_kkt_val(lqr_ps, T, ax):
    dLs = compute_kkt_val(lqr_ps, T)
    print(f"n:{n}, T:{T}, (<|DLDX|>, <|DLDU|>, <|DLDΛ|>):{tuple(_abs_mean_val(x) for x in dLs)}")
    ax.plot(dLs[1])


if __name__ == "__main__":
    horizon_iterations = [40,800,1600, 2000, 3000]
    n_iterations = [2,8,20,80,160]
    n_dim = 60
    m_dim = n_dim
    
    # (Q, R, q, r, A, B, a)
    print("(Q, R, q, r, A, B, a)")
    fig, axes = subplots(5, 5, figsize=(12, 12), sharey=True, sharex=True)
    for i, n in enumerate(n_iterations):
        m_dim = n
        lqr_ps = _init_lqr_params(n, m_dim, seed=100)
        plot_partial = partial(plot_kkt_val, lqr_ps)
        list(map(plot_partial, horizon_iterations, axes[i, :]))
    
    # (Q,R,q,r,A,B,a)
    print("(Q,R,q,r,A,B,a)")
    fig, axes = subplots(5,5, figsize=(12,12),sharey=True, sharex=True)
    for i, n in enumerate(n_iterations):
        m_dim = n
        lqr_ps = _init_lqr_params(n, m_dim, seed=1000)
        for j, T in enumerate(horizon_iterations):
            # horizon = horizon_iterations[i]
            dLs = compute_kkt_val(lqr_ps, T)
            print(f"n:{n}, T:{T}, (<|DLDX|>, <|DLDU|>, <|DLDΛ|>):{tuple(_abs_mean_val(x) for x in dLs)}")
            axes[i, j].plot(dLs[1])
    
    
    # (Q,R,q,A,B,a)
    print("(Q,R,q,A,B,a)")
    fig, axes = subplots(5, 5, figsize=(12, 12), sharey=True, sharex=True)
    for i, n in enumerate(n_iterations):
        m_dim = n
        lqr_ps = _init_lqr_params(n, m_dim, seed=100)
        lqr_ps = lqr_ps._replace(lqr=lqr_ps.lqr._replace(r=lqr_ps.lqr.r*0.))
        plot_partial = partial(plot_kkt_val, lqr_ps)
        list(map(plot_partial, horizon_iterations, axes[i, :]))
    
    
    # (Q,R,A,B,a)
    print("(Q,R,A,B,a)")
    fig, axes = subplots(5, 5, figsize=(12, 12), sharey=True, sharex=True)
    for i, n in enumerate(n_iterations):
        m_dim = n
        lqr_ps = _init_lqr_params(n, m_dim, seed=100)
        lqr_ps = lqr_ps._replace(lqr=lqr_ps.lqr._replace(r=lqr_ps.lqr.r*0.))
        lqr_ps = lqr_ps._replace(lqr=lqr_ps.lqr._replace(q=lqr_ps.lqr.q*0.))
        plot_partial = partial(plot_kkt_val, lqr_ps)
        list(map(plot_partial, horizon_iterations, axes[i, :]))
    
    
    # (Q,R,A,B)
    print("(Q,R,A,B)")
    fig, axes = subplots(5, 5, figsize=(12, 12), sharey=True, sharex=True)
    for i, n in enumerate(n_iterations):
        m_dim = n
        lqr_ps = _init_lqr_params(n, m_dim, seed=100)
        lqr_ps = lqr_ps._replace(lqr=lqr_ps.lqr._replace(r=lqr_ps.lqr.r*0.))
        lqr_ps = lqr_ps._replace(lqr=lqr_ps.lqr._replace(q=lqr_ps.lqr.q*0.))
        lqr_ps = lqr_ps._replace(lqr=lqr_ps.lqr._replace(a=lqr_ps.lqr.a*0.))
        plot_partial = partial(plot_kkt_val, lqr_ps)
        list(map(plot_partial, horizon_iterations, axes[i, :]))
    
    
    # (Q,R,r,A,B,a)
    print("(Q,R,r,A,B,a)")
    fig, axes = subplots(5, 5, figsize=(12, 12), sharey=True, sharex=True)
    for i, n in enumerate(n_iterations):
        m_dim = n
        lqr_ps = _init_lqr_params(n, m_dim, seed=100)
        lqr_ps = lqr_ps._replace(lqr=lqr_ps.lqr._replace(q=lqr_ps.lqr.q*0.))
        plot_partial = partial(plot_kkt_val, lqr_ps)
        list(map(plot_partial, horizon_iterations, axes[i, :]))
    
    
    # (Q,R,r,A,B)
    print("(Q,R,r,A,B)")
    fig, axes = subplots(5, 5, figsize=(12, 12), sharey=True, sharex=True)
    for i, n in enumerate(n_iterations):
        m_dim = n
        lqr_ps = _init_lqr_params(n, m_dim, seed=100)
        lqr_ps = lqr_ps._replace(lqr=lqr_ps.lqr._replace(q=lqr_ps.lqr.q*0.))
        lqr_ps = lqr_ps._replace(lqr=lqr_ps.lqr._replace(a=lqr_ps.lqr.a*0.))
        plot_partial = partial(plot_kkt_val, lqr_ps)
        list(map(plot_partial, horizon_iterations, axes[i, :]))


# make a time invariant LQR problem

    # iterate and tile through different horizons

    # swap out different params in order (Q,R,q,r,A,B,a), (Q,R,q,A,B,a), (Q,R,A,B,a), (Q,R,A,B,a), (Q,R,A,B)
    
# each iteration shoulds solver then calculate the KKT conditions
# calc abs mean of KKT conditions for each variable
# plot the results