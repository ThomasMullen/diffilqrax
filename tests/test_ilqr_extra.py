"""Test functions in diffilqrax/ilqr.py"""
import unittest
from functools import partial
from typing import Any, NamedTuple, Tuple
from pathlib import Path
from os import getcwd
import chex
import jax
from jax import lax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
from matplotlib.pyplot import subplots, close, style

from diffilqrax.utils import keygen, initialise_stable_dynamics
from diffilqrax import ilqr, utils
from diffilqrax import lqr
from diffilqrax.typs import (
    iLQRParams,
    LQR,
    LQRParams,
    System,
    ModelDims,
    Theta,
)

#jax.config.update('jax_default_device', jax.devices('cpu')[0])
#jax.config.update("jax_enable_x64", True)  # double precision

PLOT_URL = ("https://gist.githubusercontent.com/"
       "ThomasMullen/e4a6a0abd54ba430adc4ffb8b8675520/"
       "raw/1189fbee1d3335284ec5cd7b5d071c3da49ad0f4/"
       "figure_style.mplstyle")
style.use(PLOT_URL)

# helper functions
def rk4(dynamics, dt=0.01):
    def integrator(x, u):
        dt2 = dt / 2.0
        k1 = dynamics(x, u)
        k2 = dynamics(x + dt2 * k1, u)
        k3 = dynamics(x + dt2 * k2, u)
        k4 = dynamics(x + dt * k3, u)
        nx_x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return nx_x, nx_x
    return integrator

def euler(dynamics, dt=0.01):
    def integrator(x, u):
        nx_x = x + dynamics(x, u) * dt
        return nx_x, nx_x
    return integrator



# define different nonlinear systems
# -------------------------------------
class LorenzeParams(NamedTuple):
    sigma:float # = 10.
    rho:float # = 28.
    beta:float # = 8. / 3.


def lorenz_system(current_state:Array, u:Array, theta:LorenzeParams):
    # positions of x, y, z in space at the current time point
    x, y, z = current_state
    u1, u2 = u

    # define the 3 ordinary differential equations known as the lorenz equations
    dx_dt = theta.sigma * (y - x) + u1
    dy_dt = x * (theta.rho - z) - y + u2
    dz_dt = x * y - theta.beta * z + u1

    # return a list of the equations that describe the system
    nx_state = jnp.array([dx_dt, dy_dt, dz_dt])
    return nx_state

def high_dim_double_integrator(x: Array, u: Array, dt:float = 0.1) -> Array:
    n = x.shape[0] // 2
    pos = x[:n]
    vel = x[n:]
    
    next_pos = pos + vel * dt + pos * u * dt**2 / 2
    next_vel = vel + u * dt
    
    nx = jnp.r_[next_pos, next_vel]
    return nx


# inverted pendulum dynamics
# ----------------
class PendulumParams(NamedTuple):
    m: float
    l: float
    g: float

def pendulum_dynamics(x: Array, u: Array, theta: PendulumParams,dt:float=0.1):
    """simulate the dynamics of a pendulum. x0 is sin(theta), x1 is cos(theta), x2 is theta_dot. 
    u is the torque applied to the pendulum.

    Args:
        t (int): _description_
        x (Array): state params
        u (Array): external input
        theta (PendulumParams): parameters
    """
    sin_theta = x[0]
    cos_theta = x[1]
    theta_dot = x[2]
    torque = u
    
    # Deal with angle wrap-around.
    theta_ang = jnp.arctan2(sin_theta, cos_theta)

    # Define acceleration.
    theta_dot_dot = -3.0 * theta.g / (2 * theta.l) * jnp.sin(theta_ang + jnp.pi)
    theta_dot_dot += 3.0 / (theta.m * theta.l**2) * torque

    next_theta = theta_ang + theta_dot * dt
    
    next_state = jnp.array([jnp.sin(next_theta), jnp.cos(next_theta), theta_dot + theta_dot_dot * dt])
    return next_state, next_state


if __name__ == "__main__":
    
    # generate target lorenz trajectory
    theta = LorenzeParams(10., 28., 8./3.)
    x0 = jnp.array([-8., 8., 27.])
    T = 100
    mod_dims = ModelDims(n=3, m=2, horizon=T, dt=0.1)
    us_lorenz = jnp.zeros((T,2), dtype=jnp.float64)
    lorenz_dyn = rk4(partial(lorenz_system, theta=theta), dt=.01)
    x_targ = lax.scan(f=lorenz_dyn, init=x0, xs=us_lorenz)[1].squeeze()
    
    fig, ax = subplots()
    ax.plot(x_targ.squeeze())
    
    # define cost quadratic and integrated lorzens system
    
    # define system
    ilqr_params = iLQRParams(x0=jnp.array([-8., 8., 25.]), theta=theta)

    def cost(t: int, x: Array, u: Array, theta: Any):
        return 0.5*jnp.sum((x_targ[t]-x)**2) + 0.5*jnp.sum(u**2)

    def costf(x: Array, theta: Theta):
        return 0.5*jnp.sum((x_targ[-1]-x)**2)

    def dynamics(t: int, x: Array, u: Array, theta: Theta):
        return lorenz_dyn(x, u)[0]

    model = System(
        cost, costf, dynamics, mod_dims
    )
    # ilqr solver
    (Xs_stars, Us_stars, Lambs_stars), total_cost, cost_log = ilqr.ilqr_solver(
            model,
            ilqr_params,
            us_lorenz,
            max_iter=80,
            convergence_thresh=1e-13,
            alpha_init=1.,
            verbose=True,
            use_linesearch=True,
        )

    fig, ax = subplots(1,4,figsize=(12,3))
    ax[0].plot(x_targ.squeeze())
    ax[1].plot(Xs_stars.squeeze())
    ax[2].plot(Us_stars.squeeze())
    ax[3].plot(Lambs_stars.squeeze())
    
    opt_lqr = ilqr.approx_lqr(model, Xs_stars, Us_stars, ilqr_params)

    dL = lqr.kkt(LQRParams(Xs_stars[0],opt_lqr), Xs_stars, Us_stars, Lambs_stars)

    fig, ax = subplots(1,3,figsize=(9,3))
    ax[0].plot(dL[0].squeeze())
    ax[1].plot(dL[1].squeeze())
    ax[2].plot(dL[2].squeeze())




    # # generate target lorenz trajectory
    # theta = PendulumParams(1.5, 2., 9.81)
    # x0 = jnp.array([.4, 1.2, -.5])
    # T = 1000
    # mod_dims = ModelDims(n=3, m=1, horizon=T, dt=0.1)
    # us_pendulum = jnp.zeros((T,mod_dims.m), dtype=jnp.float64)
    # pendulum_dyn = partial(pendulum_dynamics, theta=theta, dt=mod_dims.dt)
    # x_targ = lax.scan(f=pendulum_dyn, init=x0, xs=us_pendulum)[1].squeeze()
    
    # fig, ax = subplots()
    # ax.plot(x_targ.squeeze())
    
    # # define cost quadratic and integrated lorzens system
    
    # # define system
    # ilqr_params = iLQRParams(x0=jnp.array([.4, .2, .5]), theta=theta)

    # def cost(t: int, x: Array, u: Array, theta: Any):
    #     return 0.5*jnp.sum((x_targ[t]-x)**2) + 0.5*jnp.sum(u**2)

    # def costf(x: Array, theta: Theta):
    #     return 0.5*jnp.sum((x_targ[-1]-x)**2)

    # def dynamics(t: int, x: Array, u: Array, theta: Theta):
    #     return pendulum_dyn(x, u)[0]

    # model = System(
    #     cost, costf, dynamics, mod_dims
    # )
    # # ilqr solver
    # (Xs_stars, Us_stars, Lambs_stars), total_cost, cost_log = ilqr.ilqr_solver(
    #         model,
    #         ilqr_params,
    #         us_pendulum,
    #         max_iter=80,
    #         convergence_thresh=1e-13,
    #         alpha_init=1.,
    #         verbose=True,
    #         use_linesearch=True,
    #     )

    # fig, ax = subplots(1,4,figsize=(12,3))
    # ax[0].plot(x_targ.squeeze())
    # ax[1].plot(Xs_stars.squeeze())
    # ax[2].plot(Us_stars.squeeze())
    # ax[3].plot(Lambs_stars.squeeze())
    
    # opt_lqr = ilqr.approx_lqr(model, Xs_stars, Us_stars, ilqr_params)

    # dL = lqr.kkt(LQRParams(Xs_stars[0],opt_lqr), Xs_stars, Us_stars, Lambs_stars)

    # fig, ax = subplots(1,3,figsize=(9,3))
    # ax[0].plot(dL[0].squeeze())
    # ax[1].plot(dL[1].squeeze())
    # ax[2].plot(dL[2].squeeze())
    
    pass