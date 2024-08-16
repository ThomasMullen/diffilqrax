"""Test functions in diffilqrax/ilqr.py"""
import unittest
from typing import Any
from pathlib import Path
from os import getcwd
import chex
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
from matplotlib.pyplot import subplots, close, style

from diffilqrax.utils import keygen, initialise_stable_dynamics
from diffilqrax import ilqr, pilqr
from diffilqrax import lqr
from diffilqrax.typs import (
    iLQRParams,
    LQR,
    LQRParams,
    System,
    ModelDims,
    Theta,
)

# jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_enable_x64", True)  # double precision

PLOT_URL = ("https://gist.githubusercontent.com/"
       "ThomasMullen/e4a6a0abd54ba430adc4ffb8b8675520/"
       "raw/1189fbee1d3335284ec5cd7b5d071c3da49ad0f4/"
       "figure_style.mplstyle")
style.use(PLOT_URL)


class TestiLQRStructs(unittest.TestCase):
    """Test LQR dimensions and data structures"""

    def setUp(self):
        """Setup LQR problem"""
        key = jr.PRNGKey(seed=234)
        key, skeys = keygen(key, 3)

        dt = 0.1
        Uh = jnp.array([[1, dt], [-1 * dt, 1 - 0.5 * dt]])
        Wh = jnp.array([[0, 0], [1, 0]]) * dt
        Q = jnp.eye(2)
        # initialise params
        self.theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros((2)), Q=Q)
        self.params = iLQRParams(x0=jnp.array([0.3, 0.0]), theta=self.theta)

        # define model
        def cost(t: int, x: Array, u: Array, theta: Theta):
            return jnp.log(x**2) + jnp.sum(u**2)

        def costf(x: Array, theta: Theta):
            # return jnp.sum(jnp.abs(x))
            return jnp.sum(x**2)

        def dynamics(t: int, x: Array, u: Array, theta: Theta):
            return theta.Uh @ x + theta.Wh @ u

        self.model = System(
            cost, costf, dynamics, ModelDims(horizon=100, n=2, m=2, dt=dt)
        )
        self.dims = chex.Dimensions(T=100, N=2, M=2, X=1)
        self.Us_init = 0.1 * jr.normal(
            next(skeys), (self.model.dims.horizon, self.model.dims.m)
        )
        # define linesearch parameters
        self.ls_kwargs = {
        "beta": 0.8,
        "max_iter_linesearch": 16,
        "tol": 1e0,
        "alpha_min": 0.0001,
        }


    def test_pilQR_solver(self):
        """test ilqr solver with integrater dynamics"""
        # setup
        fig_dir = Path(Path(getcwd()), "fig_dump")
        fig_dir.mkdir(exist_ok=True)
        (Xs_init, _), initial_cost = ilqr.ilqr_simulate(
            self.model, self.Us_init, self.params
        )
        # exercise
        (Xs_stars, Us_stars, Lambs_stars), converged_cost, cost_log = pilqr.pilqr_solver(
            self.model,
            self.params,
            self.Us_init,
            max_iter=70,
            convergence_thresh=1e-8,
            alpha_init=1.0,
            verbose=True,
            use_linesearch=True,
            **self.ls_kwargs,
        )
        fig, ax = subplots(2, 2, sharey=True)
        ax[0, 0].plot(Xs_init)
        ax[0, 0].set(title="X")
        ax[0, 1].plot(self.Us_init)
        ax[0, 1].set(title="U")
        ax[1, 0].plot(Xs_stars)
        ax[1, 1].plot(Us_stars)
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/pilqr_solver.png")
        close()
        lqr_params_stars = ilqr.approx_lqr(self.model, Xs_stars, Us_stars, self.params)
        lqr_tilde_params = LQRParams(Xs_stars[0], lqr_params_stars)
        dLdXs, dLdUs, dLdLambs = lqr.kkt(
            lqr_tilde_params, Xs_stars, Us_stars, Lambs_stars
        )
        fig, ax = subplots(2, 3, figsize=(10, 3), sharey=False)
        ax[0, 0].plot(Xs_stars)
        ax[0, 0].set(title="X")
        ax[0, 1].plot(Us_stars)
        ax[0, 1].set(title="U")
        ax[0, 2].plot(Lambs_stars)
        ax[0, 2].set(title="λ")
        ax[1, 0].plot(dLdXs)
        ax[1, 0].set(title="dLdX")
        ax[1, 1].plot(dLdUs)
        ax[1, 1].set(title="dLdUs")
        ax[1, 2].plot(dLdLambs)
        ax[1, 2].set(title="dLdλ")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/ilqr_kkt.png")
        close()
        fig, ax = subplots()
        ax.scatter(jnp.arange(cost_log.size), cost_log)
        ax.set(xlabel="Iteration", ylabel="Total cost")
        fig.savefig(f"{fig_dir}/ilqr_cost_log.png")
        close()

        # verify
        assert converged_cost < initial_cost
        # assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-03, atol=1e-04)
        # assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-03, atol=1e-04)
        # assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-03, atol=1e-04)