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

jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_enable_x64", True)  # double precision


class TestiLQRStructs(unittest.TestCase):
    """Test LQR dimensions and data structures"""

    def setUp(self):
        """Setup LQR problem"""
        key = jr.PRNGKey(seed=234)
        key, skeys = keygen(key, 3)

        dt = 0.1
        Uh = jnp.array([[1, dt, 0.01], [-1 * dt, 1 - 0.1 * dt, 0.],  [-1 * dt, 1 - 0.1 * dt, 0.05]])
        Wh = jnp.eye(3) #jnp.array([[0.5, 2., 0.], [1., -1.2, 0.1]]).T * dt
        Q = jnp.eye(3)
        # initialise params
        self.theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros((2)), Q=Q)
        self.params = iLQRParams(x0=jnp.array([2., 0.1, -0.5]), theta=self.theta)

        # define model
        def cost(t: int, x: Array, u: Array, theta: Theta):
            return jnp.sum(jnp.log(1 + x**2)) + jnp.sum(x**2)  + jnp.sum(u**2) #+ jnp.sum(x) #+ jnp.sum(jnp.log(1 + u**2)) + 0*jnp.log(1 + x**2)

        def costf(x: Array, theta: Theta):
            # return jnp.sum(jnp.abs(x))
            return 0*jnp.sum(jnp.log(1 + x**2)) + 0*jnp.sum(x**4)#+ jnp.sum(x))

        def dynamics(t: int, x: Array, u: Array, theta: Theta):
            return theta.Uh @ x + theta.Wh @ u

        self.model = System(
            cost, costf, dynamics, ModelDims(horizon=100, n=3, m=3, dt=dt)
        )
        self.dims = chex.Dimensions(T=100, N=3, M=3, X=1)
        self.Us_init = 0.1 * jr.normal(
            next(skeys), (self.model.dims.horizon, self.model.dims.m)
        )
        # define linesearch parameters
        self.ls_kwargs = {
        "beta": 0.8,
        "max_iter_linesearch": 10,
        "tol": 0.1,
        "alpha_min": 0.0001,
        }


    def test_pilQR_solver(self):
        """test ilqr solver with integrater dynamics"""
        # setup
        fig_dir = Path(Path(getcwd()), "fig_dump")
        fig_dir.mkdir(exist_ok=True)
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
        #difference when Us_init is not 0...
        (Xs_stars_ilqr, Us_stars_ilqr, _), converged_cost, cost_log = ilqr.ilqr_solver(
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
        ax[0, 1].plot(Us_stars)
        ax[0, 0].plot(Xs_stars)
        ax[0, 1].set(title="U (parallel)")
        ax[0, 0].set(title="X (parallel)")
        ax[1, 0].plot(Xs_stars_ilqr)
        ax[1, 1].plot(Us_stars_ilqr)
        ax[1, 1].set(title="U (normal)")
        ax[1, 0].set(title="X (normal)")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/pilqr_solver.png")
        chex.assert_trees_all_close(Xs_stars, Xs_stars_ilqr, rtol=1e-03, atol=1e-02)
        
     ##TODO : add more thorough tests including possible offset in dynamics
     ##add speed test of pilqr   
if __name__ == "__main__":
    unittest.main()