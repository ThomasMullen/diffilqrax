import unittest
from typing import NamedTuple
import chex
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
from jaxopt import linear_solve, implicit_diff

from src.diff_ilqr import dilqr

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_disable_jit", False)  # double precision

from src.lqr import (
    LQR,
    LQRParams,
    ModelDims,
    solve_lqr,
    kkt,
)
from src.ilqr import iLQRParams, System, ilQR_solver

import numpy as onp
from src.exact import quad_solve, exact_solve
from src.utils import keygen, initialise_stable_dynamics

from src.typs import *

is_jax_Array = lambda arr: isinstance(arr, jnp.ndarray) and not isinstance(
    arr, onp.ndarray
)
printing_on = True


class TestDILQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        key = jr.PRNGKey(seed=234)
        key, skeys = keygen(key, 3)
        n = 20
        m = 5
        self.n = n
        self.m = m
        dt = 0.1
        Uh = jax.random.normal(key, (n,n))*0.5/jnp.sqrt(n) #jnp.array([[1, dt], [-1 * dt, 1 - 0.5 * dt]])
        Wh = jax.random.normal(key, (n,m))*0.5/jnp.sqrt(m) #jnp.array([[0, 0], [1, 0]]) * dt
        L = jax.random.normal(key, (n,n))*0.5/jnp.sqrt(n)
        Q = L@L.T
        # initialise params
        self.theta = Thetax0(x0 = jnp.zeros(n), Q = Q, Uh=Uh, Wh=Wh, sigma=jnp.zeros((2)))
        self.params = iLQRParams(x0=jnp.array([0.3, 0.0]), theta=self.theta)

        # define model
        def cost(t: int, x: Array, u: Array, theta: Theta):
            x_tgt = jnp.ones(self.n)
            return (jnp.sum((x.squeeze() - x_tgt.squeeze())@Q@(x.squeeze() - x_tgt.squeeze()).T) + jnp.sum(u**2)) + 0.3*jnp.sum(x**4)

        def costf(x: Array, theta: Theta):
            # return jnp.sum(jnp.abs(x))
            return jnp.sum(x**2)

        def dynamics(t: int, x: Array, u: Array, theta: Theta):
            return jax.nn.sigmoid(theta.Uh @ x + theta.Wh @ u)

        self.model = System(
            cost, costf, dynamics, ModelDims(horizon=200, n=n, m=m, dt=dt)
        )
        self.dims = chex.Dimensions(T=100, N=n, M=m, X=1)
        self.Us_init = 0.1 * jr.normal(
            next(skeys), (self.model.dims.horizon, self.model.dims.m)
        )
        # define linesearch parameters
        self.ls_kwargs = {
        "beta": 0.5,
        "max_iter_linesearch": 16,
        "tol": 0.1,
        "alpha_min": 0.00001,
        }

    def test_dilqr(self):
        #@jax.jit
        def implicit_loss(p):
            theta = Theta(Q = p.Q, Uh=p.Uh, Wh=p.Wh, sigma=jnp.zeros((self.n)))
            params = iLQRParams(x0=p.x0, theta=theta)
            tau_star = dilqr(
            self.model,
            params,
            self.Us_init,
            max_iter=70,
            convergence_thresh=1e-8,
            alpha_init=1.0,
            use_linesearch=True,
            verbose= True,
            **self.ls_kwargs,)
            Us_lqr = tau_star[:, self.dims.N :]
            x_tgt = jnp.ones(self.n).squeeze()
            Xs_lqr = tau_star[:, : self.dims.N].squeeze() - x_tgt
            return jnp.linalg.norm(Xs_lqr)**2 + jnp.linalg.norm(Us_lqr)**2

        implicit_val, implicit_g = jax.value_and_grad(implicit_loss)(self.theta)
        chex.assert_trees_all_equal_shapes_and_dtypes(implicit_g, self.theta)
        
        def direct_loss(prms):
            theta = Theta(Uh=prms.Uh, Wh=prms.Wh, Q = prms.Q, sigma=jnp.zeros((2)))
            x_tgt = jnp.ones(self.n).squeeze()
            params = iLQRParams(x0=prms.x0, theta=theta)
            (Xs_stars, Us_stars, Lambs_stars), total_cost, costs = ilQR_solver(
            self.model,
            params,
            self.Us_init,
             max_iter=70,
            convergence_thresh=1e-8,
            alpha_init=1.0,
            use_linesearch=True,
            verbose= False,
            **self.ls_kwargs
        )   
            return jnp.linalg.norm(Us_stars)**2 + jnp.linalg.norm(Xs_stars.squeeze() - x_tgt.squeeze())**2
        direct_val, direct_g = jax.value_and_grad(direct_loss)(self.theta)
        chex.assert_trees_all_equal_shapes_and_dtypes(direct_g, self.theta)
        chex.assert_trees_all_close(direct_val, implicit_val, rtol=1e-1)
        chex.assert_trees_all_close(direct_g, implicit_g, rtol=1e-1)
        

    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")

if __name__ == "__main__":
    unittest.main()
