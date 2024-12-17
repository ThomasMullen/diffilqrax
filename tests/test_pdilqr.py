"""
Unit tests for the differentiable iLQR solver
"""

import unittest
import chex
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import numpy as onp

from diffilqrax.parallel_dilqr import parallel_dilqr
from diffilqrax.parallel_ilqr import pilqr_solver
from diffilqrax.exact import quad_solve, exact_solve
from diffilqrax.utils import keygen
from diffilqrax.typs import (
    iLQRParams,
    System,
    ParallelSystem,
    ModelDims,
    Thetax0,
    Theta,
)
from diffilqrax.parallel_ilqr import parallel_forward_lin_integration_ilqr, parallel_feedback_lin_dyn_ilqr


jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_disable_jit", False)  # double precision

PRINTING_ON = True


def is_jax_array(arr: Array) -> bool:
    """validate jax array type"""
    return isinstance(arr, jnp.ndarray) and not isinstance(arr, onp.ndarray)


class TestPDILQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        key = jr.PRNGKey(seed=234)
        key, skeys = keygen(key, 3)
        n = 5
        m = 3
        self.n = n
        self.m = m
        dt = 0.1
        Uh = (
            jax.random.normal(key, (n, n)) * 0.5 / jnp.sqrt(n)
        )  # jnp.array([[1, dt], [-1 * dt, 1 - 0.5 * dt]])
        Wh = (
            jax.random.normal(key, (n, m)) * 0.5 / jnp.sqrt(m)
        )  # jnp.array([[0, 0], [1, 0]]) * dt
        L = jax.random.normal(key, (n, n)) * 0.5 / jnp.sqrt(n)
        Q = L @ L.T
        # initialise params
        self.theta = Thetax0(x0=jnp.zeros(n), Q=Q, Uh=Uh, Wh=Wh, sigma=jnp.zeros((2)))
        self.params = iLQRParams(x0=jnp.array([0.3, 0.0]), theta=self.theta)

        # define model
        def cost(t: int, x: Array, u: Array, theta: Theta):
            x_tgt = jnp.ones(self.n)
            return (
                jnp.sum(
                    (x.squeeze() - x_tgt.squeeze())
                    @ Q
                    @ (x.squeeze() - x_tgt.squeeze()).T
                )
                + jnp.sum(jnp.log(1 + u**2))
            ) + 0.3 * jnp.sum(x**4)

        def costf(x: Array, theta: Theta):
            # return jnp.sum(jnp.abs(x))
            return jnp.sum(x**2)

        def dynamics(t: int, x: Array, u: Array, theta: Theta):
            return theta.Uh @ x + theta.Wh @ u

        self.model = System(
            cost, costf, dynamics, ModelDims(horizon=50, n=n, m=m, dt=dt)
        )
        self.parallel_model = ParallelSystem(self.model, parallel_forward_lin_integration_ilqr, parallel_feedback_lin_dyn_ilqr)
        self.dims = chex.Dimensions(T=50, N=n, M=m, X=1)
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

    def test_pdilqr(self):
        """test grads and values of dilqr with implicit and direct differentiation"""
        # @jax.jit
        def implicit_loss(p):
            theta = Theta(Q=p.Q, Uh=p.Uh, Wh=p.Wh, sigma=jnp.zeros((self.n)))
            params = iLQRParams(x0=p.x0, theta=theta)
            tau_star = parallel_dilqr(
                self.parallel_model,
                params,
                self.Us_init,
                max_iter=70,
                convergence_thresh=1e-8,
                alpha_init=1.0,
                use_linesearch=True,
                verbose=True,
                **self.ls_kwargs,
            )
            Us_lqr = tau_star[:, self.model.dims.n :]
            x_tgt = jnp.ones(self.n).squeeze()
            Xs_lqr = tau_star[:, : self.model.dims.n].squeeze() - x_tgt
            return jnp.linalg.norm(Xs_lqr) ** 2 + jnp.linalg.norm(Us_lqr) ** 2

        implicit_val, implicit_g = jax.value_and_grad(implicit_loss)(self.theta)
        chex.assert_trees_all_equal_shapes_and_dtypes(implicit_g, self.theta)

        def direct_loss(prms):
            theta = Theta(Uh=prms.Uh, Wh=prms.Wh, Q=prms.Q, sigma=jnp.zeros((2)))
            x_tgt = jnp.ones(self.n).squeeze()
            params = iLQRParams(x0=prms.x0, theta=theta)
            (Xs_stars, Us_stars, Lambs_stars), total_cost, costs = pilqr_solver(
                self.parallel_model,
                params,
                self.Us_init,
                max_iter=70,
                convergence_thresh=1e-8,
                alpha_init=1.0,
                use_linesearch=True,
                verbose=False,
                **self.ls_kwargs,
            )
            return (
                jnp.linalg.norm(Us_stars) ** 2
                + jnp.linalg.norm(Xs_stars.squeeze() - x_tgt.squeeze()) ** 2
            )

        direct_val, direct_g = jax.value_and_grad(direct_loss)(self.theta)
        chex.assert_trees_all_equal_shapes_and_dtypes(direct_g, self.theta)
        chex.assert_trees_all_close(direct_val, implicit_val, rtol=2e-1)
        chex.assert_trees_all_close(direct_g, implicit_g, rtol=2e-1)

    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")


if __name__ == "__main__":
    unittest.main()
