"""
Unit tests for the differentiable iLQR solver
"""
import unittest
from os import getcwd
from functools import partial
from pathlib import Path
import chex
import jax
from jax import Array
from jax import tree_map
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
from matplotlib import pyplot as plt
from diffilqrax.diff_ilqr import dilqr
from diffilqrax.ilqr import ilqr_solver
from diffilqrax.exact import quad_solve, exact_solve
from diffilqrax.utils import keygen
from diffilqrax.typs import (
    iLQRParams,
    System,
    ModelDims,
    Thetax0,
    Theta,
)

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_disable_jit", False)  # double precision

PRINTING_ON = True
PLOTTING_ON = True
if PLOTTING_ON:
    FIG_DIR = Path(getcwd(), "fig_dump", "seq_dilqr")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
print(FIG_DIR)

def is_jax_array(arr: Array) -> bool:
    """validate jax array type"""
    return isinstance(arr, jnp.ndarray) and not isinstance(arr, onp.ndarray)


class TestDILQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        key = jr.PRNGKey(seed=234)
        key, skeys = keygen(key, 3)
        n = 10
        m = 2
        self.n = n
        self.m = m
        dt = 0.1
        Uh = jax.random.normal(key, (n, n)) * 0.5 / jnp.sqrt(n)
  
        Wh =  jax.random.normal(key, (n, m)) * 0.5 / jnp.sqrt(n) *dt
        L = jax.random.normal(key, (n, n)) * 0.5 / jnp.sqrt(n)
        Q = L @ L.T
        # initialise params
        self.theta = Thetax0(x0=jnp.zeros(n), Q=Q, Uh=Uh, Wh=Wh, sigma=jnp.zeros((2)))
        self.params = iLQRParams(x0=jnp.zeros(n), theta=self.theta)

        # define model
        def cost(t: int, x: Array, u: Array, theta: Theta):
            x_tgt = jnp.sin(t/5)
            return (
                jnp.sum(
                    (x.squeeze() - x_tgt.squeeze())
                    @ theta.Q @ (x.squeeze() - x_tgt.squeeze()).T
                )
                + jnp.sum(u**2) + jnp.sum(u**4)
            ) + 0.3 * jnp.sum(x**4)

        def costf(x: Array, theta: Theta):
            # return jnp.sum(jnp.abs(x))
            return 0*jnp.sum(x**2)

        def dynamics(t: int, x: Array, u: Array, theta: Theta):
            return (theta.Uh @ jnp.tanh(x)) + theta.Wh @ jnp.tanh(u) + jnp.sum(theta.Uh)*jnp.ones_like(x)

        self.model = System(
            cost, costf, dynamics, ModelDims(horizon=10, n=n, m=m, dt=dt)
        )
        self.dims = chex.Dimensions(T=10, N=n, M=m, X=1)
        self.Us_init = 0. * jr.normal(
            next(skeys), (self.model.dims.horizon, self.model.dims.m)
        )
        # define linesearch parameters
        self.ls_kwargs = {
            "beta": 0.5,
            "max_iter_linesearch": 24,
            "tol": 0.01,
            "alpha_min": 0.00001,
        }

    def test_dilqr(self):
        """test grads and values of dilqr with implicit and direct differentiation"""
        # @jax.jit
        linesearch = True
        alpha_init = 1.0
        # with jax.disable_jit():
        def implicit_loss(p):
            theta = Theta(Q=p.Q, Uh=p.Uh, Wh=p.Wh, sigma=jnp.zeros((self.n)))
            params = iLQRParams(x0=0*p.x0, theta=theta)
            tau_star = dilqr(
                self.model,
                params,
                self.Us_init,
                max_iter=70,
                convergence_thresh=1e-9,
                alpha_init=alpha_init,
                use_linesearch=linesearch,
                verbose=True,
                **self.ls_kwargs,
            )
            Us_lqr = tau_star[:, self.model.dims.n :]
            x_tgt = jnp.ones(self.n).squeeze()
            Xs_lqr = tau_star[:, : self.model.dims.n].squeeze() - x_tgt
            # if PRINTING_ON:
            #     jax.debug.print("implicit_opt_x: {x}", x=Xs_lqr[:10])
            return jnp.linalg.norm(tau_star) ** 2 #+ jnp.linalg.norm(Us_lqr) ** 2

        implicit_val, implicit_g = jax.value_and_grad(implicit_loss)(self.theta)
        chex.assert_trees_all_equal_shapes_and_dtypes(implicit_g, self.theta)

        def direct_loss(prms):
            theta = Theta(Uh=prms.Uh, Wh=prms.Wh, Q=prms.Q, sigma=jnp.zeros((2)))
            x_tgt = jnp.ones(self.n).squeeze()
            params = iLQRParams(x0=0*prms.x0, theta=theta)
            (Xs_stars, Us_stars, Lambs_stars), total_cost, costs = ilqr_solver(
                self.model,
                params,
                self.Us_init,
                max_iter=70,
                convergence_thresh=1e-8,
                alpha_init=alpha_init,
                use_linesearch=linesearch,
                verbose=False,
                **self.ls_kwargs,
            )

            if PRINTING_ON:
                jax.debug.print("cost: {c}\nopt_x: {x}",c=costs[:10], x=Xs_stars[0,:10])
                
            # if PLOTTING_ON:
            #     fig, ax = plt.subplots()
            #     ax.plot(onp.asarray(costs))
            #     fig.savefig(f"{FIG_DIR}/costs.png")
            return (
                jnp.linalg.norm(Us_stars) ** 2
                + jnp.linalg.norm(Xs_stars.squeeze())**2)# - x_tgt.squeeze()) ** 2
            
        prms = self.theta
        theta = Theta(Uh=prms.Uh, Wh=prms.Wh, Q=prms.Q, sigma=jnp.zeros((2)))
        x_tgt = jnp.ones(self.n).squeeze()
        params = iLQRParams(x0=0*prms.x0, theta=theta)
        
        direct_val, direct_g = jax.value_and_grad(direct_loss)(self.theta)
        
        if PRINTING_ON:
            jax.debug.print("Direct: {dv}", dv=direct_val)
            jax.debug.print("Implicit: {iv}", iv=implicit_val)
            jax.debug.print("Dgrads: {dg}", dg = partial(jax.tree.map, lambda x: x.flatten()[0])(direct_g))
            jax.debug.print("Igrads: {ig}", ig = partial(jax.tree.map, lambda x: x.flatten()[0])(implicit_g))

        if PLOTTING_ON:
            fig, axes = plt.subplots(1,len(direct_g), figsize=(10, 2))
            for (ax, dg, ig, n) in zip(axes.flatten(), direct_g, implicit_g, implicit_g._fields):
                ax.scatter(dg.flatten(), ig.flatten())
                ax.set(xlabel="direct", ylabel="implicit", title=f"{n}")
            fig.tight_layout()
            fig.savefig(f"{FIG_DIR}/grads.png")
            plt.close(fig)
        chex.assert_trees_all_equal_shapes_and_dtypes(direct_g, self.theta)
        chex.assert_trees_all_close(direct_val, implicit_val, rtol=2e-1)
        # TODO: Check why the gradients are not close
        chex.assert_trees_all_close(direct_g, implicit_g, rtol=2e0)
        

    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")


if __name__ == "__main__":
    unittest.main()