# """Test functions in src/ilqr.py"""
# import unittest
# import pytest
# import chex, jax
# from jax import Array, grad
# import jax.random as jr
# import jax.numpy as jnp
# import numpy as onp
# from os import getcwd
# from pathlib import Path
# from matplotlib.pyplot import subplots, close, style
# from typing import Union, Any

# from diffilqrax.utils import keygen, initialise_stable_dynamics
# import diffilqrax.ilqr as ilqr
# import diffilqrax.lqr as lqr
# from diffilqrax.typs import *

# jax.config.update('jax_default_device', jax.devices('cpu')[0])
# jax.config.update("jax_enable_x64", True)  # double precision

# style.use("https://gist.githubusercontent.com/ThomasMullen/e4a6a0abd54ba430adc4ffb8b8675520/raw/1189fbee1d3335284ec5cd7b5d071c3da49ad0f4/figure_style.mplstyle")

# class TestiLQRStructs(unittest.TestCase):
#     """Test LQR dimensions and data structures"""

#     def setUp(self):
#         """Setup LQR problem"""
#         key = jr.PRNGKey(seed=234)
#         key, skeys = keygen(key, 3)

#         dt = 0.1
#         Uh = jnp.array([[1, dt], [-1 * dt, 1 - 0.5 * dt]])
#         Wh = jnp.array([[0, 0], [1, 0]]) * dt
#         # initialise params
#         self.theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros((2)))
#         self.params = iLQRParams(x0=jnp.array([0.3, 0.0]), theta=self.theta)

#         # define model
#         def cost(t: int, x: Array, u: Array, theta: Theta):
#             return jnp.sum(x**2) + jnp.sum(u**2)

#         def costf(x: Array, theta: Theta):
#             # return jnp.sum(jnp.abs(x))
#             return jnp.sum(x**2)

#         def dynamics(t: int, x: Array, u: Array, theta: Theta):
#             return jnp.tanh(theta.Uh @ x + theta.Wh @ u)

#         self.model = System(
#             cost, costf, dynamics, ModelDims(horizon=100, n=2, m=2, dt=dt)
#         )
#         self.dims = chex.Dimensions(T=100, N=2, M=2, X=1)
#         self.Us_init = 0.1 * jr.normal(
#             next(skeys), (self.model.dims.horizon, self.model.dims.m)
#         )
#         # define linesearch parameters
#         self.ls_kwargs = {
#         "beta": 0.8,
#         "max_iter_linesearch": 16,
#         "tol": 1e0,
#         "alpha_min": 0.0001,
#         }

#     def test_vectorise_fun_in_time(self):
#         # setup
#         (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
#         Xs = Xs[:-1]
#         Us = Us[:]
#         tps = jnp.arange(self.model.dims.horizon)
#         # exercise
#         (Fx, Fu) = ilqr.vectorise_fun_in_time(ilqr.linearise(self.model.dynamics))(
#             tps, Xs, Us, self.theta
#         )
#         # verify
#         chex.assert_shape(Fx, self.dims["TNN"])
#         chex.assert_shape(Fu, self.dims["TNM"])

#     def test_quadratise(self):
#         # setup
#         (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
#         Xs = Xs[0]
#         Us = Us[0]
#         # exercise
#         (Cxx, Cxu), (Cux, Cuu) = ilqr.quadratise(self.model.cost)(
#             0, Xs, Us, self.params.theta
#         )
#         # verify
#         chex.assert_shape(Cxx, self.dims["NN"])
#         chex.assert_shape(Cxu, self.dims["NM"])
#         chex.assert_shape(Cuu, self.dims["MM"])
#         chex.assert_shape(Cux, self.dims["MN"])
#         chex.assert_type(J0.dtype, float)

#     def test_linearise(self):
#         # setup
#         (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
#         Xs = Xs[0]
#         Us = Us[0]
#         # exercise
#         Fx, Fu = ilqr.linearise(self.model.dynamics)(0, Xs, Us, self.params.theta)
#         Cx, Cu = ilqr.linearise(self.model.cost)(0, Xs, Us, self.params.theta)
#         # verify
#         chex.assert_shape(Fx, self.dims["NN"])
#         chex.assert_shape(Fu, self.dims["NM"])
#         chex.assert_shape(Cx, self.dims["N"])
#         chex.assert_shape(Cu, self.dims["M"])

#     def test_approx_lqr(self):
#         # setup
#         (Xs, _), _ = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
#         # exercise
#         lqr_tilde = ilqr.approx_lqr(
#             model=self.model, Xs=Xs, Us=self.Us_init, params=self.params
#         )
#         # verify
#         assert isinstance(lqr_tilde, LQR)
#         # check shape
#         chex.assert_shape(lqr_tilde.A, self.dims["TNN"])
#         chex.assert_shape(lqr_tilde.B, self.dims["TNM"])
#         chex.assert_shape(lqr_tilde.Q, self.dims["TNN"])
#         chex.assert_shape(lqr_tilde.R, self.dims["TMM"])
#         chex.assert_shape(lqr_tilde.S, self.dims["TNM"])
#         chex.assert_shape(lqr_tilde.Qf, self.dims["NN"])

#     def test_ilqr_simulate(self):
#         # setup
#         Xs_lqr_sim = lqr.simulate_trajectory(
#             self.model.dynamics, self.Us_init, self.params, self.model.dims
#         )
#         # exercise
#         (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
#         # verify
#         chex.assert_trees_all_equal(Us, self.Us_init)
#         chex.assert_shape(Xs, (self.dims["T"][0] + 1,) + self.dims["N"])
#         chex.assert_trees_all_equal(Xs, Xs_lqr_sim)

#     def test_ilqr_forward_pass(self):
#         # setup
#         (old_Xs, _), initial_cost = ilqr.ilqr_simulate(
#             self.model, self.Us_init, self.params
#         )
#         lqr_params = ilqr.approx_lqr(self.model, old_Xs, self.Us_init, self.params)
#         exp_cost_red, gains = lqr.lqr_backward_pass(
#             lqr_params, dims=self.model.dims, expected_change=False, verbose=False
#         )
#         exp_change_J0 = lqr.calc_expected_change(exp_cost_red, alpha=1.0)
#         # exercise
#         (new_Xs, new_Us), new_total_cost = ilqr.ilqr_forward_pass(
#             self.model, self.params, gains, old_Xs, self.Us_init, alpha=1.0
#         )
#         # verify
#         print(
#             f"\nInitial J0: {initial_cost}, New J0: {new_total_cost}, Expected ΔJ0 (α=1): {exp_change_J0}"
#         )
#         chex.assert_shape(new_Xs, old_Xs.shape)
#         chex.assert_shape(new_Us, self.Us_init.shape)
#         assert new_total_cost < initial_cost
#         assert new_total_cost - initial_cost < exp_change_J0

#     def test_ilQR_solver(self):
#         # setup
#         (Xs_init, _), initial_cost = ilqr.ilqr_simulate(
#             self.model, self.Us_init, self.params
#         )
#         # exercise
#         (Xs_stars, Us_stars, Lambs_stars), converged_cost, cost_log = ilqr.ilqr_solver(
#             self.model,
#             self.params,
#             self.Us_init,
#             max_iter=70,
#             convergence_thresh=1e-8,
#             alpha_init=1.0,
#             verbose=True,
#             use_linesearch=True,
#             **self.ls_kwargs,
#         )
#         fig, ax = subplots(2, 2, sharey=True)
#         ax[0, 0].plot(Xs_init)
#         ax[0, 0].set(title="X")
#         ax[0, 1].plot(self.Us_init)
#         ax[0, 1].set(title="U")
#         ax[1, 0].plot(Xs_stars)
#         ax[1, 1].plot(Us_stars)
#         fig.tight_layout()
#         fig.savefig(f"{FIG_DIR}/ilqr_solver.png")
#         close()
#         lqr_params_stars = ilqr.approx_lqr(self.model, Xs_stars, Us_stars, self.params)
#         lqr_tilde_params = LQRParams(Xs_stars[0], lqr_params_stars)
#         dLdXs, dLdUs, dLdLambs = lqr.kkt(
#             lqr_tilde_params, Xs_stars, Us_stars, Lambs_stars
#         )
        # if PLOTTING_ON:
        #     fig = _plot_kkt(Xs_stars, Us_stars, Lambs_stars, dLdXs, dLdUs, dLdLambs)
        #     fig.savefig(f"{FIG_DIR}/ilqr_kkt.png")
        #     close()
#         fig, ax = subplots()
#         ax.scatter(jnp.arange(cost_log.size), cost_log)
#         ax.set(xlabel="Iteration", ylabel="Total cost")
#         fig.savefig(f"{FIG_DIR}/ilqr_cost_log.png")
#         close()

#         # verify
#         assert converged_cost < initial_cost
#         # assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-03, atol=1e-04)
#         # assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-03, atol=1e-04)
#         # assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-03, atol=1e-04)


# # load test data
# # description of tajax data


# class TestiLQRExactSolution(unittest.TestCase):
#     """Test iLQR solver with exact solution"""

#     def setUp(self):
#         # load fixtures
#         self.fixtures = onp.load("tests/fixtures/ilqr_exact_solution.npz")

#         # dimensions
#         self.dims = chex.Dimensions(T=100, N=8, M=2, X=1)

#         # set-up model
#         key = jr.PRNGKey(seed=234)
#         key, skeys = keygen(key, 5)
#         Uh = initialise_stable_dynamics(next(skeys), *self.dims["NT"], 0.6)[0]
#         Wh = jr.normal(next(skeys), self.dims["NM"])
#         chex.assert_trees_all_equal(self.fixtures["Uh"], Uh)
#         chex.assert_trees_all_equal(self.fixtures["Wh"], Wh)
#         theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros(self.dims["N"]))
#         self.params = iLQRParams(
#             x0=jr.normal(next(skeys), self.dims["N"]), theta=theta
#         )
#         self.Us = jnp.zeros(self.dims["TM"])
#         assert jnp.allclose(self.fixtures["x0"], self.params.x0)
        
#         # define linesearch hyper parameters
#         self.ls_kwargs = {
#             "beta":0.8,
#             "max_iter_linesearch":16,
#             "tol":1e0,
#             "alpha_min":0.0001,
#             }

#         def cost(t: int, x: Array, u: Array, theta: Any):
#             return jnp.sum(x**2) + jnp.sum(u**2)

#         def costf(x: Array, theta: Theta):
#             return jnp.sum(x**2)

#         def dynamics(t: int, x: Array, u: Array, theta: Theta):
#             return jnp.tanh(theta.Uh @ x + theta.Wh @ u)

#         self.model = ilqr.System(
#             cost, costf, dynamics, ModelDims(*self.dims["NMT"], dt=0.1)
#         )

#     def test_ilqr_nolinesearch(self):
#         # exercise rollout
#         (Xs_init, Us_init), cost_init = ilqr.ilqr_simulate(
#             self.model, self.Us, self.params
#         )
#         # verify
#         chex.assert_trees_all_equal(self.fixtures["X_orig"], Xs_init[1:])

#         # exercise ilqr solver
#         (Xs_stars, Us_stars, Lambs_stars), total_cost, _ = ilqr.ilqr_solver(
#             self.model,
#             self.params,
#             self.Us,
#             max_iter=70,
#             tol=1e-8,
#             alpha_init=0.8,
#             verbose=True,
#             use_linesearch=False,
#         )
        
#         fig, ax = subplots(2, 2, sharey=True)
#         ax[0, 0].plot(Xs_init)
#         ax[0, 0].set(title="X")
#         ax[0, 1].plot(self.Us)
#         ax[0, 1].set(title="U")
#         ax[1, 0].plot(Xs_stars)
#         ax[1, 1].plot(Us_stars)
#         fig.tight_layout()
#         fig.savefig(f"{FIG_DIR}/ilqr_ls_solver.png")
#         close()
        
#         # verify
#         chex.assert_trees_all_close(Xs_stars, self.fixtures["X"], rtol=1e-03, atol=1e-03)
#         chex.assert_trees_all_close(Us_stars, self.fixtures["U"], rtol=1e-03, atol=1e-03)
#         print(f"iLQR solver cost:\t{total_cost:.6f}\nOther solver cost:\t{self.fixtures['obj']:.6f}")
#         assert jnp.allclose(total_cost, self.fixtures['obj'], rtol=1e-04, atol=1e-04)

#     def test_ilqr_linesearch(self):
#         # exercise ilqr solver
#         (Xs_stars, Us_stars, Lambs_stars), total_cost, _ = ilqr.ilqr_solver(
#             self.model,
#             self.params,
#             self.Us,
#             max_iter=40,
#             convergence_thresh=1e-8,
#             alpha_init=1.,
#             verbose=True,
#             use_linesearch=True,
#             **self.ls_kwargs,
#         )
#         # verify
#         chex.assert_trees_all_close(Xs_stars, self.fixtures["X"], rtol=1e-06, atol=1e-04)
#         chex.assert_trees_all_close(Us_stars, self.fixtures["U"], rtol=1e-06, atol=1e-04)
#         print(f"iLQR solver cost:\t{total_cost:.6f}\nOther solver cost:\t{self.fixtures['obj']:.6f}")
#         assert jnp.allclose(total_cost, self.fixtures['obj'], rtol=1e-06, atol=1e-06)

#     def test_ilqr_kkt_solution(self):
#         # exercise ilqr solver
#         (Xs_stars, Us_stars, Lambs_stars), total_cost, cost_log = ilqr.ilqr_solver(
#             self.model,
#             self.params,
#             self.Us,
#             max_iter=80,
#             convergence_thresh=1e-13,
#             alpha_init=1.,
#             verbose=True,
#             use_linesearch=True,
#             **self.ls_kwargs,
#         )
#         lqr_tilde = ilqr.approx_lqr(model=self.model, Xs=Xs_stars, Us=Us_stars, params=self.params)
#         lqr_approx_params = LQRParams(Xs_stars[0], lqr_tilde)
#         # verify
#         dLdXs, dLdUs, dLdLambs = lqr.kkt(lqr_approx_params, Xs_stars, Us_stars, Lambs_stars)
#         print(jnp.mean(jnp.abs(dLdXs)), jnp.mean(jnp.abs(dLdUs)), jnp.mean(jnp.abs(dLdLambs)))
#         # plot kkt
        # if PLOTTING_ON:
        #     fig = _plot_kkt(Xs_stars, Us_stars, Lambs_stars, dLdXs, dLdUs, dLdLambs)
        #     fig.savefig(f"{FIG_DIR}/ilqr_ls_kkt.png")
        #     close()
        
#         fig, ax = subplots()
#         ax.scatter(jnp.arange(cost_log.size), cost_log)
#         ax.set(xlabel="Iteration", ylabel="Total cost")
#         fig.savefig(f"{FIG_DIR}/ilqr_ls_cost_log.png")
#         close()
        
#         # Verify that the average KKT conditions are satisfied
#         assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-04, atol=1e-05)
#         assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-04, atol=1e-05)
#         # assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-02, atol=1e-02)
        
#         # Verify that the terminal state KKT conditions is satisfied
#         assert jnp.allclose(dLdXs[-1], 0.0, rtol=1e-04, atol=1e-05), "Terminal X state not satisfied"
        
#         # Verify that all KKT conditions are satisfied
#         assert jnp.allclose(dLdUs, 0.0, rtol=1e-04, atol=1e-05)
#         assert jnp.allclose(dLdXs, 0.0, rtol=1e-04, atol=1e-05)
#         # assert jnp.allclose(dLdLambs, 0.0, rtol=1e-05, atol=1e-08)



# class TestiLQRWithLQRProblem(unittest.TestCase):
#     """Test iLQR solver with exact solution"""

#     def setUp(self):
#         # dimensions
#         self.dims = chex.Dimensions(T=100, N=2, M=2, X=1)
#         self.sys_dims = ModelDims(*self.dims["NMT"], dt=0.1)
#         dt = self.sys_dims.dt
#         self.Us = jnp.zeros(self.dims["TM"])
#         self.x0 = jnp.array([0.3, 0.])
        
#         # load LQR problem
#         span_time_m=self.dims["TXX"]
#         span_time_v=self.dims["TX"]
#         A = jnp.tile(jnp.array([[1,dt],[-1*dt,1-0.5*dt]]), span_time_m)
#         B = jnp.tile(jnp.array([[0,0],[1,0]]), span_time_m)*dt
#         a = jnp.zeros(self.dims["TN"])
#         Qf = 1. *jnp.eye(self.dims["N"][0])
#         qf = 0.   * jnp.ones(self.dims["N"])
#         Q = 1. * jnp.tile(jnp.eye(self.dims["N"][0]), span_time_m)
#         q = 0. * jnp.tile(jnp.ones(self.dims["N"]), span_time_v)
#         R = 1. * jnp.tile(jnp.eye(self.dims["M"][0]), span_time_m)
#         r = 0. * jnp.tile(jnp.ones(self.dims["M"]), span_time_v)
#         S = 0. * jnp.tile(jnp.ones(self.dims["NM"]), span_time_m)
#         self.lqr_struct = LQR(A, B, a, Q, q, Qf, qf, R, r, S)()
#         self.lqr_params = LQRParams(self.x0, self.lqr_struct)


#         # set-up lqr in the model
#         Uh = self.lqr_struct.A[0]
#         Wh = self.lqr_struct.B[0]
#         theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros(self.dims["N"]))
#         self.ilqr_params = iLQRParams(x0=self.x0, theta=theta)

#         def cost(t: int, x: Array, u: Array, theta: Any):
#             return 0.5*jnp.sum(x**2) + 0.5*jnp.sum(u**2)

#         def costf(x: Array, theta: Theta):
#             return 0.5*jnp.sum(x**2)

#         def dynamics(t: int, x: Array, u: Array, theta: Theta):
#             return theta.Uh @ x + theta.Wh @ u

#         self.model = System(
#             cost, costf, dynamics, ModelDims(*self.dims["NMT"], dt=0.1)
#         )

#     def test_lqr_solution(self):
#         # setup: simulate dynamics
#         lqr_Xs_sim = lqr.simulate_trajectory(
#             dynamics=lqr.lin_dyn_step, 
#             Us=self.Us, 
#             params=self.lqr_params, 
#             dims=self.sys_dims
#             )
#         # exercise rollout
#         (Xs_init, Us_init), cost_init = ilqr.ilqr_simulate(self.model, self.Us, self.ilqr_params)
#         # verify
#         chex.assert_trees_all_equal(Xs_init, lqr_Xs_sim)
        
#         # setup: lqr solver
#         gains_lqr, Xs_lqr, Us_lqr, Lambs_lqr = lqr.solve_lqr(self.lqr_params, self.sys_dims)
#         # exercise ilqr solver
#         (Xs_stars, Us_stars, Lambs_stars), total_cost, _ = ilqr.ilqr_solver(
#             self.model,
#             self.ilqr_params,
#             self.Us,
#             max_iter=70,
#             tol=1e-8,
#             alpha_init=1.,
#             verbose=True,
#             use_linesearch=False,
#         )
#         # verify
#         chex.assert_trees_all_close(Xs_stars, Xs_lqr, rtol=1e-04, atol=1e-04)
#         chex.assert_trees_all_close(Us_stars, Us_lqr, rtol=1e-04, atol=1e-04)

#         # exercise lqr approximation
#         lqr_tilde = ilqr.approx_lqr(model=self.model, Xs=Xs_init, Us=self.Us, params=self.ilqr_params)
#         # verify
#         # chex.assert_trees_all_close(lqr_tilde, self.lqr_struct)
#         chex.assert_trees_all_close(lqr_tilde.A, self.lqr_struct.A)
#         chex.assert_trees_all_close(lqr_tilde.B, self.lqr_struct.B)
#         chex.assert_trees_all_close(lqr_tilde.a, self.lqr_struct.a)
#         chex.assert_trees_all_close(lqr_tilde.Q, self.lqr_struct.Q)
#         chex.assert_trees_all_close(lqr_tilde.Qf, self.lqr_struct.Qf)
#         chex.assert_trees_all_close(lqr_tilde.R, self.lqr_struct.R)
#         chex.assert_trees_all_close(lqr_tilde.S, self.lqr_struct.S)
#         chex.assert_trees_all_close(lqr_tilde.r, self.lqr_struct.r)
#         # chex.assert_trees_all_close(lqr_tilde.q, self.lqr_struct.q)
#         # chex.assert_trees_all_close(lqr_tilde.qf, self.lqr_struct.qf)
        
        
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
jax.config.update("jax_disable_jit", False)  # double precision

PLOT_URL = ("https://gist.githubusercontent.com/"
       "ThomasMullen/e4a6a0abd54ba430adc4ffb8b8675520/"
       "raw/1189fbee1d3335284ec5cd7b5d071c3da49ad0f4/"
       "figure_style.mplstyle")
PRINTING_ON = True
PLOTTING_ON = False
if PLOTTING_ON:
    style.use(PLOT_URL)
    FIG_DIR = Path(getcwd(), "fig_dump", "seq_ilqr")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(FIG_DIR)

def _plot_kkt(*args):
    """plot KKT conditions and state trajectories"""
    x, u, lamb, dl_dx, dl_du, dl_dlamb = args
    fig, ax = subplots(2,3, figsize=(10,3), sharey=False)
    ax[0,0].plot(x.squeeze())
    ax[0,0].set(title="X")
    ax[0,1].plot(u.squeeze())
    ax[0,1].set(title="U")
    ax[0,2].plot(lamb.squeeze())
    ax[0,2].set(title="λ")
    ax[1,0].plot(dl_dx.squeeze())
    ax[1,0].set(title="dLdX")
    ax[1,1].plot(dl_du.squeeze())
    ax[1,1].set(title="dLdUs")
    ax[1,2].plot(dl_dlamb.squeeze())
    ax[1,2].set(title="dLdλ")
    fig.tight_layout()
    return fig


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
            return jnp.sum(x**2) + jnp.sum(u**2)

        def costf(x: Array, theta: Theta):
            # return jnp.sum(jnp.abs(x))
            return jnp.sum(x**2)

        def dynamics(t: int, x: Array, u: Array, theta: Theta):
            return jnp.tanh(theta.Uh @ x + theta.Wh @ u) + jnp.ones(2)

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

    def test_vectorise_fun_in_time(self):
        """test vectorise function in time"""
        # setup
        (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        Xs = Xs[:-1]
        Us = Us[:]
        tps = jnp.arange(self.model.dims.horizon)
        # exercise
        (Fx, Fu) = ilqr.time_map(utils.linearise(self.model.dynamics))(
            tps, Xs, Us, self.theta
        )
        # verify
        chex.assert_shape(Fx, self.dims["TNN"])
        chex.assert_shape(Fu, self.dims["TNM"])

    def test_quadratise(self):
        """test quadratise function shape"""
        # setup
        (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        Xs = Xs[0]
        Us = Us[0]
        # exercise
        (Cxx, Cxu), (Cux, Cuu) = utils.quadratise(self.model.cost)(
            0, Xs, Us, self.params.theta
        )
        # verify
        chex.assert_shape(Cxx, self.dims["NN"])
        chex.assert_shape(Cxu, self.dims["NM"])
        chex.assert_shape(Cuu, self.dims["MM"])
        chex.assert_shape(Cux, self.dims["MN"])
        chex.assert_type(J0.dtype, float)

    def test_linearise(self):
        """test linearise function shape"""
        # setup
        (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        Xs = Xs[0]
        Us = Us[0]
        # exercise
        Fx, Fu = utils.linearise(self.model.dynamics)(0, Xs, Us, self.params.theta)
        Cx, Cu = utils.linearise(self.model.cost)(0, Xs, Us, self.params.theta)
        # verify
        chex.assert_shape(Fx, self.dims["NN"])
        chex.assert_shape(Fu, self.dims["NM"])
        chex.assert_shape(Cx, self.dims["N"])
        chex.assert_shape(Cu, self.dims["M"])

    def test_approx_lqr(self):
        """test approx_lqr function shape"""
        # setup
        (Xs, _), _ = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        # exercise
        lqr_tilde = ilqr.approx_lqr(
            model=self.model, Xs=Xs, Us=self.Us_init, params=self.params
        )
        # verify
        assert isinstance(lqr_tilde, LQR)
        # check shape
        chex.assert_shape(lqr_tilde.A, self.dims["TNN"])
        chex.assert_shape(lqr_tilde.B, self.dims["TNM"])
        chex.assert_shape(lqr_tilde.Q, self.dims["TNN"])
        chex.assert_shape(lqr_tilde.R, self.dims["TMM"])
        chex.assert_shape(lqr_tilde.S, self.dims["TNM"])
        chex.assert_shape(lqr_tilde.Qf, self.dims["NN"])

    def test_ilqr_simulate(self):
        """test ilqr simulate trajectory shape"""
        # setup
        Xs_lqr_sim = lqr.simulate_trajectory(
            self.model.dynamics, self.Us_init, self.params, self.model.dims
        )
        # exercise
        (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        # verify
        chex.assert_trees_all_equal(Us, self.Us_init)
        chex.assert_shape(Xs, (self.dims["T"][0] + 1,) + self.dims["N"])
        chex.assert_trees_all_equal(Xs, Xs_lqr_sim)

    def test_ilqr_forward_pass(self):
        """test ilqr forward pass shape and cost reduction"""
        # setup
        (old_Xs, _), initial_cost = ilqr.ilqr_simulate(
            self.model, self.Us_init, self.params
        )
        lqr_params = ilqr.approx_lqr(self.model, old_Xs, self.Us_init, self.params)
        exp_cost_red, gains = lqr.lqr_backward_pass(lqr_params)
        exp_change_J0 = lqr.calc_expected_change(exp_cost_red, alpha=1.0)
        # exercise
        (new_Xs, new_Us), new_total_cost = ilqr.ilqr_forward_pass(
            self.model, self.params, gains, old_Xs, self.Us_init, alpha=1.0
        )
        # verify
        print(
            f"\nInitial J0: {initial_cost}, New J0: {new_total_cost}", 
            f"Expected ΔJ0 (α=1): {exp_change_J0}"
        )
        chex.assert_shape(new_Xs, old_Xs.shape)
        chex.assert_shape(new_Us, self.Us_init.shape)
        assert new_total_cost < initial_cost
        assert new_total_cost - initial_cost < exp_change_J0

    def test_ilQR_solver(self):
        """test ilqr solver with integrater dynamics"""
        # setup
        (Xs_init, _), initial_cost = ilqr.ilqr_simulate(
            self.model, self.Us_init, self.params
        )
        # exercise
        (Xs_stars, Us_stars, Lambs_stars), converged_cost, cost_log = ilqr.ilqr_solver(
            self.model,
            self.params,
            self.Us_init,
            max_iter=70,
            convergence_thresh=1e-8,
            alpha_init=1.0,
            verbose=False,
            use_linesearch=True,
            **self.ls_kwargs,
        )
        lqr_params_stars = ilqr.approx_lqr(self.model, Xs_stars, Us_stars, self.params)
        lqr_tilde_params = LQRParams(Xs_stars[0], lqr_params_stars)
        dLdXs, dLdUs, dLdLambs = lqr.kkt(
            lqr_tilde_params, Xs_stars, Us_stars, Lambs_stars
        )        
        if PLOTTING_ON:
            fig, ax = subplots(2, 2, sharey=True)
            ax[0, 0].plot(Xs_init)
            ax[0, 0].set(title="X")
            ax[0, 1].plot(self.Us_init)
            ax[0, 1].set(title="U")
            ax[1, 0].plot(Xs_stars)
            ax[1, 1].plot(Us_stars)
            fig.tight_layout()
            fig.savefig(f"{FIG_DIR}/ilqr_solver.png")
            close()
        
            fig = _plot_kkt(Xs_stars, Us_stars, Lambs_stars, dLdXs, dLdUs, dLdLambs)
            fig.savefig(f"{FIG_DIR}/ilqr_kkt.png")
            close()
        
            fig, ax = subplots()
            ax.scatter(jnp.arange(cost_log.size), cost_log)
            ax.set(xlabel="Iteration", ylabel="Total cost")
            fig.savefig(f"{FIG_DIR}/ilqr_cost_log.png")
            close()

        # verify
        assert converged_cost < initial_cost
        # assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-03, atol=1e-04)
        # assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-03, atol=1e-04)
        # assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-03, atol=1e-04)


# load test data
# description of tajax data


class TestiLQRExactSolution(unittest.TestCase):
    """Test iLQR solver with exact solution"""

    def setUp(self):
        # load fixtures
        self.fixtures = onp.load("tests/fixtures/ilqr_exact_solution.npz")

        # dimensions
        self.dims = chex.Dimensions(T=50, N=2, M=2, X=1)

        # set-up model
        key = jr.PRNGKey(seed=234)
        key, skeys = keygen(key, 5)
        Uh = initialise_stable_dynamics(next(skeys), *self.dims["NT"], .6)[0]
        Wh = jr.normal(next(skeys), self.dims["NM"])
        # chex.assert_trees_all_equal(self.fixtures["Uh"], Uh)
        # chex.assert_trees_all_equal(self.fixtures["Wh"], Wh)
        Q = jnp.eye(*self.dims["N"])
        # initialise params
        self.theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros((2)), Q=Q)
        theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros(self.dims["N"]), Q=Q)
        self.params = iLQRParams(
            x0=jr.normal(next(skeys), self.dims["N"]), theta=theta
        )
        self.Us = jnp.ones(self.dims["TM"])

        # define linesearch hyper parameters
        self.ls_kwargs = {
            "beta":0.8,
            "max_iter_linesearch":16,
            "tol":0.1,
            "alpha_min":0.0001,
            }
        x_targ = jnp.sin(jnp.linspace(0,2,self.dims["T"][0]+1))
        x_targ = jnp.tile(x_targ, (self.dims["N"][0], 1)).T
        self.x_targ = x_targ
        def cost(t: int, x: Array, u: Array, theta: Any):
            return jnp.sum((x-x_targ[t])**2) + 0.1*jnp.sum((u**2))

        def costf(x: Array, theta: Theta):
            return 1.0*jnp.sum((x-x_targ[-1])**2)

        def dynamics(t: int, x: Array, u: Array, theta: Theta):
            return  theta.Uh @ x + theta.Wh @ u + jnp.ones_like(x)

        self.model = System(
            cost, costf, dynamics, ModelDims(*self.dims["NMT"], dt=0.1)
        )

    # def test_ilqr_nolinesearch(self):
    #     """test ilqr solver without linesearch"""
    #     # exercise rollout
    #     (Xs_init, Us_init), cost_init = ilqr.ilqr_simulate(
    #         self.model, self.Us, self.params
    #     )
    #     # verify
    #     # chex.assert_trees_all_equal(self.fixtures["X_orig"], Xs_init[1:])
    #     chex.assert_trees_all_close(self.fixtures["X_orig"], Xs_init[1:], rtol=1e-11, atol=1e-11)

    #     # exercise ilqr solver
    #     (Xs_stars, Us_stars, Lambs_stars), total_cost, _ = ilqr.ilqr_solver(
    #         self.model,
    #         self.params,
    #         self.Us,
    #         max_iter=70,
    #         tol=1e-8,
    #         alpha_init=0.8,
    #         verbose=True,
    #         use_linesearch=False,
    #     )

    #     fig, ax = subplots(2, 2, sharey=True)
    #     ax[0, 0].plot(Xs_init)
    #     ax[0, 0].set(title="X")
    #     ax[0, 1].plot(self.Us)
    #     ax[0, 1].set(title="U")
    #     ax[1, 0].plot(Xs_stars)
    #     ax[1, 1].plot(Us_stars)
    #     fig.tight_layout()
    #     fig.savefig(f"{FIG_DIR}/ilqr_ls_solver.png")
    #     close()

    #     # verify
    #     chex.assert_trees_all_close(Xs_stars, self.fixtures["X"], rtol=1e-03, atol=1e-03)
    #     chex.assert_trees_all_close(Us_stars, self.fixtures["U"], rtol=1e-03, atol=1e-03)
    #     print(f"iLQR solver cost:\t{total_cost:.6f}"
    #           f"\nOther solver cost:\t{self.fixtures['obj']:.6f}")
    #     assert jnp.allclose(total_cost, self.fixtures['obj'], rtol=1e-04, atol=1e-04)

    # def test_ilqr_linesearch(self):
    #     """test ilqr solver with linesearch"""
    #     # exercise ilqr solver
    #     (Xs_stars, Us_stars, Lambs_stars), total_cost, _ = ilqr.ilqr_solver(
    #         self.model,
    #         self.params,
    #         self.Us,
    #         max_iter=40,
    #         convergence_thresh=1e-8,
    #         alpha_init=1.,
    #         verbose=True,
    #         use_linesearch=True,
    #         **self.ls_kwargs,
    #     )
    #     # verify
    #     chex.assert_trees_all_close(Xs_stars, self.fixtures["X"], rtol=1e-06, atol=1e-04)
    #     chex.assert_trees_all_close(Us_stars, self.fixtures["U"], rtol=1e-06, atol=1e-04)
    #     print(f"iLQR solver cost:\t{total_cost:.6f}"
    #           f"\nOther solver cost:\t{self.fixtures['obj']:.6f}")
    #     assert jnp.allclose(total_cost, self.fixtures['obj'], rtol=1e-06, atol=1e-06)

    def test_lqr_on_ilqr(self):
        T = self.model.dims.horizon
        # set up lqr
        print(T, self.x_targ.shape)
        lqr_thetas = LQR(
                        A=jnp.tile(self.theta.Uh,(T,1,1)),
                        B=jnp.tile(self.theta.Wh,(T,1,1)),
                        a=jnp.ones((T,self.model.dims.n)),
                        q=2*-self.x_targ[:-1],
                        qf=2*-self.x_targ[-1],
                        Qf=2*jnp.eye(self.model.dims.n),
                        Q=2*jnp.tile(jnp.eye(self.model.dims.n),(T,1,1)),
                        R=2*0.1*jnp.tile(jnp.eye(self.model.dims.m),(T,1,1)),
                        r=2*0.1*jnp.zeros((T,self.model.dims.m)),
                        S=jnp.zeros((T,self.model.dims.n, self.model.dims.m))
                        )
        lqr_params = LQRParams(self.params.x0, lqr_thetas)
        # run lqr
        Xs_lqr, Us_lqr, Lambs_lqr = lqr.solve_lqr(lqr_params)
        
        # run ilqr
        (Xs_stars, Us_stars, Lambs_stars), total_cost, cost_log = ilqr.ilqr_solver(
            self.model,
            self.params,
            self.Us,
            max_iter=80,
            convergence_thresh=1e-13,
            alpha_init=1.,
            verbose=True,
            use_linesearch=False,
            **self.ls_kwargs,
        )
        if PLOTTING_ON:
            fig, ax = subplots(1,3)
            ax[0].plot(Xs_stars,c='b', linestyle=":",alpha=.6)
            ax[0].plot(Xs_lqr,c='k', linestyle="--",alpha=.6)
            ax[1].plot(Us_stars,c='b', linestyle=":",alpha=.6)
            ax[1].plot(Us_lqr,c='k', linestyle="--",alpha=.6)
            ax[2].plot(Lambs_stars,c='b', linestyle=":",alpha=.6)
            ax[2].plot(Lambs_lqr,c='k', linestyle="--",alpha=.6)
            [a_.set(title=l_) for a_,l_ in zip(ax.flatten(), ["x","u","λ"])]
            fig.savefig(f"{FIG_DIR}/ilqr_vs_lqr.png")
            close(fig)
        
        # test
        chex.assert_trees_all_close(Xs_stars, Xs_lqr, rtol=1e-04, atol=1e-04)
        chex.assert_trees_all_close(Us_stars, Us_lqr, rtol=1e-04, atol=1e-04)
        chex.assert_trees_all_close(Lambs_stars, Lambs_lqr, rtol=1e-04, atol=1e-04)
        

    def test_ilqr_kkt_solution(self):
        """test ilqr solver with kkt optimality conditions"""
        # exercise ilqr solver
        (Xs_stars, Us_stars, Lambs_stars), total_cost, cost_log = ilqr.ilqr_solver(
            self.model,
            self.params,
            self.Us,
            max_iter=80,
            convergence_thresh=1e-13,
            alpha_init=1.,
            verbose=True,
            use_linesearch=False,
            **self.ls_kwargs,
        )
        # lqr_tilde = ilqr.approx_lqr_dyn(model=self.model, Xs=Xs_stars, Us=Us_stars, params=self.params)
        lqr_tilde = ilqr.approx_lqr_offset(model=self.model, Xs=Xs_stars, Us=Us_stars, params=self.params)
        lqr_approx_params = LQRParams(Xs_stars[0], lqr_tilde)
        # verify
        print(jnp.linalg.eigvals(self.params.theta.Uh))
        print(jnp.linalg.eigvals(lqr_tilde.A[0]))
        dLdXs, dLdUs, dLdLambs = lqr.kkt(lqr_approx_params, Xs_stars, Us_stars, Lambs_stars)
        # plot kkt
        if PLOTTING_ON:
            fig = _plot_kkt(Xs_stars, Us_stars, Lambs_stars, dLdXs, dLdUs, dLdLambs)
            fig.savefig(f"{FIG_DIR}/ilqr_ls_kkt_sin_2_short_nl.png")
            close()
        
            fig, ax = subplots()
            ax.plot(lqr_tilde.r)
            fig.savefig(f"{FIG_DIR}/ilqr_ls_kkt_sin_2_r_short_nl.png")
            close()
        
            fig, ax = subplots()
            ax.scatter(jnp.arange(cost_log.size), cost_log)
            ax.set(xlabel="Iteration", ylabel="Total cost")
            fig.savefig(f"{FIG_DIR}/ilqr_ls_cost_log.png")
            close()

        print(jnp.mean(jnp.abs(dLdXs)))
        print(jnp.mean(jnp.abs(dLdUs)))
        print(jnp.mean(jnp.abs(dLdLambs)))
        # Verify that the average KKT conditions are satisfied
        assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-04, atol=1e-05)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-04, atol=1e-05)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-02, atol=1e-02)

        # Verify that the terminal state KKT conditions is satisfied
        assert jnp.allclose(dLdXs[-1], 0.0, rtol=1e-04, atol=1e-05), "Terminal X not satisfied"

        # Verify that all KKT conditions are satisfied
        assert jnp.allclose(dLdUs, 0.0, rtol=1e-04, atol=1e-05)
        assert jnp.allclose(dLdXs, 0.0, rtol=1e-04, atol=1e-05)
        # assert jnp.allclose(dLdLambs, 0.0, rtol=1e-05, atol=1e-08)



class TestiLQRWithLQRProblem(unittest.TestCase):
    """Test iLQR solver with exact solution"""

    def setUp(self):
        # dimensions
        self.dims = chex.Dimensions(T=100, N=2, M=2, X=1)
        self.sys_dims = ModelDims(*self.dims["NMT"], dt=0.1)
        dt = self.sys_dims.dt
        self.Us = jnp.zeros(self.dims["TM"])
        self.x0 = jnp.array([0.3, 0.])

        # load LQR problem
        span_time_m=self.dims["TXX"]
        span_time_v=self.dims["TX"]
        A = jnp.tile(jnp.array([[1,dt],[-1*dt,1-0.5*dt]]), span_time_m)
        B = jnp.tile(jnp.array([[0,0],[1,0]]), span_time_m)*dt
        a = jnp.zeros(self.dims["TN"])
        Qf = 1. *jnp.eye(self.dims["N"][0])
        qf = 0.   * jnp.ones(self.dims["N"])
        Q = 1. * jnp.tile(jnp.eye(self.dims["N"][0]), span_time_m)
        q = 0. * jnp.tile(jnp.ones(self.dims["N"]), span_time_v)
        R = 1. * jnp.tile(jnp.eye(self.dims["M"][0]), span_time_m)
        r = 0. * jnp.tile(jnp.ones(self.dims["M"]), span_time_v)
        S = 0. * jnp.tile(jnp.ones(self.dims["NM"]), span_time_m)
        self.lqr_struct = LQR(A, B, a, Q, q, R, r, S, Qf, qf)()
        self.lqr_params = LQRParams(self.x0, self.lqr_struct)


        # set-up lqr in the model
        Uh = self.lqr_struct.A[0]
        Wh = self.lqr_struct.B[0]
        theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros(self.dims["N"]), Q=Q[0])
        self.ilqr_params = iLQRParams(x0=self.x0, theta=theta)

        def cost(t: int, x: Array, u: Array, theta: Any):
            return 0.5*jnp.sum(x**2) + 0.5*jnp.sum(u**2)

        def costf(x: Array, theta: Theta):
            return 0.5*jnp.sum(x**2)

        def dynamics(t: int, x: Array, u: Array, theta: Theta):
            return theta.Uh @ x + theta.Wh @ u

        self.model = System(
            cost, costf, dynamics, ModelDims(*self.dims["NMT"], dt=0.1)
        )
        print(self.model)

    def test_lqr_solution(self):
        """test lqr solution with ilqr solver"""
        # setup: simulate dynamics
        lqr_Xs_sim = lqr.simulate_trajectory(
            dynamics=lqr.lin_dyn_step,
            Us=self.Us,
            params=self.lqr_params,
            dims=self.sys_dims
            )
        # exercise rollout
        (Xs_init, Us_init), cost_init = ilqr.ilqr_simulate(self.model, self.Us, self.ilqr_params)
        # verify
        chex.assert_trees_all_close(Xs_init, lqr_Xs_sim)

        # setup: lqr solver
        Xs_lqr, Us_lqr, Lambs_lqr = lqr.solve_lqr(self.lqr_params) #, self.sys_dims)
        # exercise ilqr solver
        (Xs_stars, Us_stars, Lambs_stars), total_cost, _ = ilqr.ilqr_solver(
            self.model,
            self.ilqr_params,
            self.Us,
            max_iter=70,
            tol=1e-8,
            alpha_init=1.,
            verbose=True,
            use_linesearch=False,
        )
        # verify
        chex.assert_trees_all_close(Xs_stars, Xs_lqr, rtol=1e-04, atol=1e-04)
        chex.assert_trees_all_close(Us_stars, Us_lqr, rtol=1e-04, atol=1e-04)

        # exercise lqr approximation
        lqr_tilde = ilqr.approx_lqr(
            model=self.model,
            Xs=Xs_init,
            Us=self.Us,
            params=self.ilqr_params
            )
        # verify
        # chex.assert_trees_all_close(lqr_tilde, self.lqr_struct)
        chex.assert_trees_all_close(lqr_tilde.A, self.lqr_struct.A)
        chex.assert_trees_all_close(lqr_tilde.B, self.lqr_struct.B)
        chex.assert_trees_all_close(lqr_tilde.a, self.lqr_struct.a)
        chex.assert_trees_all_close(lqr_tilde.Q, self.lqr_struct.Q)
        chex.assert_trees_all_close(lqr_tilde.Qf, self.lqr_struct.Qf)
        chex.assert_trees_all_close(lqr_tilde.R, self.lqr_struct.R)
        chex.assert_trees_all_close(lqr_tilde.S, self.lqr_struct.S)
        chex.assert_trees_all_close(lqr_tilde.r, self.lqr_struct.r)
        # chex.assert_trees_all_close(lqr_tilde.q, self.lqr_struct.q)
        # chex.assert_trees_all_close(lqr_tilde.qf, self.lqr_struct.qf)
