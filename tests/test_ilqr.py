"""Test functions in src/ilqr.py"""
import unittest
import pytest
import chex, jax 
from jax import Array, grad
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
from os import getcwd
from pathlib import Path
from matplotlib.pyplot import subplots, close

from src.utils import keygen
import src.ilqr as ilqr
import src.lqr as lqr

class TestiLQRStructs(unittest.TestCase):
    """Test LQR dimensions and data structures"""
    def setUp(self):
        """Setup LQR problem"""
        key = jr.PRNGKey(seed=234)
        key, skeys = keygen(key, 3)
        
        dt=0.1
        Uh = jnp.array([[1,dt],[-1*dt,1-0.5*dt]])
        Wh = jnp.array([[0,0],[1,0]])*dt
        # initialise params
        self.theta = ilqr.Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros((2, 1)))
        self.params = ilqr.Params(x0=jnp.array([[0.3], [0.]]), theta=self.theta)
        # define model
        def cost(t: int, x: Array, u: Array, theta: ilqr.Theta):
            return jnp.sum(x**2) + jnp.sum(u**2)
        def costf(x: Array, theta: ilqr.Theta):
            # return jnp.sum(jnp.abs(x))
            return jnp.sum(x**2)
        def dynamics(t: int, x: Array, u: Array, theta: ilqr.Theta):
            return jnp.tanh(theta.Uh @ x + theta.Wh @ u)
        
        self.model = ilqr.System(cost, costf, dynamics, lqr.ModelDims(horizon=100, n=2, m=2, dt=dt))
        self.dims = chex.Dimensions(T=100, N=2, M=2, X=1)
        self.Us_init = 0.1 * jr.normal(next(skeys), (self.model.dims.horizon, self.model.dims.m, 1))
        
    def test_vectorise_fun_in_time(self):
        # setup
        (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        Xs = Xs[:-1].squeeze()
        Us = Us[:].squeeze()
        tps = jnp.arange(self.model.dims.horizon)
        # exercise
        (Fx, Fu) = ilqr.vectorise_fun_in_time(ilqr.linearise(self.model.dynamics))(tps, Xs, Us, self.theta)
        # verify
        chex.assert_shape(Fx, self.dims["TNN"])
        chex.assert_shape(Fu, self.dims["TNM"])
        
    def test_quadratise(self):
        # setup
        (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        Xs = Xs[0].squeeze()
        Us = Us[0].squeeze()
        # exercise
        (Cxx, Cxu), (Cux, Cuu) = ilqr.quadratise(self.model.cost)(0,Xs, Us, self.params.theta)
        # verify
        chex.assert_shape(Cxx, self.dims["NN"])
        chex.assert_shape(Cxu, self.dims["NM"])
        chex.assert_shape(Cuu, self.dims["MM"])
        chex.assert_shape(Cux, self.dims["MN"])
        chex.assert_type(J0.dtype, float)
        
    def test_linearise(self):
        # setup
        (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        Xs = Xs[0].squeeze()
        Us = Us[0].squeeze()
        # exercise
        Fx, Fu = ilqr.linearise(self.model.dynamics)(0,Xs, Us, self.params.theta)
        Cx, Cu = ilqr.linearise(self.model.cost)(0,Xs, Us, self.params.theta)
        # verify
        chex.assert_shape(Fx, self.dims["NN"])
        chex.assert_shape(Fu, self.dims["NM"])
        chex.assert_shape(Cx, self.dims["N"])
        chex.assert_shape(Cu, self.dims["M"])

    def test_approx_lqr(self):
        # setup
        (Xs, _), _ = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        # exercise
        lqr_tilde = ilqr.approx_lqr(model=self.model, Xs=Xs, Us=self.Us_init, params=self.params)
        # verify
        assert isinstance(lqr_tilde, lqr.LQR)
        # check shape
        chex.assert_shape(lqr_tilde.A, self.dims["TNN"])
        chex.assert_shape(lqr_tilde.B, self.dims["TNM"])
        chex.assert_shape(lqr_tilde.Q, self.dims["TNN"])
        chex.assert_shape(lqr_tilde.R, self.dims["TMM"])
        chex.assert_shape(lqr_tilde.S, self.dims["TNM"])
        chex.assert_shape(lqr_tilde.Qf, self.dims["NN"])
        
    def test_ilqr_simulate(self):
        # setup
        Xs_lqr_sim = lqr.simulate_trajectory(self.model.dynamics, self.Us_init, self.params, self.model.dims)
        # exercise
        (Xs, Us), J0 = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        # verify
        chex.assert_trees_all_equal(Us, self.Us_init)
        chex.assert_shape(Xs, (self.dims["T"][0]+1,) + self.dims["NX"])
        chex.assert_trees_all_equal(Xs, Xs_lqr_sim)
        
        
    def test_ilqr_forward_pass(self):
        # setup
        (old_Xs,_), initial_cost = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        lqr_params = ilqr.approx_lqr(self.model, old_Xs, self.Us_init, self.params)
        exp_cost_red, gains = lqr.lqr_backward_pass(
            lqr_params, dims=self.model.dims, expected_change=False, verbose=False
        )
        exp_change_J0 = lqr.calc_expected_change(exp_cost_red, alpha=1.0)
        # exercise
        (new_Xs, new_Us), new_total_cost = ilqr.ilqr_forward_pass(
            self.model, self.params, gains, old_Xs, self.Us_init, alpha=1.0
        )
        # verify
        print(f"\nInitial J0: {initial_cost}, New J0: {new_total_cost}, Expected ΔJ0 (α=1): {exp_change_J0}")
        chex.assert_shape(new_Xs, old_Xs.shape)
        chex.assert_shape(new_Us, self.Us_init.shape)
        assert new_total_cost < initial_cost
        assert new_total_cost-initial_cost < exp_change_J0
        
        
    def test_ilQR_solver(self):
        # setup
        fig_dir = Path(Path(getcwd()), "fig_dump")
        fig_dir.mkdir(exist_ok=True)
        (Xs_init,_), initial_cost = ilqr.ilqr_simulate(self.model, self.Us_init, self.params)
        # exercise
        (Xs_stars, Us_stars, Lambs_stars), converged_cost = ilqr.ilQR_solver(
            self.model, self.params, Xs_init, self.Us_init, max_iter=20, tol=1e-2, verbose=True
        )
        fig, ax = subplots(2,2, sharey=True)
        ax[0,0].plot(Xs_init.squeeze())
        ax[0,0].set(title="X")
        ax[0,1].plot(self.Us_init.squeeze())
        ax[0,1].set(title="U")
        ax[1,0].plot(Xs_stars.squeeze())
        ax[1,1].plot(Us_stars.squeeze())
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/ilqr_solver.png")
        close()
        lqr_params_stars = ilqr.approx_lqr(self.model, Xs_stars, Us_stars, self.params)
        lqr_tilde_params = lqr.Params(Xs_stars[0],lqr_params_stars)
        dLdXs, dLdUs, dLdLambs = lqr.kkt(lqr_tilde_params, Xs_stars, Us_stars, Lambs_stars)
        fig, ax = subplots(2,3, figsize=(10,3), sharey=False)
        ax[0,0].plot(Xs_stars.squeeze())
        ax[0,0].set(title="X")
        ax[0,1].plot(Us_stars.squeeze())
        ax[0,1].set(title="U")
        ax[0,2].plot(Lambs_stars.squeeze())
        ax[0,2].set(title="λ")
        ax[1,0].plot(dLdXs.squeeze())
        ax[1,0].set(title="dLdX")
        ax[1,1].plot(dLdUs.squeeze())
        ax[1,1].set(title="dLdUs")
        ax[1,2].plot(dLdLambs.squeeze())
        ax[1,2].set(title="dLdλ")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/ilqr_kkt.png")
        close()

        # verify
        assert converged_cost < initial_cost
        # assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-03, atol=1e-04)
        # assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-03, atol=1e-04)
        # assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-03, atol=1e-04)


class TestiLQRExactSolution(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_1(self):
        pass
    def test_2(self):
        pass
    def test_3(self):
        pass
    
    