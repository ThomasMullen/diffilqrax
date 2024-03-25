import unittest
import pytest
import chex, jax 
from jax import Array, grad
from typing import Tuple
import jax.random as jr
import jax.numpy as jnp
from jaxopt import linear_solve, implicit_diff

import numpy as onp
from src.exact import quad_solve, exact_solve

jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_enable_x64", True)  # double precision

from src import (
    Gains,
    CostToGo,
    LQR,
    Params,
    ModelDims,
    simulate_trajectory,
    lqr_adjoint_pass,
    lin_dyn_step,
    lqr_forward_pass,
    lqr_tracking_forward_pass,
    lqr_backward_pass,
    solve_lqr,
    kkt,
)

from src import keygen, initialise_stable_dynamics

is_jax_Array = lambda arr: isinstance(arr, jnp.ndarray) and not isinstance(arr, onp.ndarray)

def setup_lqr(dims: chex.Dimensions,
              pen_weight: dict = {"Q": 1e-0, "R": 1e-3, "Qf": 1e0, "S": 1e-3}) -> LQR:
    """Setup LQR problem"""
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    # initialise dynamics
    span_time=dims["TXX"]
    A = initialise_stable_dynamics(next(skeys), dims['N'][0], dims['T'][0],radii=0.6)
    B = jnp.tile(jr.normal(next(skeys), dims['NM']), span_time)
    a = jnp.tile(jr.normal(next(skeys), dims['NX']), span_time)
    # define cost matrices
    Q = pen_weight["Q"] * jnp.tile(jnp.eye(dims['N'][0]), span_time)
    q = 2*1e-1 * jnp.tile(jnp.ones(dims['NX']), span_time)
    R = pen_weight["R"] * jnp.tile(jnp.eye(dims['M'][0]), span_time)
    r = 1e-6 * jnp.tile(jnp.ones(dims['MX']), span_time)
    S = pen_weight["S"] * jnp.tile(jnp.ones(dims['NM']), span_time)
    Qf = pen_weight["Q"] * jnp.eye(dims['N'][0])
    qf = 2*1e-1 * jnp.ones(dims['NX'])
    # construct LQR
    lqr = LQR(A, B, a, Q, q, Qf, qf, R, r, S)
    return lqr()


class TestLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        """Instantiate dummy LQR"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=60, N=3, M=2, X=1)
        self.sys_dims = ModelDims(self.dims["N"][0], self.dims["M"][0], self.dims["T"][0], dt=0.1)
        print("Model dimensionality", self.dims["TNMX"])
        print("\nMake LQR struct")
        self.lqr = setup_lqr(self.dims)

        print("\nMake initial state x0 and input U")
        self.x0 = jnp.array([[2.0], [1.0], [1.0]])
        Us = jnp.zeros(self.dims["TMX"], dtype=float)
        Us = Us.at[2].set(1.0)
        self.Us = Us

    def test_lqr_params_struct(self):
        print("Construct params")
        params = Params(self.x0, self.lqr)
        chex.assert_trees_all_close(params[0], self.x0)
        chex.assert_trees_all_close(params.x0, self.x0)
        chex.assert_type(params.x0, float)
        assert isinstance(params.lqr, LQR)
        assert isinstance(params[1], LQR)
        assert isinstance(params[-1], LQR)
        print("Params struct passed.")

    def test_lqr_struct(self):
        """Test test shape of LQR"""
        print("Running test_lqr_struct")

        # test cost is positive symmetric
        chex.assert_trees_all_close(self.lqr.Q[0], self.lqr.Q[0].T)
        chex.assert_trees_all_close(self.lqr.R[0], self.lqr.R[0].T)
        # check if Q is positive definite
        assert jnp.all(jnp.linalg.eigvals(self.lqr.Q[0]) > 0), "Q is not positive definite"
        assert jnp.all(jnp.linalg.eigvals(self.lqr.R[0]) > 0), "R is not positive definite"

        # check shape
        chex.assert_shape(self.lqr.A, self.dims["TNN"])
        chex.assert_shape(self.lqr.B, self.dims["TNM"])
        chex.assert_shape(self.lqr.Q, self.dims["TNN"])
        chex.assert_shape(self.lqr.R, self.dims["TMM"])
        chex.assert_shape(self.lqr.S, self.dims["TNM"])

        # check dtypes
        chex.assert_type(self.lqr.S.dtype, float)
        chex.assert_type(self.lqr.Qf.dtype, float)
        # test jax arrays
        assert is_jax_Array(self.lqr.Q)
        assert is_jax_Array(self.lqr.R)
        assert is_jax_Array(self.lqr.S)
        assert is_jax_Array(self.lqr.Qf)
        assert is_jax_Array(self.lqr.A)
        assert is_jax_Array(self.lqr.B)
        print("LQR struct passed.")

    def test_simulate_trajectory(self):
        print("Running test_simulate_trajectory")
        params = Params(self.x0, self.lqr)
        Xs = simulate_trajectory(lin_dyn_step, self.Us, params, self.sys_dims)
        chex.assert_type(Xs, float)
        chex.assert_shape(Xs, (self.dims["T"][0] + 1,) + self.dims["NX"])

    def test_lqr_adjoint_pass(self):
        print("Running test_lqr_adjoint_pass")
        params = Params(self.x0, self.lqr)
        Xs_sim = simulate_trajectory(lin_dyn_step, self.Us, params, self.sys_dims)
        Lambs = lqr_adjoint_pass(Xs_sim, self.Us, params)
        chex.assert_type(Lambs, float)
        chex.assert_shape(Lambs, (self.dims["T"][0] + 1,) + self.dims["NX"])

    def test_lqr_backward_pass(self):
        params = Params(self.x0, self.lqr)
        (dJ, Ks), exp_dJ = lqr_backward_pass(
            lqr=params.lqr, dims=self.sys_dims, expected_change=True, verbose=False
        )
        chex.assert_type(Ks.K, float)
        chex.assert_shape(Ks.K, self.dims["TMN"])
        chex.assert_type(Ks.k, float)
        chex.assert_shape(Ks.k, self.dims["TMX"])
        chex.assert_type(dJ, float)
        chex.assert_type(exp_dJ, float)

    def test_lqr_forward_pass(self):
        params = Params(self.x0, self.lqr)
        (dJ, Ks), exp_dJ = lqr_backward_pass(
            lqr=params.lqr, dims=self.sys_dims, expected_change=True, verbose=False
        )
        Xs_lqr, Us_lqr = lqr_forward_pass(gains=Ks, params=params)
        chex.assert_type(Xs_lqr, float)
        chex.assert_shape(Xs_lqr, (self.dims["T"][0] + 1,) + self.dims["NX"])
        chex.assert_type(Us_lqr, float)
        chex.assert_shape(Us_lqr, self.dims["TMX"])

    def test_solve_lqr(self):
        params = Params(self.x0, self.lqr)
        gains_lqr, Xs_lqr, Us_lqr, Lambs_lqr = solve_lqr(params, self.sys_dims)
        chex.assert_type(gains_lqr.K, float)
        chex.assert_shape(gains_lqr.K, self.dims["TMN"])
        chex.assert_type(gains_lqr.k, float)
        chex.assert_shape(gains_lqr.k, self.dims["TMX"])
        chex.assert_type(Xs_lqr, float)
        chex.assert_shape(Xs_lqr, (self.dims["T"][0] + 1,) + self.dims["NX"])
        chex.assert_type(Us_lqr, float)
        chex.assert_shape(Us_lqr, self.dims["TMX"])
        chex.assert_type(Lambs_lqr, float)
        chex.assert_shape(Lambs_lqr, (self.dims["T"][0] + 1,) + self.dims["NX"])
        # verify output jax arrays
        assert is_jax_Array(gains_lqr.K)
        assert is_jax_Array(gains_lqr.k)
        assert is_jax_Array(Xs_lqr)
        assert is_jax_Array(Us_lqr)
        assert is_jax_Array(Lambs_lqr)
        
    def test_solution_output(self):
        import os
        import matplotlib.pyplot as plt
        from pathlib import Path
        params = Params(self.x0, self.lqr)
        gains_lqr, Xs_lqr, Us_lqr, Lambs_lqr = solve_lqr(params, self.sys_dims)
        fig_dir = Path(Path(os.getcwd()), "fig_dump")
        fig_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(1,3,figsize=(10,3))
        ax[0].plot(Xs_lqr.squeeze())
        ax[1].plot(Us_lqr.squeeze())
        ax[2].plot(Lambs_lqr.squeeze())
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/lqr_solution.png")
        plt.close()
        
        
    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")


class TestLQRSolution(unittest.TestCase):
    """Test LQR solution using jaxopt conjugate gradient solution"""

    def setUp(self):
        """Instantiate dummy LQR"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=20, N=3, M=2, X=1)
        self.sys_dims = ModelDims(self.dims["N"][0], self.dims["M"][0], self.dims["T"][0], dt=0.1)
        print(self.dims["TNMX"])

        print("\nMake LQR struct")
        key = jr.PRNGKey(seed=234)
        key, skeys = keygen(key, 3)

        A = initialise_stable_dynamics(next(skeys), self.dims["N"][0], self.dims["T"][0],radii=0.6)
        B = jnp.tile(jr.normal(next(skeys), self.dims["NM"]), self.dims["TXX"])
        a = jnp.tile(jr.normal(next(skeys), self.dims["NX"]), self.dims["TXX"])
        # B = jr.normal(next(skeys), self.dims["TNM"])
        # a = jr.normal(next(skeys), self.dims["TNX"])
        Qf = jnp.eye(self.dims["N"][0])
        qf = 2*1e-1 * jnp.ones(self.dims["NX"])
        Q = jnp.tile(jnp.eye(self.dims["N"][0]), self.dims["TXX"])
        q = 2*1e-1 * jnp.tile(jnp.ones(self.dims["NX"]), self.dims["TXX"])
        R = 1e-4 * jnp.tile(jnp.eye(self.dims["M"][0]), self.dims["TXX"])
        r = 1e-2 * jnp.tile(jnp.ones(self.dims["MX"]), self.dims["TXX"])
        S = 1e-4 * jnp.tile(jnp.ones(self.dims["NM"]), self.dims["TXX"])
        self.lqr = LQR(A, B, a, Q, q, Qf, qf, R, r, S)()

        print("\nMake x0 and input U")
        self.x0 = jnp.array([[2.0], [1.0], [1.0]])
        Us = jnp.zeros(self.dims["TMX"]) * 1.0
        Us = Us.at[2].set(1.0)
        self.Us = Us
        self.params = Params(self.x0, self.lqr)

    def test_lqr_solution(self):
        """test LQR solution using jaxopt conjugate gradient solution"""
        # Exercise the LQR solver function
        K_dir, Xs_dir, Us_dir, Lambs_dir = solve_lqr(params=self.params, sys_dims=self.sys_dims)
        K_impl, Xs_impl, Us_impl, Lambs_impl = implicit_diff.custom_root(
            kkt, linear_solve.solve_cg
        )(solve_lqr)(self.params)
        # Verify that the two solutions are close
        assert jnp.allclose(Xs_dir, Xs_impl, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(Us_dir, Us_impl, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(Lambs_dir, Lambs_impl, rtol=1e-05, atol=1e-08)


    def test_kkt_optimal(self):
        # Setup the LQR problem
        K_dir, Xs_dir, Us_dir, Lambs_dir = solve_lqr(params=self.params, sys_dims=self.sys_dims)
        # Exercise the KKT function
        dLdXs, dLdUs, dLdLambs = kkt(self.params, Xs_dir, Us_dir, Lambs_dir)
        # Verify that the KKT conditions are satisfied
        assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-01, atol=1e-01)
        #TODO: Inspect why U not near 0 and how to set threhold
        assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-01, atol=1e-01) 
        assert jnp.allclose(dLdLambs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdXs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdXs[-1], 0.0, rtol=1e-05, atol=1e-08), "Terminal X state not satisfied"
        assert jnp.allclose(dLdUs, 0.0, rtol=1e-05, atol=1e-08)
        
    def test_gradients(self):
        # Setup the LQR problem
        # define a loss function
        def loss(params, Xs, Us, Lambds):
            return jnp.sum(Xs**2) + jnp.sum(Us**2) + jnp.sum(Lambds**2) + jnp.sum(params.x0**2) + jnp.sum(params.lqr.A**2)
        def dir_loss(params):
            s = solve_lqr(params=params)[1:]
            return loss(params, *s)
        def impl_loss(params):
            s = implicit_diff.custom_root(
                kkt, linear_solve.solve_cg
            )(solve_lqr)(params)[1:]
            return loss(params, *s)
        # Exercise - take gradients of implicit and explicit solutions
        dir_param_grads = grad(dir_loss, allow_int=True)(self.params)
        impl_param_grads = grad(impl_loss, allow_int=True)(self.params)
        # Verify
        assert jnp.allclose(dir_param_grads.x0, impl_param_grads.x0, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(dir_param_grads.lqr.A, impl_param_grads.lqr.A, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(dir_param_grads.lqr.B, impl_param_grads.lqr.B, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(dir_param_grads.lqr.a, impl_param_grads.lqr.a, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(dir_param_grads.lqr.Q, impl_param_grads.lqr.Q, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(dir_param_grads.lqr.S, impl_param_grads.lqr.S, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(dir_param_grads.lqr.R, impl_param_grads.lqr.R, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(dir_param_grads.lqr.q, impl_param_grads.lqr.q, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(dir_param_grads.lqr.r, impl_param_grads.lqr.r, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(dir_param_grads.lqr.Qf, impl_param_grads.lqr.Qf, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(dir_param_grads.lqr.qf, impl_param_grads.lqr.qf, rtol=1e-01, atol=1e-01)
        # chex.assert_numerical_grads
        pass


class TestLQRSolutionExact(unittest.TestCase):
    """Test LQR solution comparing to the exact solution using a CG solve (in a case in which the dynamics are constant)"""

    def setUp(self):
        """Instantiate LQR example using the pendulum example to compare against Ocaml"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=100, N=2, M=2, X=1)
        self.sys_dims = ModelDims(self.dims["N"][0], self.dims["M"][0], self.dims["T"][0], dt=0.1)
        dt = self.sys_dims.dt

        A = jnp.float64(jnp.tile(jnp.array([[1,dt],[-1*dt,1-0.5*dt]]), self.dims["TXX"]))
        B = jnp.tile(jnp.array([[0,0],[1,0]]), self.dims["TXX"])*dt
        a = jnp.zeros(self.dims["TNX"])
        Qf = 0. *jnp.eye(self.dims["N"][0])
        qf = 0. * jnp.ones(self.dims["NX"])
        Q = 2. * jnp.tile(jnp.eye(self.dims["N"][0]), self.dims["TXX"])
        q = 0. * jnp.tile(jnp.ones(self.dims["NX"]), self.dims["TXX"])
        R = 0.5 * jnp.tile(jnp.eye(self.dims["M"][0]), self.dims["TXX"])
        r = 0. * jnp.tile(jnp.ones(self.dims["MX"]), self.dims["TXX"])
        S = 0. * jnp.tile(jnp.ones(self.dims["NM"]), self.dims["TXX"])
        self.lqr = LQR(A, B, a, Q, q, Qf, qf, R, r, S)()

        print("\nMake initial state x0 and input U")
        self.x0 = jnp.array([[0.3], [0.]])
        Us = jnp.zeros(self.dims["TMX"]) * 1.0
        Us = Us.at[2].set(1.0)
        self.Us = Us
        self.params = Params(self.x0, self.lqr)
        
    def test_lqr_solution(self):
        """test LQR solution using jaxopt conjugate gradient solution"""
        # Exercise the LQR solver function
        K_dir, Xs_dir, Us_dir, Lambs_dir = solve_lqr(params=self.params)
        Xs_quad, Us_quad = quad_solve(self.params, self.dims["N"][0], self.dims["M"][0], self.dims["T"][0], self.x0) 
        Xs_exact, Us_exact = exact_solve(self.params, self.dims["N"][0], self.dims["M"][0], self.dims["T"][0], self.x0)   
        onp.savetxt("Us_ext", Us_exact[...,0]) 
        onp.savetxt("Xs_ext", Xs_exact[...,0]) 
        onp.savetxt("Us_quad", Us_quad[...,0]) 
        onp.savetxt("Xs_quad", Xs_quad[...,0]) 
        onp.savetxt("Us_dir", Us_dir[...,0]) 
        onp.savetxt("Xs_dir", Xs_dir[...,0])
        # Verify that the two solutions are close
        assert jnp.allclose(Us_dir[:-1], Us_exact, rtol=1e-05, atol=1e-08)
        
        
if __name__ == "__main__":
    unittest.main()