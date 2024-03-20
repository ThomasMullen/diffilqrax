import unittest
import pytest
import chex
from jax import Array
import jax.random as jr
import jax.numpy as jnp
from jaxopt import linear_solve, implicit_diff

# from tests.fixtures import sys_matrices, sys_dims
from src.lqr import (
    Gains,
    CostToGo,
    LQR,
    Params,
    simulate_trajectory,
    lqr_adjoint_pass,
    lin_dyn_step,
    lqr_forward_pass,
    lqr_tracking_forward_pass,
    lqr_backward_pass,
    solve_lqr,
    kkt,
)


# chex.assert_type(...,  Array)
# chex.assert_trees_all_close(lqr.Q[0], lqr.Q[0].T)
# dims = chex.Dimensions(T=20, N=3, M=2, X=1)
# chex.assert_shape(x, dims['TNX'])


class TestLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        """Instantiate dummy LQR"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=20, N=3, M=2, X=1)
        print(self.dims["TNMX"])

        print("\nMake LQR struct")
        key = jr.PRNGKey(seed=234)
        subkeys = jr.split(key, 11)

        A = jr.normal(subkeys[0], self.dims["TNN"])
        B = jr.normal(subkeys[1], self.dims["TNM"])
        a = jr.normal(subkeys[2], self.dims["TNX"])
        Qf = jnp.eye(self.dims["N"][0])
        qf = 0.5 * jnp.ones(self.dims["NX"])
        Q = jnp.tile(jnp.eye(self.dims["N"][0]), self.dims["TXX"])
        q = 0.5 * jnp.tile(jnp.ones(self.dims["NX"]), self.dims["TXX"])
        R = jnp.tile(jnp.eye(self.dims["M"][0]), self.dims["TXX"])
        r = 0.5 * jnp.tile(jnp.ones(self.dims["MX"]), self.dims["TXX"])
        S = 0.5 * jnp.tile(jnp.ones(self.dims["NM"]), self.dims["TXX"])

        self.lqr = LQR(A, B, a, Q, q, Qf, qf, R, r, S)()
        print("LQR Q shape", self.lqr.Q.shape)

        print("\nMake initial state x0 and input U")
        self.x0 = jnp.array([[2.0], [1.0], [1.0]])
        Us = jnp.zeros(self.dims["TMX"]) * 1.0
        Us = Us.at[2].set(1.0)
        self.Us = Us

    def test_lqr_params_struct(self):
        print("Construct params")
        params = Params(self.x0, self.dims["T"][0], self.lqr)
        chex.assert_equal(params.horizon, self.dims["T"][0])
        chex.assert_equal(params[1], self.dims["T"][0])
        chex.assert_type(params.horizon, int)
        chex.assert_trees_all_close(params[0], self.x0)
        chex.assert_trees_all_close(params.x0, self.x0)
        chex.assert_type(params.x0, float)
        assert isinstance(params.lqr, LQR)
        assert isinstance(params[2], LQR)
        print("Params struct passed.")

    def test_lqr_struct(self):
        """Test test shape of LQR"""
        print("Running test_lqr_struct")

        # test cost is positive symmetric
        chex.assert_trees_all_close(self.lqr.Q[0], self.lqr.Q[0].T)
        chex.assert_trees_all_close(self.lqr.R[0], self.lqr.R[0].T)

        # check shape
        chex.assert_shape(self.lqr.A, self.dims["TNN"])
        chex.assert_shape(self.lqr.B, self.dims["TNM"])
        chex.assert_shape(self.lqr.Q, self.dims["TNN"])
        chex.assert_shape(self.lqr.R, self.dims["TMM"])
        chex.assert_shape(self.lqr.S, self.dims["TNM"])

        # check dtypes
        chex.assert_type(self.lqr.S.dtype, float)
        chex.assert_type(self.lqr.Qf.dtype, float)

    def test_simulate_trajectory(self):
        print("Running test_simulate_trajectory")
        params = Params(self.x0, self.dims["T"][0], self.lqr)
        Xs = simulate_trajectory(lin_dyn_step, self.Us, params)
        print(Xs.shape)
        chex.assert_type(Xs, float)
        chex.assert_shape(Xs, (self.dims["T"][0] + 1,) + self.dims["NX"])

    def test_lqr_adjoint_pass(self):
        print("Running test_lqr_adjoint_pass")
        params = Params(self.x0, self.dims["T"][0], self.lqr)
        Xs_sim = simulate_trajectory(lin_dyn_step, self.Us, params)
        Lambs = lqr_adjoint_pass(Xs_sim, self.Us, params)
        chex.assert_type(Lambs, float)
        chex.assert_shape(Lambs, (self.dims["T"][0] + 1,) + self.dims["NX"])

    def test_lqr_backward_pass(self):
        params = Params(self.x0, self.dims["T"][0], self.lqr)
        (dJ, Ks), exp_dJ = lqr_backward_pass(
            lqr=params.lqr, T=params.horizon, expected_change=True, verbose=False
        )
        chex.assert_type(Ks.K, float)
        chex.assert_shape(Ks.K, self.dims["TMN"])
        chex.assert_type(Ks.k, float)
        chex.assert_shape(Ks.k, self.dims["TMX"])
        chex.assert_type(dJ, float)
        chex.assert_type(exp_dJ, float)

    def test_lqr_forward_pass(self):
        params = Params(self.x0, self.dims["T"][0], self.lqr)
        (dJ, Ks), exp_dJ = lqr_backward_pass(
            lqr=params.lqr, T=params.horizon, expected_change=True, verbose=False
        )
        Xs_lqr, Us_lqr = lqr_forward_pass(gains=Ks, params=params)
        chex.assert_type(Xs_lqr, float)
        chex.assert_shape(Xs_lqr, (self.dims["T"][0] + 1,) + self.dims["NX"])
        chex.assert_type(Us_lqr, float)
        chex.assert_shape(Us_lqr, self.dims["TMX"])

    def test_solve_lqr(self):
        params = Params(self.x0, self.dims["T"][0], self.lqr)
        gains_lqr, Xs_lqr, Us_lqr, Lambs_lqr = solve_lqr(params)
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

        pass

    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")


class TestLQRSolution(unittest.TestCase):
    """Test LQR solution using jaxopt conjugate gradient solution"""

    def setUp(self):
        """Instantiate dummy LQR"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=20, N=3, M=2, X=1)
        print(self.dims["TNMX"])

        print("\nMake LQR struct")
        key = jr.PRNGKey(seed=234)
        subkeys = jr.split(key, 11)

        A = jr.normal(subkeys[0], self.dims["TNN"])
        B = jr.normal(subkeys[1], self.dims["TNM"])
        a = jr.normal(subkeys[2], self.dims["TNX"])
        Qf = jnp.eye(self.dims["N"][0])
        qf = 0.5 * jnp.ones(self.dims["NX"])
        Q = jnp.tile(jnp.eye(self.dims["N"][0]), self.dims["TXX"])
        q = 0.5 * jnp.tile(jnp.ones(self.dims["NX"]), self.dims["TXX"])
        R = jnp.tile(jnp.eye(self.dims["M"][0]), self.dims["TXX"])
        r = 0.5 * jnp.tile(jnp.ones(self.dims["MX"]), self.dims["TXX"])
        S = 0.5 * jnp.tile(jnp.ones(self.dims["NM"]), self.dims["TXX"])

        self.lqr = LQR(A, B, a, Q, q, Qf, qf, R, r, S)()
        print("LQR Q shape", self.lqr.Q.shape)

        print("\nMake initial state x0 and input U")
        self.x0 = jnp.array([[2.0], [1.0], [1.0]])
        Us = jnp.zeros(self.dims["TMX"]) * 1.0
        Us = Us.at[2].set(1.0)
        self.Us = Us
        self.params = Params(self.x0, self.dims["T"][0], self.lqr)

    def test_lqr_solution(self):
        """test LQR solution using jaxopt conjugate gradient solution"""
        # Exercise the LQR solver function
        K_dir, Xs_dir, Us_dir, Lambs_dir = solve_lqr(params=self.params)
        K_impl, Xs_impl, Us_impl, Lambs_impl = implicit_diff.custom_root(
            kkt, linear_solve.solve_cg
        )(solve_lqr)(self.params)
        # Verify that the two solutions are close
        assert jnp.allclose(Xs_dir, Xs_impl, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(Us_dir, Us_impl, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(Lambs_dir, Lambs_impl, rtol=1e-05, atol=1e-08)


    def test_kkt_optimal(self):
        # Setup the LQR problem
        K_dir, Xs_dir, Us_dir, Lambs_dir = solve_lqr(params=self.params)
        # Exercise the KKT function
        dLdXs, dLdUs, dLdLambs = kkt(self.params, Xs_dir, Us_dir, Lambs_dir)
        # Verify that the KKT conditions are satisfied
        assert jnp.allclose(dLdXs[:-1], 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdXs[-1], 0.0, rtol=1e-05, atol=1e-08), "Terminal X state not satisfied"
        assert jnp.allclose(dLdXs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdUs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdLambs, 0.0, rtol=1e-05, atol=1e-08)
        
    def test_gradients(self):
        # Setup the LQR problem
        
        # Exercise
        
        # Verify
        # chex.assert_numerical_grads
        pass


if __name__ == "__main__":
    unittest.main()
