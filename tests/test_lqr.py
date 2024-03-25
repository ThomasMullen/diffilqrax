import unittest
import pytest
import chex, jax 
from jax import Array, grad
from typing import Tuple
import jax.random as jr
import jax.numpy as jnp
from jaxopt import linear_solve, implicit_diff
import numpy as np
from src.exact import quad_solve

#jax.config.update('jax_default_device', jax.devices('cpu')[0])
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


def keygen(key, nkeys):
    """Generate randomness that JAX can use by splitting the JAX keys.

    Args:
    key : the random.PRNGKey for JAX
    nkeys : how many keys in key generator

    Returns:
    2-tuple (new key for further generators, key generator)
    """
    keys = jr.split(key, nkeys+1)
    return keys[0], (k for k in keys[1:])


def initialise_stable_dynamics(key:Tuple[int,int], n_dim:int, T:int, radii:float=0.6)->Array:
    """Generate a state matrix with stable dynamics (eigenvalues < 1)

    Args:
        key (Tuple[int,int]): random key
        n_dim (int): state dimensions
        radii (float, optional): spectral radius. Defaults to 0.6.

    Returns:
        Array: matrix A with stable dynamics.
    """
    mat = jr.normal(key,(n_dim,n_dim))*radii
    mat /= jnp.sqrt(n_dim)
    mat -= jnp.eye(n_dim)
    return jnp.tile(mat,(T,1,1))


def initialise_stable_time_varying_dynamics(key:Tuple[int,int], n_dim:int, T:int, radii:float=0.6)->Array:
    """Generate a state matrix with stable dynamics (eigenvalues < 1)

    Args:
        key (Tuple[int,int]): random key
        n_dim (int): state dimensions
        radii (float, optional): spectral radius. Defaults to 0.6.

    Returns:
        Array: matrix A with stable dynamics.
    """
    mat = jr.normal(key,(T,n_dim,n_dim))*radii
    mat /= jnp.sqrt(n_dim)
    mat -= jnp.eye(n_dim)
    return mat


class TestLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        """Instantiate dummy LQR"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=20, N=3, M=2, X=1)
        print(self.dims["TNMX"])

        print("\nMake LQR struct")
        key = jr.PRNGKey(seed=234)
        key, skeys = keygen(key, 3)

        # A = jr.normal(next(skeys), self.dims["TNN"])
        A = initialise_stable_dynamics(next(skeys), self.dims["N"][0], self.dims["T"][0],radii=0.6)
        B = jnp.tile(jr.normal(next(skeys), self.dims["NM"]), self.dims["TXX"])
        # a = jr.normal(next(skeys), self.dims["TNX"])
        a = jnp.tile(jr.normal(next(skeys), self.dims["NX"]), self.dims["TXX"])
        Qf = jnp.eye(self.dims["N"][0])
        qf = 2*1e-1 * jnp.ones(self.dims["NX"])
        Q = jnp.tile(jnp.eye(self.dims["N"][0]), self.dims["TXX"])
        q = 2*1e-1 * jnp.tile(jnp.ones(self.dims["NX"]), self.dims["TXX"])
        R = 1e-4 * jnp.tile(jnp.eye(self.dims["M"][0]), self.dims["TXX"])
        r = 1e-2 * jnp.tile(jnp.ones(self.dims["MX"]), self.dims["TXX"])
        S = 1e-4 * jnp.tile(jnp.ones(self.dims["NM"]), self.dims["TXX"])

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
        key, skeys = keygen(key, 3)

        # A = jr.normal(next(skeys), self.dims["TNN"])
        A = initialise_stable_dynamics(next(skeys), self.dims["N"][0], self.dims["T"][0],radii=0.6)
        B = jnp.tile(jr.normal(next(skeys), self.dims["NM"]), self.dims["TXX"])
        # a = jr.normal(next(skeys), self.dims["TNX"])
        a = jnp.tile(jr.normal(next(skeys), self.dims["NX"]), self.dims["TXX"])
        Qf = jnp.eye(self.dims["N"][0])
        qf = 2*1e-1 * jnp.ones(self.dims["NX"])
        Q = jnp.tile(jnp.eye(self.dims["N"][0]), self.dims["TXX"])
        q = 2*1e-1 * jnp.tile(jnp.ones(self.dims["NX"]), self.dims["TXX"])
        R = 1e-4 * jnp.tile(jnp.eye(self.dims["M"][0]), self.dims["TXX"])
        r = 1e-2 * jnp.tile(jnp.ones(self.dims["MX"]), self.dims["TXX"])
        S = 1e-4 * jnp.tile(jnp.ones(self.dims["NM"]), self.dims["TXX"])


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
        assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-01, atol=1e-01)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-01, atol=1e-01)
        #TODO: Inspect why U not near 0 and how to set threhold
        # assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-01, atol=1e-01) 
        # assert jnp.allclose(dLdLambs, 0.0, rtol=1e-05, atol=1e-08)
        # assert jnp.allclose(dLdXs, 0.0, rtol=1e-05, atol=1e-08)
        # assert jnp.allclose(dLdXs[-1], 0.0, rtol=1e-05, atol=1e-08), "Terminal X state not satisfied"
        # assert jnp.allclose(dLdUs, 0.0, rtol=1e-05, atol=1e-08)
        
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
        """Instantiate dummy LQR"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=10, N=3, M=2, X=1)
        print(self.dims["TNMX"])

        print("\nMake LQR struct")
        key = jr.PRNGKey(seed=234)
        subkeys = jr.split(key, 11)

        A = jnp.tile(((jr.normal(subkeys[0], self.dims["NN"]) / np.sqrt(self.dims["N"])*0.8) - jnp.eye(self.dims["N"][0])), self.dims["TXX"])
        B = jnp.tile(jnp.ones(self.dims["NM"]), self.dims["TXX"])
        a = jnp.zeros(self.dims["TNX"])
        Qf = jnp.eye(self.dims["N"][0])
        qf = 0.5 * jnp.ones(self.dims["NX"])
        Q = 0.5 * jnp.tile(jnp.eye(self.dims["N"][0]), self.dims["TXX"])
        q = 0.5 * jnp.tile(jnp.ones(self.dims["NX"]), self.dims["TXX"])
        R = jnp.tile(jnp.eye(self.dims["M"][0]), self.dims["TXX"])
        r = 0.5 * jnp.tile(jnp.ones(self.dims["MX"]), self.dims["TXX"])
        S = 0. * jnp.tile(jnp.ones(self.dims["NM"]), self.dims["TXX"])

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
        Xs_exact, Us_exact = quad_solve(self.params, self.dims["N"][0], self.dims["M"][0], self.dims["T"][0], self.x0)   
        #np.savetxt("Us_ext", Us_exact[...,0]) 
        #np.savetxt("Xs_ext", Xs_exact[...,0]) 
        #np.savetxt("Us_dir", Us_dir[...,0]) 
        #np.savetxt("Xs_dir", Xs_dir[...,0])
        # Verify that the two solutions are close
        assert jnp.allclose(Us_dir, Us_exact, rtol=1e-05, atol=1e-08)
        
        
if __name__ == "__main__":
    unittest.main()