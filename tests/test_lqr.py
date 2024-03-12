import unittest
import pytest
import chex
from jax import Array
import jax.random as jr
import jax.numpy as jnp

from tests.fixtures import sys_matrices, sys_dims
from src.lqr import Gains, LQR, Params, simulate_trajectory, lqr_adjoint_pass, lin_dyn_step, lqr_forward_pass, lqr_tracking_forward_pass, lqr_backward_pass, solve_lqr



# chex.assert_type(...,  Array)
# chex.assert_trees_all_close(lqr.Q[0], lqr.Q[0].T)
# dims = chex.Dimensions(T=20, N=3, M=2, X=1)
# chex.assert_shape(x, dims['TNX'])


class TestLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""
    def setUp(self):
        """Instantiate dummy LQR
        """
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=20, N=3, M=2, X=1)
        print(self.dims['TNMX'])
        
        print("\nMake LQR struct")
        key = jr.PRNGKey(seed=234)
        subkeys = jr.split(key,11)
    
        A = jr.normal(subkeys[0], self.dims['TNN'])
        B = jr.normal(subkeys[1], self.dims['TNM'])
        a = jr.normal(subkeys[2], self.dims['TN'])
        Qf = jnp.eye(self.dims['N'][0])
        qf = 0.5 * jnp.ones(self.dims['N'][0])
        Q = jnp.tile(jnp.eye(self.dims['N'][0]), self.dims['TXX'])
        q = 0.5 * jnp.tile(jnp.ones(self.dims['N']), self.dims['TX'])
        R = jnp.tile(jnp.eye(self.dims['M'][0]), self.dims['TXX'])
        r = 0.5 * jnp.tile(jnp.ones(self.dims['M']), self.dims['TX'])
        S = 0.5 * jnp.tile(jnp.ones(self.dims['NM']), self.dims['TXX'])
        
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
        assert(isinstance(params.lqr, LQR))
        assert(isinstance(params[2], LQR))
        print("Params struct passed.")
        

    def test_lqr_struct(self):
        """Test test shape of LQR
        """
        print("Running test_lqr_struct")
        
        # test cost is positive symmetric
        chex.assert_trees_all_close(self.lqr.Q[0], self.lqr.Q[0].T)
        chex.assert_trees_all_close(self.lqr.R[0], self.lqr.R[0].T)
        
        # check shape
        chex.assert_shape(self.lqr.A, self.dims['TNN'])
        chex.assert_shape(self.lqr.B, self.dims['TNM'])
        chex.assert_shape(self.lqr.Q, self.dims['TNN'])
        chex.assert_shape(self.lqr.R, self.dims['TMM'])
        chex.assert_shape(self.lqr.S, self.dims['TNM'])
        
        # check dtypes
        chex.assert_type(self.lqr.S.dtype,  float)
        chex.assert_type(self.lqr.Qf.dtype,  float)
        
        
    def test_simulate_trajectory(self):
        print("Running test_simulate_trajectory")
        params = Params(self.x0, self.dims["T"][0], self.lqr)
        Xs = simulate_trajectory(lin_dyn_step, self.Us, params)
        print(Xs.shape)
        chex.assert_type(Xs,  float)
        chex.assert_shape(Xs,  (self.dims["T"][0]+1,) + self.dims["NX"])
        
        pass
    
    
    def test_lqr_adjoint_pass(self):
        pass
        
    def test_lqr_forward_pass(self):
        pass
        
        
    def test_solve_lqr(self):
        pass
        

    def tearDown(self):
        """Destruct test class
        """
        print("Running tearDown method...")
class TestLQRSolution(unittest.TestCase):
    """Test LQR solution using jaxopt conjugate gradient solution"""
    pass

if __name__ == "__main__":
    unittest.main()