import unittest
import pytest
import chex
from jax import Array

from ..src import Gains, LQR, simulate_trajectory, lqr_adjoint_pass, lin_dyn_step, lqr_forward_pass, lqr_tracking_forward_pass, lqr_backward_pass, solve_lqr
from .fixtures import sys_matrices, sys_dims



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
        self.LQR = sys_matrices()
        self.dims = chex.Dimensions(T=20, N=3, M=2, X=1)
        

    def tearDown(self):
        """Destruct test class
        """
        print("Running tearDown method...")


    def test_lqr_struct(self):
        """Test test shape of LQR
        """
        print("Running test_lqr_struct")
        
        # test cost is positive symmetric
        chex.assert_trees_all_close(self.lqr.Q[0], self.lqr.Q[0].T)
        chex.assert_trees_all_close(self.lqr.R[0], self.lqr.R[0].T)
        
        # check shape
        chex.assert_shape(self.lqr.Q, self.dims['TNN'])
        chex.assert_shape(self.lqr.R, self.dims['TMM'])
        chex.assert_shape(self.lqr.S, self.dims['TNM'])
        
        
    def test_simulate_trajectory(self):
        pass
    
    
    def test_lqr_adjoint_pass(self):
        pass
        
    def test_lqr_forward_pass(self):
        pass
        
        
    def test_solve_lqr(self):
        pass
        
        
class TestLQRSolution(unittest.TestCase):
    """Test LQR solution using jaxopt conjugate gradient solution"""
    pass

if __name__ == "__main__":
    unittest.main()