import unittest
import pytest
import chex, jax 
from jax import Array, grad
from typing import Tuple
import jax.random as jr
import jax.numpy as jnp
from jaxopt import linear_solve, implicit_diff
from matplotlib.pyplot import subplots, close
from os import getcwd
from pathlib import Path

import numpy as onp
from src.exact import quad_solve, exact_solve
from src.diff_lqr import dlqr
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
        self.params = Params(self.x0, self.lqr)
        # Verify that the average KKT conditions are satisfied
        
    def test_dlqr(self):
        def loss(params):
            Us_lqr = dlqr(self.sys_dims, params, self.x0)
            return jnp.linalg.norm(params.lqr.A) + jnp.linalg.norm(Us_lqr)
        #Us_lqr is Dims here, weird
        val, g = jax.value_and_grad(loss)(self.params)
        chex.assert_shape(g.lqr.B.shape, self.params.lqr.B.shape)
        chex.assert_shape(g.lqr.A.shape, self.params.lqr.A.shape)
        chex.assert_shape(g.lqr.Q.shape, self.params.lqr.Q.shape)
        chex.assert_shape(g.lqr.q.shape, self.params.lqr.q.shape)
        chex.assert_shape(g.lqr.r.shape, self.params.lqr.r.shape)
        chex.assert_shape(g.lqr.a.shape, self.params.lqr.a.shape)
    #print(Us_lqr.shape)
        

        
    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")
        
        
if __name__ == "__main__":
    unittest.main()