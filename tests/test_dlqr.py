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

def setup_lqr_time_varying(dims: chex.Dimensions,
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

def setup_lqr(dims: chex.Dimensions,
    pen_weight: dict = {"Q": 1e-0, "R": 1e-3, "Qf": 1e0, "S": 1e-3}) -> LQR:
    dt = 0.1
    A = jnp.float64(jnp.tile(jnp.array([[1-0.5*dt,0],[0,1-0.5*dt]]), dims["TXX"]))
    B = jnp.tile(jnp.array([[1.,0],[0,1.]]), dims["TXX"])#*dt
    a = jnp.zeros(dims["TNX"])
    Qf = 0. *jnp.eye(dims["N"][0])
    qf = 0.   * jnp.ones(dims["NX"])
    Q = 1. * jnp.tile(jnp.eye(dims["N"][0]), dims["TXX"])
    q = 0. * jnp.tile(jnp.ones(dims["NX"]), dims["TXX"])
    R = 1. * jnp.tile(jnp.eye(dims["M"][0]), dims["TXX"])
    r = 0. * jnp.tile(jnp.ones(dims["MX"]), dims["TXX"])
    S = 0. * jnp.tile(jnp.ones(dims["NM"]), dims["TXX"])
    return LQR(A, B, a, Q, q, Qf, qf, R, r, S)()

class TestLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        """Instantiate dummy LQR"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=60, N=2, M=2, X=1)
        self.sys_dims = ModelDims(self.dims["N"][0], self.dims["M"][0], self.dims["T"][0], dt=0.1)
        print("Model dimensionality", self.dims["TNMX"])
        print("\nMake LQR struct")
        self.lqr = setup_lqr(self.dims)

        print("\nMake initial state x0 and input U")
        self.x0 = jnp.array([[2.0], [1.0]])
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
        chex.assert_shape(g.lqr.B, self.params.lqr.B.shape)
        chex.assert_shape(g.lqr.A, self.params.lqr.A.shape)
        chex.assert_shape(g.lqr.Q, self.params.lqr.Q.shape)
        chex.assert_shape(g.lqr.q, self.params.lqr.q.shape)
    #print(Us_lqr.shape)
        

        
    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")
      
      
from jaxopt import implicit_diff, linear_solve
from typing import NamedTuple, Tuple
  
  
class State(NamedTuple):
    Xs : Array
    Us: Array
    Lambs : Array
 
 
class Prms(NamedTuple):
    q : Array
    r : Array
    Q: Array
    R: Array

def state_kkt(Xs: jnp.ndarray, Us: jnp.ndarray, Lambs: jnp.ndarray, params: Params):
    Xs, Us, Lambs = Xs
    dLdXs, dLdUs, dLdLambs = kkt(params, Xs, Us, Lambs)
    return dLdXs, dLdUs, dLdLambs #State(Xs=dLdXs, Us=dLdUs, Lambs=dLdLambs)
    
class TestDLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        """Instantiate dummy LQR"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=15, N=2, M=2, X=1)
        self.sys_dims = ModelDims(self.dims["N"][0], self.dims["M"][0], self.dims["T"][0], dt=0.1)
        print("Model dimensionality", self.dims["TNMX"])
        print("\nMake LQR struct")
        self.lqr = setup_lqr(self.dims)

        print("\nMake initial state x0 and input U")
        self.x0 = jnp.array([[100.0], [100.0]])
        Us = jnp.zeros(self.dims["TMX"], dtype=float)
        Us = Us.at[2].set(1.0)
        self.Us = Us
        self.params = Params(self.x0, self.lqr)
        # Verify that the average KKT conditions are satisfied
        
    
    def test_dlqr(self):
        def replace_params(p):
            lqr = LQR(A = self.params.lqr.A, B = self.params.lqr.B, a = self.params.lqr.a, Q = p.Q, q = p.q, Qf = self.params.lqr.Qf, qf = self.params.lqr.qf, R = p.R, r = p.r, S = self.params.lqr.S)
            return Params(self.params.x0, lqr)
        
        def loss(prms):
            tau_lqr = dlqr(self.sys_dims, replace_params(prms), self.x0)
            Us_lqr = tau_lqr[:,self.dims.N:]
            Xs_lqr = tau_lqr[:,:self.dims.N]
            return jnp.linalg.norm(Us_lqr)**2 + jnp.linalg.norm(Xs_lqr)**2
        #Us_lqr is Dims here, weird
        
    
        @implicit_diff.custom_root(state_kkt, solve=linear_solve.solve_cg)
        def implicit_solve_lqr(Xs, Us, Lambs, params):
            gains, Xs, Us, Lambs = solve_lqr(params, self.sys_dims)
            return Xs, Us, Lambs #State(Xs=Xs,Us=Us,Lambs=Lambs)   
        
        def implicit_loss(prms):
            Xs_lqr, Us_lqr, _Us_lqr = implicit_solve_lqr(None, None, None, replace_params(prms))
            return jnp.linalg.norm(Us_lqr)**2 + jnp.linalg.norm(Xs_lqr)**2
        
        def direct_loss(prms):
            gains, Xs, Us, Lambs = solve_lqr(replace_params(prms), self.sys_dims)
            return jnp.linalg.norm(Us)**2 + jnp.linalg.norm(Xs)**2

        prms = Prms(R = self.params.lqr.R, Q = self.params.lqr.Q, q = 10*jnp.ones(self.dims["TNX"]), r = 1*jnp.ones(self.dims["TNX"]))
        lqr_val, lqr_g = jax.value_and_grad(loss)(prms)
        implicit_val, implicit_g = jax.value_and_grad(implicit_loss)(prms)
        direct_val, direct_g = jax.value_and_grad(direct_loss)(prms)
        print("\n || cat || \n ")
        print(implicit_g.q[:4])
        print(direct_g.q[:4])
        print(lqr_g.q[:4])
        print(direct_g.Q[:4])
        print(lqr_g.Q[:4])
        assert jnp.allclose(lqr_g.q, direct_g.q)
        assert jnp.allclose(implicit_g.q, direct_g.q, rtol=1e-03, atol=1e-01)
        assert jnp.allclose(lqr_g.r[:-1], direct_g.r)
        assert jnp.allclose(implicit_g.r, direct_g.r, rtol=1e-03, atol=1e-01)
        assert jnp.allclose(lqr_g.Q, direct_g.Q)
        assert jnp.allclose(implicit_g.Q, direct_g.Q, rtol=1e-03, atol=1e-01)
        assert jnp.allclose(lqr_g.R[:-1], direct_g.R)
        #assert jnp.allclose(lqr_g.A, direct_g.A)
        

        
    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")
        
        
             
if __name__ == "__main__":
    unittest.main()