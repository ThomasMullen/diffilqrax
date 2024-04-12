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
printing_on = True

def setup_lqr_time_varying(dims: chex.Dimensions,
              pen_weight: dict = {"Q": 1e-0, "R": 1e-3, "Qf": 1e0, "S": 1e-3}) -> LQR:
    """Setup LQR problem"""
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    # initialise dynamics
    span_time_m=dims["TXX"]
    span_time_v=dims["TX"]
    A = initialise_stable_dynamics(next(skeys), *dims['NT'],radii=0.6)
    B = jnp.tile(jr.normal(next(skeys), dims['NM']), span_time_m)
    a = jnp.tile(jr.normal(next(skeys), dims['N']), span_time_v)
    # define cost matrices
    Q = pen_weight["Q"] * jnp.tile(jnp.eye(dims['N'][0]), span_time_m)
    q = 2*1e-1 * jnp.tile(jnp.ones(dims['N']), span_time_v)
    R = pen_weight["R"] * jnp.tile(jnp.eye(dims['M'][0]), span_time_m)
    r = 1e-6 * jnp.tile(jnp.ones(dims['M']), span_time_v)
    S = pen_weight["S"] * jnp.tile(jnp.ones(dims['NM']), span_time_m)
    Qf = pen_weight["Q"] * jnp.eye(dims['N'][0])
    qf = 2*1e-1 * jnp.ones(dims['N'])
    # construct LQR
    lqr = LQR(A, B, a, Q, q, Qf, qf, R, r, S)
    return lqr()

def setup_lqr(dims: chex.Dimensions,
    pen_weight: dict = {"Q": 1e-0, "R": 1e-3, "Qf": 1e0, "S": 1e-3}) -> LQR:
    dt = 0.1
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    span_time_m=dims["TXX"]
    span_time_v=dims["TX"]
    A = initialise_stable_dynamics(next(skeys), *dims['NT'],radii=0.6) #jnp.float64(jnp.tile(jnp.array([[1-0.5*dt,0],[0.2*dt,1-0.5*dt]]), dims["TXX"]))
    B = jnp.tile(jr.normal(next(skeys), dims['NM']), span_time_m)
    a = jnp.tile(jr.normal(next(skeys), dims['N']), span_time_v)
    #B = jnp.tile(jnp.array([[1.,0],[0,1.]]), dims["TXX"])#*dt
    #a = jnp.zeros(dims["TNX"])
    Qf = 1. *jnp.eye(dims["N"][0])
    qf = 1.   * jnp.ones(dims["N"])
    Q = 1. * jnp.tile(jnp.eye(dims["N"][0]), span_time_m)
    q = 0. * jnp.tile(jnp.ones(dims["N"]), span_time_v)
    R = 1. * jnp.tile(jnp.eye(dims["M"][0]), span_time_m)
    r = 0. * jnp.tile(jnp.ones(dims["M"]), span_time_v)
    S = 0.0 * jnp.tile(jnp.ones(dims["NM"]), span_time_m)
    return LQR(A, B, a, Q, q, Qf, qf, R, r, S)()

class TestLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        self.dims = chex.Dimensions(T=60, N=2, M=2, X=1)
        self.sys_dims = ModelDims(*self.dims["NMT"], dt=0.1)
        self.lqr = setup_lqr(self.dims)
        self.x0 = jnp.array([2.0, 1.0])
        Us = jnp.zeros(self.dims["TM"], dtype=float)
        Us = Us.at[2].set(1.0)
        self.Us = Us
        self.params = Params(self.x0, self.lqr)
        # Verify that the average KKT conditions are satisfied
        
    def test_dlqr(self):
        @jax.jit
        def loss(p):
            Us_lqr = dlqr(ModelDims(*self.dims["NMT"], dt=0.1), p, jnp.array([2.0, 1.0]))
            return jnp.linalg.norm(p.lqr.A) + jnp.linalg.norm(Us_lqr)
        val, g = jax.value_and_grad(loss)(self.params)
        chex.assert_trees_all_equal_shapes_and_dtypes(g, self.params)
        
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
    A: Array

def state_kkt(Xs: jnp.ndarray, Us: jnp.ndarray, Lambs: jnp.ndarray, params: Params):
    Xs, Us, Lambs = Xs
    dLdXs, dLdUs, dLdLambs = kkt(params, Xs, Us, Lambs)
    return dLdXs, dLdUs, dLdLambs #State(Xs=dLdXs, Us=dLdUs, Lambs=dLdLambs)
    
class TestDLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        """Instantiate dummy LQR"""
        self.dims = chex.Dimensions(T=15, N=2, M=2, X=1)
        self.sys_dims = ModelDims(*self.dims["NMT"], dt=0.1)
        self.lqr = setup_lqr(self.dims)
        self.x0 = jnp.array([100.0, 100.0])
        Us = jnp.zeros(self.dims["TM"], dtype=float)
        Us = Us.at[2].set(1.0)
        self.Us = Us
        self.params = Params(self.x0, self.lqr)
        # Verify that the average KKT conditions are satisfied
        
    
    def test_dlqr(self):
        def replace_params(p):
            lqr = LQR(A = p.A, B = self.params.lqr.B, a = self.params.lqr.a, Q = p.Q, q = p.q, Qf = self.params.lqr.Qf, qf = self.params.lqr.qf, R = p.R, r = p.r, S = self.params.lqr.S)
            return Params(self.params.x0, lqr)
        @jax.jit
        def loss(prms):
            tau_lqr = dlqr(self.sys_dims, replace_params(prms), self.x0)
            Us_lqr = tau_lqr[:,self.dims.N:]
            Xs_lqr = tau_lqr[:,:self.dims.N]
            return jnp.linalg.norm(Us_lqr)**2
        
    
        @implicit_diff.custom_root(state_kkt, solve=linear_solve.solve_cg)
        def implicit_solve_lqr(Xs, Us, Lambs, params):
            gains, Xs, Us, Lambs = solve_lqr(params, self.sys_dims)
            return Xs, Us, Lambs  
        
        def implicit_loss(prms):
            Xs_lqr, Us_lqr, _Us_lqr = implicit_solve_lqr(None, None, None, replace_params(prms))
            return jnp.linalg.norm(Us_lqr)**2
        
        def direct_loss(prms):
            gains, Xs, Us, Lambs = solve_lqr(replace_params(prms), self.sys_dims)
            return jnp.linalg.norm(Us)**2 

        prms = Prms(A = self.params.lqr.A, R = self.params.lqr.R, Q = self.params.lqr.Q, q = 10*jnp.ones(self.dims["TN"]), r = jnp.ones(self.dims["TN"]))
        lqr_val, lqr_g = jax.value_and_grad(loss)(prms)
        implicit_val, implicit_g = jax.value_and_grad(implicit_loss)(prms)
        direct_val, direct_g = jax.value_and_grad(direct_loss)(prms)
        if printing_on : 
            print("\n || Printing grads || \n ")
            print("\n || Printing  q || \n ")
            print(implicit_g.q[:4])
            print(direct_g.q[:4])
            print(lqr_g.q[:4])
            print("\n || Printing  Q || \n ")
            print(direct_g.Q[:4])
            print(lqr_g.Q[:4])
            print(implicit_g.Q[:4])
            print("\n || Printing  A || \n ")
            print(direct_g.A[:4])
            print(lqr_g.A[:4])
        
        # assert jnp.allclose(lqr_g.q, direct_g.q)
        # assert jnp.allclose(lqr_g.r[:-1], direct_g.r)
        # assert jnp.allclose(lqr_g.Q, direct_g.Q)
        # assert jnp.allclose(lqr_g.R[:-1], direct_g.R)
        # assert jnp.allclose(lqr_g.A, direct_g.A)
        
        # verify shapes and dtypes
        chex.assert_trees_all_equal_shapes_and_dtypes(lqr_val, direct_val)
        chex.assert_trees_all_equal_shapes_and_dtypes(lqr_g, direct_g)
        chex.assert_trees_all_equal_shapes_and_dtypes(implicit_val, direct_val)
        chex.assert_trees_all_equal_shapes_and_dtypes(implicit_g, direct_g)
        # verify numerics
        chex.assert_trees_all_close(lqr_val, direct_val)
        chex.assert_trees_all_close(lqr_g, direct_g)
        chex.assert_trees_all_close(implicit_val, direct_val)
        chex.assert_trees_all_close(implicit_g, direct_g)
        
    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")
        
        
             
if __name__ == "__main__":
    unittest.main()