"""
Unit tests for the differentiable LQR solver
"""

from typing import NamedTuple
import unittest
import chex
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
from jaxopt import linear_solve, implicit_diff

from diffilqrax.typs import LQRParams, ModelDims, LQR
from diffilqrax.diff_lqr import dlqr
from diffilqrax.lqr import solve_lqr, kkt
from diffilqrax.utils import keygen, initialise_stable_dynamics

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # double precision

PRINTING_ON = True


def is_jax_array(arr: Array) -> bool:
    """validate jax array type"""
    return isinstance(arr, jnp.ndarray) and not isinstance(arr, onp.ndarray)


def setup_lqr_time_varying(
    dims: chex.Dimensions,
    pen_weight: dict = {"Q": 1e-0, "R": 1e-3, "Qf": 1e0, "S": 1e-3},
) -> LQR:
    """Setup LQR problem"""
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 4)
    # initialise dynamics
    span_time_m = dims["TXX"]
    span_time_v = dims["TX"]
    A = initialise_stable_dynamics(next(skeys), *dims["NT"], radii=0.6)
    B = jnp.tile(jr.normal(next(skeys), dims["NM"]), span_time_m)
    a = jnp.tile(jr.normal(next(skeys), dims["N"]), span_time_v)
    # define cost matrices
    Q = pen_weight["Q"] * jnp.tile(jnp.eye(dims["N"][0]), span_time_m)
    q = 2 * 1e-1 * jnp.tile(jnp.ones(dims["N"]), span_time_v)
    R = pen_weight["R"] * jnp.tile(jnp.eye(dims["M"][0]), span_time_m)
    r = 1e-6 * jnp.tile(jnp.ones(dims["M"]), span_time_v)
    S = pen_weight["S"] * jnp.tile(jr.normal(next(skeys), dims["NM"]), span_time_m)
    Qf = pen_weight["Q"] * jnp.eye(dims["N"][0])
    qf = 2 * 1e-1 * jnp.ones(dims["N"])
    # construct LQR
    lqr = LQR(A, B, a, Q, q, R, r, S, Qf, qf)
    return lqr()


def setup_lqr(
    dims: chex.Dimensions,
    pen_weight: dict = {"Q": 1e-0, "R": 1e-3, "Qf": 1e0, "S": 1e-3},
) -> LQR:
    """set up time invariant LQR problem"""
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    span_time_m = dims["TXX"]
    span_time_v = dims["TX"]
    A = initialise_stable_dynamics(next(skeys), *dims["NT"], radii=0.6)
    B = jnp.tile(jr.normal(next(skeys), dims["NM"]), span_time_m)
    a = 0*jnp.tile(jr.normal(next(skeys), dims["N"]), span_time_v)
    Qf = 1.0 * jnp.eye(dims["N"][0])
    qf = 1.0 * jnp.ones(dims["N"])
    Q = 1.0 * jnp.tile(jnp.eye(dims["N"][0]), span_time_m)
    q = 1.0 * jnp.tile(jnp.ones(dims["N"]), span_time_v)
    R = 1.0 * jnp.tile(jnp.eye(dims["M"][0]), span_time_m)
    r = 1.0 * jnp.tile(jnp.ones(dims["M"]), span_time_v)
    S = 0.5 * jnp.tile(jnp.ones(dims["NM"]), span_time_m)
    return LQR(A, B, a, Q, q, R, r, S, Qf, qf)()


class TestLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        """set up LQR problem"""
        self.dims = chex.Dimensions(T=60, N=2, M=2, X=1)
        self.sys_dims = ModelDims(*self.dims["NMT"], dt=0.1)
        self.lqr = setup_lqr(self.dims)
        self.x0 = jnp.array([2.0, 1.0])
        Us = jnp.zeros(self.dims["TM"], dtype=float)
        Us = Us.at[2].set(1.0)
        self.Us = Us
        self.params = LQRParams(self.x0, self.lqr)
        # Verify that the average KKT conditions are satisfied

    def test_dlqr(self):
        """test dlqr struct and shapes"""
        @jax.jit
        def loss(p):
            Us_lqr = dlqr(
                ModelDims(*self.dims["NMT"], dt=0.1), p, jnp.array([2.0, 1.0])
            )
            return jnp.linalg.norm(p.lqr.A) + jnp.linalg.norm(Us_lqr)

        _, g = jax.value_and_grad(loss)(self.params)
        chex.assert_trees_all_equal_shapes_and_dtypes(g, self.params)

    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")


class State(NamedTuple):
    """helper class for state"""
    Xs: Array
    Us: Array
    Lambs: Array


class Prms(NamedTuple):
    """helper class for params"""
    q: Array
    r: Array
    Q: Array
    R: Array
    A: Array
    S: Array
    x0: Array
    a: Array


def state_kkt(Xs: jnp.ndarray, Us: jnp.ndarray, Lambs: jnp.ndarray, params: LQRParams):
    """calculate KKT conditions for LQR problem"""
    Xs, Us, Lambs = Xs
    dLdXs, dLdUs, dLdLambs = kkt(params, Xs, Us, Lambs)
    return dLdXs, dLdUs, dLdLambs  # State(Xs=dLdXs, Us=dLdUs, Lambs=dLdLambs)


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
        self.params = LQRParams(self.x0, self.lqr)
        # Verify that the average KKT conditions are satisfied

    def test_dlqr(self):
        """
        test dlqr struct, shapes and solutions compared to direct gradients,
        and implicit gradients
        """
        def replace_params(p:Prms)->LQRParams:
            lqr = LQR(
                A=p.A,
                B=self.params.lqr.B,
                a=p.a,
                Q=p.Q,
                q=p.q,
                R=p.R,
                r=p.r,
                S=p.S,
                Qf=self.params.lqr.Qf,
                qf=self.params.lqr.qf,
            )
            return LQRParams(p.x0, lqr)

        @jax.jit
        def loss(prms):
            tau_lqr = dlqr(self.sys_dims, replace_params(prms), self.x0)
            Us_lqr = tau_lqr[:, self.sys_dims.n :]
            Xs_lqr = tau_lqr[:, : self.sys_dims.n]
            return jnp.linalg.norm(Us_lqr) ** 2 + jnp.linalg.norm(Xs_lqr - 1) ** 2

        @implicit_diff.custom_root(state_kkt, solve=linear_solve.solve_cg)
        def implicit_solve_lqr(Xs, Us, Lambs, params):
            Xs, Us, Lambs = solve_lqr(params)
            return Xs, Us, Lambs

        def implicit_loss(prms):
            Xs_lqr, Us_lqr, _Us_lqr = implicit_solve_lqr(
                None, None, None, replace_params(prms)
            )
            print(Xs_lqr.shape)
            return jnp.linalg.norm(Us_lqr) ** 2 + jnp.linalg.norm(Xs_lqr - 1) ** 2

        def direct_loss(prms):
            Xs, Us, Lambs = solve_lqr(replace_params(prms))
            return jnp.linalg.norm(Us) ** 2 + jnp.linalg.norm(Xs - 1) ** 2

        prms = Prms(
            a=self.params.lqr.a,
            x0=jnp.array([100.0, 100.0]),
            S=self.params.lqr.S,
            A=self.params.lqr.A,
            R=self.params.lqr.R,
            Q=self.params.lqr.Q,
            q=self.params.lqr.q,
            r=self.params.lqr.r,
        )  # 10*jnp.ones(self.dims["TNX"]), r = jnp.ones(self.dims["TNX"]))
        lqr_val, lqr_g = jax.value_and_grad(loss)(prms)
        implicit_val, implicit_g = jax.value_and_grad(implicit_loss)(prms)
        direct_val, direct_g = jax.value_and_grad(direct_loss)(prms)
        if PRINTING_ON:
            print("\n || Printing grads S || \n ")
            print("\n || Implicit || \n ")
            print(lqr_g.S[:4])
            print("\n || Direct || \n ")
            print(direct_g.S[:4])
            print("\n || Printing grads a || \n ")
            print("\n || Implicit || \n ")
            print(lqr_g.a[:4])
            print("\n || Direct || \n ")
            print(direct_g.a[:4])
            print("\n || Printing  Q || \n ")
            print(direct_g.Q[:4])
            print(lqr_g.Q[:4])
            print(implicit_g.Q[:4])
            print("\n || Printing  a || \n ")
            print(direct_g.a)
            print(lqr_g.a)
            print("\n || Printing  x0 || \n ")
            print(direct_g.x0)
            print(lqr_g.x0)
        assert jnp.allclose(lqr_g.a, direct_g.a)
        assert jnp.allclose(lqr_g.q, direct_g.q)
        assert jnp.allclose(lqr_g.r, direct_g.r)
        assert jnp.allclose(lqr_g.Q, direct_g.Q)
        assert jnp.allclose(lqr_g.R, direct_g.R)
        assert jnp.allclose(lqr_g.A, direct_g.A)
        # verify shapes and dtypes
        chex.assert_trees_all_equal_shapes_and_dtypes(lqr_val, direct_val)
        chex.assert_trees_all_equal_shapes_and_dtypes(lqr_g, direct_g)
        chex.assert_trees_all_equal_shapes_and_dtypes(implicit_val, direct_val)
        chex.assert_trees_all_equal_shapes_and_dtypes(implicit_g, direct_g)
        # verify numerics
        chex.assert_trees_all_close(lqr_val, direct_val, rtol=1e-3)
        chex.assert_trees_all_close(lqr_g, direct_g, rtol=1e-3)
        chex.assert_trees_all_close(implicit_val, direct_val, rtol = 1e-3)

    # chex.assert_trees_all_close(implicit_g, direct_g, rtol = 1e-3)

    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")


if __name__ == "__main__":
    unittest.main()
