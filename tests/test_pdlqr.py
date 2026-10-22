"""
Gradients of the differentiable parallel LQR solver (pdlqr, pdllqr), for both adjoint solvers, against the
sequential dlqr/dllqr and direct differentiation through solve_lqr.
"""

import os
import pytest
import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.test_util import check_grads

from diffilqrax.typs import ModelDims
from diffilqrax.lqr import solve_lqr
from diffilqrax.diff_lqr import dlqr, dllqr
from diffilqrax.parallel_dlqr import pdlqr, pdllqr
from problems import random_lqr
from test_dlqr import GRAD_DIMS, sym_lqr_grads

jax.config.update("jax_default_device", jax.devices(os.environ.get("DIFFILQRAX_TEST_PLATFORM", "cpu"))[0])
jax.config.update("jax_enable_x64", True)

ADJOINTS = ["parallel", "sequential"]


def setup(n: int, m: int, horizon: int = 30):
    """Random time-varying LQR with cross term, and a smooth loss of the state-input trajectory"""
    params = random_lqr(n, m, horizon, seed=1)
    dims = ModelDims(n, m, horizon, dt=0.1)
    weights = jr.normal(jr.PRNGKey(5), (horizon + 1, n + m))
    return dims, params, lambda tau: jnp.sum(weights * jnp.sin(tau))


def test_invalid_adjoint():
    dims, params, _ = setup(3, 2)
    with pytest.raises(AssertionError):
        pdlqr(dims, params, adjoint="banana")


@pytest.mark.parametrize("adjoint", ADJOINTS)
@pytest.mark.parametrize("n,m", GRAD_DIMS)
def test_pdlqr_matches_dlqr(n, m, adjoint):
    """Value and gradients wrt every LQR field and x0 equal the sequential dlqr and direct autodiff"""
    dims, params, loss = setup(n, m)
    v_par, g_par = jax.value_and_grad(lambda p: loss(pdlqr(dims, p, adjoint=adjoint)))(params)
    v_seq, g_seq = jax.value_and_grad(lambda p: loss(dlqr(dims, p, None)))(params)

    def direct(p):
        Xs, Us, _ = solve_lqr(p)
        return loss(jnp.c_[Xs, jnp.r_[Us, jnp.zeros((1, m))]])

    g_dir = jax.grad(direct)(params)
    chex.assert_trees_all_close(v_par, v_seq, rtol=1e-10)
    chex.assert_trees_all_close(g_par, g_seq, rtol=1e-8, atol=1e-10)
    chex.assert_trees_all_close(sym_lqr_grads(g_par), sym_lqr_grads(g_dir), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("adjoint", ADJOINTS)
@pytest.mark.parametrize("n,m", GRAD_DIMS[:3])
def test_pdlqr_check_grads(n, m, adjoint):
    """Implicit gradients agree with central finite differences, with symmetric Q, R and Qf directions"""
    dims, params, loss = setup(n, m)
    sym = lambda x: (x + jnp.swapaxes(x, -1, -2)) / 2

    def sym_loss(p):
        p = p._replace(lqr=p.lqr._replace(Q=sym(p.lqr.Q), R=sym(p.lqr.R), Qf=sym(p.lqr.Qf)))
        return loss(pdlqr(dims, p, adjoint=adjoint))

    check_grads(sym_loss, (params,), order=1, modes=["rev"], eps=1e-6, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("adjoint", ADJOINTS)
def test_pdllqr_matches_dllqr(adjoint):
    """pdllqr returns tau_star unchanged and has the gradients of dllqr"""
    dims, params, loss = setup(8, 3)
    Xs, Us, _ = solve_lqr(params)
    tau_star = jnp.c_[Xs, jnp.r_[Us, jnp.zeros((1, 3))]]
    v_par, g_par = jax.value_and_grad(lambda p: loss(pdllqr(dims, p, tau_star, adjoint=adjoint)))(params)
    v_seq, g_seq = jax.value_and_grad(lambda p: loss(dllqr(dims, p, tau_star)))(params)
    chex.assert_trees_all_equal(pdllqr(dims, params, tau_star, adjoint=adjoint), tau_star)
    chex.assert_trees_all_close(v_par, v_seq, rtol=1e-12)
    chex.assert_trees_all_close(g_par, g_seq, rtol=1e-8, atol=1e-10)


def test_pdlqr_jit_and_long_horizon():
    """Both adjoints agree under jit on a long horizon, where the scans have many levels"""
    dims, params, loss = setup(5, 2, horizon=1000)
    grad_fn = jax.jit(lambda p, adjoint: jax.grad(lambda q: loss(pdlqr(dims, q, adjoint=adjoint)))(p), static_argnums=1)
    grads = [grad_fn(params, a) for a in ADJOINTS]  # pylint: disable=not-callable
    chex.assert_trees_all_close(*grads, rtol=1e-8, atol=1e-10)
