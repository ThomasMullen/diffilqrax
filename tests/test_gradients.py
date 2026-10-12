"""
Gradient tests for the differentiable iLQR solver: implicit (custom VJP) gradients against finite
differences and against direct differentiation through the unrolled solver.
"""

import pytest
import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.test_util import check_grads

from diffilqrax.ilqr import ilqr_solver
from diffilqrax.diff_ilqr import dilqr
from diffilqrax.typs import iLQRParams
from problems import rnn_problem

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

MAX_ITER = 400


def rnn_losses(n: int, m: int):
    """Implicit and direct losses ||X*||^2 + ||U*||^2 of the RNN problem as functions of theta"""
    model, params = rnn_problem(n, m, horizon=30)
    Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (model.dims.horizon, m))

    def implicit_loss(theta):
        tau = dilqr(model, iLQRParams(params.x0, theta), Us_init, max_iter=MAX_ITER)
        return jnp.sum(tau**2)

    def direct_loss(theta):
        (Xs, Us, _), _, _ = ilqr_solver(model, iLQRParams(params.x0, theta), Us_init, max_iter=MAX_ITER)
        return jnp.sum(Xs**2) + jnp.sum(Us**2)

    return implicit_loss, direct_loss, params.theta


def test_implicit_gradient_matches_finite_differences():
    """Reverse-mode implicit gradients agree with central finite differences along random directions"""
    implicit_loss, _, theta = rnn_losses(5, 3)
    check_grads(implicit_loss, (theta,), order=1, modes=["rev"], eps=1e-5, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("n,m", [(5, 3), (20, 5)])
def test_implicit_matches_direct_gradient(n, m):
    """Implicit gradients (exact Lagrangian Hessian) match differentiation through the unrolled solver"""
    implicit_loss, direct_loss, theta = rnn_losses(n, m)
    v_imp, g_imp = jax.value_and_grad(implicit_loss)(theta)
    v_dir, g_dir = jax.value_and_grad(direct_loss)(theta)
    chex.assert_trees_all_close(v_imp, v_dir, rtol=1e-10)
    chex.assert_trees_all_close(g_imp, g_dir, rtol=1e-6, atol=1e-9)
