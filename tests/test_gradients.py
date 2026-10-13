"""
Gradient tests for the differentiable iLQR solver: implicit (custom VJP) gradients against finite
differences and against direct differentiation through the unrolled solver, wrt both x0 and theta.
"""

import pytest
import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.test_util import check_grads

from diffilqrax.ilqr import ilqr_solver
from diffilqrax.diff_ilqr import dilqr
from problems import rnn_problem, linear_rnn_problem, pendulum_problem, van_der_pol, inverted_pendulum

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

MAX_ITER = 400

RNN_DIMS = [(5, 3), (20, 5), (50, 10), pytest.param(100, 20, marks=pytest.mark.slow)]

PROBLEMS = {
    "pendulum": lambda: pendulum_problem(horizon=50),
    "van_der_pol": lambda: van_der_pol(horizon=60),
    "inverted_pendulum": lambda: inverted_pendulum(horizon=100),
}


def losses(model, params, exact_hessian: bool = True, seed: int = 0):
    """Implicit and direct losses ||X*||^2 + ||U*||^2 as functions of iLQRParams (x0 and theta)"""
    Us_init = 0.1 * jr.normal(jr.PRNGKey(seed), (model.dims.horizon, model.dims.m))

    def implicit_loss(p):
        tau = dilqr(model, p, Us_init, exact_hessian=exact_hessian, max_iter=MAX_ITER)
        return jnp.sum(tau**2)

    def direct_loss(p):
        (Xs, Us, _), _, _ = ilqr_solver(model, p, Us_init, max_iter=MAX_ITER)
        return jnp.sum(Xs**2) + jnp.sum(Us**2)

    return implicit_loss, direct_loss, params


def assert_grads_close(g, g_ref, rtol: float = 1e-6, atol: float = 1e-12):
    """Normwise relative error per leaf, ||g - g_ref|| <= rtol ||g_ref|| + atol. Element-wise rtol is too strict
    for near-zero entries of large gradients, whose absolute error sits at the solver's round-off."""
    for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(g), jax.tree_util.tree_leaves(g_ref)):
        err, ref = jnp.linalg.norm(a - b), jnp.linalg.norm(b)
        assert err <= rtol * ref + atol, f"{jax.tree_util.keystr(path)}: relative error {err / ref:.2e} > {rtol:.0e}"


def assert_implicit_matches_direct(implicit_loss, direct_loss, params):
    """Values agree to rtol 1e-8 and gradients wrt x0 and theta to normwise rtol 1e-6"""
    v_imp, g_imp = jax.value_and_grad(implicit_loss)(params)
    v_dir, g_dir = jax.value_and_grad(direct_loss)(params)
    chex.assert_trees_all_close(v_imp, v_dir, rtol=1e-8)
    assert_grads_close(g_imp, g_dir)


def test_implicit_gradient_matches_finite_differences():
    """Reverse-mode implicit gradients agree with central finite differences along random directions"""
    implicit_loss, _, params = losses(*rnn_problem(5, 3))
    check_grads(implicit_loss, (params,), order=1, modes=["rev"], eps=1e-5, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("n,m", RNN_DIMS)
def test_implicit_matches_direct_gradient(n, m):
    """Implicit gradients (exact Lagrangian Hessian) match differentiation through the unrolled solver"""
    assert_implicit_matches_direct(*losses(*rnn_problem(n, m)))


@pytest.mark.parametrize("name", PROBLEMS)
def test_implicit_matches_direct_gradient_control_problems(name):
    """Same check on the low-dimensional control problems (gradients wrt x0, and theta where present)"""
    assert_implicit_matches_direct(*losses(*PROBLEMS[name]()))


@pytest.mark.parametrize("n,m", [(5, 3), (20, 5)])
def test_gauss_newton_exact_for_linear_dynamics(n, m):
    """With linear dynamics the dynamics curvature vanishes, so exact_hessian=False is also exact"""
    model, params = linear_rnn_problem(n, m)
    implicit_exact, direct_loss, _ = losses(model, params, exact_hessian=True)
    implicit_gn, _, _ = losses(model, params, exact_hessian=False)
    g_exact = jax.grad(implicit_exact)(params)
    g_gn = jax.grad(implicit_gn)(params)
    g_dir = jax.grad(direct_loss)(params)
    assert_grads_close(g_gn, g_exact, rtol=1e-10)
    assert_grads_close(g_gn, g_dir)


@pytest.mark.parametrize("max_iter", [3, 10, MAX_ITER])
def test_value_independent_of_differentiation(max_iter):
    """dilqr returns the solver's trajectory whether or not it is differentiated, even when unconverged"""
    model, params = rnn_problem(5, 3)
    Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (model.dims.horizon, model.dims.m))

    def loss(p):
        return jnp.sum(dilqr(model, p, Us_init, max_iter=max_iter) ** 2)

    chex.assert_trees_all_close(jax.value_and_grad(loss)(params)[0], loss(params), rtol=1e-14)


def test_warm_start_is_not_differentiated():
    """Us_init gets no gradient, and a theta-dependent warm start neither changes the gradient nor
    makes reverse mode store the solver iterations"""
    model, params = rnn_problem(5, 3)
    Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (model.dims.horizon, model.dims.m))

    def loss(theta, warm):
        warm_start = Us_init + warm * 0.01 * jnp.sum(theta.Uh)
        return jnp.sum(dilqr(model, params._replace(theta=theta), warm_start, max_iter=MAX_ITER) ** 2)

    def loss_us(us):
        return jnp.sum(dilqr(model, params, us, max_iter=MAX_ITER) ** 2)

    assert jnp.all(jax.grad(loss_us)(Us_init) == 0.0)
    assert_grads_close(jax.grad(loss)(params.theta, 1.0), jax.grad(loss)(params.theta, 0.0), rtol=1e-10)
    temp_bytes = {
        warm: jax.jit(jax.grad(loss)).lower(params.theta, warm).compile().memory_analysis().temp_size_in_bytes
        for warm in (0.0, 1.0)
    }
    assert temp_bytes[1.0] <= 1.5 * temp_bytes[0.0], temp_bytes
