"""
Gradients of the differentiable parallel iLQR solver (pdilqr), for both adjoint solvers and both pilqr
rollouts, against the sequential dilqr (itself validated against direct differentiation in test_gradients.py),
on linear dynamics with non-quadratic costs and on non-linear dynamics. The implicit-gradient path through
pdllqr is also checked on the ilqr solution, independently of the solver.
"""

import os
import pytest
import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.test_util import check_grads

from diffilqrax.typs import LQRParams
from diffilqrax.ilqr import ilqr_solver, approx_lqr_offset
from diffilqrax.diff_ilqr import dilqr, add_dynamics_curvature
from diffilqrax.parallel_dilqr import pdilqr
from diffilqrax.parallel_dlqr import pdllqr
from problems import (
    NONCONVEX_COSTS,
    linear_nonconvex_problem,
    linear_rnn_problem,
    rnn_problem,
    pendulum_problem,
    van_der_pol,
    inverted_pendulum,
    as_parallel,
)
from test_gradients import assert_grads_close

jax.config.update("jax_default_device", jax.devices(os.environ.get("DIFFILQRAX_TEST_PLATFORM", "cpu"))[0])
jax.config.update("jax_enable_x64", True)

MAX_ITER = 300
ADJOINTS = ["parallel", "sequential"]

LINEAR_PROBLEMS = {
    **{kind: (lambda kind=kind: linear_nonconvex_problem(kind)) for kind in NONCONVEX_COSTS},
    "linear_rnn": lambda: linear_rnn_problem(5, 3),
}

NONLINEAR_PROBLEMS = {
    "pendulum": lambda: pendulum_problem(horizon=50),
    "van_der_pol": lambda: van_der_pol(horizon=60),
    "inverted_pendulum": lambda: inverted_pendulum(horizon=100),
    "rnn": lambda: rnn_problem(5, 3),
}


def init_inputs(model, seed: int = 0):
    return 0.1 * jr.normal(jr.PRNGKey(seed + 100), (model.dims.horizon, model.dims.m))


def value_and_grads(solve, params):
    """Value and gradient wrt x0 and theta of the loss sum(tau**2)"""
    return jax.value_and_grad(lambda p: jnp.sum(solve(p) ** 2))(params)


@pytest.mark.parametrize("adjoint", ADJOINTS)
@pytest.mark.parametrize("name", list(LINEAR_PROBLEMS))
def test_pdilqr_matches_dilqr(name, adjoint):
    """Linear dynamics: pdilqr has the value and gradients of dilqr"""
    model, params = LINEAR_PROBLEMS[name]()
    Us_init = init_inputs(model)
    pmodel = as_parallel(model)
    v_par, g_par = value_and_grads(lambda p: pdilqr(pmodel, p, Us_init, adjoint=adjoint, max_iter=MAX_ITER), params)
    v_seq, g_seq = value_and_grads(lambda p: dilqr(model, p, Us_init, max_iter=MAX_ITER), params)
    chex.assert_trees_all_close(v_par, v_seq, rtol=1e-8)
    assert_grads_close(g_par, g_seq, rtol=1e-6)


def test_pdilqr_adjoints_agree():
    """The parallel and sequential adjoint solvers give the same gradients to round-off"""
    model, params = LINEAR_PROBLEMS["track_log"]()
    Us_init, pmodel = init_inputs(model), as_parallel(model)
    g_par, g_seq = (
        value_and_grads(lambda p, a=a: pdilqr(pmodel, p, Us_init, adjoint=a, max_iter=MAX_ITER), params)[1] for a in ADJOINTS
    )
    assert_grads_close(g_par, g_seq, rtol=1e-10)


def test_pdilqr_check_grads():
    """Implicit gradients agree with central finite differences wrt x0 and theta"""
    model, params = LINEAR_PROBLEMS["linear_rnn"]()
    Us_init, pmodel = init_inputs(model), as_parallel(model)
    loss = lambda p: jnp.sum(pdilqr(pmodel, p, Us_init, max_iter=MAX_ITER, kkt_tol=1e-12) ** 2)
    check_grads(loss, (params,), order=1, modes=["rev"], eps=1e-6, atol=1e-6, rtol=1e-6)


def test_pdilqr_us_init_gets_no_gradient():
    """Us_init is stop-gradiented, so a warm start that depends on the parameters adds no gradient"""
    model, params = LINEAR_PROBLEMS["linear_rnn"]()
    Us_init, pmodel = init_inputs(model), as_parallel(model)
    loss = lambda p, us: jnp.sum(pdilqr(pmodel, p, us, max_iter=MAX_ITER) ** 2)
    g = jax.grad(lambda p: loss(p, Us_init))(params)
    g_warm = jax.grad(lambda p: loss(p, Us_init + jnp.sum(p.theta.Wh)))(params)
    assert_grads_close(g_warm, g, rtol=1e-8)


@pytest.mark.parametrize("exact_hessian", [True, False])
@pytest.mark.parametrize("adjoint", ADJOINTS)
@pytest.mark.parametrize("name", list(NONLINEAR_PROBLEMS))
def test_pdllqr_nonlinear_matches_dilqr(name, adjoint, exact_hessian):
    """Non-linear dynamics: the local LQR at the ilqr solution, with (exact) or without (Gauss-Newton) the
    dynamics curvature, differentiated through pdllqr gives the gradients of dilqr. The Lagrangian Hessian is
    indefinite on these problems, so this also checks that the parallel adjoint scan does not need a convex LQR."""
    model, params = NONLINEAR_PROBLEMS[name]()
    Us_init = init_inputs(model)
    sol = ilqr_solver(model, params, Us_init, max_iter=400)
    (Xs, Us, Lambs), _, _ = jax.lax.stop_gradient(sol)
    tau = jnp.c_[Xs, jnp.r_[Us, jnp.zeros((1, model.dims.m))]]

    def solve(p):
        local = approx_lqr_offset(model, Xs, Us, p)
        if exact_hessian:
            local = add_dynamics_curvature(model, local, Xs, Us, Lambs, p)
        return pdllqr(model.dims, LQRParams(p.x0, local), tau, adjoint=adjoint)

    v_par, g_par = value_and_grads(solve, params)
    v_seq, g_seq = value_and_grads(lambda p: dilqr(model, p, Us_init, exact_hessian=exact_hessian, max_iter=400), params)
    chex.assert_trees_all_close(v_par, v_seq, rtol=1e-12)
    assert_grads_close(g_par, g_seq, rtol=1e-8)


@pytest.mark.parametrize("exact_hessian", [True, False])
@pytest.mark.parametrize("rollout", ["feedback", "parallel"])
@pytest.mark.parametrize("adjoint", ADJOINTS)
@pytest.mark.parametrize("name", list(NONLINEAR_PROBLEMS))
def test_pdilqr_nonlinear_matches_dilqr(name, adjoint, rollout, exact_hessian):
    """Non-linear dynamics end to end: pdilqr has the value and gradients of dilqr"""
    model, params = NONLINEAR_PROBLEMS[name]()
    Us_init = init_inputs(model)
    kw = dict(exact_hessian=exact_hessian, max_iter=400)
    v_par, g_par = value_and_grads(lambda p: pdilqr(model, p, Us_init, adjoint=adjoint, rollout=rollout, **kw), params)
    v_seq, g_seq = value_and_grads(lambda p: dilqr(model, p, Us_init, **kw), params)
    chex.assert_trees_all_close(v_par, v_seq, rtol=1e-8)
    assert_grads_close(g_par, g_seq, rtol=1e-6)
