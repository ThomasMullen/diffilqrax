"""
Optimality tests for the LQR and iLQR solvers: KKT conditions across the dimension grid, agreement
with exact solutions and an independent optimiser, second-order conditions, multi-start agreement,
the quadratic growth of the cost around the optimum, the expected cost change model, smooth L1
sparsity costs, and float32 behaviour.
"""

import os
import pytest
import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from diffilqrax import ilqr, lqr
from diffilqrax.diff_ilqr import add_dynamics_curvature
from diffilqrax.typs import LQR, System
from diffilqrax.utils import smooth_l1
from problems import (
    DIMS,
    SLOW_SIZE,
    random_lqr,
    exact_lqr_inputs,
    lqr_as_system,
    pendulum_problem,
    rnn_problem,
    ocaml_pendulum,
    vanilla_rnn,
    inverted_pendulum,
    van_der_pol,
    rnn_tracking,
    total_cost,
    nonlinear_kkt,
    lbfgs_reference,
    perturbation_slope,
)

jax.config.update("jax_default_device", jax.devices(os.environ.get("DIFFILQRAX_TEST_PLATFORM", "cpu"))[0])
jax.config.update("jax_enable_x64", True)

KKT_ATOL = 1e-8


def grid(dims=DIMS):
    """Parametrise over (n, m), marking the large cases slow"""
    return [pytest.param(n, m, marks=pytest.mark.slow) if n * m >= SLOW_SIZE else (n, m) for n, m in dims]


def assert_kkt(model, params, Xs, Us, Lambs):
    """All KKT residuals of the non-linear problem vanish to KKT_ATOL"""
    for res in nonlinear_kkt(model, params, Xs, Us, Lambs):
        chex.assert_trees_all_close(res, jnp.zeros_like(res), atol=KKT_ATOL, rtol=0.0)


def assert_second_order(model, params, Xs, Us, Lambs):
    """The Lagrangian Hessian LQR has a positive definite Huu at every step (strict local minimum)"""
    local = add_dynamics_curvature(model, ilqr.approx_lqr_offset(model, Xs, Us, params), Xs, Us, Lambs, params)
    assert bool(lqr.regularised_backward_pass(local, mu=0.0)[2]), "reduced Hessian is not positive definite"


def random_inputs(model, scale=0.1, seed=0):
    return scale * jr.normal(jr.PRNGKey(seed), (model.dims.horizon, model.dims.m))


# --------------------------------------------------------------------------------------------
# LQR
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,m", grid())
def test_lqr_kkt_and_exact_solution(n, m):
    """LQR satisfies all KKT conditions and matches the dense exact QP solution"""
    params = random_lqr(n, m, horizon=20)
    Xs, Us, Lambs = lqr.solve_lqr(params)
    dLdXs, dLdUs, dLdLambs = lqr.kkt(params, Xs, Us, Lambs)
    chex.assert_trees_all_close(dLdXs, jnp.zeros_like(dLdXs), atol=KKT_ATOL, rtol=0.0)
    chex.assert_trees_all_close(dLdUs, jnp.zeros_like(dLdUs), atol=KKT_ATOL, rtol=0.0)
    chex.assert_trees_all_close(dLdLambs, jnp.zeros_like(dLdLambs), atol=KKT_ATOL, rtol=0.0)
    chex.assert_trees_all_close(Us, exact_lqr_inputs(params), atol=1e-8, rtol=1e-8)


def test_expected_cost_change_is_exact_for_lqr():
    """The quadratic model of the cost reduction is exact when the problem is an LQR"""
    model, ilqr_params = lqr_as_system(random_lqr(6, 3, horizon=15, seed=1))
    Us = jnp.zeros((model.dims.horizon, model.dims.m))
    (Xs, _), cost0 = ilqr.ilqr_simulate(model, Us, ilqr_params)
    dJ, gains = lqr.lqr_backward_pass(ilqr.approx_lqr(model, Xs, Us, ilqr_params))
    for alpha in (1.0, 0.5, 0.1):
        _, cost = ilqr.ilqr_forward_pass(model, ilqr_params, gains, Xs, Us, alpha=alpha)
        chex.assert_trees_all_close(cost0 - cost, lqr.calc_expected_change(dJ, alpha), rtol=1e-8)


def test_regularisation_matches_ocaml():
    """increase/decrease reproduce OCaml Regularisation.increase/decrease"""
    delta, mu = ilqr.increase_regularisation(1.0, 0.0)
    assert (float(delta), float(mu)) == (2.0, 1e-6)
    delta, mu = ilqr.increase_regularisation(delta, mu)
    assert (float(delta), float(mu)) == (4.0, 4e-6)
    delta, mu = ilqr.decrease_regularisation(delta, mu)
    assert (float(delta), float(mu)) == (0.5, 2e-6)
    delta, mu = ilqr.decrease_regularisation(delta, mu)
    assert (float(delta), float(mu)) == (0.25, 0.0)


def test_backward_pass_pd_flag():
    """The positive definite flag detects an indefinite Huu and regularisation restores it"""
    params = random_lqr(4, 2, horizon=5, seed=2)
    lqr_struct: LQR = params.lqr._replace(R=params.lqr.R - 5.0 * jnp.eye(2))
    assert not bool(lqr.regularised_backward_pass(lqr_struct, mu=0.0)[2])
    assert bool(lqr.regularised_backward_pass(lqr_struct, mu=1e3)[2])


# --------------------------------------------------------------------------------------------
# iLQR across the dimension grid
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,m", grid())
def test_ilqr_recovers_lqr_solution(n, m):
    """iLQR on a problem that is exactly an LQR matches the LQR and the dense exact QP solutions"""
    params = random_lqr(n, m, horizon=15, seed=3)
    model, ilqr_params = lqr_as_system(params)
    (Xs, Us, Lambs), _, _ = ilqr.ilqr_solver(model, ilqr_params, jnp.zeros((15, m)))
    Xs_lqr, Us_lqr, Lambs_lqr = lqr.solve_lqr(params)
    chex.assert_trees_all_close((Xs, Us, Lambs), (Xs_lqr, Us_lqr, Lambs_lqr), atol=1e-9, rtol=1e-9)
    chex.assert_trees_all_close(Us, exact_lqr_inputs(params), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("n,m", grid())
def test_ilqr_rnn_kkt_and_second_order(n, m):
    """iLQR on the non-convex RNN reaches a strict local minimum (KKT + PD reduced Hessian) at every size"""
    model, params = rnn_problem(n, m, horizon=30)
    (Xs, Us, Lambs), _, _ = ilqr.ilqr_solver(model, params, random_inputs(model), max_iter=800)
    assert_kkt(model, params, Xs, Us, Lambs)
    assert_second_order(model, params, Xs, Us, Lambs)


@pytest.mark.parametrize("n", [5, 20, 50])
def test_ilqr_rnn_matches_lbfgs(n):
    """iLQR and L-BFGS started from the same inputs reach the same optimal cost"""
    model, params = rnn_problem(n, m=min(n, 5), horizon=30)
    Us_init = random_inputs(model)
    _, cost, _ = ilqr.ilqr_solver(model, params, Us_init, max_iter=400)
    ref_cost, _ = lbfgs_reference(model, params, Us_init)
    assert jnp.allclose(cost, ref_cost, rtol=1e-10, atol=0.0), (float(cost), ref_cost)


@pytest.mark.parametrize("n", [5, 20, 50])
def test_ilqr_rnn_multistart_agreement(n):
    """Different initial inputs reach the same optimal cost and inputs"""
    model, params = rnn_problem(n, m=min(n, 5), horizon=30)
    sols = [ilqr.ilqr_solver(model, params, random_inputs(model, 0.5, seed), max_iter=400) for seed in range(4)]
    for (_, Us, _), cost, _ in sols[1:]:
        chex.assert_trees_all_close(cost, sols[0][1], rtol=1e-10)
        chex.assert_trees_all_close(Us, sols[0][0][1], atol=1e-6)


# --------------------------------------------------------------------------------------------
# control problems
# --------------------------------------------------------------------------------------------

# (name, builder, use_linesearch, max_iter, initial input scale). Full α=1 steps on the non-convex
# RNNs only converge linearly (Gauss-Newton overshoot), so the no-linesearch mode is checked on the
# pendulums only. The inverted pendulum starts from random inputs: hanging still is a stationary point.
CONTROL_PROBLEMS = [
    ("pendulum", pendulum_problem, True, 100, 0.1),
    ("pendulum_no_linesearch", pendulum_problem, False, 100, 0.1),
    ("ocaml_pendulum", ocaml_pendulum, True, 300, 0.0),
    ("ocaml_pendulum_no_linesearch", ocaml_pendulum, False, 400, 0.0),
    ("vanilla_rnn", vanilla_rnn, True, 200, 0.1),
    ("inverted_pendulum", inverted_pendulum, True, 300, 0.5),
    ("van_der_pol", van_der_pol, True, 200, 0.1),
    pytest.param("rnn_tracking", rnn_tracking, True, 600, 0.1, marks=pytest.mark.slow),
]


@pytest.mark.parametrize("name,builder,use_linesearch,max_iter,scale", CONTROL_PROBLEMS)
def test_control_problem_optimality(name, builder, use_linesearch, max_iter, scale):
    """KKT + PD reduced Hessian, and L-BFGS warm-started at the iLQR solution cannot improve on it"""
    model, params = builder()
    Us_init = random_inputs(model, scale) if scale else jnp.zeros((model.dims.horizon, model.dims.m))
    (Xs, Us, Lambs), cost, _ = ilqr.ilqr_solver(model, params, Us_init, max_iter=max_iter, use_linesearch=use_linesearch)
    assert_kkt(model, params, Xs, Us, Lambs)
    assert_second_order(model, params, Xs, Us, Lambs)
    ref_cost, _ = lbfgs_reference(model, params, Us)
    assert float(cost) <= ref_cost * (1 + 1e-10), (name, float(cost), ref_cost)


@pytest.mark.parametrize(
    "builder,scale",
    [(pendulum_problem, 0.5), (vanilla_rnn, 0.5), (inverted_pendulum, 0.5), (van_der_pol, 0.5)],
)
def test_control_problem_multistart_agreement(builder, scale):
    """Random initial inputs reach the same optimal cost"""
    model, params = builder()
    costs = [ilqr.ilqr_solver(model, params, random_inputs(model, scale, seed), max_iter=300)[1] for seed in range(3)]
    for cost in costs[1:]:
        chex.assert_trees_all_close(cost, costs[0], rtol=1e-10)


# The OCaml pendulum is left out: its open-loop dynamics are unstable over 10 s, so even
# eps = 1e-9 input perturbations grow ~e^31-fold and the eps^2 regime is not reachable.
@pytest.mark.parametrize(
    "builder,scale",
    [(pendulum_problem, 0.1), (rnn_problem, 0.1), (vanilla_rnn, 0.1), (inverted_pendulum, 0.5), (van_der_pol, 0.1)],
)
def test_cost_grows_quadratically_around_optimum(builder, scale):
    """J(U* + eps dU) - J(U*) > 0 and scales like eps^2 (slope 2 on a log-log plot), not eps"""
    model, params = builder()
    (_, Us, _), _, _ = ilqr.ilqr_solver(model, params, random_inputs(model, scale), max_iter=300)
    for seed in range(3):
        _, dJ, slope = perturbation_slope(model, params, Us, seed=seed)
        assert (dJ > 0).all(), dJ
        assert abs(slope - 2.0) < 0.1, slope


# --------------------------------------------------------------------------------------------
# costs and precision
# --------------------------------------------------------------------------------------------


def test_ilqr_smooth_l1_sparsity():
    """Annealed smooth L1 input cost gives a KKT point with sparse inputs"""
    model, params = pendulum_problem(horizon=60)
    Us = jnp.zeros((model.dims.horizon, model.dims.m))
    l1_weight = 2.0
    for eps in (1e-2, 1e-4, 1e-6):
        sparse_model = System(
            lambda t, x, u, th, eps=eps: jnp.sum(x**2) + l1_weight * smooth_l1(u, eps),
            model.costf,
            model.dynamics,
            model.dims,
        )
        (Xs, Us, Lambs), _, _ = ilqr.ilqr_solver(sparse_model, params, Us, max_iter=200)
        assert_kkt(sparse_model, params, Xs, Us, Lambs)
    # with eps -> 0 a sizeable fraction of inputs is (numerically) zero
    assert jnp.mean(jnp.abs(Us) < 1e-2) > 0.25


def test_ilqr_float32_best_effort():
    """float32 converges and reaches a loose KKT tolerance"""
    model, params = pendulum_problem()
    params = jax.tree.map(lambda x: x.astype(jnp.float32), params)
    # dtype-preserving dynamics: jnp.ones defaults to float64 when x64 is enabled
    model = System(
        model.cost,
        model.costf,
        lambda t, x, u, th: jnp.tanh(th.Uh @ x + th.Wh @ u) + jnp.ones_like(x),
        model.dims,
    )
    Us_init = random_inputs(model).astype(jnp.float32)
    (Xs, Us, Lambs), cost, _ = ilqr.ilqr_solver(model, params, Us_init, kkt_tol=1e-3)
    assert cost < total_cost(model, params, Us_init)
    dLdUs = nonlinear_kkt(model, params, Xs, Us, Lambs)[1]
    assert float(jnp.max(jnp.abs(dLdUs))) < 1e-3
