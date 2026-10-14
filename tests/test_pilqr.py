"""
Parallel iLQR (pilqr_solver) on non-linear and linear problems, for both rollouts: "feedback" (sequential
non-linear rollout) and "parallel" (multiple shooting with a merit line search). The solutions must match the
sequential ilqr_solver and primal_dual_ilqr (arXiv:2403.00748) and satisfy the KKT conditions.
"""

from pathlib import Path
import pytest
import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp

from diffilqrax.ilqr import ilqr_solver, ilqr_simulate, approx_lqr
from diffilqrax.parallel_ilqr import pilqr_solver, dynamics_defects, multiple_shooting_step, total_cost
from diffilqrax.plqr import plqr_backward_pass
from diffilqrax.typs import LQRParams
from problems import (
    rnn_problem,
    linear_rnn_problem,
    pendulum_problem,
    van_der_pol,
    inverted_pendulum,
    linear_nonconvex_problem,
    nonlinear_kkt,
    as_parallel,
)

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

MAX_ITER = 400
ROLLOUTS = ["feedback", "parallel"]
FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("pdilqr_ilqr_*.npz"))

PROBLEMS = {
    "pendulum": lambda: pendulum_problem(horizon=50),
    "van_der_pol": lambda: van_der_pol(horizon=60),
    "inverted_pendulum": lambda: inverted_pendulum(horizon=100),
    "rnn": lambda: rnn_problem(5, 3),
    "linear_rnn": lambda: linear_rnn_problem(5, 3),
    "track_log": lambda: linear_nonconvex_problem("track_log"),
    "sharp_log_u": lambda: linear_nonconvex_problem("sharp_log_u"),
}


def init_inputs(model, seed: int = 0, scale: float = 0.1):
    return scale * jr.normal(jr.PRNGKey(seed), (model.dims.horizon, model.dims.m))


def max_kkt(model, params, Xs, Us, Lambs) -> float:
    return max(float(jnp.max(jnp.abs(r))) for r in nonlinear_kkt(model, params, Xs, Us, Lambs))


@pytest.mark.parametrize("rollout", ROLLOUTS)
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_pilqr_matches_ilqr(name, rollout):
    """Same solution, cost and adjoints as ilqr_solver; KKT conditions and dynamics hold"""
    model, params = PROBLEMS[name]()
    Us_init = init_inputs(model)
    (Xs, Us, Lambs), J, _ = ilqr_solver(model, params, Us_init, max_iter=MAX_ITER)
    (Xp, Up, Lp), Jp, costs = pilqr_solver(model, params, Us_init, max_iter=MAX_ITER, rollout=rollout)
    chex.assert_trees_all_close(Jp, J, rtol=1e-10)
    chex.assert_trees_all_close((Xp, Up, Lp), (Xs, Us, Lambs), atol=1e-7, rtol=1e-7)
    assert max_kkt(model, params, Xp, Up, Lp) < 1e-8
    assert float(jnp.max(jnp.abs(dynamics_defects(model, params, Xp, Up)))) < 1e-10
    chex.assert_trees_all_close(costs[-1], Jp)


@pytest.mark.parametrize("rollout", ROLLOUTS)
@pytest.mark.parametrize("seed", range(3))
def test_pilqr_multistart(rollout, seed):
    """Large random initial inputs on the tanh RNN: pilqr still reaches a feasible KKT point"""
    model, params = PROBLEMS["rnn"]()
    (Xp, Up, Lp), _, _ = pilqr_solver(model, params, init_inputs(model, seed, 2.0), max_iter=MAX_ITER, rollout=rollout)
    assert max_kkt(model, params, Xp, Up, Lp) < 1e-8
    assert float(jnp.max(jnp.abs(dynamics_defects(model, params, Xp, Up)))) < 1e-10


@pytest.mark.parametrize("rollout", ROLLOUTS)
def test_pilqr_cost_history(rollout):
    """Costs are finite; feedback iterates are feasible, so their cost decreases monotonically"""
    model, params = PROBLEMS["pendulum"]()
    _, _, costs = pilqr_solver(model, params, init_inputs(model), max_iter=60, rollout=rollout)
    if rollout == "feedback":
        assert bool(jnp.all(jnp.diff(costs) <= 1e-9 * jnp.abs(costs[1:])))
    assert bool(jnp.all(jnp.isfinite(costs)))


@pytest.mark.parametrize("rollout", ROLLOUTS)
def test_pilqr_without_linesearch(rollout):
    """Full steps (α = 1) converge on a mildly non-linear problem"""
    model, params = PROBLEMS["rnn"]()
    Us_init = init_inputs(model)
    _, J, _ = ilqr_solver(model, params, Us_init, max_iter=MAX_ITER)
    (Xp, Up, Lp), Jp, _ = pilqr_solver(model, params, Us_init, max_iter=MAX_ITER, rollout=rollout, use_linesearch=False)
    chex.assert_trees_all_close(Jp, J, rtol=1e-10)
    assert max_kkt(model, params, Xp, Up, Lp) < 1e-8


def test_parallel_system_still_accepted():
    """A ParallelSystem wrapper gives the same result as the bare System"""
    model, params = PROBLEMS["pendulum"]()
    Us_init = init_inputs(model)
    out = pilqr_solver(model, params, Us_init, max_iter=MAX_ITER)
    chex.assert_trees_all_close(pilqr_solver(as_parallel(model), params, Us_init, max_iter=MAX_ITER), out)


def test_legacy_linesearch_kwargs_warn():
    model, params = PROBLEMS["pendulum"]()
    with pytest.warns(DeprecationWarning):
        pilqr_solver(model, params, init_inputs(model), max_iter=5, beta=0.5, tol=0.1)


def test_invalid_rollout():
    model, params = PROBLEMS["pendulum"]()
    with pytest.raises(AssertionError):
        pilqr_solver(model, params, init_inputs(model), rollout="linearised")


def test_multiple_shooting_step():
    """At an infeasible trajectory the step satisfies the linearised dynamics with the defects as offsets and
    zeroes the linearised defects; its multipliers satisfy dL/dx = 0; and the merit slope gᵀp - λᵀd - ρ‖d‖²
    equals -pᵀHp - ρ‖d‖² < 0"""
    model, params = PROBLEMS["pendulum"]()
    Us = init_inputs(model)
    (Xs, _), _ = ilqr_simulate(model, Us, params)
    Xs = Xs.at[1:].add(0.05 * jr.normal(jr.PRNGKey(3), Xs[1:].shape))
    Us = Us + 0.05 * jr.normal(jr.PRNGKey(4), Us.shape)
    lqr = approx_lqr(model, Xs, Us, params)
    d = dynamics_defects(model, params, Xs, Us)
    delta_lqr = lqr._replace(a=d)
    _, gains, is_pd, values = plqr_backward_pass(LQRParams(jnp.zeros(2), delta_lqr), 0.0, return_values=True)
    assert bool(is_pd)
    dX, dU, lambs = multiple_shooting_step(delta_lqr, gains, values)
    bmv = lambda M, v: jnp.einsum("tij,tj->ti", M, v)
    chex.assert_trees_all_close(dX[1:], bmv(lqr.A, dX[:-1]) + bmv(lqr.B, dU) + d, atol=1e-12)
    assert float(jnp.max(jnp.abs(dX[0]))) == 0.0
    # stationarity in x and u of the LQR Lagrangian
    lx = bmv(lqr.Q, dX[:-1]) + lqr.q + bmv(lqr.S, dU) + bmv(lqr.A.transpose(0, 2, 1), lambs[1:]) - lambs[:-1]
    lu = bmv(lqr.R, dU) + lqr.r + bmv(lqr.S.transpose(0, 2, 1), dX[:-1]) + bmv(lqr.B.transpose(0, 2, 1), lambs[1:])
    chex.assert_trees_all_close((lx, lu), (jnp.zeros_like(lx), jnp.zeros_like(lu)), atol=1e-10)
    rho = 1.0
    g_p = jnp.sum(lqr.q * dX[:-1]) + jnp.sum(lqr.r * dU) + lqr.qf @ dX[-1]
    pHp = (
        jnp.sum(dX[:-1] * bmv(lqr.Q, dX[:-1]))
        + 2 * jnp.sum(dX[:-1] * bmv(lqr.S, dU))
        + jnp.sum(dU * bmv(lqr.R, dU))
        + dX[-1] @ lqr.Qf @ dX[-1]
    )
    slope = g_p - jnp.sum(lambs[1:] * d) - rho * jnp.sum(d**2)
    chex.assert_trees_all_close(slope, -pHp - rho * jnp.sum(d**2), rtol=1e-10)
    assert float(slope) < 0


def test_total_cost_matches_rollout():
    model, params = PROBLEMS["rnn"]()
    Us = init_inputs(model)
    (Xs, _), J = ilqr_simulate(model, Us, params)
    chex.assert_trees_all_close(total_cost(model, params, Xs, Us), J, rtol=1e-14)


@pytest.mark.skipif(not FIXTURES, reason="run benchmarks/make_pdilqr_fixtures.py")
@pytest.mark.parametrize("rollout", ROLLOUTS)
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_pilqr_matches_primal_dual_ilqr(path, rollout):
    """Same solution as primal_dual_ilqr (Lagrangian-Hessian SQP with multiple shooting) from the same start"""
    d = dict(onp.load(path))
    model, params = {"pendulum": PROBLEMS["pendulum"], "rnn": PROBLEMS["rnn"]}[str(d["kind"])]()
    chex.assert_trees_all_close((params.theta.Uh, params.theta.Wh, params.x0), (d["Uh"], d["Wh"], d["x0"]))
    (Xp, Up, _), Jp, _ = pilqr_solver(model, params, jnp.asarray(d["U_init"]), max_iter=MAX_ITER, rollout=rollout)
    chex.assert_trees_all_close(Jp, d["cost"], rtol=1e-10)
    chex.assert_trees_all_close((Xp, Up), (d["X"], d["U"]), atol=1e-6)


def test_parallel_rollout_unstable_long_horizon():
    """Linearised dynamics with spectral radius ~1.5 over T = 2000: the open-loop adjoint recursion overflows,
    so the KKT residual is not finite for any solver, but the multiple-shooting multipliers come from the value
    functions and the parallel rollout still reaches the ilqr optimum"""
    model, params = rnn_problem(8, 3, horizon=2000)
    Us_init = jnp.zeros((2000, 3))
    _, J, _ = ilqr_solver(model, params, Us_init, max_iter=60)
    (Xp, Up, _), Jp, _ = pilqr_solver(model, params, Us_init, max_iter=60, rollout="parallel")
    chex.assert_trees_all_close(Jp, J, rtol=1e-10)
    assert float(jnp.max(jnp.abs(dynamics_defects(model, params, Xp, Up)))) < 1e-10
