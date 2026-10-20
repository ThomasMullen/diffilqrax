"""
Batched solves (diffilqrax.batch) and the while-loop solvers they rely on.

* loop="while" and loop="scan" give the same solution and cost history, and the parallel line search the
  same optimum as backtracking;
* batched solves equal unbatched ones for every strategy, on tracking problems with references simulated
  from the system itself and on the zebrafish MGU2 fixture;
* implicit gradients through a batch equal the sum of per-problem gradients, without reverse mode
  entering the while loops.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from diffilqrax import batch
from diffilqrax.diff_ilqr import dilqr
from diffilqrax.ilqr import ilqr_solver
from diffilqrax.parallel_ilqr import pilqr_solver
from diffilqrax.typs import iLQRParams

from mgu2_model import load_fixture, make_system, make_theta
from problems import TRACKING_AXES, pendulum_problem, rnn_problem, simulated_tracking
from test_gradients import assert_grads_close

jax.config.update("jax_enable_x64", True)

MAX_ITER = 100
FIXTURE = Path(__file__).parent / "fixtures" / "ocaml_mgu.npz"
N_BEG = 12
PROBLEMS = {"rnn": lambda: rnn_problem(5, 3), "pendulum": pendulum_problem}


def zero_inputs(model):
    return jnp.zeros((model.dims.horizon, model.dims.m))


def take(params: iLQRParams, i: int) -> iLQRParams:
    """Problem i of a batch built by simulated_tracking"""
    return iLQRParams(x0=params.x0[i], theta=params.theta._replace(Q=params.theta.Q[i]))


def assert_solutions_close(out, ref, atol: float = 1e-8):
    """Trajectories, adjoints and cost agree"""
    chex.assert_trees_all_close(out[:2], ref[:2], rtol=1e-10, atol=atol)


# --------------------------------------------------------------------------------------------
# while loops and line searches
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROBLEMS)
def test_while_matches_scan(name):
    """ilqr_solver: identical solution, cost and cost history with both loops"""
    model, params = PROBLEMS[name]()
    out = {loop: ilqr_solver(model, params, zero_inputs(model), max_iter=MAX_ITER, loop=loop) for loop in ("while", "scan")}
    chex.assert_trees_all_close(out["while"], out["scan"], rtol=0, atol=0)


@pytest.mark.parametrize("rollout", ["feedback", "parallel"])
@pytest.mark.parametrize("name", PROBLEMS)
def test_pilqr_while_matches_scan(name, rollout):
    """pilqr_solver: identical solution, cost and cost history with both loops"""
    model, params = PROBLEMS[name]()
    solve = lambda loop: pilqr_solver(model, params, zero_inputs(model), max_iter=MAX_ITER, rollout=rollout, loop=loop)
    chex.assert_trees_all_close(solve("while"), solve("scan"), rtol=0, atol=0)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("name", PROBLEMS)
def test_parallel_linesearch_matches_backtracking(name, parallel):
    """Trying several step sizes at once reaches the optimum of one-at-a-time backtracking"""
    model, params = PROBLEMS[name]()
    solver = pilqr_solver if parallel else ilqr_solver
    out = {ls: solver(model, params, zero_inputs(model), max_iter=MAX_ITER, linesearch=ls) for ls in ("backtrack", "parallel")}
    assert_solutions_close(out["parallel"], out["backtrack"])


def test_parallel_linesearch_falls_back_to_backtracking():
    """With fewer parallel step sizes than needed, the search continues one step size at a time"""
    model, params = PROBLEMS["pendulum"]()
    ref = ilqr_solver(model, params, zero_inputs(model), max_iter=MAX_ITER)
    out = ilqr_solver(model, params, zero_inputs(model), max_iter=MAX_ITER, linesearch="parallel", n_parallel=1)
    assert_solutions_close(out, ref)


def test_return_converged():
    """The convergence flag is True at the optimum and False when max_iter cuts the solve short"""
    model, params = PROBLEMS["rnn"]()
    *_, converged = ilqr_solver(model, params, zero_inputs(model), max_iter=MAX_ITER, return_converged=True)
    *_, truncated = ilqr_solver(model, params, zero_inputs(model), max_iter=3, return_converged=True)
    assert bool(converged) and not bool(truncated)


def test_unknown_loop_raises():
    model, params = PROBLEMS["rnn"]()
    with pytest.raises(AssertionError):
        ilqr_solver(model, params, zero_inputs(model), loop="fori")
    with pytest.raises(AssertionError):
        ilqr_solver(model, params, zero_inputs(model), linesearch="golden")


# --------------------------------------------------------------------------------------------
# batched solves of simulated tracking problems
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=["rnn", "pendulum"])
def tracking(request):
    """Odd batch size, so that sharding over 4 devices needs padding"""
    model, params, Us_true = simulated_tracking(request.param, n_batch=5)
    solve = jax.jit(lambda p: ilqr_solver(model, p, zero_inputs(model), max_iter=MAX_ITER, return_converged=True))
    refs = [solve(take(params, i)) for i in range(5)]
    return request.param, model, params, Us_true, jax.tree.map(lambda *xs: jnp.stack(xs), *refs)


@pytest.mark.parametrize("strategy,kwargs", [("vmap", {}), ("map", {}), ("map", {"batch_size": 2})])
def test_batched_matches_unbatched(tracking, strategy, kwargs):
    """Every strategy gives the per-problem solutions, which track the simulated references"""
    name, model, params, Us_true, ref = tracking
    out = batch.batched_ilqr_solver(
        model, params, zero_inputs(model), TRACKING_AXES, strategy=strategy, max_iter=MAX_ITER, return_converged=True, **kwargs
    )
    assert_solutions_close(out, ref)
    assert bool(jnp.all(out[3]))
    (Xs, Us, _), *_ = out
    refs = params.theta.Q
    rel_x = jnp.linalg.norm(Xs - refs, axis=(1, 2)) / jnp.linalg.norm(refs - refs.mean(1, keepdims=True), axis=(1, 2))
    assert float(rel_x.max()) < 0.05
    if name == "rnn":  # the pendulum's second input does not drive the system, so only the rnn inputs are identifiable
        rel_u = jnp.linalg.norm(Us - Us_true, axis=(1, 2)) / jnp.linalg.norm(Us_true, axis=(1, 2))
        assert float(rel_u.max()) < 0.05


def test_batched_parallel_solver(tracking):
    """parallel=True batches pilqr_solver, which reaches the same optima"""
    _, model, params, _, ref = tracking
    out = batch.batched_ilqr_solver(model, params, zero_inputs(model), TRACKING_AXES, parallel=True, max_iter=MAX_ITER)
    assert_solutions_close(out, ref, atol=1e-7)


def test_batch_map_in_axes():
    """batch_map accepts an int in_axes and rejects axes other than 0 and None"""
    xs = jnp.arange(6.0).reshape(3, 2)
    chex.assert_trees_all_close(batch.batch_map(jnp.sum, 0, strategy="map")(xs), xs.sum(1))
    with pytest.raises(AssertionError):
        batch.batch_map(jnp.sum, 1, strategy="map")(xs)


# --------------------------------------------------------------------------------------------
# gradients through a batch
# --------------------------------------------------------------------------------------------


def batched_tracking_loss(model, params, strategy: str = "vmap", **kwargs):
    """Loss sum_b sum(tau_b^2) as a function of the shared dynamics (Uh, Wh)"""

    def loss(UhWh):
        p = params._replace(theta=params.theta._replace(Uh=UhWh[0], Wh=UhWh[1]))
        tau = batch.batched_dilqr(model, p, zero_inputs(model), TRACKING_AXES, strategy=strategy, max_iter=MAX_ITER, **kwargs)
        return jnp.sum(tau**2)

    return loss


@pytest.mark.parametrize("strategy", ["vmap", "map"])
def test_batched_gradient_is_sum_of_gradients(strategy):
    """Implicit gradients through the batched while-loop solver equal the summed per-problem gradients"""
    model, params, _ = simulated_tracking("rnn", n_batch=3)
    UhWh = (params.theta.Uh, params.theta.Wh)

    def single_loss(UhWh, i):
        p = take(params, i)
        p = p._replace(theta=p.theta._replace(Uh=UhWh[0], Wh=UhWh[1]))
        return jnp.sum(dilqr(model, p, zero_inputs(model), max_iter=MAX_ITER) ** 2)

    g_sum = jax.tree.map(lambda *gs: sum(gs), *[jax.grad(single_loss)(UhWh, i) for i in range(3)])
    g_batch = jax.grad(batched_tracking_loss(model, params, strategy))(UhWh)
    assert_grads_close(g_batch, g_sum, rtol=1e-8)


def test_batched_gradient_matches_unrolled():
    """With loop="while" the implicit gradient equals reverse mode through the unrolled loop="scan" solver"""
    model, params, _ = simulated_tracking("rnn", n_batch=2)
    theta = params.theta

    def direct_loss(Uh):
        def one(p):
            (Xs, Us, _), _, _ = ilqr_solver(model, p, zero_inputs(model), max_iter=MAX_ITER, loop="scan")
            return jnp.sum(Xs**2) + jnp.sum(Us**2)

        return jnp.sum(jax.vmap(one, in_axes=TRACKING_AXES[:1])(params._replace(theta=theta._replace(Uh=Uh))))

    def implicit_loss(Uh):
        p = params._replace(theta=theta._replace(Uh=Uh))
        return jnp.sum(batch.batched_dilqr(model, p, zero_inputs(model), TRACKING_AXES, max_iter=MAX_ITER) ** 2)

    assert_grads_close(jax.grad(implicit_loss)(theta.Uh), jax.grad(direct_loss)(theta.Uh))


def test_reverse_mode_through_while_loop_raises():
    """Differentiating the while-loop solver directly is not supported: use dilqr (or loop="scan")"""
    model, params = PROBLEMS["rnn"]()
    loss = lambda x0: ilqr_solver(model, params._replace(x0=x0), zero_inputs(model), max_iter=5)[1]
    with pytest.raises(ValueError, match="Reverse-mode differentiation does not work"):
        jax.grad(loss)(params.x0)


# --------------------------------------------------------------------------------------------
# sharding over (fake CPU) devices
# --------------------------------------------------------------------------------------------

SHARD_SCRIPT = textwrap.dedent(
    """
    import jax, jax.numpy as jnp, chex
    jax.config.update("jax_enable_x64", True)
    assert len(jax.devices()) == 4, jax.devices()
    from diffilqrax import batch
    from problems import TRACKING_AXES, simulated_tracking
    from test_batch import MAX_ITER, assert_solutions_close, batched_tracking_loss, zero_inputs
    from test_gradients import assert_grads_close

    model, params, _ = simulated_tracking("rnn", n_batch=5)
    solve = lambda **kw: batch.batched_ilqr_solver(model, params, zero_inputs(model), TRACKING_AXES, max_iter=MAX_ITER, **kw)
    ref = solve()
    for inner in ("vmap", "map"):
        assert_solutions_close(jax.jit(lambda: solve(strategy="shard", inner=inner))(), ref)
    UhWh = (params.theta.Uh, params.theta.Wh)
    g_ref = jax.grad(batched_tracking_loss(model, params))(UhWh)
    assert_grads_close(jax.grad(batched_tracking_loss(model, params, "shard"))(UhWh), g_ref, rtol=1e-8)
    print("ok")
    """
)


def test_shard_strategy():
    """shard_map over 4 CPU devices, with padding of the odd batch, gives the vmap solutions and gradients"""
    env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=4", "JAX_PLATFORMS": "cpu"}
    res = subprocess.run(
        [sys.executable, "-c", SHARD_SCRIPT], cwd=Path(__file__).parent, env=env, capture_output=True, text=True, check=False
    )
    assert res.returncode == 0 and res.stdout.strip().endswith("ok"), res.stderr[-3000:]


# --------------------------------------------------------------------------------------------
# zebrafish MGU2 fixture
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zebrafish():
    """Model, shared parameters and stacked observations of the 6 fixture bouts"""
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE.name} missing; generate it with dilqr_jax_apps zebrafish/ocaml/make_fixture.py")
    ocaml_params, data = load_fixture(FIXTURE)
    theta = make_theta(ocaml_params, data["saved_tail"][0], N_BEG)
    model = make_system(theta.uh.shape[0], theta.B.shape[0], data["ilqr_u"].shape[1])
    params = iLQRParams(x0=jnp.zeros(theta.uh.shape[0]), theta=theta)
    axes = (iLQRParams(x0=None, theta=jax.tree.map(lambda _: None, theta)._replace(obs=0)), 0)
    return model, params, axes, data


def zebrafish_batch(params, data, bouts):
    return params._replace(theta=params.theta._replace(obs=jnp.asarray(data["saved_tail"][bouts])))


def test_zebrafish_batched_warm_start(zebrafish):
    """Two bouts started near the OCaml optimum: the vmapped solve equals the per-bout solves"""
    model, params, axes, data = zebrafish
    bouts = [0, 5]
    noise = 1e-2 * jax.random.normal(jax.random.PRNGKey(0), data["ilqr_u"][bouts].shape)
    Us_init = jnp.asarray(data["ilqr_u"][bouts]) + noise
    kw = dict(max_iter=30, alpha_init=2.0, kkt_tol=1e-6)
    out = jax.jit(lambda p, u: batch.batched_ilqr_solver(model, p, u, axes, **kw))(
        zebrafish_batch(params, data, bouts), Us_init
    )
    solve = jax.jit(lambda obs, u: ilqr_solver(model, params._replace(theta=params.theta._replace(obs=obs)), u, **kw))
    refs = [solve(jnp.asarray(data["saved_tail"][b]), Us_init[i]) for i, b in enumerate(bouts)]
    assert_solutions_close(out, jax.tree.map(lambda *xs: jnp.stack(xs), *refs), atol=1e-7)


@pytest.mark.slow
def test_zebrafish_all_bouts_from_zero(zebrafish):
    """All 6 bouts from zero inputs in one vmapped solve: converged, with costs within 2.6e-3 of OCaml's"""
    model, params, axes, data = zebrafish
    bouts = onp.arange(data["saved_tail"].shape[0])
    Us_init = jnp.zeros((model.dims.horizon, model.dims.m))
    solve = jax.jit(
        lambda p: batch.batched_ilqr_solver(
            model, p, Us_init, (axes[0], None), max_iter=1000, alpha_init=2.0, kkt_tol=1e-6, return_converged=True
        )
    )
    _, costs, _, converged = solve(zebrafish_batch(params, data, bouts))
    assert bool(jnp.all(converged))
    rel = onp.abs(onp.asarray(costs) - data["ilqr_cost"]) / onp.abs(data["ilqr_cost"])
    assert rel.max() <= 2.6e-3, rel
