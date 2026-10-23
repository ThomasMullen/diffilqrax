"""
Tests for the associative-scan LQR solver: agreement with the sequential solver and the dense QP
solution across the dimension grid, KKT conditions, the cross-term change of variables, the
associativity of the scan operators, and cross-checks against primal_dual_ilqr (arXiv:2403.00748).
"""

import os
from pathlib import Path

import pytest
import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp

from diffilqrax import lqr
from diffilqrax.plqr import (
    solve_plqr,
    solve_plqr_swap_x0,
    remove_cross_term,
    associative_riccati_scan,
    associative_opt_traj_scan,
    optimal_gains,
    get_dcosts,
    parallel_forward_lin_integration,
    parallel_reverse_lin_integration,
    riccati_operator,
    dynamic_operator,
)
from diffilqrax.typs import LQRParams, ModelDims
from tests.problems import DIMS, SLOW_SIZE, random_lqr, exact_lqr_inputs, lqr_total_cost

jax.config.update("jax_default_device", jax.devices(os.environ.get("DIFFILQRAX_TEST_PLATFORM", "cpu"))[0])
jax.config.update("jax_enable_x64", True)

FIXTURES = Path(__file__).parent / "fixtures"
RTOL = 1e-10
KKT_ATOL = 1e-10


def grid(dims=DIMS):
    """Parametrise over (n, m), marking the large cases slow"""
    return [pytest.param(n, m, marks=pytest.mark.slow) if n * m >= SLOW_SIZE else (n, m) for n, m in dims]


def assert_rel_close(a, b, rtol=RTOL):
    """Normwise relative error ||a - b|| <= rtol ||b||"""
    err, ref = jnp.linalg.norm(a - b), jnp.linalg.norm(b)
    assert err <= rtol * ref, f"relative error {err / ref:.2e} > {rtol:.0e}"


def assert_solutions_close(sol, ref, rtol=RTOL):
    """Each array of sol is close to the matching array of ref"""
    for a, b in zip(sol, ref):
        assert_rel_close(a, b, rtol)


def without_cross_term(params: LQRParams) -> LQRParams:
    """The LQR with its state-input cross term S set to zero"""
    return params._replace(lqr=params.lqr._replace(S=jnp.zeros_like(params.lqr.S)))


# --------------------------------------------------------------------------------------------
# solution
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,m", grid())
@pytest.mark.parametrize("cross_term", [True, False])
def test_plqr_matches_lqr(n, m, cross_term):
    """States, inputs and adjoints agree with the sequential Riccati solver"""
    params = random_lqr(n, m, horizon=50)
    params = params if cross_term else without_cross_term(params)
    assert_solutions_close(solve_plqr(params), lqr.solve_lqr(params))


@pytest.mark.parametrize("horizon", [1, 2, 3, 17, 128, 1000])
def test_plqr_horizons(horizon):
    """Horizons that are short, odd, a power of two and long"""
    params = random_lqr(5, 3, horizon=horizon)
    assert_solutions_close(solve_plqr(params), lqr.solve_lqr(params))


@pytest.mark.parametrize("n,m", [(5, 1), (5, 5), (20, 5)])
def test_plqr_matches_dense_qp(n, m):
    """Inputs equal the dense QP solution, and perturbing them increases the cost"""
    params = random_lqr(n, m, horizon=20)
    _, Us, _ = solve_plqr(params)
    assert_rel_close(Us, exact_lqr_inputs(params), rtol=1e-8)
    dU = 1e-3 * jr.normal(jr.PRNGKey(1), Us.shape)
    assert lqr_total_cost(params, Us + dU) > lqr_total_cost(params, Us)


@pytest.mark.parametrize("n,m", grid())
def test_plqr_kkt(n, m):
    """All KKT residuals vanish to round-off, relative to the scale of the solution"""
    params = random_lqr(n, m, horizon=50)
    Xs, Us, Lambs = solve_plqr(params)
    scale = jnp.max(jnp.abs(jnp.r_[Xs.ravel(), Us.ravel(), Lambs.ravel()]))
    for res in lqr.kkt(params, Xs, Us, Lambs):
        chex.assert_trees_all_close(res / scale, jnp.zeros_like(res), atol=KKT_ATOL, rtol=0.0)


def test_plqr_swap_x0():
    """plqr with a swapped initial state matches the sequential solver"""
    params = random_lqr(5, 3, horizon=30)
    assert_solutions_close(solve_plqr_swap_x0(params), lqr.solve_lqr_swap_x0(params))


def test_plqr_float32():
    """Single precision is accurate to float32 round-off over a moderate horizon"""
    params = jax.tree.map(lambda x: x.astype(jnp.float32), random_lqr(5, 3, horizon=100))
    sol = solve_plqr(params)
    assert all(x.dtype == jnp.float32 for x in sol)
    ref = lqr.solve_lqr(jax.tree.map(lambda x: x.astype(jnp.float64), params))
    assert_solutions_close(sol, ref, rtol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("horizon", [256, 1024, 4096, 16384])
@pytest.mark.parametrize("gain", [1.0, 2.5], ids=["stable", "unstable"])
def test_plqr_long_horizon_accuracy(horizon, gain):
    """The error of the associative scan does not grow with the horizon, in double and single precision, for
    stable dynamics (spectral radius ~0.6) and unstable ones (~1.5). For unstable dynamics the open-loop
    adjoint recursions of both solvers amplify round-off like 1.5^T, so only the states and inputs are compared"""
    params = random_lqr(5, 3, horizon=horizon)
    params = params._replace(lqr=params.lqr._replace(A=gain * params.lqr.A))
    ref = lqr.solve_lqr(params)
    n_compared = 3 if gain == 1.0 else 2
    assert_solutions_close(solve_plqr(params)[:n_compared], ref[:n_compared], rtol=1e-13)
    sol32 = solve_plqr(jax.tree.map(lambda x: x.astype(jnp.float32), params))
    assert_solutions_close(sol32[:n_compared], ref[:n_compared], rtol=1e-5)


# --------------------------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------------------------


def test_remove_cross_term_preserves_cost():
    """Under u = ũ - Gx - g the transformed problem has the same cost up to the constant -½ Σ rᵀR⁻¹r"""
    params = random_lqr(5, 3, horizon=20)
    new_params = LQRParams(params.x0, remove_cross_term(params.lqr)[0])
    _, G, g = remove_cross_term(params.lqr)
    Us_tilde = jr.normal(jr.PRNGKey(0), (20, 3))
    Xs = lqr.simulate_trajectory(lqr.lin_dyn_step, Us_tilde, new_params, ModelDims(5, 3, 20, 0.1))
    Us = Us_tilde - jnp.einsum("tij,tj->ti", G, Xs[:-1]) - g
    assert_rel_close(lqr.simulate_trajectory(lqr.lin_dyn_step, Us, params, ModelDims(5, 3, 20, 0.1)), Xs)
    const = -0.5 * jnp.sum(params.lqr.r * g)
    chex.assert_trees_all_close(lqr_total_cost(new_params, Us_tilde) + const, lqr_total_cost(params, Us), rtol=1e-12)


def random_riccati_elements(key, n, batch=1):
    """Random elements of the associative Riccati scan, with PSD blocks where required"""
    keys = jr.split(key, 5)

    def sym_psd(k):
        L = jr.normal(k, (batch, n, n))
        return L @ L.transpose(0, 2, 1) / n

    return (
        jr.normal(keys[0], (batch, n, n)) / jnp.sqrt(n),
        jr.normal(keys[1], (batch, n)),
        sym_psd(keys[2]),
        jr.normal(keys[3], (batch, n)),
        sym_psd(keys[4]),
    )


@pytest.mark.parametrize("operator,n_parts", [(riccati_operator, 5), (dynamic_operator, 2)])
def test_operator_is_associative(operator, n_parts):
    """(a∘b)∘c == a∘(b∘c) on random elements, the requirement for associative_scan"""
    a, b, c = (random_riccati_elements(k, 4)[:n_parts] for k in jr.split(jr.PRNGKey(0), 3))
    left = operator(operator(a, b), c)
    right = operator(a, operator(b, c))
    chex.assert_trees_all_close(left, right, rtol=1e-10, atol=1e-12)


def test_adjoint_via_reverse_integration():
    """Adjoints from the value functions equal a reverse associative scan of the adjoint equation"""
    params = random_lqr(5, 3, horizon=40)
    Xs, Us, Lambs = solve_plqr(params)
    assert_rel_close(parallel_reverse_lin_integration(params, Xs, Us), Lambs)


def test_forward_integration_matches_closed_loop():
    """Integrating the open-loop inputs with the forward associative scan reproduces the states"""
    params = random_lqr(5, 3, horizon=40)
    Xs, Us, _ = solve_plqr(params)
    assert_rel_close(parallel_forward_lin_integration(params, Us, params.lqr.a), Xs)


def test_gains_match_sequential():
    """Feedback gains, and the expected cost change used by the linesearch, match the sequential pass"""
    params = random_lqr(5, 3, horizon=40)
    etas, Js = associative_riccati_scan(params)
    gains, _ = optimal_gains(params, etas, Js)
    dJ, ref_gains = lqr.lqr_backward_pass(params.lqr)
    assert_rel_close(gains.K, ref_gains.K)
    assert_rel_close(gains.k, ref_gains.k)
    dJ_par = get_dcosts(params, etas, Js)
    chex.assert_trees_all_close(dJ_par.V, dJ.V, rtol=RTOL)
    chex.assert_trees_all_close(dJ_par.v, dJ.v, rtol=RTOL)


def test_linesearch_alpha_scales_feedforward():
    """alpha scales the feedforward term only: u = Kx + αk along the resulting trajectory"""
    params = random_lqr(5, 3, horizon=40)
    etas, Js = associative_riccati_scan(params)
    cs, gains = associative_opt_traj_scan(params, etas, Js, alpha=0.3)
    Xs = jnp.r_[params.x0[None], cs]
    Us = jnp.einsum("tij,tj->ti", gains.K, Xs[:-1]) + 0.3 * gains.k
    assert_rel_close(lqr.simulate_trajectory(lqr.lin_dyn_step, Us, params, ModelDims(5, 3, 40, 0.1)), Xs)


# --------------------------------------------------------------------------------------------
# primal_dual_ilqr cross-check
# --------------------------------------------------------------------------------------------

PDILQR_FIXTURES = sorted(FIXTURES.glob("pdilqr_lqr_*.npz"))


@pytest.mark.skipif(not PDILQR_FIXTURES, reason="run benchmarks/make_pdilqr_fixtures.py")
@pytest.mark.parametrize("path", PDILQR_FIXTURES, ids=lambda p: p.stem)
@pytest.mark.parametrize("variant", ["seq", "par"])
def test_matches_primal_dual_ilqr(path, variant):
    """Value functions, gains and trajectory match primal_dual_ilqr's sequential and parallel TVLQR"""
    d = dict(onp.load(path))
    n, m, horizon = d["B"].shape[1], d["B"].shape[2], d["B"].shape[0]
    params = random_lqr(n, m, horizon, seed=int(d["seed"]))
    chex.assert_trees_all_close(onp.asarray(params.lqr.A), d["A"], rtol=1e-11, atol=1e-13)
    etas, Js = associative_riccati_scan(params)
    gains, _ = optimal_gains(params, etas, Js)
    Xs, Us, _ = solve_plqr(params)
    rtol = 1e-10 if variant == "seq" else 1e-8
    assert_rel_close(Js, d[f"{variant}_P"], rtol)
    assert_rel_close(-etas, d[f"{variant}_p"], rtol)
    assert_rel_close(gains.K, d[f"{variant}_K"], rtol)
    assert_rel_close(gains.k, d[f"{variant}_k"], rtol)
    assert_rel_close(Xs, d[f"{variant}_X"], rtol)
    assert_rel_close(Us, d[f"{variant}_U"], rtol)
