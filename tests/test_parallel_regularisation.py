"""
Regularisation of the parallel (associative-scan) LQR and iLQR solvers.

The regulariser R -> R + μI keeps the local problem an exact LQR, so the associative scan solves it
to round-off, and it adds μI to every Huu. pilqr adapts μ as ilqr does; it must leave the solution
unchanged and let pilqr converge where the unregularised local LQR is indefinite. The optional PSD
projection of the local LQR (make_psd) must also leave the solution unchanged.
"""

import os
import pytest
import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from diffilqrax import ilqr, lqr
from diffilqrax.parallel_ilqr import pilqr_solver
from diffilqrax.plqr import plqr_backward_pass, regularise_lqr, project_psd, project_psd_lqr
from problems import NONCONVEX_COSTS, random_lqr, linear_nonconvex_problem, as_parallel

jax.config.update("jax_default_device", jax.devices(os.environ.get("DIFFILQRAX_TEST_PLATFORM", "cpu"))[0])
jax.config.update("jax_enable_x64", True)

MAX_ITER = 200


def kkt_residual(model, params, Xs, Us) -> float:
    return float(jnp.max(jnp.abs(ilqr.kkt_input_residual(ilqr.approx_lqr(model, Xs, Us, params)))))


def iterations(costs, J) -> int:
    """Iterations until the cost is within round-off of its final value"""
    return int(jnp.argmax(jnp.abs(costs - J) <= 1e-10 * jnp.abs(J))) + 1


def solve_both(kind: str, seed: int = 0, scale: float = 0.1, regularise: bool = True):
    model, params = linear_nonconvex_problem(kind, seed=seed)
    Us_init = scale * jr.normal(jr.PRNGKey(seed + 100), (model.dims.horizon, model.dims.m))
    seq = ilqr.ilqr_solver(model, params, Us_init, max_iter=MAX_ITER)
    par = pilqr_solver(as_parallel(model), params, Us_init, max_iter=MAX_ITER, regularise=regularise)
    return model, params, seq, par


@pytest.mark.parametrize("mu", [0.0, 0.1, 10.0])
def test_regularised_backward_pass_is_exact(mu):
    """Gains equal the sequential solution of the regularised LQR, and the expected cost terms match"""
    params = random_lqr(5, 3, horizon=30)
    dJ, gains, is_pd = plqr_backward_pass(params, mu)
    dJ_seq, gains_seq = lqr.lqr_backward_pass(regularise_lqr(params.lqr, mu))
    chex.assert_trees_all_close(gains, gains_seq, rtol=1e-10, atol=1e-12)
    chex.assert_trees_all_close(dJ.v, dJ_seq.v, rtol=1e-10)
    # V uses the unregularised Huu: ½ kᵀ(Huu_reg - μI)k
    chex.assert_trees_all_close(dJ.V, dJ_seq.V - 0.5 * mu * jnp.sum(gains.k**2), rtol=1e-10)
    assert bool(is_pd)


def test_pd_flag():
    """An indefinite R gives an indefinite Huu, which a large enough μ makes positive definite"""
    params = random_lqr(5, 3, horizon=30)
    params = params._replace(lqr=params.lqr._replace(R=params.lqr.R - 5.0 * jnp.eye(3)))
    assert not bool(plqr_backward_pass(params, 0.0)[2])
    assert bool(plqr_backward_pass(params, 100.0)[2])


@pytest.mark.parametrize("kind", list(NONCONVEX_COSTS))
def test_regularised_pilqr_matches_ilqr(kind):
    """With adaptive μ, pilqr converges to the same KKT point as the sequential solver"""
    model, params, ((_, Us, _), J, _), ((Xp, Up, _), Jp, _) = solve_both(kind)
    chex.assert_trees_all_close(Jp, J, rtol=1e-10)
    chex.assert_trees_all_close(Up, Us, atol=1e-6, rtol=1e-6)
    assert kkt_residual(model, params, Xp, Up) < 1e-9


@pytest.mark.parametrize("seed", range(4))
def test_regularised_pilqr_multistart(seed):
    """Large random initial inputs on the tracking problem: pilqr still reaches a KKT point"""
    model, params, _, ((Xp, Up, _), _, _) = solve_both("track_log", seed=seed, scale=1.0)
    assert kkt_residual(model, params, Xp, Up) < 1e-9


def test_unregularised_pilqr_stalls_on_indefinite_problem():
    """Without regularisation the Newton step is not a descent direction once Huu is indefinite, the
    line search fails and pilqr stops far from a stationary point"""
    model, params, (_, J, _), ((Xp, Up, _), Jp, _) = solve_both("track_log", regularise=False)
    assert float(Jp) > float(J) + 1.0
    assert kkt_residual(model, params, Xp, Up) > 1e-2


def test_project_psd():
    """Eigenvalues are clipped at delta, eigenvectors kept, and PD matrices above delta are unchanged"""
    M = jr.normal(jr.PRNGKey(0), (7, 4, 4))
    M = M + M.transpose(0, 2, 1)
    P = project_psd(M, 0.1)
    evals, evecs = jnp.linalg.eigh(M)
    chex.assert_trees_all_close(jnp.linalg.eigvalsh(P), jnp.maximum(evals, 0.1), atol=1e-12)
    chex.assert_trees_all_close(P, P.transpose(0, 2, 1))
    # same eigenvectors: P v = max(λ, δ) v
    chex.assert_trees_all_close(P @ evecs, evecs * jnp.maximum(evals, 0.1)[:, None, :], atol=1e-12)
    pd = M @ M.transpose(0, 2, 1) + jnp.eye(4)
    chex.assert_trees_all_close(project_psd(pd, 0.1), pd, atol=1e-12)


@pytest.mark.parametrize("which", ["R", "full"])
def test_project_psd_lqr(which):
    """The projected local LQR is convex where requested and keeps the linear terms"""
    params = random_lqr(5, 3, horizon=20)
    lqr_ = params.lqr._replace(R=params.lqr.R - 3.0 * jnp.eye(3), Q=params.lqr.Q - 3.0 * jnp.eye(5))
    proj = project_psd_lqr(lqr_, 1e-3, which)
    assert float(jnp.min(jnp.linalg.eigvalsh(proj.R))) >= 1e-3 - 1e-10
    chex.assert_trees_all_close((proj.q, proj.r, proj.S, proj.A), (lqr_.q, lqr_.r, lqr_.S, lqr_.A))
    schur = proj.Q - proj.S @ jnp.linalg.solve(proj.R, proj.S.transpose(0, 2, 1))
    if which == "full":
        assert float(jnp.min(jnp.linalg.eigvalsh(schur))) >= 1e-3 - 1e-9
        assert float(jnp.min(jnp.linalg.eigvalsh(proj.Qf))) >= 1e-3 - 1e-10
        # strictly convex: the unregularised scan is valid
        assert bool(plqr_backward_pass(params._replace(lqr=proj), 0.0)[2])
    else:
        chex.assert_trees_all_close((proj.Q, proj.Qf), (lqr_.Q, lqr_.Qf))


@pytest.mark.parametrize("make_psd", ["R", "full"])
@pytest.mark.parametrize("kind", ["sharp_log_u", "track_log"])
def test_projected_pilqr_reaches_kkt_point(kind, make_psd):
    """Projection changes the step but not the stationary points: pilqr still converges to KKT"""
    model, params = linear_nonconvex_problem(kind)
    Us_init = jr.normal(jr.PRNGKey(100), (model.dims.horizon, model.dims.m))
    (Xp, Up, _), _, _ = pilqr_solver(as_parallel(model), params, Us_init, max_iter=MAX_ITER, make_psd=make_psd)
    assert kkt_residual(model, params, Xp, Up) < 1e-9


@pytest.mark.parametrize("kind", ["convex", "track_log"])
def test_input_projection_inactive_when_R_is_pd(kind):
    """With R ≻ δI the input projection does nothing: identical iterates to the unprojected solver"""
    model, params = linear_nonconvex_problem(kind)
    Us_init = jr.normal(jr.PRNGKey(100), (model.dims.horizon, model.dims.m))
    solve = lambda **kw: pilqr_solver(as_parallel(model), params, Us_init, max_iter=MAX_ITER, **kw)
    chex.assert_trees_all_close(solve(make_psd="R"), solve(), rtol=1e-12, atol=1e-12)


def test_input_projection_speeds_up_indefinite_R():
    """Strongly indefinite R from a distant start: projecting R needs far fewer iterations than raising
    μ alone, and finds the same solution"""
    model, params = linear_nonconvex_problem("sharp_log_u")
    Us_init = 10.0 * jr.normal(jr.PRNGKey(100), (model.dims.horizon, model.dims.m))
    solve = lambda **kw: pilqr_solver(as_parallel(model), params, Us_init, max_iter=300, **kw)
    _, J, costs = solve()
    (Xp, Up, _), Jp, costs_p = solve(make_psd="R")
    assert kkt_residual(model, params, Xp, Up) < 1e-9
    chex.assert_trees_all_close(Jp, J, rtol=1e-10)
    assert 2 * iterations(costs_p, Jp) < iterations(costs, J)
