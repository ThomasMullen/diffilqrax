"""
Regularisation of the parallel (associative-scan) LQR and iLQR solvers.

The regulariser R -> R + μI keeps the local problem an exact LQR, so the associative scan solves it
to round-off, and it adds μI to every Huu. pilqr adapts μ as ilqr does; it must leave the solution
unchanged and let pilqr converge where the unregularised local LQR is indefinite.
"""

import pytest
import chex
import jax
import jax.numpy as jnp
import jax.random as jr

from diffilqrax import ilqr, lqr
from diffilqrax.parallel_ilqr import pilqr_solver
from diffilqrax.plqr import plqr_backward_pass, regularise_lqr
from problems import NONCONVEX_COSTS, random_lqr, linear_nonconvex_problem, as_parallel

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

MAX_ITER = 200


def kkt_residual(model, params, Xs, Us) -> float:
    return float(jnp.max(jnp.abs(ilqr.kkt_input_residual(ilqr.approx_lqr(model, Xs, Us, params)))))


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
