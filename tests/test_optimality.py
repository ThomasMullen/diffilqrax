"""
Optimality tests for the LQR and iLQR solvers: KKT conditions across dimensions, agreement with
exact solutions, second-order conditions, multi-start agreement, the expected cost change model,
smooth L1 sparsity costs, and float32 behaviour.
"""

import math
import numpy as onp
import pytest
import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from scipy.optimize import minimize

from diffilqrax import ilqr, lqr
from diffilqrax.diff_ilqr import add_dynamics_curvature
from diffilqrax.typs import LQR, LQRParams, ModelDims, System, iLQRParams, Theta
from diffilqrax.utils import keygen, initialise_stable_time_varying_dynamics, smooth_l1

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

KKT_ATOL = 1e-8
DIMS = [(n, m) for n in (5, 20, 50, 100) for m in (1, 5, 15, 30) if m <= n]


def random_lqr(n: int, m: int, horizon: int, seed: int = 0) -> LQRParams:
    """Well-posed time-varying LQR: stable dynamics, positive definite stage costs with cross term"""
    _, skeys = keygen(jr.PRNGKey(seed), 8)
    A = initialise_stable_time_varying_dynamics(next(skeys), n, horizon, radii=0.6)
    B = jr.normal(next(skeys), (horizon, n, m)) / jnp.sqrt(n)
    a = 0.1 * jr.normal(next(skeys), (horizon, n))
    L = jr.normal(next(skeys), (horizon, n + m, n + m)) / jnp.sqrt(n + m)
    C = L @ L.transpose(0, 2, 1) + 0.1 * jnp.eye(n + m)
    q = jr.normal(next(skeys), (horizon, n))
    r = jr.normal(next(skeys), (horizon, m))
    Qf = jnp.eye(n)
    qf = jr.normal(next(skeys), (n,))
    lqr_struct = LQR(A, B, a, C[:, :n, :n], q, C[:, n:, n:], r, C[:, :n, n:], Qf, qf)()
    return LQRParams(jr.normal(next(skeys), (n,)), lqr_struct)


def lqr_total_cost(params: LQRParams, Us: jax.Array) -> jax.Array:
    """Total LQR cost as a function of the inputs, with cost 0.5 [x;u]^T C [x;u] + c^T [x;u]"""
    p = params.lqr

    def step(x, inps):
        A, B, a, Q, q, R, r, S, u = inps
        c = 0.5 * x @ Q @ x + 0.5 * u @ R @ u + x @ S @ u + q @ x + r @ u
        return A @ x + B @ u + a, c

    xf, cs = jax.lax.scan(step, params.x0, (*p[:-2], Us))
    return jnp.sum(cs) + 0.5 * xf @ p.Qf @ xf + p.qf @ xf


def exact_lqr_inputs(params: LQRParams, horizon: int, m: int) -> jax.Array:
    """Exact optimal inputs: the cost is quadratic in U, so a single Newton step is exact"""
    fun = lambda u: lqr_total_cost(params, u.reshape(horizon, m))
    u0 = jnp.zeros(horizon * m)
    return (u0 - jnp.linalg.solve(jax.hessian(fun)(u0), jax.grad(fun)(u0))).reshape(horizon, m)


@pytest.mark.parametrize(
    "n,m", [pytest.param(n, m, marks=pytest.mark.slow) if n * m >= 750 else (n, m) for n, m in DIMS]
)
def test_lqr_kkt_and_exact_solution(n, m):
    """LQR satisfies all KKT conditions and matches the exact quadratic program solution"""
    horizon = 20
    params = random_lqr(n, m, horizon)
    Xs, Us, Lambs = lqr.solve_lqr(params)
    dLdXs, dLdUs, dLdLambs = lqr.kkt(params, Xs, Us, Lambs)
    chex.assert_trees_all_close(dLdXs, jnp.zeros_like(dLdXs), atol=KKT_ATOL, rtol=0.0)
    chex.assert_trees_all_close(dLdUs, jnp.zeros_like(dLdUs), atol=KKT_ATOL, rtol=0.0)
    chex.assert_trees_all_close(dLdLambs, jnp.zeros_like(dLdLambs), atol=KKT_ATOL, rtol=0.0)
    chex.assert_trees_all_close(Us, exact_lqr_inputs(params, horizon, m), atol=1e-8, rtol=1e-8)


def test_expected_cost_change_is_exact_for_lqr():
    """The quadratic model of the cost reduction is exact when the problem is an LQR"""
    n, m, horizon = 6, 3, 15
    params = random_lqr(n, m, horizon, seed=1)
    p = params.lqr
    Us = jnp.zeros((horizon, m))
    # rollout and linearise around Us=0 with the iLQR helpers via an equivalent System
    model = System(
        lambda t, x, u, th: 0.5 * x @ p.Q[t] @ x + 0.5 * u @ p.R[t] @ u + x @ p.S[t] @ u + p.q[t] @ x + p.r[t] @ u,
        lambda x, th: 0.5 * x @ p.Qf @ x + p.qf @ x,
        lambda t, x, u, th: p.A[t] @ x + p.B[t] @ u + p.a[t],
        ModelDims(n=n, m=m, horizon=horizon, dt=0.1),
    )
    ilqr_params = iLQRParams(params.x0, None)
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


def regularised_lqr_needs_mu() -> LQR:
    """LQR with an indefinite input cost so mu > 0 is required for a positive definite Huu"""
    params = random_lqr(4, 2, 5, seed=2)
    return params.lqr._replace(R=params.lqr.R - 5.0 * jnp.eye(2))


def test_backward_pass_pd_flag():
    """The positive definite flag detects an indefinite Huu and regularisation restores it"""
    lqr_struct = regularised_lqr_needs_mu()
    assert not bool(lqr.regularised_backward_pass(lqr_struct, mu=0.0)[2])
    assert bool(lqr.regularised_backward_pass(lqr_struct, mu=1e3)[2])


# --------------------------------------------------------------------------------------------
# non-linear problems
# --------------------------------------------------------------------------------------------


def pendulum_problem(horizon: int = 100):
    """tanh pendulum-like system used throughout test_ilqr.py"""
    dt = 0.1
    Uh = jnp.array([[1, dt], [-1 * dt, 1 - 0.5 * dt]])
    Wh = jnp.array([[0, 0], [1, 0]]) * dt
    theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros(2), Q=jnp.eye(2))
    model = System(
        lambda t, x, u, th: jnp.sum(x**2) + jnp.sum(u**2),
        lambda x, th: jnp.sum(x**2),
        lambda t, x, u, th: jnp.tanh(th.Uh @ x + th.Wh @ u) + jnp.ones(2),
        ModelDims(horizon=horizon, n=2, m=2, dt=dt),
    )
    return model, iLQRParams(x0=jnp.array([0.3, 0.0]), theta=theta)


def rnn_problem(n: int = 8, m: int = 3, horizon: int = 30):
    """Non-convex tanh RNN tracking problem with quartic costs"""
    key = jr.PRNGKey(7)
    k1, k2, k3 = jr.split(key, 3)
    theta = Theta(
        Uh=1.2 * jr.normal(k1, (n, n)) / jnp.sqrt(n),
        Wh=jr.normal(k2, (n, m)) / jnp.sqrt(n),
        sigma=jnp.zeros(n),
        Q=jnp.eye(n),
    )

    def cost(t, x, u, th):
        target = jnp.sin(t / 5.0) * jnp.ones(n)
        return jnp.sum((x - target) ** 2) + 0.1 * jnp.sum(u**2) + 0.05 * jnp.sum(u**4)

    model = System(
        cost,
        lambda x, th: jnp.sum(x**2),
        lambda t, x, u, th: jnp.tanh(th.Uh @ x + th.Wh @ u),
        ModelDims(horizon=horizon, n=n, m=m, dt=0.1),
    )
    return model, iLQRParams(x0=jr.normal(k3, (n,)), theta=theta)


def nonlinear_kkt(model, params, Xs, Us, Lambs):
    """KKT residuals of the non-linear problem, via the offset LQR approximation"""
    local = ilqr.approx_lqr_offset(model, Xs, Us, params)
    return lqr.kkt(LQRParams(Xs[0], local), Xs, Us, Lambs)


def lbfgs_reference(model, params, Us_init):
    """Reference optimum from L-BFGS on the inputs"""
    horizon, m = Us_init.shape
    fun = jax.jit(jax.value_and_grad(lambda u: ilqr.ilqr_simulate(model, u.reshape(horizon, m), params)[1]))
    res = minimize(
        lambda u: [onp.asarray(v) for v in fun(jnp.asarray(u))],
        onp.asarray(Us_init).ravel(),
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=20000, gtol=1e-12, ftol=1e-16),
    )
    return res.fun


@pytest.mark.parametrize(
    "problem,use_linesearch",
    # full α=1 steps on the non-convex RNN only converge linearly (Gauss-Newton overshoot), so
    # the no-linesearch mode is checked on the pendulum only
    [(pendulum_problem, True), (pendulum_problem, False), (rnn_problem, True)],
)
def test_ilqr_kkt_and_second_order(problem, use_linesearch):
    """iLQR converges to a point satisfying KKT (atol 1e-8) with a positive definite reduced Hessian"""
    model, params = problem()
    Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (model.dims.horizon, model.dims.m))
    (Xs, Us, Lambs), cost, _ = ilqr.ilqr_solver(model, params, Us_init, max_iter=100, use_linesearch=use_linesearch)
    for res in nonlinear_kkt(model, params, Xs, Us, Lambs):
        chex.assert_trees_all_close(res, jnp.zeros_like(res), atol=KKT_ATOL, rtol=0.0)
    # second-order condition: Lagrangian Hessian LQR has a positive definite Huu at every step
    local = add_dynamics_curvature(model, ilqr.approx_lqr_offset(model, Xs, Us, params), Xs, Us, Lambs, params)
    assert bool(lqr.regularised_backward_pass(local, mu=0.0)[2])
    # matches an independent optimiser
    assert jnp.allclose(cost, lbfgs_reference(model, params, Us_init), rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("problem", [pendulum_problem, rnn_problem])
def test_ilqr_multistart_agreement(problem):
    """Different initial inputs reach the same optimal cost and trajectory"""
    model, params = problem()
    sols = []
    for seed in range(4):
        Us_init = 0.5 * jr.normal(jr.PRNGKey(seed), (model.dims.horizon, model.dims.m))
        sols.append(ilqr.ilqr_solver(model, params, Us_init, max_iter=100)[:2])
    for (Xs_Us_Ls, cost) in sols[1:]:
        chex.assert_trees_all_close(cost, sols[0][1], rtol=1e-10)
        chex.assert_trees_all_close(Xs_Us_Ls[1], sols[0][0][1], atol=1e-6)


def test_ilqr_recovers_lqr_solution():
    """iLQR on a problem that is exactly an LQR matches the LQR solution"""
    n, m, horizon = 6, 3, 15
    params = random_lqr(n, m, horizon, seed=3)
    p = params.lqr
    model = System(
        lambda t, x, u, th: 0.5 * x @ p.Q[t] @ x + 0.5 * u @ p.R[t] @ u + x @ p.S[t] @ u + p.q[t] @ x + p.r[t] @ u,
        lambda x, th: 0.5 * x @ p.Qf @ x + p.qf @ x,
        lambda t, x, u, th: p.A[t] @ x + p.B[t] @ u + p.a[t],
        ModelDims(n=n, m=m, horizon=horizon, dt=0.1),
    )
    (Xs, Us, Lambs), _, _ = ilqr.ilqr_solver(model, iLQRParams(params.x0, None), jnp.zeros((horizon, m)))
    Xs_lqr, Us_lqr, Lambs_lqr = lqr.solve_lqr(params)
    chex.assert_trees_all_close((Xs, Us, Lambs), (Xs_lqr, Us_lqr, Lambs_lqr), atol=1e-9, rtol=1e-9)


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
        for res in nonlinear_kkt(sparse_model, params, Xs, Us, Lambs):
            chex.assert_trees_all_close(res, jnp.zeros_like(res), atol=KKT_ATOL, rtol=0.0)
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
    Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (model.dims.horizon, model.dims.m), dtype=jnp.float32)
    (Xs, Us, Lambs), cost, _ = ilqr.ilqr_solver(model, params, Us_init, max_iter=100, kkt_tol=1e-3)
    _, cost_init = ilqr.ilqr_simulate(model, Us_init, params)
    assert cost < cost_init
    dLdUs = nonlinear_kkt(model, params, Xs, Us, Lambs)[1]
    assert float(jnp.max(jnp.abs(dLdUs))) < 1e-3
