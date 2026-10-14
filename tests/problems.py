"""
Shared control problems for the optimality tests, the validation report and the benchmarks.

Each builder returns ``(model, params)`` for iLQR problems, or ``LQRParams`` for LQR problems.
"""

from typing import Tuple

import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jr
from scipy.optimize import minimize

from diffilqrax import ilqr, lqr
from diffilqrax.typs import LQR, LQRParams, ModelDims, ParallelSystem, System, iLQRParams, Theta
from diffilqrax.utils import keygen, initialise_stable_time_varying_dynamics, smooth_l1

jax.config.update("jax_enable_x64", True)

DIMS = [(n, m) for n in (5, 20, 50, 100) for m in (1, 5, 15, 30) if m <= n]
SLOW_SIZE = 750  # n * m at or above which a grid case is marked slow

# --------------------------------------------------------------------------------------------
# LQR problems
# --------------------------------------------------------------------------------------------


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


def exact_lqr_inputs(params: LQRParams) -> jax.Array:
    """Exact optimal inputs from a dense QP solve: the cost is quadratic in U, so one Newton step is exact"""
    horizon, _, m = params.lqr.B.shape
    fun = lambda u: lqr_total_cost(params, u.reshape(horizon, m))
    u0 = jnp.zeros(horizon * m)
    return (u0 - jnp.linalg.solve(jax.hessian(fun)(u0), jax.grad(fun)(u0))).reshape(horizon, m)


def lqr_as_system(params: LQRParams) -> Tuple[System, iLQRParams]:
    """Non-linear System wrapper of an LQR problem, so iLQR can be checked against the exact solution"""
    p = params.lqr
    horizon, n, m = p.B.shape
    model = System(
        lambda t, x, u, th: 0.5 * x @ p.Q[t] @ x + 0.5 * u @ p.R[t] @ u + x @ p.S[t] @ u + p.q[t] @ x + p.r[t] @ u,
        lambda x, th: 0.5 * x @ p.Qf @ x + p.qf @ x,
        lambda t, x, u, th: p.A[t] @ x + p.B[t] @ u + p.a[t],
        ModelDims(n=n, m=m, horizon=horizon, dt=0.1),
    )
    return model, iLQRParams(params.x0, None)


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


def rnn_problem(n: int = 8, m: int = 3, horizon: int = 30, seed: int = 7):
    """Non-convex tanh RNN tracking problem with quartic input costs"""
    k1, k2, k3 = jr.split(jr.PRNGKey(seed), 3)
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


def linear_rnn_problem(n: int = 8, m: int = 3, horizon: int = 30, seed: int = 7):
    """rnn_problem with linear dynamics x' = Uh x + Wh u: the Gauss-Newton and exact implicit gradients coincide"""
    model, params = rnn_problem(n, m, horizon, seed)
    model = System(model.cost, model.costf, lambda t, x, u, th: 0.5 * th.Uh @ x + th.Wh @ u, model.dims)
    return model, params


def ocaml_pendulum(friction: float = 1.0, horizon: int = 2000):
    """
    Exact port of dilqr examples/pendulum.ml: Euler-discretised pendulum with dt=5e-3.

    x = [angle, velocity], u is 2-d but only u[0] drives the system. Running loss
    0.5 * 1e-3 |u|^2 + 2 |x|^2 and zero final loss; ``friction`` is OCaml's ``sum' theta``.
    """
    dt, g, mu = 5e-3, 9.8, 0.01
    theta = jnp.array([friction])

    def dynamics(t, x, u, th):
        dx2 = g * jnp.sin(x[0]) - jnp.sum(th) * mu * x[1] + u[0]
        return x + dt * jnp.stack([x[1], dx2])

    model = System(
        lambda t, x, u, th: 0.5 * 1e-3 * jnp.sum(u**2) + 2.0 * jnp.sum(x**2),
        lambda x, th: 0.0 * jnp.sum(x**2),
        dynamics,
        ModelDims(horizon=horizon, n=2, m=2, dt=dt),
    )
    return model, iLQRParams(x0=jnp.array([2.0, 1.1]), theta=theta)


def vanilla_rnn_params(n: int = 20, m: int = 5, horizon: int = 50, seed: int = 11) -> dict:
    """Parameters of the vanilla RNN problem as numpy arrays, shared with the OCaml reference program"""
    rng = onp.random.default_rng(seed)
    W = 0.9 * rng.standard_normal((n, n)) / onp.sqrt(n)
    B = rng.standard_normal((n, m)) / onp.sqrt(m)
    x0 = 0.5 * rng.standard_normal(n)
    t = onp.arange(horizon)[:, None]
    phases = onp.linspace(0.0, onp.pi, n)[None, :]
    target = 0.5 * onp.sin(2 * onp.pi * t / 25.0 + phases)
    return {"W": W, "B": B, "x0": x0, "target": target}


def vanilla_rnn(n: int = 20, m: int = 5, horizon: int = 50, seed: int = 11, linear: bool = False):
    """
    Vanilla tanh RNN x' = tanh(W x + B u) tracking phase-shifted sinusoids.

    Running loss |x - target_t|^2 + 0.1 |u|^2, final loss |x|^2. Mirrors examples/rnn_ref.ml and
    examples/rnn_grad_ref.ml; ``linear`` drops the tanh (rnn_grad_ref.ml -linear).
    """
    act = (lambda z: z) if linear else jnp.tanh
    p = vanilla_rnn_params(n, m, horizon, seed)
    target = jnp.asarray(p["target"])
    theta = Theta(Uh=jnp.asarray(p["W"]), Wh=jnp.asarray(p["B"]), sigma=jnp.zeros(n), Q=target)
    model = System(
        lambda t, x, u, th: jnp.sum((x - th.Q[t]) ** 2) + 0.1 * jnp.sum(u**2),
        lambda x, th: jnp.sum(x**2),
        lambda t, x, u, th: act(th.Uh @ x + th.Wh @ u),
        ModelDims(horizon=horizon, n=n, m=m, dt=0.1),
    )
    return model, iLQRParams(x0=jnp.asarray(p["x0"]), theta=theta)


def inverted_pendulum(horizon: int = 100, u_max: float = 3.0):
    """
    Single-link pendulum swing-up from hanging down (angle 0) to upright (angle pi).

    The input torque is limited to |u| <= u_max by a smooth convex barrier, so the swing-up needs
    several pumps. Euler dynamics with dt=0.05, unit mass and length, damping 0.1.
    """
    dt, g, damping = 0.05, 9.8, 0.1

    def dynamics(t, x, u, th):
        acc = -g * jnp.sin(x[0]) - damping * x[1] + u[0]
        return x + dt * jnp.stack([x[1], acc])

    def barrier(u):
        return jnp.sum(jax.nn.softplus(5.0 * (u - u_max)) ** 2 + jax.nn.softplus(5.0 * (-u - u_max)) ** 2)

    def cost(t, x, u, th):
        return 0.1 * (1.0 + jnp.cos(x[0])) + 0.01 * x[1] ** 2 + 0.01 * jnp.sum(u**2) + barrier(u)

    def costf(x, th):
        return 50.0 * (1.0 + jnp.cos(x[0])) + 5.0 * x[1] ** 2

    model = System(cost, costf, dynamics, ModelDims(horizon=horizon, n=2, m=1, dt=dt))
    return model, iLQRParams(x0=jnp.array([0.1, 0.0]), theta=None)


def van_der_pol(horizon: int = 100, mu: float = 1.0):
    """Van der Pol oscillator steered from its limit cycle to the origin (Euler, dt=0.05)"""
    dt = 0.05

    def dynamics(t, x, u, th):
        return x + dt * jnp.stack([x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0] + u[0]])

    model = System(
        lambda t, x, u, th: jnp.sum(x**2) + 0.1 * jnp.sum(u**2),
        lambda x, th: 10.0 * jnp.sum(x**2),
        dynamics,
        ModelDims(horizon=horizon, n=2, m=1, dt=dt),
    )
    return model, iLQRParams(x0=jnp.array([2.0, 0.0]), theta=None)


def rnn_tracking(n: int = 50, m: int = 10, horizon: int = 60, seed: int = 3):
    """
    Large tanh RNN tracking a sum of sines with a step change, with non-quadratic costs.

    State error uses log-cosh, inputs pay quadratic + quartic + smooth-L1 costs.
    """
    k1, k2, k3 = jr.split(jr.PRNGKey(seed), 3)
    theta = Theta(
        Uh=1.1 * jr.normal(k1, (n, n)) / jnp.sqrt(n),
        Wh=jr.normal(k2, (n, m)) / jnp.sqrt(m),
        sigma=jnp.zeros(n),
        Q=jnp.eye(n),
    )
    phases = jnp.linspace(0.0, jnp.pi, n)

    def target(t):
        smooth = 0.4 * jnp.sin(2 * jnp.pi * t / 30.0 + phases) + 0.2 * jnp.sin(2 * jnp.pi * t / 11.0)
        return smooth + jnp.where(t >= horizon // 2, 0.3, -0.3)

    def cost(t, x, u, th):
        err = x - target(t)
        return (
            jnp.sum(jnp.log(jnp.cosh(3.0 * err))) / 3.0
            + 0.05 * jnp.sum(u**2)
            + 0.01 * jnp.sum(u**4)
            + 0.05 * smooth_l1(u, 1e-2)
        )

    model = System(
        cost,
        lambda x, th: jnp.sum(x**2),
        lambda t, x, u, th: jnp.tanh(th.Uh @ x + th.Wh @ u),
        ModelDims(horizon=horizon, n=n, m=m, dt=0.1),
    )
    return model, iLQRParams(x0=0.5 * jr.normal(k3, (n,)), theta=theta)


NONCONVEX_COSTS = {
    "convex": lambda t, x, u: jnp.sum((x - 1.0) ** 2) + jnp.sum(u**2),
    "log_u": lambda t, x, u: jnp.sum((x - 1.0) ** 2) + jnp.sum(jnp.log(1 + u**2)) + 0.3 * jnp.sum(x**4),
    "sharp_log_u": lambda t, x, u: jnp.sum((x - 1.0) ** 2) + jnp.sum(jnp.log(1 + (u / 0.2) ** 2)),
    "log_x": lambda t, x, u: jnp.sum(jnp.log(1 + x**2)) + jnp.sum(x**2) + jnp.sum(u**2),
    "track_log": lambda t, x, u: jnp.sum(jnp.log(1 + (x - t / 10) ** 2)) + 0.1 * jnp.sum(u**2),
}


def linear_nonconvex_problem(kind: str = "track_log", n: int = 5, m: int = 3, horizon: int = 50, seed: int = 0):
    """
    Linear dynamics x' = Uh x + Wh u + offset with a non-convex cost from NONCONVEX_COSTS. The parallel
    (linearised) rollout is exact for linear dynamics, so these isolate the effect of an indefinite
    local LQR: "log_u" has an indefinite R ("sharp_log_u" strongly so, wherever |u| > 0.2), "log_x" an
    indefinite Q and "track_log" indefinite Huu.
    """
    k1, k2 = jr.split(jr.PRNGKey(seed))
    theta = Theta(
        Uh=0.5 * jr.normal(k1, (n, n)) / jnp.sqrt(n),
        Wh=0.5 * jr.normal(k2, (n, m)) / jnp.sqrt(m),
        sigma=jnp.zeros(n),
        Q=jnp.eye(n),
    )
    offset = 1.0 if kind == "log_x" else 0.0
    cost = NONCONVEX_COSTS[kind]
    model = System(
        lambda t, x, u, th: cost(t, x, u),
        lambda x, th: jnp.sum(x**2),
        lambda t, x, u, th: th.Uh @ x + th.Wh @ u + offset,
        ModelDims(horizon=horizon, n=n, m=m, dt=0.1),
    )
    return model, iLQRParams(x0=jnp.ones(n), theta=theta)


def as_parallel(model: System) -> ParallelSystem:
    """Wrap a System for the parallel solvers"""
    return ParallelSystem(model)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def total_cost(model: System, params: iLQRParams, Us: jax.Array) -> jax.Array:
    """Total cost of an input sequence"""
    return ilqr.ilqr_simulate(model, Us, params)[1]


def nonlinear_kkt(model, params, Xs, Us, Lambs):
    """KKT residuals of the non-linear problem, via the offset LQR approximation"""
    local = ilqr.approx_lqr_offset(model, Xs, Us, params)
    return lqr.kkt(LQRParams(Xs[0], local), Xs, Us, Lambs)


def lbfgs_reference(model, params, Us_init, gtol: float = 1e-12) -> Tuple[float, onp.ndarray]:
    """Reference optimum (cost, inputs) from L-BFGS on the inputs"""
    horizon, m = Us_init.shape
    fun = jax.jit(jax.value_and_grad(lambda u: total_cost(model, params, u.reshape(horizon, m))))
    res = minimize(
        lambda u: [onp.asarray(v) for v in fun(jnp.asarray(u))],  # pylint: disable=not-callable
        onp.asarray(Us_init).ravel(),
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=50000, maxfun=100000, gtol=gtol, ftol=1e-16),
    )
    return float(res.fun), res.x.reshape(horizon, m)


def perturbation_slope(model, params, Us, epsilons=(1e-1, 1e-2, 1e-3, 1e-4), seed: int = 0):
    """
    Cost increase J(U + eps dU) - J(U) for a random unit direction dU, and its log-log slope in eps.

    At a strict local minimum the increase is positive and scales like eps^2 (slope 2).
    """
    dU = jr.normal(jr.PRNGKey(seed), Us.shape)
    dU = dU / jnp.linalg.norm(dU)
    J0 = total_cost(model, params, Us)
    eps = onp.asarray(epsilons)
    dJ = onp.array([float(total_cost(model, params, Us + e * dU) - J0) for e in eps])
    slope = onp.polyfit(onp.log(eps), onp.log(onp.abs(dJ)), 1)[0]
    return eps, dJ, slope
