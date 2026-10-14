"""
Generate reference LQR solutions from the primal-dual iLQR repository (arXiv:2403.00748), which has
both a sequential (``tvlqr``) and an associative-scan (``tvlqr_gpu``, ``rollout_gpu``) TVLQR solver.

The problems are built here from ``tests/problems.random_lqr``; the solve runs in that repository's
own virtual environment so that its JAX version and dependencies are used. Its conventions match
ours: cost ½xᵀQx + qᵀx + ½uᵀRu + rᵀu + xᵀMu with M = S, policy u = Kx + k, value function
½xᵀPx + pᵀx, so P = J and p = -η.

It also solves two non-linear problems (``tests/problems.pendulum_problem`` and ``rnn_problem``) with
``primal_dual_ilqr.optimizers.primal_dual_ilqr``; their dynamics and costs are re-implemented in the solve
script from the saved parameters.

Usage: ``venv/bin/python benchmarks/make_pdilqr_fixtures.py``
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as onp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from problems import random_lqr, pendulum_problem, rnn_problem  # pylint: disable=wrong-import-position

PDILQR_ROOT = Path.home() / "VSCodeProjects" / "primal_dual_ilqr"
PDILQR_PYTHON = PDILQR_ROOT / "venv" / "bin" / "python"
FIXTURE_DIR = ROOT / "tests" / "fixtures"
SIZES = [(5, 3, 50), (20, 5, 200), (50, 10, 100)]

SOLVE = """
import sys, time
import numpy as onp
import jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, sys.argv[3])
from primal_dual_ilqr.primal_tvlqr import tvlqr, tvlqr_gpu, rollout, rollout_gpu

d = dict(onp.load(sys.argv[1]))
Q = onp.concatenate([d["Q"], d["Qf"][None]])
q = onp.concatenate([d["q"], d["qf"][None]])
args = (Q, q, d["R"], d["r"], d["S"], d["A"], d["B"], d["a"])
out = {}
for name, solver, roll in (("seq", tvlqr, rollout), ("par", tvlqr_gpu, rollout_gpu)):
    t0 = time.perf_counter()
    K, k, P, p = solver(*args)
    X, U = roll(K, k, d["x0"], d["A"], d["B"], d["a"])
    jax.block_until_ready(U)
    out.update({f"{name}_K": K, f"{name}_k": k, f"{name}_P": P, f"{name}_p": p, f"{name}_X": X, f"{name}_U": U,
                f"{name}_seconds": time.perf_counter() - t0})
onp.savez(sys.argv[2], **{k: onp.asarray(v) for k, v in out.items()})
"""


ILQR_PROBLEMS = {
    "pendulum": lambda: pendulum_problem(horizon=50),
    "rnn": lambda: rnn_problem(5, 3),
}

SOLVE_ILQR = """
import sys
import numpy as onp
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, sys.argv[3])
from primal_dual_ilqr.optimizers import primal_dual_ilqr

d = dict(onp.load(sys.argv[1]))
kind, Uh, Wh, x0 = str(d["kind"]), jnp.asarray(d["Uh"]), jnp.asarray(d["Wh"]), jnp.asarray(d["x0"])
T, n = d["U_init"].shape[0], x0.shape[0]
if kind == "pendulum":
    dynamics = lambda x, u, t: jnp.tanh(Uh @ x + Wh @ u) + 1.0
    cost = lambda x, u, t: jnp.sum(x**2) + jnp.where(t == T, 0.0, jnp.sum(u**2))
else:
    dynamics = lambda x, u, t: jnp.tanh(Uh @ x + Wh @ u)

    def cost(x, u, t):
        target = jnp.sin(t / 5.0) * jnp.ones(n)
        running = jnp.sum((x - target) ** 2) + 0.1 * jnp.sum(u**2) + 0.05 * jnp.sum(u**4)
        return jnp.where(t == T, jnp.sum(x**2), running)

X, U, V, iterations, g, c, no_errors = primal_dual_ilqr(
    cost, dynamics, x0, d["X_init"], d["U_init"], onp.zeros_like(d["X_init"]),
    max_iterations=500, slope_threshold=1e-12, c_sq_threshold=1e-24,
)
onp.savez(sys.argv[2], X=X, U=U, V=V, iterations=iterations, cost=g, max_defect=onp.max(onp.abs(c)), no_errors=no_errors)
"""


def make_ilqr_fixture(kind: str) -> Path:
    """Solve a non-linear problem from ``ILQR_PROBLEMS`` with primal_dual_ilqr from 0.1 * N(0, 1) inputs"""
    import jax  # pylint: disable=import-outside-toplevel
    from diffilqrax.ilqr import ilqr_simulate  # pylint: disable=import-outside-toplevel

    jax.config.update("jax_enable_x64", True)
    model, params = ILQR_PROBLEMS[kind]()
    U_init = 0.1 * jax.random.normal(jax.random.PRNGKey(100), (model.dims.horizon, model.dims.m))
    (X_init, _), _ = ilqr_simulate(model, U_init, params)
    problem = {"x0": params.x0, "Uh": params.theta.Uh, "Wh": params.theta.Wh, "X_init": X_init, "U_init": U_init}
    problem = {k: onp.asarray(v) for k, v in problem.items()}
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp, "problem.npz"), Path(tmp, "solution.npz")
        onp.savez(inp, kind=kind, **problem)
        t0 = time.perf_counter()
        subprocess.run([str(PDILQR_PYTHON), "-I", "-c", SOLVE_ILQR, str(inp), str(out), str(PDILQR_ROOT)], check=True, cwd=tmp)
        solution = dict(onp.load(out))
    path = FIXTURE_DIR / f"pdilqr_ilqr_{kind}.npz"
    onp.savez(path, kind=kind, **problem, **solution)
    print(f"wrote {path.name} ({time.perf_counter() - t0:.1f}s, {int(solution['iterations'])} iterations)")
    return path


def make_lqr_fixture(n: int, m: int, horizon: int, seed: int = 0) -> Path:
    """Solve ``random_lqr(n, m, horizon, seed)`` with primal_dual_ilqr and save problem and solution"""
    params = random_lqr(n, m, horizon, seed)
    problem = {"x0": params.x0, **params.lqr._asdict()}
    problem = {k: onp.asarray(v) for k, v in problem.items()}
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp, "problem.npz"), Path(tmp, "solution.npz")
        onp.savez(inp, **problem)
        t0 = time.perf_counter()
        subprocess.run([str(PDILQR_PYTHON), "-I", "-c", SOLVE, str(inp), str(out), str(PDILQR_ROOT)], check=True, cwd=tmp)
        solution = dict(onp.load(out))
    path = FIXTURE_DIR / f"pdilqr_lqr_n{n}_m{m}_T{horizon}.npz"
    onp.savez(path, seed=seed, **problem, **solution)
    print(f"wrote {path.name} ({time.perf_counter() - t0:.1f}s)")
    return path


if __name__ == "__main__":
    for size in SIZES:
        make_lqr_fixture(*size)
    for kind in ILQR_PROBLEMS:
        make_ilqr_fixture(kind)
