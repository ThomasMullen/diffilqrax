"""
Generate reference LQR solutions from the primal-dual iLQR repository (arXiv:2403.00748), which has
both a sequential (``tvlqr``) and an associative-scan (``tvlqr_gpu``, ``rollout_gpu``) TVLQR solver.

The problems are built here from ``tests/problems.random_lqr``; the solve runs in that repository's
own virtual environment so that its JAX version and dependencies are used. Its conventions match
ours: cost ½xᵀQx + qᵀx + ½uᵀRu + rᵀu + xᵀMu with M = S, policy u = Kx + k, value function
½xᵀPx + pᵀx, so P = J and p = -η.

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

from problems import random_lqr  # pylint: disable=wrong-import-position

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
