"""
Effect of adaptive regularisation (R -> R + μI) on the parallel iLQR solver.

For each linear-dynamics non-convex problem in ``tests/problems.NONCONVEX_COSTS`` and several random
initial inputs, runs the sequential ilqr, pilqr with adaptive μ and pilqr with μ = 0, and records the
cost history, the final cost gap to the best solution found and the KKT residual max|dL/dU|.
Linear dynamics keep the parallel rollout exact, so differences come from the regularisation only.

Usage: ``venv/bin/python benchmarks/parallel_regularisation.py [--seeds 8]``
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
jax.config.update("jax_enable_x64", True)

# pylint: disable=wrong-import-position
from problems import NONCONVEX_COSTS, linear_nonconvex_problem, as_parallel
from diffilqrax import ilqr
from diffilqrax.parallel_ilqr import pilqr_solver

MAX_ITER = 200
METHODS = {
    "ilqr": lambda model, params, U0: ilqr.ilqr_solver(model, params, U0, max_iter=MAX_ITER),
    "pilqr (adaptive μ)": lambda model, params, U0: pilqr_solver(as_parallel(model), params, U0, max_iter=MAX_ITER),
    "pilqr (μ = 0)": lambda model, params, U0: pilqr_solver(
        as_parallel(model), params, U0, max_iter=MAX_ITER, regularise=False
    ),
}


def kkt_residual(model, params, Xs, Us) -> float:
    return float(jnp.max(jnp.abs(ilqr.kkt_input_residual(ilqr.approx_lqr(model, Xs, Us, params)))))


def run(n_seeds: int, scale: float):
    rows, histories = [], {}
    for kind in NONCONVEX_COSTS:
        model, params = linear_nonconvex_problem(kind)
        for seed in range(n_seeds):
            U0 = scale * jr.normal(jr.PRNGKey(seed + 100), (model.dims.horizon, model.dims.m))
            for name, solve in METHODS.items():
                (Xs, Us, _), cost, costs = solve(model, params, U0)
                rows.append(
                    dict(problem=kind, seed=seed, method=name, cost=float(cost), kkt=kkt_residual(model, params, Xs, Us))
                )
                histories[(kind, seed, name)] = jnp.asarray(costs)
            print(kind, seed, "  ".join(f"{r['method']}: J={r['cost']:.8f} kkt={r['kkt']:.1e}" for r in rows[-3:]))
    best = {}
    for r in rows:
        best[r["problem"]] = min(best.get(r["problem"], float("inf")), r["cost"])
    for r in rows:
        r["gap"] = r["cost"] - best[r["problem"]]
    return rows, histories, best


def plot(rows, histories, best, out_dir: Path):
    kinds = list(NONCONVEX_COSTS)
    colours = dict(zip(METHODS, ("k", "tab:blue", "tab:red")))
    fig, axs = plt.subplots(2, len(kinds), figsize=(4.2 * len(kinds), 7.5), squeeze=False)
    for j, kind in enumerate(kinds):
        for (k, seed, name), costs in histories.items():
            if k != kind:
                continue
            gap = jnp.maximum(costs - best[kind], 1e-14)
            axs[0, j].semilogy(gap, color=colours[name], alpha=0.6, lw=1, label=name if seed == 0 else None)
        axs[0, j].set(title=f"{kind}: cost - best", xlabel="iteration", ylabel="J - J*")
        for i, name in enumerate(METHODS):
            pts = [r for r in rows if r["problem"] == kind and r["method"] == name]
            axs[1, j].scatter([i] * len(pts), [max(r["kkt"], 1e-16) for r in pts], color=colours[name], alpha=0.7)
        axs[1, j].set_yscale("log")
        axs[1, j].set_xticks(range(len(METHODS)), list(METHODS), rotation=20, fontsize=8)
        axs[1, j].set(title="final max|dL/dU| (each start)", ylabel="KKT residual")
        axs[0, j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "parallel_regularisation.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.0, help="std of the random initial inputs")
    args = parser.parse_args()
    rows, histories, best = run(args.seeds, args.scale)
    out_dir = REPO / "fig_dump" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "parallel_regularisation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, histories, best, out_dir)
    for kind in NONCONVEX_COSTS:
        for name in METHODS:
            pts = [r for r in rows if r["problem"] == kind and r["method"] == name]
            ok = sum(r["kkt"] < 1e-9 for r in pts)
            print(f"{kind:10s} {name:20s} KKT<1e-9: {ok}/{len(pts)}  max gap {max(r['gap'] for r in pts):.2e}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
