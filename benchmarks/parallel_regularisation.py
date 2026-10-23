"""
Effect of adaptive regularisation (R -> R + μI) and PSD projection of the local LQR on the parallel
iLQR solver.

For each linear-dynamics non-convex problem in ``tests/problems.NONCONVEX_COSTS`` and several random
initial inputs, runs the sequential ilqr, pilqr with adaptive μ, pilqr with adaptive μ and the PSD
projection of R ("R") or of R and Q - S R⁻¹Sᵀ ("full"), and pilqr with μ = 0. Records the cost
history, the iterations to converge, the final cost gap to the best solution found and the KKT
residual max|dL/dU|.
Linear dynamics keep the parallel rollout exact, so differences come from the regularisation only.

Usage: ``venv/bin/python -m benchmarks.parallel_regularisation [--seeds 8] [--scale 1.0]``
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

REPO = Path(__file__).resolve().parents[1]
jax.config.update("jax_enable_x64", True)

# pylint: disable=wrong-import-position
from tests.problems import NONCONVEX_COSTS, linear_nonconvex_problem, as_parallel
from diffilqrax import ilqr
from diffilqrax.parallel_ilqr import pilqr_solver

MAX_ITER = 200
METHODS = {
    "ilqr": lambda model, params, U0: ilqr.ilqr_solver(model, params, U0, max_iter=MAX_ITER),
    "pilqr (adaptive μ)": lambda model, params, U0: pilqr_solver(as_parallel(model), params, U0, max_iter=MAX_ITER),
    "pilqr (μ + PSD R)": lambda model, params, U0: pilqr_solver(
        as_parallel(model), params, U0, max_iter=MAX_ITER, make_psd="R"
    ),
    "pilqr (μ + PSD full)": lambda model, params, U0: pilqr_solver(
        as_parallel(model), params, U0, max_iter=MAX_ITER, make_psd="full"
    ),
    "pilqr (μ = 0)": lambda model, params, U0: pilqr_solver(
        as_parallel(model), params, U0, max_iter=MAX_ITER, regularise=False
    ),
}


def kkt_residual(model, params, Xs, Us) -> float:
    """Largest input KKT residual of the local LQR at (Xs, Us)"""
    return float(jnp.max(jnp.abs(ilqr.kkt_input_residual(ilqr.approx_lqr(model, Xs, Us, params)))))


def iterations(costs, J) -> int:
    """Iterations until the cost is within round-off of its final value"""
    return int(jnp.argmax(jnp.abs(costs - J) <= 1e-10 * jnp.abs(J))) + 1


def run(n_seeds: int, scale: float):
    """Solve each non-convex problem from random initial inputs with every method"""
    rows, histories = [], {}
    for kind in NONCONVEX_COSTS:
        model, params = linear_nonconvex_problem(kind)
        for seed in range(n_seeds):
            U0 = scale * jr.normal(jr.PRNGKey(seed + 100), (model.dims.horizon, model.dims.m))
            for name, solve in METHODS.items():
                (Xs, Us, _), cost, costs = solve(model, params, U0)
                rows.append(
                    dict(
                        problem=kind,
                        seed=seed,
                        method=name,
                        cost=float(cost),
                        kkt=kkt_residual(model, params, Xs, Us),
                        iters=iterations(costs, cost),
                    )
                )
                histories[(kind, seed, name)] = jnp.asarray(costs)
            print(kind, seed, "  ".join(f"{r['method']}: J={r['cost']:.8f} kkt={r['kkt']:.1e}" for r in rows[-len(METHODS) :]))
    best = {}
    for r in rows:
        best[r["problem"]] = min(best.get(r["problem"], float("inf")), r["cost"])
    for r in rows:
        r["gap"] = r["cost"] - best[r["problem"]]
    return rows, histories, best


def plot(rows, histories, best, out_dir: Path):
    """Cost histories, final cost gaps and KKT residuals of each method"""
    kinds = list(NONCONVEX_COSTS)
    colours = dict(zip(METHODS, ("k", "tab:blue", "tab:green", "tab:orange", "tab:red")))
    fig, axs = plt.subplots(3, len(kinds), figsize=(4.2 * len(kinds), 11), squeeze=False)
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
            axs[2, j].scatter([i] * len(pts), [r["iters"] for r in pts], color=colours[name], alpha=0.7)
        axs[1, j].set_yscale("log")
        axs[1, j].set_xticks(range(len(METHODS)), list(METHODS), rotation=20, fontsize=8)
        axs[1, j].set(title="final max|dL/dU| (each start)", ylabel="KKT residual")
        axs[2, j].set_xticks(range(len(METHODS)), list(METHODS), rotation=20, fontsize=8)
        axs[2, j].set(title="iterations to converge (each start)", ylabel="iterations")
        axs[0, j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "parallel_regularisation.png", dpi=150)
    plt.close(fig)


def main():
    """Parse arguments, run the comparison and save the CSV and figure"""
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
            iters = sum(r["iters"] for r in pts) / len(pts)
            print(
                f"{kind:12s} {name:21s} KKT<1e-9: {ok}/{len(pts)}  mean iters {iters:6.1f}  "
                f"max gap {max(r['gap'] for r in pts):.2e}"
            )
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
