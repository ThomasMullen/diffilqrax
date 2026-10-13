"""
Parallel (associative-scan) LQR vs sequential LQR: accuracy, KKT residuals and wall time.

The script is device-agnostic: it runs on whatever ``jax.devices()[0]`` is and records the device in
the CSV, so the same sweep can be re-run on a GPU. On a CPU the associative scan does O(T log T)
work with no parallel speedup and is expected to be slower than the sequential scan; the
O(log T) depth only pays off on a parallel accelerator.

Usage: ``venv/bin/python benchmarks/parallel_lqr.py [--quick]``
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "benchmarks"))
jax.config.update("jax_enable_x64", True)

# pylint: disable=wrong-import-position
from problems import random_lqr
from timing import timeit
from diffilqrax import lqr
from diffilqrax.plqr import solve_plqr

SOLVERS = {"lqr": jax.jit(lqr.solve_lqr), "plqr": solve_plqr}


def rel_err(a, b) -> float:
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def kkt_residual(params, sol) -> float:
    """Largest KKT residual relative to the largest entry of the solution"""
    scale = max(float(jnp.max(jnp.abs(x))) for x in sol)
    return max(float(jnp.max(jnp.abs(r))) for r in lqr.kkt(params, *sol)) / scale


def run(sweeps, repeats: int):
    device = str(jax.devices()[0])
    rows = []
    for axis, cases in sweeps.items():
        for n, m, T in cases:
            params = random_lqr(n, m, T)
            sols = {}
            for name, solve in SOLVERS.items():
                compile_t, run_t, sols[name] = timeit(solve, params, repeats=repeats)
                rows.append(
                    dict(
                        axis=axis,
                        solver=name,
                        n=n,
                        m=m,
                        T=T,
                        seconds=run_t,
                        compile=compile_t,
                        kkt=kkt_residual(params, sols[name]),
                        err_X=rel_err(sols[name][0], sols["lqr"][0]),
                        err_U=rel_err(sols[name][1], sols["lqr"][1]),
                        err_lambda=rel_err(sols[name][2], sols["lqr"][2]),
                        device=device,
                    )
                )
            r = rows[-1]
            print(
                f"{axis}: n={n:4} m={m:3} T={T:6}  lqr {rows[-2]['seconds'] * 1e3:9.2f} ms  "
                f"plqr {r['seconds'] * 1e3:9.2f} ms  err U {r['err_U']:.1e}  kkt {r['kkt']:.1e}"
            )
    return rows


def plot(rows, out_dir: Path):
    axes_names = sorted({r["axis"] for r in rows})
    fig, axs = plt.subplots(3, len(axes_names), figsize=(4.5 * len(axes_names), 10), squeeze=False)
    device = rows[0]["device"]
    for j, axis in enumerate(axes_names):
        sub = [r for r in rows if r["axis"] == axis]
        for solver, style in (("lqr", "o-"), ("plqr", "s--")):
            pts = [r for r in sub if r["solver"] == solver]
            x = [r[axis] for r in pts]
            axs[0, j].loglog(x, [r["seconds"] for r in pts], style, label=solver)
            axs[1, j].loglog(x, [r["kkt"] for r in pts], style, label=solver)
        pts = [r for r in sub if r["solver"] == "plqr"]
        for key in ("err_X", "err_U", "err_lambda"):
            axs[2, j].loglog([r[axis] for r in pts], [max(r[key], 1e-17) for r in pts], ".-", label=key)
        axs[0, j].set(title=f"wall time vs {axis} ({device})", ylabel="seconds")
        axs[1, j].set(title="relative KKT residual", ylabel="residual")
        axs[2, j].set(title="plqr relative error vs lqr", ylabel="error", xlabel=axis)
        for ax in axs[:, j]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "parallel_lqr.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="small sweep for a smoke test")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.quick:
        sweeps = {"T": [(5, 3, T) for T in (64, 256)], "n": [(n, 3, 64) for n in (5, 20)]}
    else:
        sweeps = {
            "T": [(10, 5, T) for T in (64, 256, 1024, 4096, 16384)],
            "n": [(n, 5, 1024) for n in (5, 20, 50, 100)],
            "m": [(50, m, 1024) for m in (1, 5, 15, 30)],
        }
    rows = run(sweeps, args.repeats)
    out_dir = REPO / "fig_dump" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "parallel_lqr.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, out_dir)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
