"""
Parallel (associative-scan) LQR vs sequential LQR: accuracy, KKT residuals and wall time.

With ``--grad``, times value_and_grad of the differentiable solvers instead: dlqr vs pdlqr with the
sequential and parallel adjoint, and dilqr vs pdilqr (on linear dynamics with a non-quadratic cost, which
pilqr solves exactly), and records each gradient's relative error against the sequential solver.

The script is device-agnostic: it runs on whatever ``jax.devices()[0]`` is and records the device in
the CSV, so the same sweep can be re-run on a GPU. On a CPU the associative scan does O(T log T)
work with no parallel speedup and is expected to be slower than the sequential scan; the
O(log T) depth only pays off on a parallel accelerator.

Usage: ``venv/bin/python benchmarks/parallel_lqr.py [--quick] [--grad]``
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
from problems import random_lqr, linear_nonconvex_problem, as_parallel
from timing import timeit
from diffilqrax import lqr
from diffilqrax.plqr import solve_plqr
from diffilqrax.typs import ModelDims
from diffilqrax.diff_lqr import dlqr
from diffilqrax.diff_ilqr import dilqr
from diffilqrax.parallel_dlqr import pdlqr
from diffilqrax.parallel_dilqr import pdilqr

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


def grad_err(g, g_ref) -> float:
    """Largest normwise relative error over the gradient leaves"""
    leaves = zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(g_ref))
    return max(float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-300)) for a, b in leaves)


def lqr_grad_fns(n: int, m: int, T: int):
    """value_and_grad of a smooth loss of the trajectory, through dlqr and pdlqr"""
    dims = ModelDims(n, m, T, dt=0.1)
    weights = jax.random.normal(jax.random.PRNGKey(5), (T + 1, n + m))
    loss = lambda tau: jnp.sum(weights * jnp.sin(tau))
    fns = {
        "dlqr": lambda p: loss(dlqr(dims, p, None)),
        "pdlqr (seq adjoint)": lambda p: loss(pdlqr(dims, p, adjoint="sequential")),
        "pdlqr (par adjoint)": lambda p: loss(pdlqr(dims, p, adjoint="parallel")),
    }
    return random_lqr(n, m, T), {k: jax.jit(jax.value_and_grad(f)) for k, f in fns.items()}


def ilqr_grad_fns(n: int, m: int, T: int):
    """value_and_grad of sum(tau**2) through dilqr and pdilqr on the log_u problem (linear dynamics)"""
    model, params = linear_nonconvex_problem("log_u", n=n, m=m, horizon=T)
    pmodel = as_parallel(model)
    Us_init = 0.1 * jax.random.normal(jax.random.PRNGKey(100), (T, m))
    fns = {
        "dilqr": lambda p: jnp.sum(dilqr(model, p, Us_init, max_iter=100) ** 2),
        "pdilqr (seq adjoint)": lambda p: jnp.sum(pdilqr(pmodel, p, Us_init, adjoint="sequential", max_iter=100) ** 2),
        "pdilqr (par adjoint)": lambda p: jnp.sum(pdilqr(pmodel, p, Us_init, adjoint="parallel", max_iter=100) ** 2),
    }
    return params, {k: jax.jit(jax.value_and_grad(f)) for k, f in fns.items()}


def run_grad(sweeps, repeats: int):
    device = str(jax.devices()[0])
    rows = []
    for axis, cases in sweeps.items():
        kind = axis.split("_")[0]
        for n, m, T in cases:
            params, fns = (lqr_grad_fns if kind == "lqr" else ilqr_grad_fns)(n, m, T)
            ref = None
            line = f"{axis}: n={n:4} m={m:3} T={T:6} "
            for name, fn in fns.items():
                compile_t, run_t, (_, g) = timeit(fn, params, repeats=repeats)
                ref = g if ref is None else ref
                rows.append(
                    dict(
                        axis=axis,
                        solver=name,
                        n=n,
                        m=m,
                        T=T,
                        seconds=run_t,
                        compile=compile_t,
                        grad_err=grad_err(g, ref),
                        device=device,
                    )
                )
                line += f" {name} {run_t * 1e3:9.2f} ms (err {rows[-1]['grad_err']:.1e})"
            print(line)
    return rows


def plot_grad(rows, out_dir: Path):
    axes_names = list(dict.fromkeys(r["axis"] for r in rows))
    fig, axs = plt.subplots(2, len(axes_names), figsize=(4.5 * len(axes_names), 7), squeeze=False)
    device = rows[0]["device"]
    for j, axis in enumerate(axes_names):
        sub = [r for r in rows if r["axis"] == axis]
        x_key = axis.split("_")[1]
        for solver, style in zip(dict.fromkeys(r["solver"] for r in sub), ("o-", "s--", "^:")):
            pts = [r for r in sub if r["solver"] == solver]
            x = [r[x_key] for r in pts]
            axs[0, j].loglog(x, [r["seconds"] for r in pts], style, label=solver)
            axs[1, j].loglog(x, [max(r["grad_err"], 1e-17) for r in pts], style, label=solver)
        axs[0, j].set(title=f"value_and_grad time vs {x_key} ({device})", ylabel="seconds")
        axs[1, j].set(title="gradient error vs sequential", ylabel="relative error", xlabel=x_key)
        for ax in axs[:, j]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "parallel_lqr_grad.png", dpi=150)
    plt.close(fig)


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
    parser.add_argument("--grad", action="store_true", help="time gradients of the differentiable solvers")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory, by default fig_dump/<timestamp>")
    args = parser.parse_args()
    if args.grad:
        if args.quick:
            sweeps = {"lqr_T": [(5, 3, T) for T in (64, 256)], "ilqr_T": [(5, 3, T) for T in (64, 256)]}
        else:
            sweeps = {
                "lqr_T": [(10, 5, T) for T in (64, 256, 1024, 4096, 16384)],
                "lqr_n": [(n, 5, 1024) for n in (5, 20, 50, 100)],
                "ilqr_T": [(5, 3, T) for T in (64, 256, 1024, 4096)],
            }
        rows, name, plot_fn = run_grad(sweeps, args.repeats), "parallel_lqr_grad", plot_grad
    else:
        if args.quick:
            sweeps = {"T": [(5, 3, T) for T in (64, 256)], "n": [(n, 3, 64) for n in (5, 20)]}
        else:
            sweeps = {
                "T": [(10, 5, T) for T in (64, 256, 1024, 4096, 16384)],
                "n": [(n, 5, 1024) for n in (5, 20, 50, 100)],
                "m": [(50, m, 1024) for m in (1, 5, 15, 30)],
            }
        rows, name, plot_fn = run(sweeps, args.repeats), "parallel_lqr", plot
    out_dir = args.out_dir or REPO / "fig_dump" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_fn(rows, out_dir)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
