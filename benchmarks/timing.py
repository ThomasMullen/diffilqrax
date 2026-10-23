"""
Wall-time scaling of the LQR and iLQR solvers and of their gradients.

Measures (post-JIT, median of repeats, on CPU):
  (a) solve_lqr vs state dim n, input dim m and horizon T
  (b) ilqr_solver time and iterations vs n and T on the tanh RNN problem
  (c) implicit (dilqr) vs direct-unrolled gradient time vs n and T

Run from the repository root:
    python -m benchmarks.timing [--quick]

Writes timing.csv and timing.png to fig_dump/<timestamp>/.
"""

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

REPO = Path(__file__).resolve().parents[1]
# pylint: disable=wrong-import-position
from tests.problems import random_lqr, rnn_problem
from diffilqrax import lqr
from diffilqrax.ilqr import ilqr_solver
from diffilqrax.diff_ilqr import dilqr
from diffilqrax.typs import iLQRParams

jax.config.update("jax_enable_x64", True)


def timeit(fn, *args, repeats: int = 5):
    """Compile time and median post-JIT wall time of fn(*args)"""
    start = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    compile_time = time.perf_counter() - start
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return compile_time, statistics.median(times), out


def bench_lqr(sizes, rows):
    """solve_lqr vs n (m = 5, T = 100), m (n = 50, T = 100) and T (n = 20, m = 5)"""
    solve = jax.jit(lqr.solve_lqr)
    sweeps = {
        "n": [(n, 5, 100) for n in sizes["lqr_n"]],
        "m": [(50, m, 100) for m in sizes["lqr_m"]],
        "T": [(20, 5, T) for T in sizes["lqr_T"]],
    }
    for axis, cases in sweeps.items():
        for n, m, T in cases:
            compile_t, run_t, _ = timeit(solve, random_lqr(n, m, T))
            rows.append(dict(bench="lqr", axis=axis, n=n, m=m, T=T, seconds=run_t, compile=compile_t, iters=1))
            print(f"lqr   {axis}: n={n:4} m={m:3} T={T:5}  {run_t * 1e3:9.2f} ms")


def bench_ilqr(sizes, rows):
    """ilqr_solver on the RNN vs n (m = 5, T = 50) and T (n = 20, m = 5)"""
    sweeps = {"n": [(n, 5, 50) for n in sizes["ilqr_n"]], "T": [(20, 5, T) for T in sizes["ilqr_T"]]}
    for axis, cases in sweeps.items():
        for n, m, T in cases:
            model, params = rnn_problem(n, min(n, m), horizon=T)
            Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (T, model.dims.m))
            solve = jax.jit(lambda U, model=model, params=params: ilqr_solver(model, params, U, max_iter=400))
            compile_t, run_t, (_, _, cost_log) = timeit(solve, Us_init, repeats=3)
            iters = int(jnp.sum(jnp.diff(cost_log) != 0))
            rows.append(dict(bench="ilqr", axis=axis, n=n, m=model.dims.m, T=T, seconds=run_t, compile=compile_t, iters=iters))
            print(f"ilqr  {axis}: n={n:4} m={model.dims.m:3} T={T:5}  {run_t * 1e3:9.2f} ms  {iters} iterations")


def bench_gradients(sizes, rows):
    """Implicit (dilqr) vs direct-unrolled gradient of ||X*||^2 + ||U*||^2 wrt the RNN weights"""
    sweeps = {"n": [(n, 5, 30) for n in sizes["grad_n"]], "T": [(10, 5, T) for T in sizes["grad_T"]]}
    for axis, cases in sweeps.items():
        for n, m, T in cases:
            model, params = rnn_problem(n, min(n, m), horizon=T)
            Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (T, model.dims.m))

            def implicit_loss(theta, model=model, params=params, Us_init=Us_init):
                return jnp.sum(dilqr(model, iLQRParams(params.x0, theta), Us_init, max_iter=400) ** 2)

            def direct_loss(theta, model=model, params=params, Us_init=Us_init):
                (Xs, Us, _), _, _ = ilqr_solver(model, iLQRParams(params.x0, theta), Us_init, max_iter=400)
                return jnp.sum(Xs**2) + jnp.sum(Us**2)

            for kind, loss in (("implicit", implicit_loss), ("direct", direct_loss)):
                compile_t, run_t, _ = timeit(jax.jit(jax.grad(loss)), params.theta, repeats=3)
                rows.append(
                    dict(bench=f"grad_{kind}", axis=axis, n=n, m=model.dims.m, T=T, seconds=run_t, compile=compile_t, iters=0)
                )
                print(f"grad  {axis}: n={n:4} m={model.dims.m:3} T={T:5}  {kind:8} {run_t * 1e3:9.2f} ms")


def plot(rows, out_dir: Path):
    """Log-log plots of the timings"""
    panels = [
        ("lqr", "n", "solve_lqr vs n (m=5, T=100)"),
        ("lqr", "m", "solve_lqr vs m (n=50, T=100)"),
        ("lqr", "T", "solve_lqr vs T (n=20, m=5)"),
        ("ilqr", "n", "ilqr_solver vs n (RNN, T=50)"),
        ("ilqr", "T", "ilqr_solver vs T (RNN, n=20)"),
        ("grad", "n", "gradient vs n (RNN, T=30)"),
        ("grad", "T", "gradient vs T (RNN, n=10)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (bench, axis, title) in zip(axes.flat, panels):
        benches = ["grad_implicit", "grad_direct"] if bench == "grad" else [bench]
        for name in benches:
            sel = [r for r in rows if r["bench"] == name and r["axis"] == axis]
            if sel:
                ax.loglog([r[axis] for r in sel], [r["seconds"] for r in sel], "o-", label=name)
        if bench == "ilqr":
            sel = [r for r in rows if r["bench"] == "ilqr" and r["axis"] == axis]
            ax2 = ax.twinx()
            ax2.plot([r[axis] for r in sel], [r["iters"] for r in sel], "s--", c="tab:gray", label="iterations")
            ax2.set_ylabel("iterations", color="tab:gray")
        ax.set(title=title, xlabel=axis, ylabel="seconds (median, post-JIT)")
        ax.legend(loc="upper left", fontsize=8)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "timing.png", dpi=150)


def main():
    """Parse arguments, run the timing sweeps and save the CSV and figure"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small sizes for a fast smoke run")
    args = parser.parse_args()
    # pinned here rather than at import, so that importing timeit does not move other benchmarks to the CPU
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    sizes = {
        "lqr_n": [5, 10, 20, 50, 100, 200],
        "lqr_m": [1, 5, 10, 20, 50],
        "lqr_T": [50, 100, 200, 500, 1000, 2000],
        "ilqr_n": [5, 10, 20, 50, 100],
        "ilqr_T": [25, 50, 100, 200, 400],
        "grad_n": [5, 10, 20, 50],
        "grad_T": [15, 30, 60, 120],
    }
    if args.quick:
        sizes = {key: value[:3] for key, value in sizes.items()}

    out_dir = REPO / "fig_dump" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    bench_lqr(sizes, rows)
    bench_ilqr(sizes, rows)
    bench_gradients(sizes, rows)
    with open(out_dir / "timing.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, out_dir)
    print(f"saved {out_dir / 'timing.csv'} and {out_dir / 'timing.png'}")


if __name__ == "__main__":
    main()
