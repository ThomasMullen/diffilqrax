"""
Memory and wall time of implicit (dilqr) vs direct-unrolled gradients through the iLQR solver.

The direct gradient differentiates through the ``lax.scan`` over ``max_iter`` iterations in
``ilqr_solver``, so reverse mode stores residuals for every iteration, converged or not. The implicit
gradient solves one extra LQR at the optimum and stores nothing from the iterations.

Gradient of ||X*||^2 + ||U*||^2 wrt the tanh RNN weights (tests/problems.rnn_problem), swept over
  (a) horizon T        at n = 20, m = 5,  max_iter = 100
  (b) state dim n      at T = 50, m = 5,  max_iter = 100
  (c) max_iter         at n = 20, m = 5,  T = 50
Each configuration runs in a fresh subprocess and reports
  - XLA's compiled memory analysis (temp + argument + output bytes): deterministic, the primary measure
  - peak RSS growth over the process baseline after setup (noisy, includes allocator overheads)
  - post-JIT median wall time (benchmarks/timing.timeit)
Configurations whose compiled memory exceeds --max-run-gib (default 16) are compiled but not run, so
only their compiled memory is reported (the direct gradient needs ~47 GiB at n = 100).

Run from the repository root:
    python -m benchmarks.memory [--quick]

Writes memory.csv, memory.json and memory.png to fig_dump/<timestamp>/.
"""

import argparse
import csv
import json
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

REPO = Path(__file__).resolve().parents[1]
KINDS = ("implicit", "direct")


def peak_rss_bytes() -> int:
    """Peak resident set size of this process (ru_maxrss is bytes on macOS, KiB on Linux)"""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def worker(kind: str, n: int, m: int, horizon: int, max_iter: int, max_run_bytes: int) -> dict:
    """Measure one gradient configuration in this (fresh) process; run it only if it fits in max_run_bytes"""
    # pylint: disable=import-outside-toplevel
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from tests.problems import rnn_problem
    from benchmarks.timing import timeit
    from diffilqrax.ilqr import ilqr_solver
    from diffilqrax.diff_ilqr import dilqr
    from diffilqrax.typs import iLQRParams

    model, params = rnn_problem(n, m, horizon=horizon)
    Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (horizon, m))

    def implicit_loss(theta):
        return jnp.sum(dilqr(model, iLQRParams(params.x0, theta), Us_init, max_iter=max_iter) ** 2)

    def direct_loss(theta):
        (Xs, Us, _), _, _ = ilqr_solver(model, iLQRParams(params.x0, theta), Us_init, max_iter=max_iter)
        return jnp.sum(Xs**2) + jnp.sum(Us**2)

    grad_fn = jax.jit(jax.grad(implicit_loss if kind == "implicit" else direct_loss))
    baseline = peak_rss_bytes()
    stats = grad_fn.lower(params.theta).compile().memory_analysis()
    compiled = stats.temp_size_in_bytes + stats.argument_size_in_bytes + stats.output_size_in_bytes
    nan = float("nan")
    compile_t, run_t = nan, nan
    if compiled <= max_run_bytes:
        compile_t, run_t, _ = timeit(grad_fn, params.theta, repeats=3)
    return dict(
        kind=kind,
        n=n,
        m=m,
        T=horizon,
        max_iter=max_iter,
        temp_bytes=stats.temp_size_in_bytes,
        compiled_bytes=compiled,
        rss_growth_bytes=peak_rss_bytes() - baseline if compiled <= max_run_bytes else nan,
        seconds=run_t,
        compile_seconds=compile_t,
    )


def run_config(kind: str, n: int, m: int, horizon: int, max_iter: int, max_run_bytes: int) -> dict:
    """Run worker() in a fresh Python process so peak RSS is not shared between configurations"""
    sizes = [str(v) for v in (n, m, horizon, max_iter, max_run_bytes)]
    cmd = [sys.executable, "-m", "benchmarks.memory", "--worker", kind, *sizes]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def plot(rows, out_dir: Path):
    """Compiled memory, RSS growth and wall time of both gradients against T, n and max_iter"""
    sweeps = [
        ("T", "horizon T (n=20, max_iter=100)"),
        ("n", "state dim n (T=50, max_iter=100)"),
        ("max_iter", "max_iter (n=20, T=50)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (axis, label) in enumerate(sweeps):
        for kind, color in zip(KINDS, ("tab:blue", "tab:red")):
            sel = sorted((r for r in rows if r["sweep"] == axis and r["kind"] == kind), key=lambda r, axis=axis: r[axis])
            xs = [r[axis] for r in sel]
            axes[0, col].loglog(xs, [r["compiled_bytes"] / 2**20 for r in sel], "o-", c=color, label=f"{kind}: compiled")
            axes[0, col].loglog(
                xs, [r["rss_growth_bytes"] / 2**20 for r in sel], "s--", c=color, alpha=0.6, label=f"{kind}: peak RSS growth"
            )
            axes[1, col].loglog(xs, [r["seconds"] for r in sel], "o-", c=color, label=kind)
        axes[0, col].set(title=f"memory vs {label}", xlabel=axis, ylabel="MiB")
        axes[1, col].set(title=f"gradient time vs {label}", xlabel=axis, ylabel="seconds (median, post-JIT)")
        for ax in axes[:, col]:
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "memory.png", dpi=150)
    plt.close(fig)


def main():
    """Run the memory sweeps (or one --worker configuration) and write the CSV, JSON and plot"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small sizes for a fast smoke run")
    parser.add_argument("--max-run-gib", type=float, default=16.0, help="only run configurations that fit in this")
    parser.add_argument("--worker", nargs=6, metavar=("KIND", "N", "M", "T", "MAX_ITER", "BYTES"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        kind, *dims = args.worker
        print(json.dumps(worker(kind, *map(int, dims))))
        return

    sweeps = {
        "T": [(20, 5, T, 100) for T in (25, 50, 100, 200, 400)],
        "n": [(n, 5, 50, 100) for n in (5, 20, 50, 100)],
        "max_iter": [(20, 5, 50, k) for k in (10, 25, 50, 100, 200)],
    }
    if args.quick:
        sweeps = {key: cases[:3] for key, cases in sweeps.items()}

    out_dir = REPO / "fig_dump" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sweep, cases in sweeps.items():
        for n, m, horizon, max_iter in cases:
            for kind in KINDS:
                row = dict(sweep=sweep, **run_config(kind, n, m, horizon, max_iter, int(args.max_run_gib * 2**30)))
                rows.append(row)
                print(
                    f"{sweep:8} n={n:4} T={horizon:4} max_iter={max_iter:4} {kind:8} "
                    f"compiled {row['compiled_bytes'] / 2**20:9.2f} MiB  RSS +{row['rss_growth_bytes'] / 2**20:8.1f} MiB  "
                    f"{row['seconds'] * 1e3:9.1f} ms"
                )

    with open(out_dir / "memory.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "memory.json").write_text(json.dumps(rows, indent=2))
    plot(rows, out_dir)
    print(f"saved memory.csv, memory.json and memory.png to {out_dir}")


if __name__ == "__main__":
    main()
