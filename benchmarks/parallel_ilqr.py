"""
Sequential iLQR vs parallel iLQR (pilqr_solver) with the "feedback" and "parallel" (multiple-shooting)
rollouts, over the horizon T, on the tanh pendulum and the tanh RNN: iterations, final cost, KKT residual,
dynamics defects and wall time.

The script is device-agnostic: it runs on whatever ``jax.devices()[0]`` is and records the device in the CSV,
so the same sweep can be re-run on a GPU. On a CPU the associative scans do O(T log T) work with no
parallel speedup; the O(log T) depth of the "parallel" rollout only pays off on a parallel accelerator.

Usage: ``venv/bin/python benchmarks/parallel_ilqr.py [--quick] [--repeats 3]``
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
sys.path.insert(0, str(REPO / "benchmarks"))
jax.config.update("jax_enable_x64", True)

# pylint: disable=wrong-import-position
from problems import pendulum_problem, rnn_problem, nonlinear_kkt
from timing import timeit
from diffilqrax.ilqr import ilqr_solver
from diffilqrax.parallel_ilqr import pilqr_solver, dynamics_defects

MAX_ITER = 200
PROBLEMS = {
    "pendulum": lambda T: pendulum_problem(horizon=T),
    "rnn": lambda T: rnn_problem(8, 3, horizon=T),
}
INFO = dict(max_iter=MAX_ITER, return_info=True)
METHODS = {
    "ilqr": lambda model, params, U0: ilqr_solver(model, params, U0, **INFO),
    "pilqr feedback": lambda model, params, U0: pilqr_solver(model, params, U0, rollout="feedback", **INFO),
    "pilqr parallel": lambda model, params, U0: pilqr_solver(model, params, U0, rollout="parallel", **INFO),
}


def run(horizons, repeats: int):
    device = str(jax.devices()[0])
    rows = []
    for problem, make in PROBLEMS.items():
        for T in horizons:
            model, params = make(T)
            U0 = 0.1 * jr.normal(jr.PRNGKey(0), (T, model.dims.m))
            line = f"{problem:9s} T={T:6}"
            for method, solve in METHODS.items():
                fn = jax.jit(lambda p, u, solve=solve, model=model: solve(model, p, u))
                compile_t, run_t, ((Xs, Us, Lambs), J, _, converged, n_iters) = timeit(fn, params, U0, repeats=repeats)
                kkt = max(float(jnp.max(jnp.abs(r))) for r in nonlinear_kkt(model, params, Xs, Us, Lambs))
                rows.append(
                    dict(
                        problem=problem,
                        method=method,
                        T=T,
                        n=model.dims.n,
                        m=model.dims.m,
                        iterations=int(n_iters),
                        converged=bool(converged),
                        cost=float(J),
                        kkt=kkt,
                        max_defect=float(jnp.max(jnp.abs(dynamics_defects(model, params, Xs, Us)))),
                        seconds=run_t,
                        compile=compile_t,
                        device=device,
                    )
                )
                r = rows[-1]
                line += f" | {method} {run_t:8.3f}s it={r['iterations']:3d} kkt={kkt:.0e}"
            print(line)
    return rows


def plot(rows, out_dir: Path):
    problems = list(dict.fromkeys(r["problem"] for r in rows))
    fig, axs = plt.subplots(3, len(problems), figsize=(5 * len(problems), 10), squeeze=False)
    device = rows[0]["device"]
    for j, problem in enumerate(problems):
        for method, style in zip(METHODS, ("o-", "s--", "^:")):
            pts = [r for r in rows if r["problem"] == problem and r["method"] == method]
            Ts = [r["T"] for r in pts]
            axs[0, j].loglog(Ts, [r["seconds"] for r in pts], style, label=method)
            axs[1, j].semilogx(Ts, [r["iterations"] for r in pts], style, label=method)
            axs[2, j].loglog(Ts, [max(r["kkt"], 1e-17) for r in pts], style, label=method)
        axs[0, j].set(title=f"{problem}: wall time ({device})", ylabel="seconds")
        axs[1, j].set(title="iterations to converge", ylabel="iterations")
        axs[2, j].set(title="max KKT residual", ylabel="residual", xlabel="horizon T")
        for ax in axs[:, j]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "parallel_ilqr.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="small sweep for a smoke test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory, by default fig_dump/<timestamp>")
    args = parser.parse_args()
    horizons = (64, 256) if args.quick else (64, 256, 1024, 4096, 16384)
    rows = run(horizons, args.repeats)
    out_dir = args.out_dir or REPO / "fig_dump" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "parallel_ilqr.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, out_dir)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
