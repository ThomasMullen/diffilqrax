"""
Long-horizon iLQR with the sequential and the parallel (associative-scan) solvers: a tanh RNN tracks a slow
sine target over T = 4000 steps.

Run from the repository root:
    python examples/long_horizon.py [--horizon 4000]

Solves with ilqr_solver and with pilqr_solver using the "feedback" rollout (parallel backward pass,
sequential non-linear rollout) and the "parallel" rollout (multiple shooting: everything is an associative
scan or a vmap). Prints the wall time, iterations, cost and KKT residual of each, and saves a plot of the
solutions to fig_dump/<timestamp>/long_horizon.png. On a CPU the parallel solvers are not faster; on a GPU
the "parallel" rollout has O(log T) depth per iteration.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from diffilqrax.ilqr import ilqr_solver, approx_lqr, kkt_input_residual  # pylint: disable=wrong-import-position
from diffilqrax.parallel_ilqr import pilqr_solver, dynamics_defects  # pylint: disable=wrong-import-position
from diffilqrax.typs import System, ModelDims, iLQRParams, Theta  # pylint: disable=wrong-import-position

jax.config.update("jax_enable_x64", True)

N, M = 8, 3


def make_problem(horizon: int):
    """Stable tanh RNN x' = tanh(Uh x + Wh u) tracking sin(t / 50) in every state, with a quadratic input cost.
    With unstable dynamics the open-loop adjoint in the KKT check overflows on long horizons."""
    k1, k2 = jr.split(jr.PRNGKey(0))
    theta = Theta(Uh=0.8 * jr.normal(k1, (N, N)) / jnp.sqrt(N), Wh=jr.normal(k2, (N, M)) / jnp.sqrt(N), sigma=0.0, Q=None)

    def cost(t, x, u, th):
        return jnp.sum((x - 0.5 * jnp.sin(t / 50.0)) ** 2) + 0.1 * jnp.sum(u**2)

    model = System(
        cost,
        lambda x, th: jnp.sum(x**2),
        lambda t, x, u, th: jnp.tanh(th.Uh @ x + th.Wh @ u),
        ModelDims(horizon=horizon, n=N, m=M, dt=0.1),
    )
    return model, iLQRParams(x0=jnp.zeros(N), theta=theta)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--horizon", type=int, default=4000)
    args = parser.parse_args()
    model, params = make_problem(args.horizon)
    Us_init = jnp.zeros((args.horizon, M))
    solvers = {
        "ilqr": lambda p, u: ilqr_solver(model, p, u, max_iter=200),
        "pilqr feedback": lambda p, u: pilqr_solver(model, p, u, max_iter=200, rollout="feedback"),
        "pilqr parallel": lambda p, u: pilqr_solver(model, p, u, max_iter=200, rollout="parallel"),
    }
    print(f"device: {jax.devices()[0]}, T = {args.horizon}")
    solutions = {}
    for name, solve in solvers.items():
        solve = jax.jit(solve)
        jax.block_until_ready(solve(params, Us_init))  # compile  # pylint: disable=not-callable
        start = time.perf_counter()
        (Xs, Us, _), J, costs = jax.block_until_ready(solve(params, Us_init))  # pylint: disable=not-callable
        seconds = time.perf_counter() - start
        iters = int(jnp.argmax(jnp.abs(costs - J) <= 1e-10 * jnp.abs(J))) + 1
        kkt = float(jnp.max(jnp.abs(kkt_input_residual(approx_lqr(model, Xs, Us, params)))))
        defect = float(jnp.max(jnp.abs(dynamics_defects(model, params, Xs, Us))))
        print(
            f"{name:15s} {seconds:7.3f}s  iterations {iters:3d}  cost {float(J):.10f}  max|dL/dU| {kkt:.1e}  max defect {defect:.1e}"
        )
        solutions[name] = (Xs, Us)

    out_dir = Path("fig_dump") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for (name, (Xs, Us)), style in zip(solutions.items(), ("-", "--", ":")):
        axs[0].plot(Xs[:, 0], style, label=name)
        axs[1].plot(Us[:, 0], style, label=name)
    axs[0].plot(0.5 * jnp.sin(jnp.arange(args.horizon + 1) / 50.0), "k", lw=0.5, label="target")
    axs[0].set(ylabel="x[0]")
    axs[1].set(ylabel="u[0]", xlabel="t")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "long_horizon.png", dpi=150)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
