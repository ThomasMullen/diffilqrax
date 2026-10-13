"""
Summary plots comparing the initial guess with the iLQR optimum for each control problem.

For every problem in tests/problems.py one figure shows, initial (dashed) vs optimal (solid):
  - state trajectories (first few states for high-dimensional systems, with the tracking target)
  - control inputs
  - running cost per time step, with the total costs
  - phase portrait (2-d systems) or the iLQR cost history (higher-dimensional systems)
plus an overview figure of initial vs optimal total cost and the iterations taken.

Run from the repository root:
    python benchmarks/solution_summary.py

Writes <problem>.png, overview.png and solutions.npz to fig_dump/<timestamp>/.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as onp
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
# pylint: disable=wrong-import-position
import problems as P
from diffilqrax import ilqr

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])

MAX_STATES = 4  # states/inputs plotted for high-dimensional systems

# (file name, title, builder, max_iter, initial input scale, state labels, tracking target or None)
PROBLEMS = [
    ("tanh_pendulum", "tanh pendulum", P.pendulum_problem, 100, 0.1, ["x1", "x2"], None),
    ("ocaml_pendulum", "OCaml pendulum (pendulum.ml)", P.ocaml_pendulum, 300, 0.0, ["angle", "velocity"], None),
    ("inverted_pendulum", "inverted pendulum swing-up", P.inverted_pendulum, 300, 0.5, ["angle", "velocity"], None),
    ("van_der_pol", "Van der Pol to origin", P.van_der_pol, 200, 0.1, ["x1", "x2"], None),
    ("rnn", "tanh RNN (n=8)", P.rnn_problem, 300, 0.1, None, lambda t, n: onp.sin(t / 5.0)[:, None] * onp.ones(n)),
    ("vanilla_rnn", "vanilla RNN (n=20)", P.vanilla_rnn, 200, 0.1, None, "theta.Q"),
    ("rnn_tracking", "RNN tracking (n=50)", P.rnn_tracking, 600, 0.1, None, None),
]


def running_costs(model, params, Xs, Us):
    """Per-step running costs and the final cost"""
    tps = jnp.arange(model.dims.horizon)
    stage = jax.vmap(model.cost, in_axes=(0, 0, 0, None))(tps, Xs[:-1], Us, params.theta)
    return onp.asarray(stage), float(model.costf(Xs[-1], params.theta))


def solve_problem(builder, max_iter, scale):
    """Initial rollout and iLQR optimum of one problem"""
    model, params = builder()
    shape = (model.dims.horizon, model.dims.m)
    Us_init = scale * jr.normal(jr.PRNGKey(0), shape) if scale else jnp.zeros(shape)
    (Xs_init, _), cost_init = ilqr.ilqr_simulate(model, Us_init, params)
    (Xs, Us, Lambs), cost, cost_log = ilqr.ilqr_solver(model, params, Us_init, max_iter=max_iter)
    kkt = max(float(jnp.abs(r).max()) for r in P.nonlinear_kkt(model, params, Xs, Us, Lambs))
    log = onp.asarray(cost_log)
    changed = onp.nonzero(onp.diff(log))[0]
    return {
        "model": model,
        "params": params,
        "Xs_init": onp.asarray(Xs_init),
        "Us_init": onp.asarray(Us_init),
        "cost_init": float(cost_init),
        "Xs": onp.asarray(Xs),
        "Us": onp.asarray(Us),
        "Lambs": onp.asarray(Lambs),
        "cost": float(cost),
        "cost_log": log[: changed.max() + 2] if changed.size else log[:1],
        "iterations": int(changed.size),
        "kkt": kkt,
    }


def tracking_target(spec, sol):
    """Target trajectory for tracking problems, if one is defined"""
    if spec is None:
        return None
    if spec == "theta.Q":
        return onp.asarray(sol["params"].theta.Q)
    return spec(onp.arange(sol["model"].dims.horizon), sol["model"].dims.n)


def plot_problem(name, title, labels, target_spec, sol, out_dir):
    """Initial vs optimal states, inputs, running costs and phase portrait / cost history"""
    model = sol["model"]
    dt = model.dims.dt
    t_x = onp.arange(model.dims.horizon + 1) * dt
    t_u = t_x[:-1]
    n_x = min(model.dims.n, MAX_STATES)
    n_u = min(model.dims.m, MAX_STATES)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    labels = labels or [f"x{i + 1}" for i in range(n_x)]
    target = tracking_target(target_spec, sol)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ax = axes[0, 0]
    for i in range(n_x):
        ax.plot(t_x, sol["Xs_init"][:, i], "--", c=colors[i], alpha=0.6)
        ax.plot(t_x, sol["Xs"][:, i], "-", c=colors[i], label=labels[i])
        if target is not None:
            ax.plot(t_u, target[:, i], ":", c=colors[i], lw=1)
    shown = f" (first {n_x} of {model.dims.n})" if model.dims.n > n_x else ""
    ax.set(title=f"states{shown}", xlabel="time")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    for i in range(n_u):
        ax.plot(t_u, sol["Us_init"][:, i], "--", c=colors[i], alpha=0.6)
        ax.plot(t_u, sol["Us"][:, i], "-", c=colors[i], label=f"u{i + 1}")
    shown = f" (first {n_u} of {model.dims.m})" if model.dims.m > n_u else ""
    ax.set(title=f"inputs{shown}", xlabel="time")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    stage_init, final_init = running_costs(model, sol["params"], sol["Xs_init"], sol["Us_init"])
    stage_opt, final_opt = running_costs(model, sol["params"], sol["Xs"], sol["Us"])
    ax.plot(t_u, stage_init, "--", c="tab:gray", label=f"initial: J = {sol['cost_init']:.4g} (final {final_init:.3g})")
    ax.plot(t_u, stage_opt, "-", c="tab:green", label=f"optimal: J = {sol['cost']:.4g} (final {final_opt:.3g})")
    ax.set(title="running cost per step", xlabel="time", yscale="log" if stage_init.min() > 0 else "linear")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if model.dims.n == 2:
        ax.plot(sol["Xs_init"][:, 0], sol["Xs_init"][:, 1], "--", c="tab:gray", label="initial")
        ax.plot(sol["Xs"][:, 0], sol["Xs"][:, 1], "-", c="tab:green", label="optimal")
        ax.plot(*sol["Xs"][0], "ko", label="x0")
        ax.plot(*sol["Xs"][-1], "g*", ms=12, label="optimal x_T")
        ax.set(title="phase portrait", xlabel=labels[0], ylabel=labels[1])
    else:
        gap = sol["cost_log"] - sol["cost_log"].min()
        ax.semilogy(gap + 1e-13, ".-", c="tab:green", label="J - J*")
        ax.set(title="iLQR convergence", xlabel="iteration", ylabel="J - J*")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"{title}: J {sol['cost_init']:.4g} → {sol['cost']:.4g} in {sol['iterations']} iterations, "
        f"max|KKT| = {sol['kkt']:.1e}    (dashed: initial, solid: optimal, dotted: target)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def plot_overview(results, out_dir):
    """Initial vs optimal total cost and iterations for every problem"""
    names = list(results)
    x = onp.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].bar(x - 0.2, [results[k]["cost_init"] for k in names], 0.4, label="initial", color="tab:gray")
    axes[0].bar(x + 0.2, [results[k]["cost"] for k in names], 0.4, label="optimal", color="tab:green")
    axes[0].set(yscale="log", xticks=x, ylabel="total cost", title="Total cost: initial vs optimal")
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].legend()
    axes[1].bar(x, [results[k]["iterations"] for k in names], color="tab:blue")
    for xi, k in zip(x, names):
        axes[1].text(xi, results[k]["iterations"], f"{results[k]['kkt']:.0e}", ha="center", va="bottom", fontsize=8)
    axes[1].set(xticks=x, ylabel="iterations", title="iLQR iterations (label: max |KKT residual|)")
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=150)
    plt.close(fig)


def main():
    out_dir = REPO / "fig_dump" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    results, arrays = {}, {}
    for name, title, builder, max_iter, scale, labels, target_spec in PROBLEMS:
        sol = solve_problem(builder, max_iter, scale)
        plot_problem(name, title, labels, target_spec, sol, out_dir)
        results[title] = sol
        for key in ("Xs_init", "Us_init", "Xs", "Us", "Lambs", "cost_log"):
            arrays[f"{name}/{key}"] = sol[key]
        print(
            f"{title:30} J {sol['cost_init']:12.4f} -> {sol['cost']:12.6f}  "
            f"{sol['iterations']:4} iterations  max|KKT| {sol['kkt']:.1e}"
        )
    plot_overview(results, out_dir)
    onp.savez(out_dir / "solutions.npz", **arrays)
    print(f"saved plots to {out_dir}")


if __name__ == "__main__":
    main()
