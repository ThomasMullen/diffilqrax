"""
Validation report: re-solves the test problems and saves figures plus a JSON summary.

Figures (fig_dump/<timestamp>/):
  kkt_residuals.png        dL/dX, dL/dU, dL/dλ over time for each control problem
  kkt_grid.png             max KKT residual over the (n, m) grid for LQR and for iLQR on the RNN
  ocaml_comparison.png     pendulum and RNN: Python vs OCaml dilqr trajectories and cost histories
  lbfgs_comparison.png     iLQR cost history vs the L-BFGS optimum for each control problem
  perturbation.png         J(U* + eps dU) - J(U*) vs eps with fitted log-log slopes
  gradients.png            implicit vs direct gradients, exact Hessian vs Gauss-Newton
  gradient_methods.png     exact / Gauss-Newton / OCaml-style implicit gradient error vs direct, per problem
  ocaml_gradients.png      OCaml dilqr implicit gradients vs Python, both cases, for n in 5..100
  dlqr_gradients.png       dlqr implicit vs direct gradient relative error per LQR field
  summary.json             the key numbers behind the figures

Memory and time of implicit vs unrolled gradients are in benchmarks/memory.py (memory.png/json).

Run from the repository root:
    python benchmarks/validation_report.py [--skip-grid]
"""

import argparse
import json
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
from test_ocaml_reference import gauss_newton_loss, ocaml_rollout_loss
from test_dlqr import dlqr_losses, sym_lqr_grads
from diffilqrax import ilqr, lqr
from diffilqrax.diff_ilqr import add_dynamics_curvature, dilqr
from diffilqrax.typs import iLQRParams

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])

# (label, builder, use_linesearch, max_iter, initial input scale)
CONTROL_PROBLEMS = [
    ("tanh pendulum", P.pendulum_problem, True, 100, 0.1),
    ("OCaml pendulum", P.ocaml_pendulum, True, 300, 0.0),
    ("vanilla RNN (n=20)", P.vanilla_rnn, True, 200, 0.1),
    ("inverted pendulum", P.inverted_pendulum, True, 300, 0.5),
    ("Van der Pol", P.van_der_pol, True, 200, 0.1),
    ("RNN tracking (n=50)", P.rnn_tracking, True, 600, 0.1),
]


def solve(builder, use_linesearch=True, max_iter=300, scale=0.1, seed=0):
    model, params = builder()
    shape = (model.dims.horizon, model.dims.m)
    Us_init = scale * jr.normal(jr.PRNGKey(seed), shape) if scale else jnp.zeros(shape)
    sol = ilqr.ilqr_solver(model, params, Us_init, max_iter=max_iter, use_linesearch=use_linesearch)
    return model, params, Us_init, sol


def max_kkt(model, params, Xs, Us, Lambs):
    return max(float(jnp.abs(r).max()) for r in P.nonlinear_kkt(model, params, Xs, Us, Lambs))


def is_strict_min(model, params, Xs, Us, Lambs):
    local = add_dynamics_curvature(model, ilqr.approx_lqr_offset(model, Xs, Us, params), Xs, Us, Lambs, params)
    return bool(lqr.regularised_backward_pass(local, mu=0.0)[2])


def control_problems(out_dir, summary):
    """KKT residual traces, L-BFGS comparison and perturbation plots for each control problem"""
    n = len(CONTROL_PROBLEMS)
    fig_kkt, ax_kkt = plt.subplots(n, 3, figsize=(12, 2.2 * n), squeeze=False)
    fig_lb, ax_lb = plt.subplots(2, 3, figsize=(13, 7))
    fig_pt, ax_pt = plt.subplots(1, 1, figsize=(6, 4.5))
    summary["control_problems"] = {}
    for row, (label, builder, use_ls, max_iter, scale) in enumerate(CONTROL_PROBLEMS):
        model, params, Us_init, ((Xs, Us, Lambs), cost, cost_log) = solve(builder, use_ls, max_iter, scale)
        residuals = P.nonlinear_kkt(model, params, Xs, Us, Lambs)
        for col, (res, name) in enumerate(zip(residuals, ("dL/dX", "dL/dU", "dL/dλ"))):
            ax = ax_kkt[row, col]
            ax.semilogy(onp.abs(onp.asarray(res)).max(axis=1) + 1e-18)
            ax.axhline(1e-8, ls=":", c="r")
            ax.set(title=f"{label}: max |{name}|" if col == 0 else f"max |{name}|", xlabel="t")
        lbfgs_cold, _ = P.lbfgs_reference(model, params, Us_init)
        lbfgs_warm, _ = P.lbfgs_reference(model, params, Us)
        ax = ax_lb.flat[row]
        log = onp.asarray(cost_log)
        ax.semilogy(log - min(log.min(), lbfgs_cold, lbfgs_warm) + 1e-12, label="iLQR J - J*")
        ax.axhline(abs(lbfgs_cold - float(cost)) + 1e-12, ls="--", c="tab:orange", label="|L-BFGS (cold) - iLQR|")
        ax.set(title=label, xlabel="iteration", ylabel="cost gap")
        ax.legend(fontsize=7)
        entry = {
            "cost": float(cost),
            "iterations": int(onp.sum(onp.diff(log) != 0)),
            "max_kkt": max_kkt(model, params, Xs, Us, Lambs),
            "strict_local_min": is_strict_min(model, params, Xs, Us, Lambs),
            "lbfgs_cold_cost": lbfgs_cold,
            "lbfgs_warm_cost": lbfgs_warm,
        }
        if label != "OCaml pendulum":  # open-loop unstable: eps^2 regime unreachable, see test_optimality
            eps, dJ, slope = P.perturbation_slope(model, params, Us)
            ax_pt.loglog(eps, onp.abs(dJ), "o-", label=f"{label} (slope {slope:.2f})")
            entry["perturbation_slope"] = float(slope)
        summary["control_problems"][label] = entry
        print(f"{label}: J={entry['cost']:.10f} kkt={entry['max_kkt']:.1e} L-BFGS cold={lbfgs_cold:.10f}")
    eps = onp.logspace(-4, -1, 10)
    ax_pt.loglog(eps, eps**2, "k:", label="∝ eps²")
    ax_pt.loglog(eps, eps, "k--", label="∝ eps")
    ax_pt.set(xlabel="eps", ylabel="J(U* + eps dU) - J(U*)", title="Cost growth around the optimum")
    ax_pt.legend(fontsize=7)
    for fig, name in ((fig_kkt, "kkt_residuals.png"), (fig_lb, "lbfgs_comparison.png"), (fig_pt, "perturbation.png")):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)


def kkt_grid(out_dir, summary):
    """Max KKT residual over the dimension grid, for LQR and for iLQR on the RNN"""
    ns = sorted({n for n, _ in P.DIMS})
    ms = sorted({m for _, m in P.DIMS})
    lqr_grid = onp.full((len(ns), len(ms)), onp.nan)
    ilqr_grid = onp.full((len(ns), len(ms)), onp.nan)
    for n, m in P.DIMS:
        params = P.random_lqr(n, m, horizon=20)
        Xs, Us, Lambs = lqr.solve_lqr(params)
        lqr_grid[ns.index(n), ms.index(m)] = max(float(jnp.abs(r).max()) for r in lqr.kkt(params, Xs, Us, Lambs))
        model, ilqr_params = P.rnn_problem(n, m, horizon=30)
        Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (30, m))
        (Xs, Us, Lambs), _, _ = ilqr.ilqr_solver(model, ilqr_params, Us_init, max_iter=800)
        ilqr_grid[ns.index(n), ms.index(m)] = max_kkt(model, ilqr_params, Xs, Us, Lambs)
        print(f"grid n={n} m={m}: LQR {lqr_grid[ns.index(n), ms.index(m)]:.1e} iLQR {ilqr_grid[ns.index(n), ms.index(m)]:.1e}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title in ((axes[0], lqr_grid, "LQR (T=20)"), (axes[1], ilqr_grid, "iLQR on tanh RNN (T=30)")):
        im = ax.imshow(onp.log10(data), cmap="viridis", vmin=-16, vmax=-8)
        for i in range(len(ns)):
            for j in range(len(ms)):
                if not onp.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.0e}", ha="center", va="center", color="w", fontsize=8)
        ax.set(xticks=range(len(ms)), xticklabels=ms, yticks=range(len(ns)), yticklabels=ns, xlabel="m", ylabel="n")
        ax.set_title(f"max KKT residual, {title}")
        fig.colorbar(im, ax=ax, label="log10")
    fig.tight_layout()
    fig.savefig(out_dir / "kkt_grid.png", dpi=150)
    plt.close(fig)
    summary["kkt_grid"] = {"n": ns, "m": ms, "lqr": lqr_grid.tolist(), "ilqr_rnn": ilqr_grid.tolist()}


def ocaml_comparison(out_dir, summary):
    """Python vs OCaml dilqr on the pendulum (no linesearch, as pendulum.ml) and the vanilla RNN"""
    cases = [
        ("pendulum", "ocaml_pendulum", P.ocaml_pendulum, False),
        ("vanilla RNN", "ocaml_rnn", P.vanilla_rnn, True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    summary["ocaml"] = {}
    for row, (label, fixture, builder, use_ls) in enumerate(cases):
        path = REPO / "tests" / "fixtures" / f"{fixture}.npz"
        if not path.exists():
            print(f"skipping {label}: {path.name} missing (run benchmarks/make_ocaml_fixtures.py)")
            continue
        ref = onp.load(path)
        model, params = builder()
        Us_init = jnp.zeros((model.dims.horizon, model.dims.m))
        (Xs, Us, _), cost, cost_log = ilqr.ilqr_solver(model, params, Us_init, max_iter=400, use_linesearch=use_ls)
        k = min(Xs.shape[1], 3)
        axes[row, 0].plot(ref["Xs"][:, :k], lw=3, alpha=0.4)
        axes[row, 0].plot(onp.asarray(Xs[:, :k]), "k:", lw=1)
        axes[row, 0].set(title=f"{label}: states (thick OCaml, dotted Python)", xlabel="t")
        axes[row, 1].plot(ref["Us"][:, :k], lw=3, alpha=0.4)
        axes[row, 1].plot(onp.asarray(Us[:, :k]), "k:", lw=1)
        axes[row, 1].set(title="inputs", xlabel="t")
        J_star = min(float(cost), ref["costs"].min())
        axes[row, 2].semilogy(ref["costs"] - J_star + 1e-13, "o-", label="OCaml dilqr")
        log = onp.asarray(cost_log)
        last = int(onp.nonzero(onp.diff(log))[0].max()) + 2 if onp.any(onp.diff(log)) else 1
        axes[row, 2].semilogy(log[:last] - J_star + 1e-13, ".-", label="diffilqrax")
        axes[row, 2].set(title="cost - J*", xlabel="iteration")
        axes[row, 2].legend(fontsize=8)
        summary["ocaml"][label] = {
            "python_cost": float(cost),
            "ocaml_cost": float(ref["costs"][-1]),
            "max_abs_dX": float(onp.abs(onp.asarray(Xs) - ref["Xs"]).max()),
            "max_abs_dU": float(onp.abs(onp.asarray(Us) - ref["Us"]).max()),
            "ocaml_iterations": int(len(ref["costs"]) - 1),
            "ocaml_seconds": float(ref["seconds"]),
        }
        print(f"OCaml {label}: {summary['ocaml'][label]}")
    fig.tight_layout()
    fig.savefig(out_dir / "ocaml_comparison.png", dpi=150)
    plt.close(fig)


def gradients(out_dir, summary):
    """Implicit (exact Lagrangian Hessian and Gauss-Newton) vs direct-unrolled gradients on the RNN"""
    model, params = P.rnn_problem(20, 5, horizon=30)
    Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (30, 5))

    def implicit(theta, exact):
        return jnp.sum(dilqr(model, iLQRParams(params.x0, theta), Us_init, exact_hessian=exact, max_iter=400) ** 2)

    def direct(theta):
        (Xs, Us, _), _, _ = ilqr.ilqr_solver(model, iLQRParams(params.x0, theta), Us_init, max_iter=400)
        return jnp.sum(Xs**2) + jnp.sum(Us**2)

    g_dir = jax.grad(direct)(params.theta)
    g_exact = jax.grad(implicit)(params.theta, True)
    g_gn = jax.grad(implicit)(params.theta, False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    summary["gradients"] = {}
    for col, field in enumerate(("Uh", "Wh")):
        d = onp.asarray(getattr(g_dir, field)).ravel()
        for g, name, marker in ((g_exact, "implicit, exact Hessian", "o"), (g_gn, "implicit, Gauss-Newton", "x")):
            axes[col].scatter(d, onp.asarray(getattr(g, field)).ravel(), s=10, marker=marker, label=name)
        lim = [d.min(), d.max()]
        axes[col].plot(lim, lim, "k:", lw=1)
        axes[col].set(title=f"dL/d{field}", xlabel="direct (unrolled)", ylabel="implicit")
        axes[col].legend(fontsize=8)
        summary["gradients"][field] = {
            name: float(onp.linalg.norm(onp.asarray(getattr(g, field)).ravel() - d) / onp.linalg.norm(d))
            for g, name in ((g_exact, "exact_hessian"), (g_gn, "gauss_newton"))
        }
    names = list(summary["gradients"])
    width = 0.35
    for i, kind in enumerate(("exact_hessian", "gauss_newton")):
        axes[2].bar(onp.arange(len(names)) + i * width, [summary["gradients"][f][kind] for f in names], width, label=kind)
    axes[2].set(yscale="log", xticks=onp.arange(len(names)) + width / 2, xticklabels=names, ylabel="relative error vs direct")
    axes[2].set_title("Gradient error (RNN n=20, T=30)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gradients.png", dpi=150)
    plt.close(fig)
    print(f"gradients: {summary['gradients']}")


def rel_err(a, b) -> float:
    """Normwise relative error ||a - b|| / ||b||"""
    a, b = onp.asarray(a).ravel(), onp.asarray(b).ravel()
    return float(onp.linalg.norm(a - b) / onp.linalg.norm(b))


def exact_loss(model, params, Us_init):
    """sum(tau*^2) with exact (Lagrangian Hessian) implicit gradients"""
    return jnp.sum(dilqr(model, params, Us_init, max_iter=400) ** 2)


def direct_loss(model, params, Us_init):
    """sum(tau*^2) differentiated through the unrolled solver"""
    (Xs, Us, _), _, _ = ilqr.ilqr_solver(model, params, Us_init, max_iter=400)
    return jnp.sum(Xs**2) + jnp.sum(Us**2)


def gradient_methods(out_dir, summary):
    """Error of each implicit gradient against direct autodiff, wrt theta and x0, on several problems"""
    cases = [(f"RNN n={n}", lambda n=n: P.rnn_problem(n, min(n, 5))) for n in (5, 10, 20)]
    cases += [("tanh pendulum", lambda: P.pendulum_problem(horizon=50)), ("vanilla RNN n=20", P.vanilla_rnn)]
    methods = {"exact": exact_loss, "gauss_newton": gauss_newton_loss, "ocaml_released": ocaml_rollout_loss}
    summary["gradient_methods"] = {}
    for label, builder in cases:
        model, params = builder()
        Us_init = 0.1 * jr.normal(jr.PRNGKey(0), (model.dims.horizon, model.dims.m))
        g_dir = jax.grad(direct_loss, argnums=1)(model, params, Us_init)
        flat_dir = onp.concatenate([onp.ravel(g_dir.x0), onp.ravel(g_dir.theta.Uh), onp.ravel(g_dir.theta.Wh)])
        entry = {}
        for name, fn in methods.items():
            g = jax.grad(fn, argnums=1)(model, params, Us_init)
            entry[name] = rel_err(onp.concatenate([onp.ravel(g.x0), onp.ravel(g.theta.Uh), onp.ravel(g.theta.Wh)]), flat_dir)
        summary["gradient_methods"][label] = entry
        print(f"gradient methods {label}: {entry}")
    labels = list(summary["gradient_methods"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.27
    for i, name in enumerate(methods):
        ax.bar(onp.arange(len(labels)) + i * width, [summary["gradient_methods"][l][name] for l in labels], width, label=name)
    ax.set(
        yscale="log",
        xticks=onp.arange(len(labels)) + width,
        xticklabels=labels,
        ylabel="relative error vs direct autodiff",
        title="Implicit gradient error wrt (x0, W, B)",
    )
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out_dir / "gradient_methods.png", dpi=150)
    plt.close(fig)


def ocaml_gradients(out_dir, summary):
    """OCaml dilqr implicit gradients vs Python: fixed trajectory vs Gauss-Newton, released vs its emulation"""
    sizes = [(5, 3), (20, 5), (50, 10), (100, 20)]
    comparisons = {
        "fixed vs python GN": ("_fixed", "gn"),
        "released vs python emulation": ("", "rollout"),
        "fixed vs python exact": ("_fixed", "exact"),
        "released vs python exact": ("", "exact"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    summary["ocaml_gradients"] = {}
    for n, m in sizes:
        path = REPO / "tests" / "fixtures" / f"ocaml_rnn_grad_n{n}.npz"
        if not path.exists():
            print(f"skipping OCaml gradients n={n}: {path.name} missing (run benchmarks/make_ocaml_fixtures.py)")
            continue
        ref = onp.load(path)
        model, params = P.vanilla_rnn(n, m)
        Us_init = jnp.zeros((model.dims.horizon, m))
        grads = {
            key: jax.grad(fn, argnums=1)(model, params, Us_init)
            for key, fn in (("gn", gauss_newton_loss), ("rollout", ocaml_rollout_loss), ("exact", exact_loss))
        }
        flat = lambda g: onp.concatenate([onp.ravel(g.theta.Uh), onp.ravel(g.theta.Wh), onp.ravel(g.x0)])
        flat_ref = lambda sfx, ref=ref: onp.concatenate(
            [ref[f"grad_W{sfx}"].ravel(), ref[f"grad_B{sfx}"].ravel(), ref[f"grad_x0{sfx}"]]
        )
        summary["ocaml_gradients"][n] = {
            label: rel_err(flat(grads[py]), flat_ref(sfx)) for label, (sfx, py) in comparisons.items()
        }
        summary["ocaml_gradients"][n]["ocaml_seconds"] = float(ref["seconds"])
        axes[0].scatter(flat_ref("_fixed"), flat(grads["gn"]), s=6, label=f"n={n}")
        axes[1].scatter(flat_ref(""), flat(grads["rollout"]), s=6, label=f"n={n}")
        print(f"OCaml gradients n={n}: {summary['ocaml_gradients'][n]}")
    for ax, title, ylabel in (
        (axes[0], "case 1: OCaml fixed trajectory", "Python Gauss-Newton"),
        (axes[1], "case 2: OCaml as released", "Python emulation of released"),
    ):
        lim = ax.get_xlim()
        ax.plot(lim, lim, "k:", lw=1)
        ax.set(title=title, xlabel="OCaml gradient entries (W, B, x0)", ylabel=ylabel)
        ax.legend(fontsize=8)
    ns = sorted(summary["ocaml_gradients"])
    for label in comparisons:
        axes[2].semilogy(ns, [max(summary["ocaml_gradients"][n][label], 1e-17) for n in ns], "o-", label=label)
    axes[2].set(xlabel="n", ylabel="relative error", title="OCaml vs Python implicit gradients")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ocaml_gradients.png", dpi=150)
    plt.close(fig)


def dlqr_gradients(out_dir, summary):
    """dlqr custom-VJP gradients vs direct autodiff through solve_lqr, per LQR field (symmetric parts)"""
    implicit_loss, direct_loss_lqr, params = dlqr_losses(20, 5)
    g_imp = sym_lqr_grads(jax.grad(implicit_loss)(params))
    g_dir = sym_lqr_grads(jax.grad(direct_loss_lqr)(params))
    errs = {"x0": rel_err(g_imp.x0, g_dir.x0)}
    errs.update({f: rel_err(getattr(g_imp.lqr, f), getattr(g_dir.lqr, f)) for f in g_dir.lqr._fields})
    summary["dlqr_gradients"] = errs
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(errs), [max(v, 1e-18) for v in errs.values()])
    ax.axhline(1e-8, ls=":", c="r", label="test tolerance")
    ax.set(yscale="log", ylabel="relative error", title="dlqr implicit vs direct gradients (n=20, m=5, T=30)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dlqr_gradients.png", dpi=150)
    plt.close(fig)
    print(f"dlqr gradients: {errs}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-grid", action="store_true", help="skip the (slow) dimension-grid heat-map")
    args = parser.parse_args()
    out_dir = REPO / "fig_dump" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"created": datetime.now().isoformat(timespec="seconds")}
    control_problems(out_dir, summary)
    ocaml_comparison(out_dir, summary)
    gradients(out_dir, summary)
    gradient_methods(out_dir, summary)
    ocaml_gradients(out_dir, summary)
    dlqr_gradients(out_dir, summary)
    if not args.skip_grid:
        kkt_grid(out_dir, summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved report to {out_dir}")


if __name__ == "__main__":
    main()
