"""
Stages 2-3: infer the inputs of a trial with the diffilqrax sequential iLQR solver and compare with OCaml.

For every trial in each ``--ref`` directory (written by ``examples/ocaml_mgu_ref.ml``):

1. objective parity: the JAX objective of the saved OCaml inputs equals OCaml's ``cost_saved_u``;
2. stationarity: the gradient of the JAX objective is small at OCaml's converged ``ilqr_u``, and a JAX
   solve started there stays there;
3. exact-cost solve from ``u = 0``: optimal cost, ``u*`` and latents against OCaml's solve. The objective
   is non-convex and the two solvers take different paths (exact Hessian with Levenberg-Marquardt
   regularisation vs OCaml's approximate Student-t Hessian), so they can converge to different local
   optima; compare costs rather than ``u*``;
4. proxy solve (Gaussian prior with the same scales): differences from the exact-cost solution.

    python ilqr_parity.py --ref /path/to/ref_test --ref /path/to/ref_train
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp

from diffilqrax import ilqr
from diffilqrax.typs import iLQRParams
from load_ocaml import load_params, load_reference, run_dirs
from mgu2 import make_system, make_theta, readout, rollout, total_cost
from plots import plot_cost_history, plot_trial

jax.config.update("jax_enable_x64", True)

N_BEG = 12
FIGS = Path(__file__).parent / "figs"


def r2(o: onp.ndarray, tail: onp.ndarray) -> float:
    """Coefficient of determination of the readout ``o`` (T, n_out) against the observed tail"""
    pred = o[N_BEG - 1 :]
    return float(1 - onp.sum((pred - tail) ** 2) / onp.sum((tail - tail.mean(axis=0)) ** 2))


def solve(theta, us_init, proxy: bool, max_iter: int, alpha_init: float, kkt_tol: float):
    """Run ``ilqr_solver`` on the MGU2 problem; returns inputs, latents, cost, history and wall time"""
    n, m = theta.uh.shape[0], theta.B.shape[0]
    model = make_system(n, m, us_init.shape[0], proxy=proxy)
    params = iLQRParams(x0=jnp.zeros(n), theta=theta)
    run = jax.jit(lambda u0: ilqr.ilqr_solver(model, params, u0, max_iter=max_iter, alpha_init=alpha_init, kkt_tol=kkt_tol))
    t0 = time.time()
    (xs, us, _), cost, hist = run(jnp.asarray(us_init))
    cost.block_until_ready()
    return onp.asarray(us), onp.asarray(xs[1:]), float(cost), onp.asarray(hist), time.time() - t0


def compare_trial(params, ref, i: int, label: str, args) -> dict:
    """All checks for trial ``i`` of a reference directory"""
    theta = make_theta(params, ref["saved_tail"][i], N_BEG)
    tail = ref["saved_tail"][i]
    J = jax.jit(total_cost, static_argnames="proxy")
    row = {"trial": f"{label}[{i}]"}

    # 1. objective parity on the saved inputs
    j_saved = float(J(theta, jnp.asarray(ref["saved_u"][i])))
    row["cost_saved_rel"] = abs(j_saved - ref["cost_saved_u"][i]) / abs(ref["cost_saved_u"][i])

    # 2. stationarity of the OCaml optimum under the JAX objective
    g = jax.grad(total_cost, argnums=1)(theta, jnp.asarray(ref["ilqr_u"][i]))
    row["grad_at_ocaml_u*"] = float(jnp.abs(g).max())

    # warm start at the OCaml optimum: a converged OCaml solution should not move
    us_w, _, cost_w, _, _ = solve(theta, ref["ilqr_u"][i], False, args.warm_iter, args.alpha_init, args.kkt_tol)
    row["warm_dJ_rel"] = (cost_w - ref["ilqr_cost"][i]) / abs(ref["ilqr_cost"][i])
    row["warm_max|du|"] = float(onp.abs(us_w - ref["ilqr_u"][i]).max())

    # 3. exact-cost solve from zero (non-convex: may reach a different local optimum than OCaml)
    us, zs, cost, hist, secs = solve(theta, onp.zeros_like(ref["ilqr_u"][i]), False, args.max_iter, args.alpha_init, args.kkt_tol)
    row["J_ocaml"] = ref["ilqr_cost"][i]
    row["J_jax"] = cost
    row["cost_rel"] = (cost - ref["ilqr_cost"][i]) / abs(ref["ilqr_cost"][i])
    row["max|du*|"] = float(onp.abs(us - ref["ilqr_u"][i]).max())
    row["max|dz*|"] = float(onp.abs(zs - ref["ilqr_z"][i]).max())
    # the history keeps repeating the final cost once no step is accepted
    row["iters_jax"] = int(onp.nonzero(hist != hist[-1])[0].max(initial=-1)) + 1
    row["iters_ocaml"] = int(onp.isfinite(ref["ilqr_cost_hist"][i]).sum()) - 1
    row["secs"] = secs
    o_jax = onp.asarray(readout(theta, jnp.asarray(zs)))

    # 4. Gaussian proxy prior
    us_p, zs_p, _, hist_p, _ = solve(theta, onp.zeros_like(us), True, args.max_iter, args.alpha_init, args.kkt_tol)
    o_p = onp.asarray(readout(theta, jnp.asarray(zs_p)))
    row["proxy_max|du|"] = float(onp.abs(us_p - us).max())
    row["proxy_dJ_exact"] = float(J(theta, jnp.asarray(us_p))) - cost
    row["R2_exact"] = r2(o_jax, tail)
    row["R2_proxy"] = r2(o_p, tail)
    row["R2_saved"] = r2(ref["fwd_o"][i], tail)

    if i < args.n_plot:
        plot_trial(
            FIGS / f"ilqr_{label}_trial{i}.png",
            f"iLQR posterior mean, {label} trial {i}",
            tail,
            {
                "OCaml iLQR": {"z": ref["ilqr_z"][i], "o": ref["ilqr_o"][i], "u": ref["ilqr_u"][i]},
                "JAX exact": {"z": zs, "o": o_jax, "u": us},
                "JAX proxy": {"z": zs_p, "o": o_p, "u": us_p},
            },
            u_dims=range(4),
        )
        plot_cost_history(
            FIGS / f"ilqr_cost_{label}_trial{i}.png",
            f"iLQR objective, {label} trial {i}",
            {"OCaml dilqr": ref["ilqr_cost_hist"][i], "JAX exact": hist, "JAX proxy (own objective)": hist_p},
            proxy="JAX proxy (own objective)",
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", type=Path, action="append", required=True, help="ocaml_mgu_ref.ml output dir")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--warm-iter", type=int, default=50, help="iterations of the solve started at OCaml's u*")
    parser.add_argument("--alpha-init", type=float, default=2.0, help="first line-search step is alpha_init / 2")
    parser.add_argument("--kkt-tol", type=float, default=1e-6)
    parser.add_argument("--n-plot", type=int, default=3)
    args = parser.parse_args()

    params = load_params(run_dirs().network_parameters / "final")
    rows = []
    for ref_dir in args.ref:
        ref = load_reference(ref_dir)
        for i in range(ref["saved_u"].shape[0]):
            rows.append(compare_trial(params, ref, i, ref_dir.name, args))
            print("  ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()), flush=True)
    print(f"figures in {FIGS}")


if __name__ == "__main__":
    main()
