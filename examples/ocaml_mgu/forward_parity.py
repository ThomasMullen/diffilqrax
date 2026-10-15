"""
Stage 1: forward parity of the JAX MGU2 rollout with the OCaml model.

Two comparisons:

* ``--ref DIR`` (repeatable): full-precision rollouts written by ``examples/ocaml_mgu_ref.ml`` from the
  same saved parameters. This is the exact check (expected max |dz| ~ 1e-12).
* ``--drive``: every trial of every MPI node in ``results/final_*``. These predictions were made with
  parameters about one Adam step away from the saved ``final.*``, so agreement is limited to ~1e-2.

    python forward_parity.py --ref /path/to/ref_test --ref /path/to/ref_train --drive
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp

from load_ocaml import load_node, load_params, load_reference, n_nodes, run_dirs
from mgu2 import make_theta, readout, rollout
from plots import plot_trial

jax.config.update("jax_enable_x64", True)

N_BEG = 12
FIGS = Path(__file__).parent / "figs"


def jax_forward(params, us: onp.ndarray, tail: onp.ndarray):
    """Rollout and readout of a batch of inputs (B, T, m)"""
    theta = make_theta(params, tail[0], N_BEG)
    zs = jax.jit(jax.vmap(lambda u: rollout(theta, u)))(jnp.asarray(us))
    return onp.asarray(zs), onp.asarray(readout(theta, zs))


def summary(name: str, dz: onp.ndarray, do: onp.ndarray, z: onp.ndarray):
    """Print max and relative errors"""
    rel = onp.sqrt(onp.mean(dz**2)) / onp.sqrt(onp.mean(z**2))
    print(f"{name:28s} trials={dz.shape[0]:5d}  max|dz|={dz.max():.3e}  max|do|={do.max():.3e}  rel rms dz={rel:.3e}")
    return dz.max(), do.max()


def check_reference(params, ref_dir: Path, n_plot: int):
    """Exact parity against ocaml_mgu_ref.ml outputs"""
    ref = load_reference(ref_dir)
    z, o = jax_forward(params, ref["saved_u"], ref["saved_tail"])
    dz, do = onp.abs(z - ref["fwd_z"]), onp.abs(o - ref["fwd_o"])
    out = summary(f"reference {ref_dir.name}", dz, do, ref["fwd_z"])
    for i in range(min(n_plot, z.shape[0])):
        plot_trial(
            FIGS / f"forward_{ref_dir.name}_trial{i}.png",
            f"forward rollout of saved u, {ref_dir.name} trial {i}",
            ref["saved_tail"][i],
            {"OCaml": {"z": ref["fwd_z"][i], "o": ref["fwd_o"][i]}, "JAX": {"z": z[i], "o": o[i]}},
        )
    return out


def check_drive(params, results: Path, n_plot: int):
    """Parity against every saved prediction (limited by the parameter mismatch)"""
    for split in ("test", "train"):
        dzs, dos, zs = [], [], []
        for node in range(n_nodes(results, split)):
            x = load_node(results, split, node)
            z, o = jax_forward(params, x["u"], x["tail"])
            dzs.append(onp.abs(z - x["z"]))
            dos.append(onp.abs(o - x["o"]))
            zs.append(x["z"])
            if node == 0:
                for i in range(n_plot):
                    plot_trial(
                        FIGS / f"forward_drive_{split}_trial{i}.png",
                        f"forward rollout of saved u, drive {split} node 0 trial {i}",
                        x["tail"][i],
                        {"OCaml": {"z": x["z"][i], "o": x["o"][i]}, "JAX": {"z": z[i], "o": o[i]}},
                    )
        summary(f"drive {split} (all nodes)", onp.concatenate(dzs), onp.concatenate(dos), onp.concatenate(zs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", type=Path, action="append", default=[], help="ocaml_mgu_ref.ml output dir")
    parser.add_argument("--drive", action="store_true", help="sweep all saved predictions on the drive")
    parser.add_argument("--n-plot", type=int, default=3)
    args = parser.parse_args()

    dirs = run_dirs()
    params = load_params(dirs.network_parameters / "final")
    for ref_dir in args.ref:
        check_reference(params, ref_dir, args.n_plot)
    if args.drive:
        check_drive(params, dirs.results, args.n_plot)
    print(f"figures in {FIGS}")


if __name__ == "__main__":
    main()
