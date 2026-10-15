"""
Write ``tests/fixtures/ocaml_mgu.npz`` from the saved OCaml parameters and ``ocaml_mgu_ref.ml`` outputs.

    python make_fixture.py --ref /path/to/ref_test --ref /path/to/ref_train
"""

import argparse
from pathlib import Path

import numpy as onp

from load_ocaml import REFERENCE_KEYS, load_params, load_reference, run_dirs

FIXTURE = Path(__file__).parents[2] / "tests" / "fixtures" / "ocaml_mgu.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", type=Path, action="append", required=True, help="ocaml_mgu_ref.ml output dir")
    args = parser.parse_args()

    params = load_params(run_dirs().network_parameters / "final")
    refs = [load_reference(d) for d in args.ref]
    max_hist = max(r["ilqr_cost_hist"].shape[1] for r in refs)
    for r in refs:
        pad = max_hist - r["ilqr_cost_hist"].shape[1]
        r["ilqr_cost_hist"] = onp.pad(r["ilqr_cost_hist"], ((0, 0), (0, pad)), constant_values=onp.nan)
    data = {f"param_{k}": onp.asarray(v) for k, v in params._asdict().items()}
    data.update({k: onp.concatenate([r[k] for r in refs]) for k in REFERENCE_KEYS})
    data["source"] = onp.concatenate([[d.name] * r["saved_u"].shape[0] for d, r in zip(args.ref, refs)])
    onp.savez_compressed(FIXTURE, **data)
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size / 1e6:.1f} MB, {data['saved_u'].shape[0]} trials)")


if __name__ == "__main__":
    main()
