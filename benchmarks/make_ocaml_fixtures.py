"""
Generate OCaml dilqr reference solutions as test fixtures.

Runs ``examples/pendulum_ref.exe`` and ``examples/rnn_ref.exe`` from a local dilqr checkout (built
in the ``dilqr-ref`` opam switch) and saves their outputs to ``tests/fixtures/ocaml_*.npz``.

Usage:
    DILQR_DIR=/path/to/dilqr python benchmarks/make_ocaml_fixtures.py

Build the executables first:
    cd $DILQR_DIR && opam exec --switch dilqr-ref -- dune build ./examples/pendulum_ref.exe ./examples/rnn_ref.exe
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as onp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
from problems import vanilla_rnn_params  # pylint: disable=wrong-import-position

DEFAULT_DILQR = Path.home() / "Documents/01 Zenith/17 IQLR/Code/zebrafish_20240416/ocaml-code/marine_code/dilqr"
DILQR_DIR = Path(os.environ.get("DILQR_DIR", DEFAULT_DILQR))
FIXTURES = REPO / "tests" / "fixtures"


def run_ocaml(exe: str, out_dir: Path, *args: str) -> float:
    """Run an OCaml reference executable in the dilqr-ref switch, returning the wall time"""
    path = DILQR_DIR / "_build" / "default" / "examples" / exe
    if not path.exists():
        raise FileNotFoundError(f"{path} not built; see the module docstring")
    cmd = ["opam", "exec", "--switch", "dilqr-ref", "--", str(path), "-d", str(out_dir), "-tmp", str(out_dir), *args]
    start = time.perf_counter()
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    print("\n".join(proc.stdout.strip().splitlines()[-3:]))
    return elapsed


def load_outputs(out_dir: Path) -> dict:
    """Load xs, us and the cost history written by an OCaml reference program"""
    return {
        "Xs": onp.loadtxt(out_dir / "xs", ndmin=2),
        "Us": onp.loadtxt(out_dir / "us", ndmin=2),
        "costs": onp.loadtxt(out_dir / "costs", ndmin=1),
    }


def make_pendulum(linesearch: bool) -> None:
    """Pendulum from pendulum.ml"""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        args = ["-linesearch"] if linesearch else []
        elapsed = run_ocaml("pendulum_ref.exe", out, *args)
        data = load_outputs(out)
    name = "ocaml_pendulum" + ("_linesearch" if linesearch else "")
    onp.savez(FIXTURES / f"{name}.npz", **data, x0=onp.array([2.0, 1.1]), friction=1.0, linesearch=linesearch, seconds=elapsed)
    print(f"saved {name}.npz: cost {data['costs'][-1]:.12f} after {len(data['costs'])} evaluations")


def make_rnn() -> None:
    """Vanilla tanh RNN with parameters shared with tests/problems.vanilla_rnn"""
    params = vanilla_rnn_params()
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "params"
        out = Path(tmp) / "out"
        pdir.mkdir()
        out.mkdir()
        for key, value in params.items():
            onp.savetxt(pdir / key, onp.atleast_2d(value), delimiter="\t", fmt="%.17g")
        elapsed = run_ocaml("rnn_ref.exe", out, "-p", str(pdir), "-linesearch")
        data = load_outputs(out)
    onp.savez(FIXTURES / "ocaml_rnn.npz", **data, **params, linesearch=True, seconds=elapsed)
    print(f"saved ocaml_rnn.npz: cost {data['costs'][-1]:.12f} after {len(data['costs'])} evaluations")


if __name__ == "__main__":
    FIXTURES.mkdir(exist_ok=True)
    make_pendulum(linesearch=False)
    make_pendulum(linesearch=True)
    make_rnn()
