"""
Generate OCaml dilqr reference solutions as test fixtures.

Runs ``examples/pendulum_ref.exe``, ``examples/rnn_ref.exe`` and ``examples/rnn_grad_ref.exe`` from a
local dilqr checkout (built in the ``dilqr-ref`` opam switch) and saves their outputs to
``tests/fixtures/ocaml_*.npz``. The gradient fixtures cover the RNN sizes in ``GRAD_SIZES``.

Usage:
    DILQR_DIR=/path/to/dilqr python benchmarks/make_ocaml_fixtures.py

Build the executables first:
    cd $DILQR_DIR && opam exec --switch dilqr-ref -- dune build ./examples/pendulum_ref.exe ./examples/rnn_ref.exe ./examples/rnn_grad_ref.exe
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


def write_params(pdir: Path, params: dict) -> None:
    """Write problem parameters as full-precision tab-separated text for the OCaml programs"""
    pdir.mkdir()
    for key, value in params.items():
        onp.savetxt(pdir / key, onp.atleast_2d(value), delimiter="\t", fmt="%.17g")


def make_rnn() -> None:
    """Vanilla tanh RNN with parameters shared with tests/problems.vanilla_rnn"""
    params = vanilla_rnn_params()
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "params"
        out = Path(tmp) / "out"
        out.mkdir()
        write_params(pdir, params)
        elapsed = run_ocaml("rnn_ref.exe", out, "-p", str(pdir), "-linesearch")
        data = load_outputs(out)
    onp.savez(FIXTURES / "ocaml_rnn.npz", **data, **params, linesearch=True, seconds=elapsed)
    print(f"saved ocaml_rnn.npz: cost {data['costs'][-1]:.12f} after {len(data['costs'])} evaluations")


GRAD_SIZES = [(5, 3), (20, 5), (50, 10), (100, 20)]


def grad_fixture_name(n: int, linear: bool = False) -> str:
    """File stem of an OCaml gradient fixture"""
    return f"ocaml_rnn_grad{'_linear' if linear else ''}_n{n}"


def make_rnn_grad(n: int, m: int, linear: bool = False) -> None:
    """
    Implicit gradients of sum(tau*^2) wrt W, B and x0 for tests/problems.vanilla_rnn(n, m, linear=linear).

    ``grad_*`` come from dilqr as released (AD through the theta-dependent rollout of the tape);
    ``grad_*_fixed`` from ``~fixed_trajectory:true`` (Gauss-Newton implicit gradients).
    """
    params = vanilla_rnn_params(n, m)
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "params"
        out = Path(tmp) / "out"
        out.mkdir()
        write_params(pdir, params)
        args = ["-p", str(pdir), "-linesearch"] + (["-linear"] if linear else [])
        elapsed = run_ocaml("rnn_grad_ref.exe", out, *args)
        data = load_outputs(out)
        for key in ("loss", "grad_W", "grad_B", "grad_x0", "grad_W_fixed", "grad_B_fixed", "grad_x0_fixed"):
            data[key] = onp.loadtxt(out / key, ndmin=2)
    data["loss"] = data["loss"].item()
    data["grad_x0"] = data["grad_x0"].ravel()
    data["grad_x0_fixed"] = data["grad_x0_fixed"].ravel()
    name = grad_fixture_name(n, linear)
    onp.savez(FIXTURES / f"{name}.npz", **data, **params, n=n, m=m, linear=linear, linesearch=True, seconds=elapsed)
    print(f"saved {name}.npz: loss {data['loss']:.12f} in {elapsed:.1f}s")


if __name__ == "__main__":
    FIXTURES.mkdir(exist_ok=True)
    make_pendulum(linesearch=False)
    make_pendulum(linesearch=True)
    make_rnn()
    make_rnn_grad(5, 3, linear=True)
    for n_, m_ in GRAD_SIZES:
        make_rnn_grad(n_, m_)
