"""
Readers for the outputs of an OCaml iLQR-VAE run (MGU2 dynamics, Student prior, Gaussian likelihood).

Three kinds of files are produced by the OCaml code:

* parameters saved by ``Owl_parameters.save_to_files``: tab-separated text, 6 significant figures,
  with row vectors written as columns (``<prefix>.generative.dynamics.wh`` ...);
* per-MPI-node prediction arrays saved with ``Arr.save_npy`` but without the ``.npy`` extension
  (``final_{u,z,o,tail,bout,fish}_{test,train}_<node>``, ``original_idx_{test,train}_<node>``);
* posterior summaries saved by ``Vae.save_results`` as gzipped text (``<prefix>.mean.u.<id>`` ...).
"""

import gzip
from pathlib import Path
from typing import Dict, NamedTuple, Union

import numpy as onp

PathLike = Union[str, Path]

DATA_ROOT = Path("/Volumes/TomOrgerLab/17 IQLR/Data/20230320_benchmark")
RUN = "full_mgu_120_10_0.008_1_110_12_10000_tanh"


class RunDirs(NamedTuple):
    """Directories of a single OCaml run"""

    results: Path
    network_parameters: Path
    tmpdir: Path


def run_dirs(root: PathLike = DATA_ROOT, run: str = RUN) -> RunDirs:
    """Results, final-parameter and checkpoint directories of ``run`` under ``root``"""
    root = Path(root)
    return RunDirs(root / "results" / run, root / "network_parameters" / run, root / "tmpdir" / run)


class OCamlParams(NamedTuple):
    """Generative parameters in OCaml (row-vector) orientation, as float64 numpy arrays"""

    wh: onp.ndarray  # (m, n) input weights, normalised inside the dynamics
    uh: onp.ndarray  # (n, n) candidate recurrent weights
    uf: onp.ndarray  # (n, n) gate recurrent weights
    bh: onp.ndarray  # (n,)
    bf: onp.ndarray  # (n,)
    c: onp.ndarray  # (n_out, n) readout
    bias: onp.ndarray  # (n_out,)
    variances: onp.ndarray  # (n_out,)
    spatial_stds: onp.ndarray  # (m,)
    first_step: onp.ndarray  # (m,)
    nu: float


def load_owl_txt(path: PathLike) -> onp.ndarray:
    """Read a matrix written by ``Mat.save_txt`` (optionally gzipped) as a 2D float64 array"""
    path = Path(path)
    opener = gzip.open if path.read_bytes()[:2] == b"\x1f\x8b" else open
    with opener(path, "rt") as f:
        rows = [line.split() for line in f if line.strip()]
    return onp.array(rows, dtype=onp.float64)


def load_npy(path: PathLike) -> onp.ndarray:
    """Read an ``Arr.save_npy`` file (no ``.npy`` extension)"""
    with open(path, "rb") as f:
        return onp.load(f)


def _vector(path: Path) -> onp.ndarray:
    """Row vectors are saved transposed, as a single column"""
    x = load_owl_txt(path)
    assert x.ndim == 2 and min(x.shape) == 1, f"{path.name} is not a vector: {x.shape}"
    return x.ravel()


def load_params(prefix: PathLike) -> OCamlParams:
    """
    Load generative parameters saved with ``Owl_parameters.save_to_files``.

    Parameters
    ----------
    prefix : PathLike
        File prefix, e.g. ``network_parameters/<run>/final`` or ``tmpdir/<run>/test_10000``.

    Returns
    -------
    OCamlParams
        Dynamics, likelihood and prior parameters.
    """
    prefix = Path(prefix)

    def f(name: str) -> Path:
        return prefix.with_name(f"{prefix.name}.generative.{name}")

    params = OCamlParams(
        wh=load_owl_txt(f("dynamics.wh")),
        uh=load_owl_txt(f("dynamics.uh")),
        uf=load_owl_txt(f("dynamics.uf")),
        bh=_vector(f("dynamics.bh")),
        bf=_vector(f("dynamics.bf")),
        c=load_owl_txt(f("likelihood.c")),
        bias=_vector(f("likelihood.bias")),
        variances=_vector(f("likelihood.variances")),
        spatial_stds=_vector(f("prior.spatial_stds")),
        first_step=_vector(f("prior.first_step")),
        nu=float(load_owl_txt(f("prior.nu")).item()),
    )
    m, n = params.wh.shape
    n_out = params.c.shape[0]
    assert params.uh.shape == params.uf.shape == (n, n)
    assert params.bh.shape == params.bf.shape == (n,)
    assert params.c.shape == (n_out, n) and params.bias.shape == params.variances.shape == (n_out,)
    assert params.spatial_stds.shape == params.first_step.shape == (m,)
    return params


def load_node(results: PathLike, split: str, node: int) -> Dict[str, onp.ndarray]:
    """
    Load the predictions saved by one MPI node.

    Returns a dict with ``u`` (B, T, m) posterior-mean inputs, ``z`` (B, T, n) latents,
    ``o`` (B, T, n_out) noiseless outputs, ``tail`` (B, T - n_beg + 1, n_out) observed data,
    ``bout`` and ``fish`` (B,) labels and ``idx`` (B,) original trial indices.
    """
    results = Path(results)
    out = {k: load_npy(results / f"final_{k}_{split}_{node}") for k in ("u", "z", "o", "tail", "bout", "fish")}
    out["idx"] = load_npy(results / f"original_idx_{split}_{node}")
    for k in ("bout", "fish", "idx"):
        out[k] = out[k].reshape(out[k].shape[0])
    return out


def n_nodes(results: PathLike, split: str = "test") -> int:
    """Number of MPI nodes that saved predictions"""
    return len(list(Path(results).glob(f"final_u_{split}_*")))


def load_mean(prefix: PathLike, trial_id: int = 0) -> Dict[str, onp.ndarray]:
    """
    Load the posterior summary written by ``Vae.save_results`` for one trial.

    ``u`` is the deterministic posterior mean (T, m). ``z`` (T - n_beg + 1, n) and ``o`` (T, n_out)
    are Monte Carlo averages over posterior samples, not the rollout of ``u``.
    """
    prefix = Path(prefix)

    def f(name: str) -> Path:
        return prefix.with_name(f"{prefix.name}.mean.{name}")

    return {
        "u": load_owl_txt(f(f"u.{trial_id}")),
        "z": load_owl_txt(f(f"z.{trial_id}")),
        "o": load_owl_txt(f(f"o.{trial_id}.tail")),
    }
