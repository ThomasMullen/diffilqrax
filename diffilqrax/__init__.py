"""Differentiable LQR and iLQR solvers in JAX."""

from importlib.metadata import PackageNotFoundError, version

from diffilqrax import (
    batch,
    diff_ilqr,
    diff_lqr,
    ilqr,
    lqr,
    parallel_dilqr,
    parallel_dlqr,
    parallel_ilqr,
    plqr,
    typs,
    utils,
)

try:
    __version__ = version("diffilqrax")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

__all__ = [
    "lqr",
    "ilqr",
    "diff_lqr",
    "diff_ilqr",
    "plqr",
    "parallel_ilqr",
    "parallel_dlqr",
    "parallel_dilqr",
    "batch",
    "typs",
    "utils",
    "__version__",
]
