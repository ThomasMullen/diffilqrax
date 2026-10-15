"""Figures comparing JAX and OCaml trajectories"""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as onp  # pylint: disable=wrong-import-position

N_BEG = 12


def plot_trial(
    path: Path,
    title: str,
    tail: onp.ndarray,
    curves: dict,
    z_dims: Sequence[int] = (0, 1, 2, 3),
    u_dims: Optional[Sequence[int]] = None,
):
    """
    One trial: selected latents, inputs, all output channels against the observed tail, and errors.

    ``curves`` maps a label to a dict with ``z`` (T, n), ``o`` (T, n_out) and optionally ``u`` (T, m).
    The first entry is the reference that errors are computed against.
    """
    labels = list(curves)
    ref = curves[labels[0]]
    styles = ["-", "--", ":", "-."]
    n_out = tail.shape[1]
    show_u = u_dims is not None and all("u" in c for c in curves.values())
    n_rows = 3 + int(show_u)
    fig, axes = plt.subplots(n_rows, max(n_out, len(z_dims)), figsize=(2.6 * max(n_out, len(z_dims)), 2.2 * n_rows))
    t_obs = onp.arange(tail.shape[0]) + N_BEG - 1
    for j in range(axes.shape[1]):
        ax = axes[0, j]
        if j < len(z_dims):
            for s, (lab, c) in zip(styles, curves.items()):
                ax.plot(c["z"][:, z_dims[j]], s, lw=1, label=lab)
            ax.set_title(f"z[{z_dims[j]}]", fontsize=8)
        else:
            ax.axis("off")
        ax = axes[1, j]
        if j < n_out:
            ax.plot(t_obs, tail[:, j], color="k", lw=0.8, alpha=0.5, label="tail")
            for s, (lab, c) in zip(styles, curves.items()):
                ax.plot(c["o"][:, j], s, lw=1, label=lab)
            ax.set_title(f"o[{j}]", fontsize=8)
        else:
            ax.axis("off")
        if show_u:
            ax = axes[2, j]
            if j < len(u_dims):
                for s, (lab, c) in zip(styles, curves.items()):
                    ax.plot(c["u"][:, u_dims[j]], s, lw=1, label=lab)
                ax.set_title(f"u[{u_dims[j]}]", fontsize=8)
            else:
                ax.axis("off")
    ax = axes[-1, 0]
    for lab in labels[1:]:
        err = onp.abs(curves[lab]["z"] - ref["z"]).max(axis=1)
        ax.semilogy(onp.maximum(err, 1e-17), lw=1, label=f"{lab} vs {labels[0]}")
    ax.set_title("max |dz| over dims", fontsize=8)
    ax.legend(fontsize=6)
    for ax in axes[-1, 1:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].legend(fontsize=6)
    for ax in axes.ravel():
        ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cost_history(path: Path, title: str, histories: dict, proxy: Optional[str] = None):
    """
    Objective per iteration: left, the cost itself; right, the gap to the lowest cost reached on the
    same objective, on a log scale. ``proxy`` names a history on a different objective, whose gap is
    taken to its own minimum.
    """
    # OCaml histories are nan-padded, diffilqrax histories zero-padded after convergence
    hs = {lab: onp.trim_zeros(onp.asarray(h)[onp.isfinite(h)], "b") for lab, h in histories.items()}
    best = min(h.min() for lab, h in hs.items() if lab != proxy)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.2))
    for lab, h in hs.items():
        ax0.plot(h, lw=1, label=lab)
        gap = h - (h.min() if lab == proxy else best)
        ax1.semilogy(onp.maximum(gap, 1e-10), lw=1, label=lab)
    lo = min(h[min(5, len(h) - 1) :].min() for h in hs.values())
    hi = max(h[min(5, len(h) - 1) :].max() for h in hs.values())
    ax0.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax0.set_xlabel("iteration")
    ax0.set_ylabel("cost (from iteration 5)")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("cost - lowest cost")
    ax1.legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
