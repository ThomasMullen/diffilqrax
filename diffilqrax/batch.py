"""
Batched iLQR solves: many problems that share a model, e.g. one input trajectory per recorded trial.

A batch is described as in ``jax.vmap``: ``in_axes`` is a pytree prefix of the arguments whose leaves are 0
(the leaf has a leading batch axis) or None (the leaf is shared by the whole batch). Three strategies map
the solver over the batch:

- "vmap": one vectorised solve. The solver's while loops stop once every batch element has converged, so
  the batch costs about as many iterations as its slowest element. Usually the fastest choice on GPU.
- "map": ``lax.map``, one element (or one chunk of ``batch_size`` elements, vectorised) at a time, so slow
  elements only hold up their own chunk and memory stays bounded. Usually the fastest choice on CPU.
- "shard": ``shard_map`` over a 1-D mesh of ``devices`` (by default ``jax.devices()``), with "vmap" or
  "map" (``inner``) on each device, so each device stops on its own shard. The batch is padded to a
  multiple of the number of devices. After ``jax.distributed.initialize()``, ``jax.devices()`` spans every
  host, so the same call runs on a multi-host cluster.

Slow elements (stragglers) cost the most under "vmap" and "shard". When the same batch is solved repeatedly,
e.g. once per training epoch, ``return_info=True`` gives the iterations each element needed. Sorting the
batch by them and solving with "map" in chunks of ``batch_size`` groups similar elements together. Starting
each solve from the previous solution (``Us_init``) usually cuts the iterations.

Gradients of :func:`batched_dilqr` are the implicit gradients of :func:`diffilqrax.diff_ilqr.dilqr`, attached
after the batched solve, so ``jax.grad`` of a loss summed over the batch works with every strategy without
differentiating the solver loops.
"""

import inspect
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as onp
from jax import lax
from jax.sharding import Mesh, PartitionSpec

from diffilqrax.diff_ilqr import implicit_dilqr
from diffilqrax.ilqr import ilqr_solver
from diffilqrax.parallel_dilqr import implicit_pdilqr
from diffilqrax.parallel_ilqr import pilqr_solver
from diffilqrax.typs import iLQRParams, System

try:  # jax >= 0.6
    from jax import shard_map as _shard_map  # pylint: disable=ungrouped-imports
except ImportError:
    from jax.experimental.shard_map import shard_map as _shard_map

# replication checks have no rules for the solver's loops, so they are switched off
_NO_REP_CHECK = {"check_vma" if "check_vma" in inspect.signature(_shard_map).parameters else "check_rep": False}

STRATEGIES = ("vmap", "map", "shard")
INNER = ("vmap", "map")


def _batch_mask(in_axes: Any, args: tuple) -> list:
    """Flat list over the leaves of ``args``: True where the leaf is batched (axis 0)"""
    if not isinstance(in_axes, (tuple, list)):
        in_axes = (in_axes,) * len(args)
    assert len(in_axes) == len(args), f"in_axes has {len(in_axes)} entries for {len(args)} arguments"

    def expand(axis, arg):
        assert axis in (0, None), f"only in_axes 0 or None are supported, got {axis}"
        return jax.tree.map(lambda _: axis is not None, arg)

    return jax.tree.leaves(jax.tree.map(expand, tuple(in_axes), args, is_leaf=lambda x: x is None))


def batch_map(
    fn: Callable,
    in_axes: Any = 0,
    strategy: str = "vmap",
    batch_size: Optional[int] = None,
    devices: Optional[Sequence[jax.Device]] = None,
    inner: str = "vmap",
) -> Callable:
    """
    Map ``fn`` over the leading axis of the arguments selected by ``in_axes``.

    Parameters
    ----------
    fn : Callable
        Function of one batch element.
    in_axes : Any, optional
        Pytree prefix of the arguments with leaves 0 (batched) or None (shared), as in ``jax.vmap``, by
        default 0 (every argument batched).
    strategy : str, optional
        "vmap" (default), "map" or "shard", see the module docstring.
    batch_size : int, optional
        Chunk size of ``lax.map`` for strategy "map" (and inner "map"); by default one element at a time.
    devices : Sequence[jax.Device], optional
        Devices of strategy "shard", by default ``jax.devices()``.
    inner : str, optional
        Mapping on each device for strategy "shard": "vmap" (default) or "map".

    Returns
    -------
    Callable
        Function of the batched arguments whose outputs have a leading batch axis.
    """
    assert strategy in STRATEGIES, f"strategy must be one of {STRATEGIES}, got {strategy}"
    assert inner in INNER, f"inner must be one of {INNER}, got {inner}"
    if strategy == "vmap":
        return jax.vmap(fn, in_axes=in_axes)

    def mapped(*args):
        mask = _batch_mask(in_axes, args)
        leaves, treedef = jax.tree.flatten(args)
        batched = [leaf for leaf, b in zip(leaves, mask) if b]
        shared = [leaf for leaf, b in zip(leaves, mask) if not b]
        assert batched, "in_axes selects no batched argument"
        n_batch = batched[0].shape[0]
        assert all(leaf.shape[0] == n_batch for leaf in batched), "batched leaves differ in their leading axis"

        def call(batched_i, shared_):
            it_b, it_s = iter(batched_i), iter(shared_)
            return fn(*treedef.unflatten([next(it_b) if b else next(it_s) for b in mask]))

        def local_map(kind, batched_, shared_):
            if kind == "vmap":
                return jax.vmap(call, in_axes=(0, None))(batched_, shared_)
            return lax.map(lambda batched_i: call(batched_i, shared_), batched_, batch_size=batch_size)

        if strategy == "map":
            return local_map("map", batched, shared)

        devs = jax.devices() if devices is None else list(devices)
        pad = -n_batch % len(devs)
        if pad:  # repeat the last element so every device gets the same number of elements
            batched = [jnp.concatenate([leaf, jnp.repeat(leaf[-1:], pad, axis=0)]) for leaf in batched]
        mesh = Mesh(onp.asarray(devs), ("batch",))
        # newer jax returns a decorator lambda when f is omitted, which pylint can infer here instead of the call
        out = _shard_map(  # pylint: disable=too-many-function-args
            lambda batched_, shared_: local_map(inner, batched_, shared_),
            mesh=mesh,
            in_specs=(PartitionSpec("batch"), PartitionSpec()),
            out_specs=PartitionSpec("batch"),
            **_NO_REP_CHECK,
        )(batched, shared)
        return jax.tree.map(lambda leaf: leaf[:n_batch], out)

    return mapped


def batched_ilqr_solver(
    model: System,
    params: iLQRParams,
    Us_init: jax.Array,
    in_axes: Any = (0, 0),
    parallel: bool = False,
    strategy: str = "vmap",
    batch_size: Optional[int] = None,
    devices: Optional[Sequence[jax.Device]] = None,
    inner: str = "vmap",
    **solver_kwargs,
):
    """
    Solve a batch of iLQR problems that share ``model``.

    Parameters
    ----------
    model : System
        The system model, shared by the batch.
    params : iLQRParams
        The iLQR parameters, with a leading batch axis on the leaves selected by ``in_axes[0]``.
    Us_init : Array
        Initial inputs, [BxTxm] if batched or [Txm] if shared (``in_axes[1] = None``).
    in_axes : Any, optional
        Batch axes of ``(params, Us_init)`` as in ``jax.vmap`` (0 or None leaves), by default (0, 0). For
        example ``(iLQRParams(x0=None, theta=theta_axes), None)`` batches only the theta leaves marked 0 in
        ``theta_axes``.
    parallel : bool, optional
        Use :func:`diffilqrax.parallel_ilqr.pilqr_solver` instead of
        :func:`diffilqrax.ilqr.ilqr_solver`, by default False.
    strategy, batch_size, devices, inner
        See :func:`batch_map`.
    **solver_kwargs
        Keyword arguments of the solver, e.g. ``max_iter``, ``kkt_tol``, ``linesearch``. With
        ``return_info=True`` the outputs end with the per-problem convergence flags and iteration counts.

    Returns
    -------
    Tuple
        The solver outputs (see :func:`diffilqrax.ilqr.ilqr_solver`), each with a leading batch axis.
    """
    solver = pilqr_solver if parallel else ilqr_solver
    return batch_map(lambda p, u: solver(model, p, u, **solver_kwargs), in_axes, strategy, batch_size, devices, inner)(
        params, Us_init
    )


def batched_dilqr(
    model: System,
    params: iLQRParams,
    Us_init: jax.Array,
    in_axes: Any = (0, 0),
    parallel: bool = False,
    strategy: str = "vmap",
    batch_size: Optional[int] = None,
    devices: Optional[Sequence[jax.Device]] = None,
    inner: str = "vmap",
    exact_hessian: bool = True,
    adjoint: str = "parallel",
    **solver_kwargs,
) -> jax.Array:
    """
    Differentiable solutions of a batch of iLQR problems that share ``model``.

    The batch is solved by :func:`batched_ilqr_solver` without gradients. The implicit gradients of
    :func:`diffilqrax.diff_ilqr.dilqr` are then attached to each solution, one LQR solve per problem, mapped
    with ``vmap`` (strategy "vmap", or "shard" with inner "vmap") or ``lax.map`` (otherwise, chunked by
    ``batch_size``). Reverse mode therefore never enters the solver loops or ``shard_map``.

    Parameters
    ----------
    model : System
        The system model, shared by the batch.
    params : iLQRParams
        The iLQR parameters, batched as in :func:`batched_ilqr_solver`.
    Us_init : Array
        Initial inputs, batched as in :func:`batched_ilqr_solver`. They receive no gradient.
    in_axes, parallel, strategy, batch_size, devices, inner
        See :func:`batched_ilqr_solver`.
    exact_hessian : bool, optional
        See :func:`diffilqrax.diff_ilqr.dilqr`, by default True.
    adjoint : str, optional
        Adjoint solver of :func:`diffilqrax.parallel_dilqr.pdilqr` if ``parallel``, by default "parallel".
    **solver_kwargs
        Keyword arguments of the solver, e.g. ``max_iter``, ``kkt_tol``, ``linesearch``.

    Returns
    -------
    Array
        The optimal state-control trajectories [Bx(T+1)x(n+m)], with implicit gradients wrt ``params``.
    """
    solver_kwargs.pop("return_converged", None)
    solution = batched_ilqr_solver(
        model,
        lax.stop_gradient(params),
        lax.stop_gradient(Us_init),
        in_axes,
        parallel,
        strategy,
        batch_size,
        devices,
        inner,
        **solver_kwargs,
    )[0]

    def attach(p, sol):
        if parallel:
            return implicit_pdilqr(model, p, sol, exact_hessian, adjoint)
        return implicit_dilqr(model, p, sol, exact_hessian)

    params_axes = in_axes[0] if isinstance(in_axes, (tuple, list)) else in_axes
    vectorised = strategy == "vmap" or (strategy == "shard" and inner == "vmap")
    return batch_map(attach, (params_axes, 0), "vmap" if vectorised else "map", batch_size)(params, solution)
