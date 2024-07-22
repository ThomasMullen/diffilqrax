"""Test runtime of sequential vs parallel LQR solver"""

from pathlib import Path
from os import getcwd
from typing import Callable
from functools import partial
import time

from tqdm.auto import trange, tqdm
from matplotlib.pyplot import subplots, close, style
import cmcrameri.cm as cmc
import numpy as np
from numpy.typing import NDArray

import jax
from jax import jit, device_put, devices, Device
import jax.numpy as jnp
import jax.random as jr

from diffilqrax.typs import (
    LQR,
    LQRParams,
    ModelDims,
)
from diffilqrax.plqr import solve_plqr
from diffilqrax.lqr import solve_lqr
from diffilqrax.utils import keygen, initialise_stable_dynamics


jax.config.update("jax_enable_x64", True)  # Match np precision

PLOT_URL = (
    "https://gist.githubusercontent.com/"
    "ThomasMullen/e4a6a0abd54ba430adc4ffb8b8675520/"
    "raw/1189fbee1d3335284ec5cd7b5d071c3da49ad0f4/"
    "figure_style.mplstyle"
)
style.use(PLOT_URL)

fig_dir = Path(Path(getcwd()), "fig_dump")
fig_dir.mkdir(exist_ok=True)

def tuple_block_until_ready(x_tuple):
    """Apply block for leaves in a pytree struct"""
    def block_each_element(x):
        return x.block_until_ready()
    return jax.tree.map(block_each_element, x_tuple)


def setup_lqr(
    dims: ModelDims,
    device: Device,
    pen_weight: dict = {"Q": 10.0, "R": 1.0, "Qf": 1e0, "S": 1e-3},
) -> LQR:
    """Setup LQR problem"""
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    # initialise dynamics
    span_time_m = (dims.horizon, 1, 1)
    span_time_v = (dims.horizon, 1)
    A = initialise_stable_dynamics(next(skeys), dims.n, dims.horizon, radii=0.6)
    B = jnp.tile(jr.normal(next(skeys), (dims.n, dims.m)), span_time_m)
    a = 2 * jnp.tile(jr.normal(next(skeys), (dims.n,)), span_time_v)
    # define cost matrices
    Q = pen_weight["Q"] * jnp.tile(jnp.eye(dims.n), span_time_m)
    q = -5 * jnp.tile(jnp.ones((dims.n,)), span_time_v)
    R = pen_weight["R"] * jnp.tile(jnp.eye(dims.m), span_time_m)
    r = 0.0 * jnp.tile(jnp.ones(dims.m), span_time_v)
    S = 0 * pen_weight["S"] * jnp.tile(jnp.ones((dims.n, dims.m)), span_time_m)
    Qf = 2 * pen_weight["Q"] * jnp.eye(dims.n)
    qf = 0 * jnp.ones((dims.n,))
    # construct LQR
    lqr_mats = LQR(A, B, a, Q, q, R, r, S, Qf, qf)
    device_params = LQR(*(device_put(jnp.asarray(val), device) for val in lqr_mats))
    return device_params()


def get_average_runtimes(
    func: Callable, n_iter: int, device: Device, state_dim: int
) -> NDArray:
    runtimes = np.empty(input_sizes.shape)
    for i, input_size in tqdm(enumerate(input_sizes), total=runtimes.shape[0]):
        dims = ModelDims(
            n=state_dim, m=state_dim, horizon=input_size, dt=0.1
        )  # define model shape
        device_model = setup_lqr(dims=dims, device=device)
        x0 = device_put(jnp.ones(dims.n), device)
        res = func(LQRParams(x0, device_model))  # compilation run
        for e in res:
            # _ = e.block_until_ready()
            _ = tuple_block_until_ready(e)
        tic = time.time()
        for _ in trange(n_iter, leave=False):
            res = func(LQRParams(x0, device_model))
            for e in res:
                # _ = e.block_until_ready()
                _ = tuple_block_until_ready(e)
        runtimes[i] = (time.time() - tic) / n_iter
    return runtimes


# jit and specify device solver
# cpu_lqr = jit(solve_lqr, static_argnames="sys_dims", backend="cpu")
cpu_lqr = jit(solve_lqr, backend="cpu")
cpu_plqr = jit(solve_plqr, backend="cpu")
gpu_lqr = jit(solve_lqr, backend="gpu")
gpu_plqr = jit(solve_plqr, backend="gpu")

cpu = devices("cpu")[0]
gpu = devices("gpu")[0]


# Single size example
n_iter = 5
log10T = 5
input_sizes = np.logspace(2, log10T, num=10, base=10).astype(int)
dim_n=4

# Test runtime for each function
cpu_sequential_runtimes = get_average_runtimes(cpu_lqr, n_iter, cpu, dim_n)
cpu_parallel_runtimes = get_average_runtimes(cpu_plqr, n_iter, cpu, dim_n)
gpu_sequential_runtimes = get_average_runtimes(gpu_lqr, n_iter, gpu, dim_n)
gpu_parallel_runtimes = get_average_runtimes(gpu_plqr, n_iter, gpu, dim_n)


# Pretty plots of results
fig, axes = subplots(ncols=2, figsize=(15, 6), sharex=True, sharey=True)
axes[0].loglog(
    input_sizes,
    cpu_sequential_runtimes,
    label="Sequential-CPU",
    linestyle="-.",
    linewidth=3,
)
axes[0].loglog(input_sizes, cpu_parallel_runtimes, label="Parallel-CPU", linewidth=3)
axes[0].legend()

axes[1].loglog(
    input_sizes,
    gpu_sequential_runtimes,
    label="Sequential-GPU",
    linestyle="-.",
    linewidth=3,
)
axes[1].loglog(input_sizes, gpu_parallel_runtimes, label="Parallel-GPU", linewidth=3)
_ = axes[0].set_ylabel("Average run time (seconds)")

for ax in axes:
    _ = ax.set_xlabel("Number of data points")

_ = fig.suptitle("Runtime comparison on CPU and GPU", size=15)
_ = axes[1].legend()

fig.savefig(f"{fig_dir}/runtime_slqr_plqr_n{dim_n:03}.png", dpi=350)




# define time length
n_iter = 5
log10T = 5
input_sizes = np.logspace(2, log10T, num=10, base=10).astype(int)
dim_ns=np.logspace(2, 7, num=6, base=2).astype(int)

run_time_log = np.empty((4,dim_ns.size, input_sizes.size))

for ix, dim_n in enumerate(dim_ns):
    # Test runtime for each function
    cpu_sequential_runtimes = get_average_runtimes(cpu_lqr, n_iter, cpu, dim_n)
    run_time_log[0,ix] = cpu_sequential_runtimes
    cpu_parallel_runtimes = get_average_runtimes(cpu_plqr, n_iter, cpu, dim_n)
    run_time_log[1,ix] = cpu_parallel_runtimes
    gpu_sequential_runtimes = get_average_runtimes(gpu_lqr, n_iter, gpu, dim_n)
    run_time_log[2,ix] = gpu_sequential_runtimes
    gpu_parallel_runtimes = get_average_runtimes(gpu_plqr, n_iter, gpu, dim_n)
    run_time_log[3,ix] = gpu_parallel_runtimes
    


fig, axes = subplots(figsize=(8, 6), sharex = True, sharey=True)
colors = cmc.navia(np.linspace(0.2, 1, dim_ns.size))
for i, n in enumerate(dim_ns) : 
    axes.plot(input_sizes, run_time_log[2][i], label = f"Sequential LQR n = {n}", color = colors[i])
    axes.plot(input_sizes, run_time_log[3][i], label = f"Parallel LQR n = {n}", color = colors[i], linestyle = "-.")
axes.legend()

fig.savefig(f"{fig_dir}/runtime_slqr_plqr.png", dpi=350)