"""
Unit test for the parallel LQR module
"""

from pathlib import Path
import unittest
from os import getcwd
from itertools import product
from tqdm import trange
import time
import logging
import chex
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
from matplotlib.pyplot import subplots, close, style

from diffilqrax.typs import (
    LQR,
    LQRParams,
    ModelDims,
)
from diffilqrax.plqr import (
    solve_plqr,
    associative_riccati_scan,
    associative_opt_traj_scan,
    parallel_reverse_lin_integration,
    parallel_forward_lin_integration,
)
from diffilqrax.lqr import solve_lqr
from diffilqrax.utils import keygen, initialise_stable_dynamics

logging.getLogger("matplotlib.font_manager").disabled = True
# jax.config.update('jax_default_device', jax.devices('cpu')[0])
# jax.config.update('jax_platform_name', 'gpu')


PLOT_URL = (
    "https://gist.githubusercontent.com/"
    "ThomasMullen/e4a6a0abd54ba430adc4ffb8b8675520/"
    "raw/1189fbee1d3335284ec5cd7b5d071c3da49ad0f4/"
    "figure_style.mplstyle"
)
PLOTTING_ON = False
if PLOTTING_ON:
    # style.use("/home/marineschimel/code/diffilqrax/paper.mplstyle")
    style.use(PLOT_URL)
    FIG_DIR = Path(getcwd(), "fig_dump", "para_lqr")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
LONG_TIME_PROFILE = False

def is_jax_array(arr: Array) -> bool:
    """validate jax array type"""
    return isinstance(arr, jnp.ndarray) and not isinstance(arr, onp.ndarray)


def _gpu_available()->bool:
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        return True
    except:
        return False

def _tuple_block_until_ready(x_tuple):
    """Apply block for leaves in a pytree struct"""
    def block_each_element(x):
        return x.block_until_ready()
    return jax.tree_util.tree_map(block_each_element, x_tuple)

def setup_lqr(
    dims: chex.Dimensions,
    pen_weight: dict = {"Q": 10.0, "R": 1.0, "Qf": 1e0, "S": 1e-3},
) -> LQR:
    """Setup LQR problem"""
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    # initialise dynamics
    span_time_m = dims["TXX"]
    span_time_v = dims["TX"]
    dt = 0.1
    Uh = jnp.array(
        [[1, dt, 0.01], [-1 * dt, 1 - 0.1 * dt, 0.0], [-1 * dt, 1 - 0.1 * dt, 0.05]]
    )
    Wh = jnp.eye(3)  # jnp.array([[0.5, 2., 0.], [1., -1.2, 0.1]]).T * dt
    Q = 1.0
    A = jnp.tile(Uh, span_time_m)
    B = jnp.tile(Wh, span_time_m)
    a = jnp.tile(jr.normal(next(skeys), dims["N"]), span_time_v)
    # define cost matrices
    Q = pen_weight["Q"] * jnp.tile(jnp.eye(dims["N"][0]), span_time_m)
    q = -5 * jnp.tile(jnp.ones(dims["N"]), span_time_v)
    R = pen_weight["R"] * jnp.tile(jnp.eye(dims["M"][0]), span_time_m)
    r = -2.0 * jnp.tile(jnp.ones(dims["M"]), span_time_v)
    S = 0 * pen_weight["S"] * jnp.tile(jnp.ones(dims["NM"]), span_time_m)
    Qf = 0 * pen_weight["Q"] * jnp.eye(dims["N"][0])
    qf = 0 * jnp.ones(dims["N"])
    # construct LQR
    lqr = LQR(A, B, a, Q, q, R, r, S, Qf, qf)
    return lqr()


def setup_lqr_time(
    dims: chex.Dimensions,
    pen_weight: dict = {"Q": 10.0, "R": 1.0, "Qf": 1e0, "S": 1e-3},
) -> LQR:
    """Setup LQR problem"""
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    # initialise dynamics
    span_time_m = dims["TXX"]
    span_time_v = dims["TX"]
    A = initialise_stable_dynamics(next(skeys), *dims["NT"], radii=0.6)
    B = jnp.tile(jr.normal(next(skeys), dims["NM"]), span_time_m)
    a = 2 * jnp.tile(jr.normal(next(skeys), dims["N"]), span_time_v)
    # define cost matrices
    Q = pen_weight["Q"] * jnp.tile(jnp.eye(dims["N"][0]), span_time_m)
    q = -5 * jnp.tile(jnp.ones(dims["N"]), span_time_v)
    R = pen_weight["R"] * jnp.tile(jnp.eye(dims["M"][0]), span_time_m)
    r = 0.0 * jnp.tile(jnp.ones(dims["M"]), span_time_v)
    S = 0 * pen_weight["S"] * jnp.tile(jnp.ones(dims["NM"]), span_time_m)
    Qf = 2 * pen_weight["Q"] * jnp.eye(dims["N"][0])
    qf = 0 * jnp.ones(dims["N"])
    # construct LQR
    lqr = LQR(A, B, a, Q, q, R, r, S, Qf, qf)
    return lqr()


class TestPLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        """Instantiate dummy LQR"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=100, N=3, M=3, X=1)
        self.sys_dims = ModelDims(*self.dims["NMT"], dt=0.01)
        print("Model dimensionality", self.dims["TNMX"])
        print("\nMake LQR struct")
        self.lqr = setup_lqr(self.dims)

        print("\nMake initial state x0 and input U")
        self.x0 = jnp.array([2.0, 1.0, 0.0])  # jnp.ones(self.dims["N"]) #
        Us = jnp.zeros(self.dims["TM"], dtype=float)
        Us = Us.at[2].set(1.0)
        self.Us = Us

    def test_solve_lqr(self):
        """test LQR solution shape and dtype"""
        params = LQRParams(self.x0, self.lqr)
        Xs_lqr, Us_lqr, Lambs_lqr = solve_lqr(params)
        # print(Xs_lqr)
        ##for this we might need to define the LQRParams class with everything of size T x ...
        xs, us, _ = solve_plqr(params)
        # verify
        chex.assert_trees_all_close(xs, Xs_lqr, rtol=1e-5, atol=1e-5)
        # Plot the KKT residuals
        if PLOTTING_ON:
            fig, ax = subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
            ax[0].plot(Xs_lqr)
            ax[1].plot(xs)
            fig.tight_layout()
            fig.savefig(f"{FIG_DIR}/TestPLQR_lqr_xs.png")
            close()
            
            fig, ax = subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
            ax[0].plot(Us_lqr)
            ax[1].plot(us)
            ax[2].plot(Us_lqr - us)
            fig.tight_layout()
            fig.savefig(f"{FIG_DIR}/TestPLQR_lqr_us.png")
            close()

    def test_lqr_adjoint(self):
        """test LQR adjoint solution"""
        params = LQRParams(self.x0, self.lqr)
        _, _, Lambs_lqr = solve_lqr(params)
        # test
        _, _, lmda = solve_plqr(params)
        
        # validate
        chex.assert_trees_all_close(lmda, Lambs_lqr, rtol=1e-5, atol=1e-5)

        if PLOTTING_ON:
            fig, axes = subplots(1, 3, figsize=(12, 3), sharey=False)
            for i, ax in enumerate(axes.flatten()):
                ax.plot(Lambs_lqr[:, i], linestyle="-")
                ax.plot(lmda[:, i], linestyle=":")
            fig.tight_layout()
            fig.savefig(f"{FIG_DIR}/TestPLQR_adjoint01.png")
            close()

            fig, ax = subplots(1, 2, sharey=True)
            ax[0].plot(Lambs_lqr)
            ax[1].plot(lmda)
            fig.tight_layout()
            fig.savefig(f"{FIG_DIR}/TestPLQR_adjoint02.png")

    def test_adjoint_via_rev_integration(self):
        """Test adjoint via reverse integration and v-funs"""
        # construct LQR and value functions
        model = LQRParams(self.x0, self.lqr)
        v_lins, v_quads = associative_riccati_scan(model)
        # test optimal pass
        _, cs, (Ks, _, _, ks), offsets = associative_opt_traj_scan(
            model, v_lins, v_quads
        )
        opt_xs = jnp.r_[model.x0[None], cs]
        opt_us = ks - jnp.einsum("bij,bj->bi", Ks, opt_xs[:-1]) + offsets
        lmbdas_vfns = jnp.einsum("bij,bj->bi", v_quads, opt_xs) - v_lins
        lmbdas_rev = parallel_reverse_lin_integration(model, opt_xs, opt_us)
        # validate
        chex.assert_trees_all_close(lmbdas_rev, lmbdas_vfns, rtol=1e-5, atol=1e-5)

    def test_lqr_direct_solve(self):
        """Test optimal u and x obtained from gains equivalent as an effective problem"""
        # construct LQR and value functions
        model = LQRParams(self.x0, self.lqr)
        v_lins, v_quads = associative_riccati_scan(model)
        # test optimal pass
        _, cs, (Ks, _, _, ks), offsets = associative_opt_traj_scan(
            model, v_lins, v_quads
        )
        opt_xs = jnp.r_[model.x0[None], cs]

        new_model = LQRParams(
            model.x0,
            LQR(
                model.lqr.A - Ks,
                model.lqr.B,
                0 * model.lqr.a,
                model.lqr.Q,
                model.lqr.q,
                model.lqr.R,
                model.lqr.r,
                model.lqr.S,
                model.lqr.Qf,
                model.lqr.qf,
            ),
        )

        new_us = ks + offsets + model.lqr.a
        new_xs = parallel_forward_lin_integration(new_model, new_us, new_model.lqr.a)

        # verify
        chex.assert_trees_all_close(opt_xs, new_xs, rtol=1e-5, atol=1e-5)
        if PLOTTING_ON:
            fig, ax = subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
            ax[0].plot(opt_xs)
            ax[1].plot(new_xs)
            fig.tight_layout()
            fig.savefig(f"{FIG_DIR}/TestPLQR_dual_method.png")
            close()

    def test_plqr_cpu_profile(self):
        cpu = jax.devices("cpu")[0]
        n_reps = 2
        if LONG_TIME_PROFILE:
            n_dims = onp.array([16,32])
        else:
            n_dims = onp.array([2,4])
        if LONG_TIME_PROFILE:
            horizon_dims = onp.array([10, 100, 200, 500, 1000, 5000, 10000, 20000,])
        else:
            horizon_dims = onp.array([10, 100])
        cpu_lqr = jax.jit(solve_lqr, device=cpu)
        cpu_plqr = jax.jit(solve_plqr, device=cpu)
        
        clock_times = onp.zeros((2, n_dims.size * horizon_dims.size), dtype=float)
        print(clock_times.shape)
        
        for ix, (n, T) in enumerate(product(n_dims, horizon_dims)):
            dims = chex.Dimensions(T=T, N=n, M=n, X=1)
            sys_dims = ModelDims(*dims["NMT"], dt=0.01)
            # dims = ModelDims(n=n, m=n, horizon=T, dt=0.1)
            lqr_model = setup_lqr_time(dims=dims)
            cpu_lqr_prms = LQR(*(jax.device_put(jnp.asarray(val), cpu) for val in lqr_model))
            x0 = jax.device_put(jnp.ones(sys_dims.n), cpu)
            # time profile linear lqr
            res = cpu_lqr(LQRParams(x0, cpu_lqr_prms))
            # skip jit compile time
            for e in res:
                _ = _tuple_block_until_ready(e)
            tic = time.time()
            for _ in trange(n_reps, leave=False):
                # res = cpu_lqr(LQRParams(x0, cpu_lqr_prms))
                for e in res:
                    _ = _tuple_block_until_ready(e)
            print(0, ix, (n, T))
            clock_times[0,ix] = (time.time() - tic) / n_reps

            # time profile parallel lqr
            res = cpu_plqr(LQRParams(x0, cpu_lqr_prms))
            # skip jit compile time
            for e in res:
                _ = _tuple_block_until_ready(e)
            tic = time.time()
            for _ in trange(n_reps, leave=False):
                # res = cpu_plqr(LQRParams(x0, cpu_lqr_prms))
                for e in res:
                    _ = _tuple_block_until_ready(e)
            print(1, ix, (n, T))
            clock_times[1,ix] = (time.time() - tic) / n_reps
            
        clock_times = clock_times.reshape(2, n_dims.size, horizon_dims.size)
        
        if PLOTTING_ON:
            fig, ax = subplots(1, figsize=(5, 3), sharex=True)
            colors = ["r", "b", "g", "magenta"]
            [ax.plot(horizon_dims, clock_times[0,i], label=f"Linear LQR {i}", color=colors[i]) for i in range(n_dims.size)]
            [ax.plot(horizon_dims, clock_times[1,i], label=f"Parallel LQR {i}", color=colors[i],linestyle="--",) for i in range(n_dims.size)]
            ax.legend(loc=(1, 0.2))
            ax.set(xlabel="N tps", ylabel="Clock Time (s)",title="cpu")
            fig.savefig(f"{FIG_DIR}/TestPLQR_lqr_cpu_time.png")
            close()
            
    def test_plqr_gpu_profile(self):
        if not _gpu_available():
            return
        
        gpu = jax.devices("gpu")[0]
        n_reps = 2
        n_dims = onp.array([12,20])
        horizon_dims = onp.array([10, 100, 200, 500, 1000, 5000,])
        gpu_lqr = jax.jit(solve_lqr, gpu)
        gpu_plqr = jax.jit(solve_plqr, gpu)
        
        clock_times = onp.zeros((2, n_dims.size * horizon_dims.size), dtype=float)
        print(clock_times.shape)
        
        for ix, (n, T) in enumerate(product(n_dims, horizon_dims)):
            dims = chex.Dimensions(T=T, N=n, M=n, X=1)
            sys_dims = ModelDims(*dims["NMT"], dt=0.01)
            # dims = ModelDims(n=n, m=n, horizon=T, dt=0.1)
            lqr_model = setup_lqr_time(dims=dims)
            gpu_lqr_prms = LQR(*(jax.device_put(jnp.asarray(val), gpu) for val in lqr_model))
            x0 = jax.device_put(jnp.ones(sys_dims.n), gpu)
            # time profile linear lqr
            res = gpu_lqr(LQRParams(x0, gpu_lqr_prms))
            # skip jit compile time
            for e in res:
                _ = _tuple_block_until_ready(e)
            tic = time.time()
            for _ in trange(n_reps, leave=False):
                # res = cpu_lqr(LQRParams(x0, cpu_lqr_prms))
                for e in res:
                    _ = _tuple_block_until_ready(e)
            print(0, ix, (n, T))
            clock_times[0,ix] = (time.time() - tic) / n_reps

            # time profile parallel lqr
            res = gpu_plqr(LQRParams(x0, gpu_lqr_prms))
            # skip jit compile time
            for e in res:
                _ = _tuple_block_until_ready(e)
            tic = time.time()
            for _ in trange(n_reps, leave=False):
                # res = cpu_plqr(LQRParams(x0, cpu_lqr_prms))
                for e in res:
                    _ = _tuple_block_until_ready(e)
            print(1, ix, (n, T))
            clock_times[1,ix] = (time.time() - tic) / n_reps
            
        clock_times = clock_times.reshape(2, n_dims.size, horizon_dims.size)
        
        if PLOTTING_ON:
            fig, ax = subplots(1, figsize=(5, 3), sharex=True)
            colors = ["r", "b", "g", "magenta"]
            [ax.plot(horizon_dims, clock_times[0,i], label=f"Linear LQR {i}", color=colors[i]) for i in range(n_dims.size)]
            [ax.plot(horizon_dims, clock_times[1,i], label=f"Parallel LQR {i}", color=colors[i],linestyle="--",) for i in range(n_dims.size)]
            ax.legend(loc=(1, 0.2))
            ax.set(xlabel="N tps", ylabel="Clock Time (s)",title="gpu")
            fig.savefig(f"{FIG_DIR}/TestPLQR_lqr_gpu_time.png")
            close()
        

if __name__ == "__main__":
    unittest.main()
