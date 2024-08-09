"""
Unit test for the parallel LQR module
"""

from pathlib import Path
import unittest
from os import getcwd
import chex
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
from matplotlib.pyplot import subplots, close, style
import time
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
from diffilqrax.typs import (
    LQR,
    LQRParams,
    ModelDims,
)
from diffilqrax.plqr import (
    solve_plqr,
)
from diffilqrax.lqr import solve_lqr
from diffilqrax.exact import quad_solve, exact_solve
from diffilqrax.utils import keygen, initialise_stable_dynamics

# jax.config.update('jax_default_device', jax.devices('cpu')[0])
# jax.config.update('jax_platform_name', 'gpu')


PLOT_URL = ("https://gist.githubusercontent.com/"
       "ThomasMullen/e4a6a0abd54ba430adc4ffb8b8675520/"
       "raw/1189fbee1d3335284ec5cd7b5d071c3da49ad0f4/"
       "figure_style.mplstyle")
#style.use("/home/marineschimel/code/diffilqrax/paper.mplstyle")


def is_jax_array(arr: Array)->bool:
    """validate jax array type"""
    return isinstance(arr, jnp.ndarray) and not isinstance(arr, onp.ndarray)


def setup_lqr(dims: chex.Dimensions,
              pen_weight: dict = {"Q": 10., "R": 1., "Qf": 1e0, "S": 1e-3}) -> LQR:
    """Setup LQR problem"""
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    # initialise dynamics
    span_time_m=dims["TXX"]
    span_time_v=dims["TX"]
    dt = 0.1
    Uh = jnp.array([[1, dt, 0.01], [-1 * dt, 1 - 0.1 * dt, 0.],  [-1 * dt, 1 - 0.1 * dt, 0.05]])
    Wh = jnp.eye(3) #jnp.array([[0.5, 2., 0.], [1., -1.2, 0.1]]).T * dt
    Q = 1.
    A = jnp.tile(Uh, span_time_m)
    B = jnp.tile(Wh, span_time_m)
    a = jnp.tile(jr.normal(next(skeys), dims['N']), span_time_v)
    # define cost matrices
    Q = pen_weight["Q"] * jnp.tile(jnp.eye(dims['N'][0]), span_time_m)
    q = -5* jnp.tile(jnp.ones(dims['N']), span_time_v)
    R = pen_weight["R"] * jnp.tile(jnp.eye(dims['M'][0]), span_time_m)
    r = -2.* jnp.tile(jnp.ones(dims['M']), span_time_v)
    S =0* pen_weight["S"] * jnp.tile(jnp.ones(dims['NM']), span_time_m)
    Qf = 0*pen_weight["Q"] * jnp.eye(dims['N'][0])
    qf = 0 * jnp.ones(dims['N'])
    # construct LQR
    lqr = LQR(A, B, a, Q, q, R, r, S, Qf, qf)
    return lqr()

def setup_lqr_time(dims: chex.Dimensions,
              pen_weight: dict = {"Q": 10., "R": 1., "Qf": 1e0, "S": 1e-3}) -> LQR:
    """Setup LQR problem"""
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    # initialise dynamics
    span_time_m=dims["TXX"]
    span_time_v=dims["TX"]
    A = initialise_stable_dynamics(next(skeys), *dims['NT'],radii=0.6) 
    B = jnp.tile(jr.normal(next(skeys), dims['NM']), span_time_m)
    a = 2*jnp.tile(jr.normal(next(skeys), dims['N']), span_time_v)
    # define cost matrices
    Q = pen_weight["Q"] * jnp.tile(jnp.eye(dims['N'][0]), span_time_m)
    q = -5* jnp.tile(jnp.ones(dims['N']), span_time_v)
    R = pen_weight["R"] * jnp.tile(jnp.eye(dims['M'][0]), span_time_m)
    r = 0. * jnp.tile(jnp.ones(dims['M']), span_time_v)
    S =0* pen_weight["S"] * jnp.tile(jnp.ones(dims['NM']), span_time_m)
    Qf = 2*pen_weight["Q"] * jnp.eye(dims['N'][0])
    qf = 0 * jnp.ones(dims['N'])
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
        self.x0 = jnp.array([2.0, 1.0,0.0]) #jnp.ones(self.dims["N"]) #
        Us = jnp.zeros(self.dims["TM"], dtype=float)
        Us = Us.at[2].set(1.0)
        self.Us = Us


    def test_solve_lqr(self):
        """test LQR solution shape and dtype"""
        params = LQRParams(self.x0, self.lqr)
        gains_lqr, Xs_lqr, Us_lqr, Lambs_lqr = solve_lqr(params)
        #print(Xs_lqr)
        ##for this we might need to define the LQRParams class with everything of size T x ...
        xs, us, _  = solve_plqr(params)
        fig_dir = Path(Path(getcwd()), "fig_dump")
        fig_dir.mkdir(exist_ok=True)
        # Plot the KKT residuals
        fig_dir = Path(Path(getcwd()), "fig_dump")
        fig_dir.mkdir(exist_ok=True)
        fig, ax = subplots(1,2,figsize=(8,3), sharex = True, sharey =True)
        ax[0].plot(Xs_lqr)
        ax[1].plot(xs)
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/TestPLQR_lqr_xs.png")
        close()
        fig_dir = Path(Path(getcwd()), "fig_dump")
        fig_dir.mkdir(exist_ok=True)
        fig, ax = subplots(1,3,figsize=(8,3), sharex = True, sharey =True)
        ax[0].plot(Us_lqr)
        ax[1].plot(us)
        ax[2].plot(Us_lqr - us)
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/TestPLQR_lqr_us.png")
        chex.assert_trees_all_close(xs, Xs_lqr, rtol=1e-5, atol=1e-5)
    
    def test_lqr_adjoint(self):
        """test LQR adjoint solution"""
        params = LQRParams(self.x0, self.lqr)
        gains_lqr, Xs_lqr, Us_lqr, Lambs_lqr = solve_lqr(params)
        # test
        xs, us, lmda = solve_plqr(params)
        # visualise 01
        fig_dir = Path(Path(getcwd()), "fig_dump")
        fig_dir.mkdir(exist_ok=True)
        fig, axes = subplots(1,3,figsize=(12,3),sharey=False)
        for i,ax in enumerate(axes.flatten()):
            ax.plot(Lambs_lqr[:,i], linestyle="-")
            ax.plot(lmda[:,i], linestyle=":")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/TestPLQR_adjoint01.png")
        close()
        # visualise 02
        fig, ax = subplots(1,2,sharey=True)
        ax[0].plot(Lambs_lqr)
        ax[1].plot(lmda)
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/TestPLQR_adjoint02.png")
        # validate
        chex.assert_trees_all_close(lmda, Lambs_lqr, rtol=1e-5, atol=1e-5)
        
    def test_time(self):
        from jax.lib import xla_bridge
        print(jax.default_backend())
        start = time.time()
        ns = [32, 33] #2,4,8,32] #,5,10] #,100]
        Ts = [10,100,200,500,1000, 5000,10000,20000]#,10000]#,100000] #, 50000, 100000, 200000, 500000, 1000000] #10000]
        parallel_lqr_times_0 = []
        normal_lqr_times_0 = []
        parallel_lqr_times = []
        normal_lqr_times = []
        for n in ns : 
            ps = []
            ls = []
            p0s = []
            l0s = []
            for T in Ts : 
                m = n
                dims = chex.Dimensions(T=T, N=n, M=m, X=1)
                sys_dims = ModelDims(*dims["NMT"], dt=0.01)
                x0 = jnp.ones(dims["N"])
                lqr = setup_lqr_time(dims)
                params = LQRParams(x0, lqr)
                for seed in [0,1]:
                    start = time.time()
                    xs = solve_plqr(params)
                    end = time.time()
                    parallel_time = end-start
                    start = time.time()
                    gains_lqr, Xs_lqr, Us_lqr, Lambs_lqr = solve_lqr(params)
                    end = time.time()
                    normal_time = end-start
                    if seed == 0:
                        p0s.append(parallel_time)
                        l0s.append(normal_time)
                    else : 
                        ps.append(parallel_time)
                        ls.append(normal_time)
            parallel_lqr_times.append(ps)
            normal_lqr_times.append(ls)
            parallel_lqr_times_0.append(p0s)
            normal_lqr_times_0.append(l0s)
        fig_dir = Path(Path(getcwd()), "fig_dump")
        fig_dir.mkdir(exist_ok=True)
        fig, axes = subplots(2,1,figsize=(5,3), sharex = True)
        colors = ['r','b','g', 'magenta']
        for i, n in enumerate(ns) : 
            axes[0].plot(Ts, parallel_lqr_times_0[i], label = f"Parallel LQR n = {n}", color = colors[i])
            axes[0].plot(Ts, normal_lqr_times_0[i], label = f"Normal LQR n = {n}", color = colors[i], linestyle = "--")
        axes[0].legend(loc = (1,0.2))
        axes[0].set_title("First run time")
        for i, n in enumerate(ns) : 
            axes[1].plot(Ts, parallel_lqr_times[i], label = f"Parallel LQR n = {n}", color = colors[i])
            axes[1].plot(Ts, normal_lqr_times[i], label = f"Normal LQR n = {n}", color = colors[i], linestyle = "--")
        axes[1].set_title("Second run time")
        axes[1].set_xlabel("Number of timesteps")
        fig.text(-0.01, 0.5, "Time (s)", va='center', rotation='vertical')
        fig.savefig(f"{fig_dir}/TestPLQR_lqr_xs_time_comp_jitting2.png")
        close()
        
         


if __name__ == "__main__":
    unittest.main()
