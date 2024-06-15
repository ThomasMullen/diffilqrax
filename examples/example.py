"""Non-linear RNN example"""

from typing import Any, Union, NamedTuple
from jax import Array, make_jaxpr
import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jr

from diffilqrax.utils import keygen
from diffilqrax.ilqr import ilqr_solver, ilqr_simulate
from diffilqrax.typs import (
    iLQRParams,
    System,
    ModelDims,
    PendulumParams,
    Theta,
)

jax.config.update("jax_enable_x64", True)  # double precision


class ThetaGRU(NamedTuple):
    """Theta params for GRU"""
    Uf: Array
    bf: Array
    Uh: Array
    Wh: Array
    bh: Array
    Wo: Array
    
def problem_setup():

    def dynamics(_, x, u, theta: ThetaGRU) -> Array:
        gh = x @ theta.Uf + theta.bf
        f = lax.logistic(gh)
        hf = x * f
        h_hat = lax.tanh((theta.bh + (hf @ theta.Uh))) + (u @ theta.Wh)
        nx = ((1 - f) * x) + (f * h_hat)
        return nx[:]


    def cost(t: int, x: Array, u: Array, theta: Any):
        return jnp.linalg.norm(x) + jnp.linalg.norm(u)


    def costf(x: Array, theta: Any):
        return jnp.linalg.norm(x)
    
    return System(cost, costf, dynamics, ModelDims(horizon=200, n=60, m=10, dt=0.1))


if __name__ == "__main__":
    key = jr.PRNGKey(seed=2340)
    key, skeys = keygen(key, 8)
    n=60
    m=10


    theta = ThetaGRU(
        Uh=(jr.normal(next(skeys), (n, n))*(0.8/jnp.sqrt(n))) - jnp.eye(n),
        Uf=(jr.normal(next(skeys), (n, n))*(0.8/jnp.sqrt(n))) - jnp.eye(n),
        Wh=jr.normal(next(skeys), (m, n)),
        Wo=jr.normal(next(skeys), (n, m)),
        bh=jr.normal(next(skeys), (n,)),
        bf=jr.normal(next(skeys), (n,)),
        )
    params = iLQRParams(x0=jr.normal(next(skeys), (n,)), theta=theta)
    
    problem = problem_setup()
    
    ls_kwargs = {
        "beta": 0.8,
        "max_iter_linesearch": 50,
        "tol": 1e0,
        "alpha_min": 0.0001,
    }
    
    # Us_init = jnp.zeros((problem.dims.horizon, problem.dims.m))
    Us_init = 0.1 * jr.normal(next(skeys), (problem.dims.horizon, problem.dims.m))
    
    with jax.profiler.trace("/Users/thomasmullen/VSCodeProjects/ilqr_vae_jax/fig_dump/tmp/trace", create_perfetto_link=True):
        # Run the operations to be profiled
        key = jax.random.key(0)
        x = jax.random.normal(key, (5000, 5000))
        y = x @ x
        y.block_until_ready()


    # with jax.profiler.start_trace("/Users/thomasmullen/VSCodeProjects/ilqr_vae_jax/fig_dump/tmp/trace", create_perfetto_link=True):
    #     (Xs_stars, Us_stars, Lambs_stars), converged_cost, cost_log = ilqr_solver(
    #         problem,
    #         params,
    #         Us_init,
    #         max_iter=400,
    #         convergence_thresh=1e-8,
    #         alpha_init=1.0,
    #         verbose=False,
    #         use_linesearch=True,
    #         **ls_kwargs,
    #     )
    #     Xs_stars.block_until_ready()
        
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2,2)
    # ax[0,0].plot(ilqr_simulate(problem, Us_init, params)[0][0])
    # ax[0,1].plot(Xs_stars)
    # ax[1,0].plot(Us_init)
    # ax[1,1].plot(Us_stars)