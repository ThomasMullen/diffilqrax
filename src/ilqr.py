from typing import Callable, Any, Optional, NamedTuple
import jax
import jax.lax as lax
import jax.numpy as np

import jax
from jax import lax
import jax.numpy as jnp
# from . import LQR
import sys
sys.path.append("/Users/thomasmullen/VSCodeProjects/ilqr_vae_jax/src")
type(sys.path)
for path in sys.path:
    print(path)
from lqr import *

# funct that linearises
def linearise(fun):
    """Function that finds jacobian w.r.t to x and u inputs. NOTE: extend to auto-vectorization.
    
    Args:
        fun (Callable): args (x, u, params)

    Returns:
        Tuple([NDARRAY, NDARRAY]): _description_
    """
    jac_x = jax.jacrev(fun, argnums=0)
    jac_u = jax.jacrev(fun, argnums=1)
    return jac_x, jac_u

# funct that quadratises
def quadratise(fun):
    hessian_xx = jax.jacfwd(jax.jacrev(fun, argnums=0), argnums=0)
    hessian_xu = jax.jacfwd(jax.jacrev(fun, argnums=1), argnums=1)
    hessian_uu = jax.jacfwd(jax.jacrev(fun, argnums=0), argnums=1)
    return hessian_xx, hessian_xu, hessian_uu

# struct: dynamics, cost, costf
class System(NamedTuple):
    """iLQR System

    cost : Callable
        running cost l(t, x, u)
    costf : Callable
        final state cost lf(xf)
    dynamics : Callable
        dynamical update f(t, x, u, params)
    """

    cost: Callable[[int, np.ndarray, np.ndarray, Optional[Any]], np.ndarray]
    costf: Callable[[np.ndarray, Optional[Any]], np.ndarray]
    dynamics: Callable[[int, np.ndarray, np.ndarray, Optional[Any]], np.ndarray]


# recursive linesearch


if __name__ == "__main__":
    
    # define non-linear function
    f = lambda x, u: np.sin(x) + np.exp(u)
    # define linearise the function w.r.t. x (arg 0) and u (arg 1)
    linearise_f = jax.jacfwd(f, (0,1))
    # vmap through time axis
    linearise_f_t = jax.vmap(linearise_f, in_axes=(1,1))
    # generate timeseries data
    xs = np.arange(0,6*np.pi, np.pi*0.5)
    us = np.zeros_like(xs)
    us=us.at[2:4].set(2)
    print(linearise_f_t(xs.reshape(1,-1), us.reshape(1,-1))[0].shape)
    # tile to 3 dimension
    Xs = np.tile(xs, (3,1))
    Us = np.tile(us, (3,1))
    print(linearise_f_t(Xs, Us)[0].shape)
