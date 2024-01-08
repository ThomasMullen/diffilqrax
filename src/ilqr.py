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


# integrator
def rk4(dynamics, dt=0.01):
    def integrator(x, u):
        dt2 = dt / 2.0
        k1 = dynamics(x, u)
        k2 = dynamics(x + dt2 * k1, u)
        k3 = dynamics(x + dt2 * k2, u)
        k4 = dynamics(x + dt * k3, u)
        nx_x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return nx_x
    return integrator

def euler(dynamics, dt=0.01):
    return lambda x, u: x + dt * dynamics(x, u)

def lorenz_system(current_state, u):
    # define the system parameters sigma, rho, and beta
    sigma = 10.
    rho = 28.
    beta = 8. / 3.

    # positions of x, y, z in space at the current time point
    x, y, z = current_state

    # define the 3 ordinary differential equations known as the lorenz equations
    dx_dt = sigma * (y - x) + u
    dy_dt = x * (rho - z) - y + u
    dz_dt = x * y - beta * z + u

    # return a list of the equations that describe the system
    nx_state = np.array([dx_dt, dy_dt, dz_dt])
    return nx_state


def generate_lorenz_data(dyn):
    lorenz_Us = np.zeros(4000)
    initial_state = np.array([[-8.], [8.], [27.]])
    def step(x, u):
        nx = dyn(x, u)
        return nx, nx     
        
    xf, lorenz_Xs = lax.scan(f=step, init=initial_state, xs=lorenz_Us)
    lorenz_Xs = lorenz_Xs.squeeze()
    return lorenz_Xs, lorenz_Us


def approx_lqr():
    initial_state = np.array([[-8.], [8.], [27.]])
    Xs, Us = generate_lorenz_data(dyn=rk4(lorenz_system, dt=0.01))
    tps, n_states = Xs.shape
    # linearise dynamics
    dynamics = rk4(lorenz_system, dt=0.01)
    linearise_dynamics = jax.jacrev(dynamics, argnums=(0,1))
    vectorised_linear_dynamics = jax.vmap(linearise_dynamics, in_axes=(0,0))
    dfdXs, dfdUs = vectorised_linear_dynamics(Xs, Us)
    # quadratise cost
    Qf = np.eye(n_states)*10
    qf = np.zeros((n_states,1))
    Q = np.eye(n_states)*10
    q = np.zeros((n_states,1))
    R = np.eye(1)*0.1
    r = np.array([0.])
    S = np.zeros((n_states,1))
    
    # set-up LQR
    lqr=LQR(
        A=dfdXs,
        B =dfdUs.reshape(dfdUs.shape+(1,)),
        a = np.zeros((tps,n_states,1)),
        Q=np.tile(Q,(tps,1,1)),
        q=np.tile(q,(tps,1,1)),
        Qf=Qf,
        qf=qf,
        R=np.tile(R,(tps,1,1)),
        r=np.tile(r,(tps,1,1)),
        S=np.tile(S,(tps,1,1)),
        )()
    # infer optimal gains
    gains = backward(lqr, 4000)
    X_hats, U_hats = forward(lqr, gains, initial_state)

if __name__ == "__main__":
    
    # define non-linear function: f(t, x, u)
    f = lambda t, x, u: np.sin(x) + np.exp(u)
    # define linearise the function w.r.t. x (arg 1) and u (arg 2)
    linearise_f = jax.jacfwd(f, (1,2))
    # vmap through time axis 0
    linearise_f_t = jax.vmap(linearise_f, in_axes=(None,0,0))
    # generate timeseries data
    xs = np.arange(0,6*np.pi, np.pi*0.5).reshape(-1,1)
    us = np.zeros_like(xs).reshape(-1,1)
    us=us.at[2:4].set(2)
    t = np.arange(xs.size)
    print(linearise_f_t(t, xs, us)[0].shape)
    # tile to 2 dimension
    Xs = np.tile(xs, (1,3))
    Us = np.tile(us, (1,3))
    print(linearise_f_t(t, Xs, Us)[0].shape)
    # obtain a set of As and Bs
    dfdx, dfdu = linearise_f_t(t, Xs, Us)
    
    lqr
