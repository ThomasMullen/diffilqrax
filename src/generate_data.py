from typing import Callable, Any, Optional, NamedTuple
import jax
import jax.lax as lax
import jax.numpy as np
from functools import partial


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
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

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
    initial_state = np.array([[-8.0], [8.0], [27.0]])

    def step(x, u):
        nx = dyn(x, u)
        return nx, nx

    xf, lorenz_Xs = lax.scan(f=step, init=initial_state, xs=lorenz_Us)
    lorenz_Xs = lorenz_Xs.squeeze()
    return lorenz_Xs, lorenz_Us


if __name__ == "__main__":
    initial_state = np.array([[-8.], [8.], [27.]])
    dynamics = rk4(lorenz_system, dt=0.01)
    Xs, Us = generate_lorenz_data(dyn=dynamics)
    tps, n_states = Xs.shape
    