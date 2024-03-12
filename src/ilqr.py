"""iterative LQR solver"""
from typing import Callable, Any, Optional, NamedTuple
from jax import Array
from jax.typing import ArrayLike
import jax
import jax.lax as lax
import jax.numpy as np
from functools import partial
import jaxopt
from . import lqr as lqr


class System(NamedTuple):
    """iLQR System

    cost : Callable
        running cost l(t, x, u, params)
    costf : Callable
        final state cost lf(xf, params)
    dynamics : Callable
        dynamical update f(t, x, u, params)
    horizon : int
        ilQR evaluate time horizon
    n : int
        state dimension
    m : int
        input dimensions
    """

    cost: Callable[[int, Array, Array, Optional[Any]], Array]
    costf: Callable[[Array, Optional[Any]], Array]
    dynamics: Callable[[int, ArrayLike, ArrayLike, Optional[Any]], Array]
    horizon: int
    n: int
    m: int


class Theta(NamedTuple):
    Uh: Array
    Wh: Array
    sigma: ArrayLike


class Params(NamedTuple):
    """Non-linear parameter struct"""

    x0: ArrayLike
    horizon: int
    theta: Any


def linearise(fun: Callable) -> Callable:
    """Function that finds jacobian w.r.t to x and u inputs.

    Args:
        fun (Callable): args (t, x, u, params)

    Returns:
        Callable[[Callable], Callable]): Jacobian tuple evaluated at args 1 and 2
    """
    return jax.jacrev(fun, argnums=(1, 2))


# funct that quadratises
def quadratise(fun: Callable) -> Callable:
    """Function that finds Hessian w.r.t to x and u inputs.

    Args:
        fun (Callable): args (t, x, u, params)

    Returns:
        Tuple([NDARRAY, NDARRAY]): Hessian tuple cross evaluated with args 1 and 2
    """
    return jax.jacfwd(jax.jacrev(fun, argnums=(1, 2)), argnums=(1, 2))


def vectorise_fun_in_time(fun: Callable) -> Callable:
    """Vectorise function in time. Assumes 0th-axis is time for x and u args of fun, the last
    arg (theta) of Callable function assumed to be time-invariant.

    Args:
        fun (Callable): function that takes args (t, x[Txn], u[Txm], theta)

    Returns:
        Callable: vectorised function along args 1 and 2 0th-axis
    """
    return jax.vmap(fun, in_axes=(None, 0, 0, None))


def approx_lqr(
    model: System, Xs: np.ndarray, Us: np.ndarray, params: Params
) -> lqr.LQR:
    """Calls linearisation and quadratisation function

    Returns:
        LQR: return the LQR structure
    """
    theta = params.theta

    Xs = Xs.squeeze()
    Us = Us.squeeze()

    (Fx, Fu) = vectorise_fun_in_time(linearise(model.dynamics))(
        None, Xs[:-1], Us, theta
    )
    (Cx, Cu) = vectorise_fun_in_time(linearise(model.cost))(None, Xs[:-1], Us, theta)
    (Cxx, Cxu), (Cux, Cuu) = vectorise_fun_in_time(quadratise(model.cost))(
        None, Xs[:-1], Us, theta
    )
    fCx = jax.jacrev(model.costf)(Xs[-1], theta)
    fCxx = jax.jacfwd(jax.jacrev(model.costf))(Xs[-1], theta)

    # set-up LQR
    lqr = lqr.LQR(
        A=Fx,
        B=Fu,
        a=np.zeros((model.horizon, model.n, 1)),
        Q=Cxx,
        q=Cx[:, :, None],
        Qf=fCxx,
        qf=fCx[:, None],
        R=Cuu,
        r=Cu[:, :, None],
        S=Cxu,
    )()

    return lqr


def define_model():
    def cost(t: int, x: np.array, u: np.array, theta: Theta):
        return np.sum(x**2) + np.sum(u**2)

    def costf(x: np.array, theta: Theta):
        return np.sum(np.abs(x))

    def dynamics(t: int, x: np.array, u: np.array, theta: Theta):
        return np.tanh(theta.Uh @ x + theta.Wh @ u)

    return System(cost, costf, dynamics, horizon=20, n=3, m=2)


if __name__ == "__main__":
    # test data
    Uh = np.array(
        [
            [-0.63462433, -1.22943886, -0.07712939],
            [-0.22857423, -1.36123108, -0.04661756],
            [-0.14380682, 1.75378683, -1.77218787],
        ]
    )
    Wh = np.array(
        [
            [0.37087464, -1.1752595],
            [-0.51433962, 1.94757307],
            [-1.29836488, -0.61030051],
        ]
    )
    # initialise params
    theta = Theta(Uh=Uh, Wh=Wh, sigma=np.zeros((3, 1)))
    params = Params(x0=np.zeros((3, 1)), theta=theta, horizon=20)
    model = define_model()
    # generate input
    Us = np.zeros((model.horizon, model.m, 1))
    Us = Us.at[2].set(2)
    # rollout model non-linear dynamics
    Xs = lqr.simulate_trajectory(model.dynamics, Us, params)
    # approx lqr with U and X trajectorys
    lqr_tilde = approx_lqr(model=model, Xs=Xs, Us=Us, params=params)

    pass
