from typing import Callable, Any, Optional, NamedTuple
import jax
import jax.lax as lax
import jax.numpy as np
from functools import partial

# from . import LQR
import sys

sys.path.append("/Users/thomasmullen/VSCodeProjects/ilqr_vae_jax/src")
type(sys.path)
for path in sys.path:
    print(path)
from lqr import *


# struct: dynamics, cost, costf
class System(NamedTuple):
    """iLQR System

    cost : Callable
        running cost l(x, u)
    costf : Callable
        final state cost lf(xf)
    dynamics : Callable
        dynamical update f(x, u, params)
    """

    cost: Callable[[np.ndarray, np.ndarray, Optional[Any]], np.ndarray]
    costf: Callable[[np.ndarray, Optional[Any]], np.ndarray]
    dynamics: Callable[[np.ndarray, np.ndarray, Optional[Any]], np.ndarray]
    horizon: int


class Theta(NamedTuple):
    Uh: np.ndarray
    Wh: np.ndarray
    sigma: np.ndarray


class Params(NamedTuple):
    """Non-linear parameter struct"""

    x0: np.ndarray
    theta: Any


# funct that linearises
def linearise(fun):
    """Function that finds jacobian w.r.t to x and u inputs.

    Args:
        fun (Callable): args (x, u, params)

    Returns:
        Tuple([NDARRAY, NDARRAY]): _description_
    """
    return jax.jacrev(fun, argnums=(0, 1))


# funct that quadratises
def quadratise(fun):
    return jax.jacfwd(jax.jacrev(fun, argnums=(0, 1)), argnums=(0, 1))


def vectorise_fun_in_time(fun):
    return jax.vmap(fun, in_axes=(0, 0, None))


def approx_lqr(model: System, Xs: np.ndarray, Us: np.ndarray, params: Params):
    """Calls linearisation and quadratisation function

    Returns:
        LQR: return the LQR structure
    """
    theta = params.theta

    Xs = Xs.squeeze()
    Us = Us.squeeze()

    (Fx, Fu) = vectorise_fun_in_time(linearise(model.dynamics))(Xs[:-1], Us, theta)
    (Cx, Cu) = vectorise_fun_in_time(linearise(model.cost))(Xs[:-1], Us, theta)
    (Cxx, Cxu), (Cux, Cuu) = vectorise_fun_in_time(quadratise(model.cost))(
        Xs[:-1], Us, theta
    )
    fCx = jax.jacrev(model.costf)(Xs[-1], theta)
    fCxx = jax.jacfwd(jax.jacrev(model.costf))(Xs[-1], theta)

    # set-up LQR
    lqr = LQR(
        A=Fx,
        B=Fu,
        a=np.zeros((model.horizon, Fx.shape[-1], 1)),
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
    def cost(x: np.array, u: np.array, theta: Theta):
        return np.sum(x**2) + np.sum(u**2)

    def costf(x: np.array, theta: Theta):
        return np.sum(np.abs(x))

    def dynamics(x: np.array, u: np.array, theta: Theta):
        return np.tanh(theta.Uh @ x + theta.Wh @ u)

    return System(cost, costf, dynamics, horizon=40)


def ddp_rollout(
    model: System,
    params: Params,
    Ks: Gains,
    Xs: np.ndarray,
    Us: np.ndarray,
    alpha: float = 1.0,
):
    x0, theta = params.x0, params.theta
    x_hat0 = x0

    def fwd_step(state, inputs):
        x_hat, nx_cost = state
        x, u, K, k = inputs

        δx = x_hat - x
        δu = K @ δx + alpha * k
        u_hat = u + δu
        nx_hat = model.dynamics(x_hat, u_hat, theta)
        nx_cost = nx_cost + model.cost(x_hat, u_hat, theta)
        return (nx_hat, nx_cost), (nx_hat, u_hat)

    (xf, nx_cost), (new_Xs, new_Us) = lax.scan(
        fwd_step, init=(x_hat0, 0.0), xs=(Xs, Us, Ks.K, Ks.k)
    )
    total_cost = nx_cost + model.costf(xf, theta)
    new_Xs = np.vstack([x0[None], new_Xs])

    return (new_Xs, new_Us), total_cost


# recursive linesearch


# overload forward function with deviation step function


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

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

    theta = Theta(Uh=Uh, Wh=Wh, sigma=np.zeros((3, 1)))
    params = Params(x0=np.zeros((3, 1)), theta=theta)
    # initialise model: dyn and costs
    model = define_model()
    # test forward step
    print(model.dynamics(np.ones((3, 1)), np.ones((2, 1)), params).shape)
    # test differentiation
    print(linearise(model.dynamics)(np.ones((3,)), np.ones((2,)), params)[0].shape)
    # test vectorisation in time
    print(
        vectorise_fun_in_time(linearise(model.dynamics))(
            np.ones(
                (
                    model.horizon,
                    3,
                )
            ),
            np.ones(
                (
                    model.horizon,
                    2,
                )
            ),
            params,
        )[0].shape
    )
    # Guess a U
    Us = np.zeros((model.horizon, 2, 1), dtype=float)
    Us = Us.at[10:14].set(2.0)
    Xs = rollout(model.dynamics, Us=Us, params=params)
    print(Xs.shape, Us.shape)

    # taylor expand and approx lqr for each tp
    lqr_approximated = approx_lqr(
        model=model, Xs=Xs.squeeze(), Us=Us.squeeze(), params=params
    )
    print(lqr_approximated.Q.shape, lqr_approximated.Qf.shape)
    print(lqr_approximated.A.shape, lqr_approximated.B.shape)

    # test backward sweep
    (dC, Ks), f = backward(lqr_approximated, 40, True)
    # test forward update sweep
    (new_Xs, new_Us), total_cost = ddp_rollout(
        model=model, params=params, Ks=Ks, Xs=Xs[:-1], Us=Us, alpha=1
    )
