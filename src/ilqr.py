"""iterative LQR solver"""
from typing import Callable, Any, Optional, NamedTuple, Tuple
from jax import Array
from jax.typing import ArrayLike
import jax
import jax.lax as lax
import jax.numpy as np
from functools import partial
import jaxopt
import src as lqr

sum_cost_to_go_struct = lambda x: x.V + x.v

# TODO separate dimesions from System and parse through ModelDims through functions

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
    dt:float


class Theta(NamedTuple):
    Uh: Array
    Wh: Array
    sigma: ArrayLike


class Params(NamedTuple):
    """Non-linear parameter struct"""

    x0: ArrayLike
    # horizon: int
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
) -> Tuple[lqr.LQR, lqr.ModelDims]:
    """Calls linearisation and quadratisation function

    Returns:
        LQR: return the LQR structure
        ModelDims: return the dimensionality of model
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
    lqr_params = lqr.LQR(
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

    return lqr_params, lqr.ModelDims(n=model.n, m=model.m, horizon=model.horizon)


def ilqr_simulate(
    model: System, Us: np.ndarray, params: Params
) -> Tuple[Tuple[Array, Array], float]:
    """Simulate forward trajectory and cost with nonlinear params

    Args:
        dynamics (Callable): function of dynamics with args t, x, u, params
        Us (ArrayLike): Input timeseries shape [Txm]
        params (Params): Parameters containing x_init, horizon and theta

    Returns:
        Tuple[[Array, Array], float]: A tuple containing the updated state trajectory and control trajectory,
        and the total cost of the trajectory.
    """
    x0, theta = params.x0, params.theta

    def fwd_step(state, u):
        x, nx_cost = state
        nx = model.dynamics(None, x, u, theta)
        nx_cost = nx_cost + model.cost(None, x, u, theta)
        return (nx, nx_cost), (nx, u)

    (xf, nx_cost), (new_Xs, new_Us) = lax.scan(fwd_step, init=(x0, 0.0), xs=Us)
    total_cost = nx_cost + model.costf(xf, theta)
    new_Xs = np.vstack([x0[None], new_Xs])

    return (new_Xs, new_Us), total_cost


def ilqr_forward_pass(
    model: System,
    params: Params,
    Ks: lqr.Gains,
    Xs: Array,
    Us: Array,
    alpha: float = 1.0,
) -> Tuple[Tuple[Array, Array], float]:
    """
    Performs a forward pass of the iterative Linear Quadratic Regulator (iLQR) algorithm. Uses the deviations of 
    target state and system generated state to update control inputs using gains obtained from LQR solver.

    Args:
        model (System): The nonlinear system model.
        params (Params): The parameters of the system.
        Ks (lqr.Gains): The gains obtained from the LQR controller.
        Xs (np.ndarray): The target state trajectory.
        Us (np.ndarray): The control trajectory.
        alpha (float, optional): The linesearch parameter (TODO: implement linesearch). Defaults to 1.0.

    Returns:
        Tuple[[np.ndarray, np.ndarray], float]: A tuple containing the updated state trajectory and control trajectory,
        and the total cost of the trajectory.
    """

    x0, theta = params.x0, params.theta
    x_hat0 = x0

    def fwd_step(state, inputs):
        x_hat, nx_cost = state
        x, u, K, k = inputs

        delta_x = x_hat - x
        delta_u = K @ delta_x + alpha * k
        u_hat = u + delta_u
        nx_hat = model.dynamics(None, x_hat, u_hat, theta)
        nx_cost = nx_cost + model.cost(None, x_hat, u_hat, theta)
        return (nx_hat, nx_cost), (nx_hat, u_hat)

    (xf, nx_cost), (new_Xs, new_Us) = lax.scan(
        fwd_step, init=(x_hat0, 0.0), xs=(Xs[:-1], Us, Ks.K, Ks.k)
    )
    total_cost = nx_cost + model.costf(xf, theta)
    new_Xs = np.vstack([x0[None], new_Xs])

    return (new_Xs, new_Us), total_cost


def ilQR_solver(
    model: System,
    params: Params,
    X_inits: Array,
    U_inits: Array,
    max_iter: int = 10,
    tol: float = 1e-6,
    verbose: bool = False,
):
    """Solves the iterative Linear Quadratic Regulator (iLQR) problem.

    This function iteratively solves the LQR problem by approximating the dynamics and cost-to-go functions
    using linearizations around the current state and control trajectories. It performs a backward pass to
    calculate the control gains and expected cost reduction, and then performs a forward pass to update the
    state and control trajectories. This process is repeated until the change in the cost-to-go function
    falls below a specified tolerance or the maximum number of iterations is reached.

    Args:
        model (System): The system model.
        params (Params): The parameters of the system.
        X_inits (Array): The initial state trajectory.
        U_inits (Array): The initial control trajectory.
        max_iter (int, optional): The maximum number of iterations. Defaults to 10.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

    Returns:
        Tuple[Array, Array, Array]: A tuple containing the final state trajectory, control trajectory, and
        the adjoint variables.
    """
    # simulate initial cost
    _, c_init = ilqr_simulate(model, U_inits, params)

    # define initial carry tuple: (Xs, Us, Total cost (old), iteration, cond)
    initial_carry = (X_inits, U_inits, c_init, 0, True)

    # define body_fun(carry_tuple)
    def lqr_iter(carry_tuple: Tuple[Array, Array, float, int, bool]):
        """lqr iteration update function"""
        # unravel carry
        old_Xs, old_Us, old_cost, n_iter, carry_on = carry_tuple
        # approximate dyn and loss to LQR with initial {u} and {x}
        lqr_params, sys_dims = approx_lqr(model, old_Xs, old_Us, params)
        # calc gains and expected dJ0
        exp_cost_red, gains = lqr.lqr_backward_pass(
            lqr_params, sys_dims, expected_change=False, verbose=False
        )
        # rollout with non-linear dynamics, α=1. (dJ, Ks), calc_expected_change(dJ=dJ)
        (new_Xs, new_Us), new_total_cost = ilqr_forward_pass(
            model, params, gains, old_Xs, old_Us, alpha=1.0
        )
        # calc change in dJ0 w.r.t old dJ0
        z = (old_cost - new_total_cost) / lqr.calc_expected_change(
            exp_cost_red, alpha=1.0
        )
        # determine cond: ΔJ0 > threshold
        carry_on = np.abs(z) > tol
        if verbose:
            print(f"z-val: {z}")

        return (new_Xs, new_Us, new_total_cost, n_iter + 1, carry_on)

    def loop_fun(carry_tuple: Tuple[Array, Array, float, int, bool], _):
        """if cond false return existing carry else run another iteration of lqr_iter"""
        updated_carry = lax.cond(carry_tuple[-1], lqr_iter, lambda x: x, carry_tuple)
        return updated_carry, _

    # scan through with max iterations
    (Xs_stars, Us_stars, total_cost, n_iters, _), _ = lax.scan(
        loop_fun, initial_carry, None, length=max_iter
    )
    if verbose:
        print(f"Converged in {n_iters}/{max_iter} iterations")
    lqr_params_stars = approx_lqr(model, Xs_stars, Us_stars, params)
    Lambs_stars = lqr.lqr_adjoint_pass(
        Xs_stars, Us_stars, lqr.Params(Xs_stars[0], model.horizon, lqr_params_stars)
    )
    return Xs_stars, Us_stars, Lambs_stars


def define_model():
    def cost(t: int, x: Array, u: Array, theta: Theta):
        return np.sum(x**2) + np.sum(u**2)

    def costf(x: Array, theta: Theta):
        return np.sum(np.abs(x))

    def dynamics(t: int, x: Array, u: Array, theta: Theta):
        return np.tanh(theta.Uh @ x + theta.Wh @ u)

    return System(cost, costf, dynamics, horizon=20, n=3, m=2, dt=0.1)


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
    params = Params(x0=np.zeros((3, 1)), theta=theta)
    model = define_model()
    # generate input
    Us = np.zeros((model.horizon, model.m, 1))
    Us = Us.at[2].set(2)
    # rollout model non-linear dynamics
    Xs = lqr.simulate_trajectory(model.dynamics, Us, params)
    # approx lqr with U and X trajectorys
    lqr_tilde = approx_lqr(model=model, Xs=Xs, Us=Us, params=params)

    Xs_stars, Us_stars, Lambs_stars = ilQR_solver(
        model, params, Xs, Us, max_iter=10, tol=1e-6
    )
