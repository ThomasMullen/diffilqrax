"""iterative LQR solver"""
from typing import Callable, Any, Optional, NamedTuple, Tuple, Union
from functools import partial
from jax import Array
from jax.typing import ArrayLike
import jax
import jax.lax as lax
import jax.numpy as jnp

import src.lqr as lqr
from src.utils import keygen, initialise_stable_dynamics
from src.typs import *


jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_disable_jit", False)  # uncomment for debugging purposes

sum_cost_to_go_struct = lambda x: x.V + x.v


def linearise(fun: Callable) -> Callable:
    """Function that finds jacobian w.r.t to x and u inputs.

    Args:
        fun (Callable): args (t, x, u, params)

    Returns:
        Callable[[Callable], Callable]): Jacobian tuple evaluated at args 1 and 2
    """
    return jax.jacrev(fun, argnums=(1, 2))


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
    return jax.vmap(fun, in_axes=(0, 0, 0, None))
    # return jax.vmap(fun, in_axes=(None, 0, 0, None))


def approx_lqr(model: System, Xs: Array, Us: Array, params: iLQRParams) -> LQR:
    """Calls linearisation and quadratisation function

    Returns:
        LQR: return the LQR structure
        ModelDims: return the dimensionality of model
    """
    theta = params.theta
    tps = jnp.arange(model.dims.horizon)

    (Fx, Fu) = vectorise_fun_in_time(linearise(model.dynamics))(tps, Xs[:-1], Us, theta)
    (Cx, Cu) = vectorise_fun_in_time(linearise(model.cost))(tps, Xs[:-1], Us, theta)
    (Cxx, Cxu), (Cux, Cuu) = vectorise_fun_in_time(quadratise(model.cost))(
        tps, Xs[:-1], Us, theta
    )
    fCx = jax.jacrev(model.costf)(Xs[-1], theta)
    fCxx = jax.jacfwd(jax.jacrev(model.costf))(Xs[-1], theta)

    # set-up LQR
    lqr_params = LQR(
        A=Fx,
        B=Fu,
        a=jnp.zeros((model.dims.horizon, model.dims.n)),
        Q=Cxx,
        q=Cx,
        Qf=fCxx,
        qf=fCx,
        R=Cuu,
        r=Cu,
        S=Cxu,
    )()

    return lqr_params


def ilqr_simulate(
    model: System, Us: Array, params: iLQRParams
) -> Tuple[Tuple[Array, Array], float]:
    """Simulate forward trajectory and cost with nonlinear params

    Args:
        dynamics (Callable): function of dynamics with args t, x, u, params
        Us (ArrayLike): Input timeseries shape [Txm]
        params (iLQRParams): Parameters containing x_init, horizon and theta

    Returns:
        Tuple[[Array, Array], float]: A tuple containing the updated state trajectory and control trajectory,
        and the total cost of the trajectory.
    """
    x0, theta = params.x0, params.theta
    tps = jnp.arange(model.dims.horizon)

    def fwd_step(state, inputs):
        t, u = inputs
        x, nx_cost = state
        nx = model.dynamics(t, x, u, theta)
        nx_cost = nx_cost + model.cost(t, x, u, theta)
        return (nx, nx_cost), (nx, u)

    (xf, nx_cost), (new_Xs, new_Us) = lax.scan(fwd_step, init=(x0, 0.0), xs=(tps, Us))
    total_cost = nx_cost + model.costf(xf, theta)
    new_Xs = jnp.vstack([x0[None], new_Xs])

    return (new_Xs, new_Us), total_cost


def ilqr_forward_pass(
    model: System,
    params: iLQRParams,
    Ks: Gains,
    Xs: Array,
    Us: Array,
    alpha: float = 1.0,
) -> Tuple[Tuple[Array, Array], float]:
    """
    Performs a forward pass of the iterative Linear Quadratic Regulator (iLQR) algorithm. Uses the deviations of
    target state and system generated state to update control inputs using gains obtained from LQR solver.

    Args:
        model (System): The nonlinear system model.
        params (iLQRParams): The parameters of the system.
        Ks (Gains): The gains obtained from the LQR controller.
        Xs (np.ndarray): The target state trajectory.
        Us (np.ndarray): The control trajectory.
        alpha (float, optional): The linesearch parameter. Defaults to 1.0.

    Returns:
        Tuple[[np.ndarray, np.ndarray], float]: A tuple containing the updated state trajectory and control trajectory,
        and the total cost of the trajectory.
    """

    x0, theta = params.x0, params.theta
    tps = jnp.arange(model.dims.horizon)
    x_hat0 = x0

    def fwd_step(state, inputs):
        x_hat, nx_cost = state
        t, x, u, K, k = inputs

        delta_x = x_hat - x
        delta_u = K @ delta_x + alpha * k
        u_hat = u + delta_u
        nx_hat = model.dynamics(t, x_hat, u_hat, theta)
        nx_cost = nx_cost + model.cost(t, x_hat, u_hat, theta)
        return (nx_hat, nx_cost), (nx_hat, u_hat)

    (xf, nx_cost), (new_Xs, new_Us) = lax.scan(
        fwd_step, init=(x_hat0, 0.0), xs=(tps, Xs[:-1], Us, Ks.K, Ks.k)
    )
    total_cost = nx_cost + model.costf(xf, theta)
    new_Xs = jnp.vstack([x0[None], new_Xs])

    return (new_Xs, new_Us), total_cost


def ilQR_solver(
    model: System,
    params: iLQRParams,
    U_inits: Array,
    max_iter: int = 40,
    convergence_thresh: float = 1e-6,
    alpha_init: float = 1.0,
    verbose: bool = False,
    use_linesearch: bool = True,
    **linesearch_kwargs,
) -> Tuple[Tuple[Array, Array, Array], float, Array]:
    """Solves the iterative Linear Quadratic Regulator (iLQR) problem.

    This function iteratively solves the LQR problem by approximating the dynamics and cost-to-go functions
    using linearizations around the current state and control trajectories. It performs a backward pass to
    calculate the control gains and expected cost reduction, and then performs a forward pass to update the
    state and control trajectories. This process is repeated until the change in the cost-to-go function
    falls below a specified convergence or the maximum number of iterations is reached.

    Args:
        model (System): The system model.
        params (iLQRParams): The parameters of the system.
        X_inits (Array): The initial state trajectory.
        U_inits (Array): The initial control trajectory.
        max_iter (int, optional): The maximum number of iterations. Defaults to 10.
        convergence_thresh (float, optional): The convergence for convergence. Defaults to 1e-6.
        alpha_init (float, optional): The initial step size for the forward pass. Defaults to 1.0.
        verbose (bool, optional): Whether to print debug information. Defaults to False.
        use_linesearch (bool, optional): Whether to use line search for the forward pass. Defaults to False.

    Returns:
        Tuple[Tuple[Array, Array, Array], float, Array]: A tuple containing the final state trajectory,
        control trajectory, and the adjoint variables. Also returns the total cost of the trajectory and the
        cost history.
    """
    # simulate initial cost
    (X_inits, _), c_init = ilqr_simulate(model, U_inits, params)

    # define initial carry tuple: (Xs, Us, Total cost (old), iteration, cond)
    initial_carry = (X_inits, U_inits, c_init, 0, True)

    rollout = partial(ilqr_forward_pass, model, params)

    # define body_fun(carry_tuple)
    def lqr_iter(carry_tuple: Tuple[Array, Array, float, int, bool]):
        """lqr iteration update function"""
        # unravel carry
        old_Xs, old_Us, old_cost, n_iter, carry_on = carry_tuple
        # approximate dyn and loss to LQR with initial {u} and {x}
        lqr_params = approx_lqr(model, old_Xs, old_Us, params)
        # calc gains and expected dold_cost
        (exp_cost_red, gains) = lqr.lqr_backward_pass(
            lqr_params, dims=model.dims, expected_change=False, verbose=False
        )

        # rollout with non-linear dynamics, α=1. (dJ, Ks), calc_expected_change(dJ=dJ)
        # wrap linesearch with rollout
        def linesearch_wrapped(*args):
            Ks, Xs_init, Us_init, alpha_init = args
            return linesearch(
                rollout,
                Ks,
                Xs_init,
                Us_init,
                alpha_init,
                cost_init=old_cost,
                expected_dJ=exp_cost_red,
                **linesearch_kwargs,
            )

        # if no line search: α = 1.0; else use dynamic line search
        (new_Xs, new_Us), new_total_cost = lax.cond(
            use_linesearch,
            linesearch_wrapped,
            rollout,
            gains,
            old_Xs,
            old_Us,
            alpha_init,
        )

        # calc change in dold_cost w.r.t old dold_cost
        z = (old_cost - new_total_cost) / old_cost

        # determine cond: Δold_cost > threshold
        carry_on = jnp.abs(z) > convergence_thresh

        return (new_Xs, new_Us, new_total_cost, n_iter + 1, carry_on)

    def loop_fun(carry_tuple: Tuple[Array, Array, float, int, bool], _):
        """if cond false return existing carry else run another iteration of lqr_iter"""
        updated_carry = lax.cond(carry_tuple[-1], lqr_iter, lambda x: x, carry_tuple)
        return updated_carry, updated_carry[2]

    # scan through with max iterations
    (Xs_stars, Us_stars, total_cost, n_iters, _), costs = lax.scan(
        loop_fun, initial_carry, None, length=max_iter
    )
    if verbose:
        jax.debug.print(f"Converged in {n_iters}/{max_iter} iterations")
        jax.debug.print(f"old_cost: {total_cost}")
    lqr_params_stars = approx_lqr(model, Xs_stars, Us_stars, params)
    Lambs_stars = lqr.lqr_adjoint_pass(
        Xs_stars, Us_stars, LQRParams(Xs_stars[0], lqr_params_stars)
    )
    return (Xs_stars, Us_stars, Lambs_stars), total_cost, costs


def linesearch(
    update: Callable,
    Ks: Gains,
    Xs_init: Array,
    Us_init: Array,
    alpha_init: float,
    cost_init: float,
    expected_dJ: CostToGo,
    beta: float = 0.8,
    max_iter_linesearch: int = 20,
    tol: float = 0.99999,
    alpha_min=0.0001,
) -> Tuple[Tuple[Array, Array], float, float, Array]:
    """Implementation of the line search backtracking algorithm for ilqr algorithm.
    Each iteration of the line search algorithm performs a forward pass with the new control inputs
    defined by the gains and an alpha value. The change in cost is compared to the expected change in cost
    and the alpha is selected based on the ratio of the two values above a given tolerance. Otherwise, the
    alpha value is reduced by a factor of beta.

    Args:
        update (Callable): rollout function which returns new Xs, Us and cost
        Ks (Gains): Gains obtained from the LQR controller.
        Xs_init (Array): state trajectory
        Us_init (Array): input trajectory
        cost_init (float): cost of initial trajectory
        alpha_init (float): initialised alpha value
        expected_dJ (CostToGo): expected change in cost from LQR controller
        beta (float): reduction factor for alpha
        max_iter_linesearch (int, optional): Maximum iterations of linesearch. Defaults to 20.
        tol (float, optional): Tolerance of ratio of actual to expected cost change to accept alpha value.
            Defaults to 0.99999.
        alpha_min (float, optional): Minimum alpha value. Defaults to 0.0001.

    Returns:
        Tuple: Returns the updated state and control trajectories, the alpha value, the total cost, and the cost history.
    """
    # initialise carry: Xs, Us, old ilqr cost, alpha, n_iter, carry_on
    initial_carry = (Xs_init, Us_init, 0.0, cost_init, alpha_init, 0, 10.0, 0.0, True)
    # jax.debug.print(f"\nLinesearch: J0={cost_init:.03f} α={alpha_init:.03f} β={beta:.03f}")

    def backtrack_iter(carry):
        """Rollout with new alpha and update alpha if z-value is above threshold"""
        # parse out carry
        Xs, Us, new_cost, old_cost, alpha, n_iter, _, _, carry_on = carry
        # rollout with alpha
        (new_Xs, new_Us), new_cost = update(Ks, Xs, Us, alpha=alpha)

        # calc expected cost reduction
        expected_delta_j = lqr.calc_expected_change(expected_dJ, alpha=alpha)
        # calc z-value
        z = (old_cost - new_cost) / expected_delta_j

        # if verbose:
        # jax.debug.print(
        # f"it={1+n_iter:02} α={alpha:.03f} z:{z:.03f} pJ:{old_cost:.03f} nJ:{new_cost:.03f} ΔJ:{old_cost-new_cost:.03f} <ΔJ>:{expected_delta_j:.03f}"
        # )

        # ensure to keep Xs and Us that reduce z-value
        new_cost = jnp.where(jnp.isnan(new_cost), cost_init, new_cost)
        # add control flow to carry on or not
        above_threshold = z > tol
        carry_on = lax.bitwise_not(jnp.logical_or(alpha < alpha_min, above_threshold))
        # Only return new trajs if leads to a strict cost decrease
        new_Xs = jnp.where(above_threshold, new_Xs, Xs)
        new_Us = jnp.where(above_threshold, new_Us, Us)
        new_cost = jnp.where(above_threshold, new_cost, old_cost)
        # update alpha
        alpha *= beta

        return (
            new_Xs,
            new_Us,
            new_cost,
            old_cost,
            alpha,
            n_iter + 1,
            z,
            expected_delta_j,
            carry_on,
        )

    def loop_fun(carry_tuple: Tuple[Array, Array, float, float, int, bool], _):
        """if cond false return existing carry else run another rollout with new alpha"""
        # assign function given carry_on condition
        updated_carry = lax.cond(
            carry_tuple[-1], backtrack_iter, lambda x: x, carry_tuple
        )
        return updated_carry, (updated_carry[2], updated_carry[3])

    # scan through with max iterations
    (Xs_opt, Us_opt, cost_opt, old_cost, alpha, its, z, exp_dj, *_), costs = lax.scan(
        loop_fun, initial_carry, None, length=max_iter_linesearch
    )
    # if verbose:
    # jax.debug.print(
    # f"Nit:{its:02} α:{alpha/beta:.03f} z:{z:.03f} J*:{cost_opt:.03f} ΔJ:{cost_init-cost_opt:.03f} <ΔJ>:{exp_dj:.03f}"
    # )
    return (Xs_opt, Us_opt), cost_opt


def define_model():
    def cost(t: int, x: Array, u: Array, theta: Any):
        return jnp.sum(x**2) + jnp.sum(u**2)

    def costf(x: Array, theta: Theta):
        # return jnp.sum(jnp.abs(x))
        return jnp.sum(x**2)

    def dynamics(t: int, x: Array, u: Array, theta: Union[Theta, PendulumParams]):
        # return pendulum_dynamics(t,x,u,theta)
        return jnp.tanh(theta.Uh @ x + theta.Wh @ u)

    # return System(cost, costf, dynamics, lqr.ModelDims(horizon=100, n=3, m=1, dt=0.1))
    return System(cost, costf, dynamics, ModelDims(horizon=100, n=8, m=2, dt=0.1))
