"""iterative LQR solver"""

from typing import Callable, Tuple, Optional
from functools import partial
from jax import Array
import jax
from jax import lax
import jax.numpy as jnp

from diffilqrax import lqr, ilqr
from diffilqrax.plqr import parallel_lin_dyn_scan, parallel_riccati_scan, parallel_forward_lin_integration
from diffilqrax.typs import (
    iLQRParams,
    System,
    LQR,
    Gains,
    CostToGo,
    LQRParams,
)


#jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_disable_jit", False)  # uncomment for debugging purposes


def sum_cost_to_go(x: CostToGo) -> Array:
    """Sum linear and quadratic components of cost-to-go tuple"""
    return x.V + x.v


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


def time_map(fun: Callable) -> Callable:
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
    """Approximate non-linear model as LQR by taylor expanding about state and
    control trajectories.

    Args:
        model (System): The system model
        Xs (Array): The state trajectory
        Us (Array): The control trajectory
        params (iLQRParams): The iLQR parameters

    Returns:
        LQR: The LQR parameters.
    """
    theta = params.theta
    tps = jnp.arange(model.dims.horizon)

    (Fx, Fu) = time_map(linearise(model.dynamics))(tps, Xs[:-1], Us, theta)
    (Cx, Cu) = time_map(linearise(model.cost))(tps, Xs[:-1], Us, theta)
    (Cxx, Cxu), (_, Cuu) = time_map(quadratise(model.cost))(tps, Xs[:-1], Us, theta)
    fCx = jax.jacrev(model.costf)(Xs[-1], theta)
    fCxx = jax.jacfwd(jax.jacrev(model.costf))(Xs[-1], theta)

    # set-up LQR
    lqr_params = LQR(
        A=Fx,
        B=Fu,
        a=jnp.zeros((model.dims.horizon, model.dims.n)),
        Q=Cxx,
        q=Cx,
        R=Cuu,
        r=Cu,
        S=Cxu,
        Qf=fCxx,
        qf=fCx,
    )()

    return lqr_params


def approx_lqr_dyn(model: System, Xs: Array, Us: Array, params: iLQRParams) -> LQR:
    """Calls linearisation and quadratisation function

    Returns:
        LQR: return the LQR structure
    """
    theta = params.theta
    tps = jnp.arange(model.dims.horizon)

    def get_diff_dyn(t, x, u, theta):
        (Fx, Fu) = linearise(model.dynamics)(t, x, u, theta)
        return model.dynamics(t, x, u, theta) - Fx @ x - Fu @ u

    (Fx, Fu) = time_map(linearise(model.dynamics))(tps, Xs[:-1], Us, theta)
    f = jax.vmap(get_diff_dyn, in_axes=(0, 0, 0, None))(tps, Xs[:-1], Us, theta)
    (Cx, Cu) = time_map(linearise(model.cost))(tps, Xs[:-1], Us, theta)
    (Cxx, Cxu), (_, Cuu) = time_map(quadratise(model.cost))(tps, Xs[:-1], Us, theta)
    fCx = jax.jacrev(model.costf)(Xs[-1], theta)
    fCxx = jax.jacfwd(jax.jacrev(model.costf))(Xs[-1], theta)

    # set-up LQR
    lqr_params = LQR(
        A=Fx,
        B=Fu,
        a=f,  # jnp.zeros((model.dims.horizon, model.dims.n)),
        Q=Cxx,
        q=Cx,  # - bmm(Cxx,Xs[:-1]) - bmm(Cxu, Us),
        r=Cu,  # - bmm(Cuu, Us) - bmm(Cxu.transpose(0, 2, 1), Xs[:-1]),
        R=Cuu,
        S=Cxu,
        Qf=fCxx,
        qf=fCx,  # - mm(fCxx, Xs[-1]),
    )()

    return lqr_params


def make_pilqr_simulate(
    parallel_fwd_integration: Callable, model: System, Us: Array, params: iLQRParams
) -> Tuple[Tuple[Array, Array], float]:
    """Simulate forward trajectory and cost with nonlinear params

    Args:
        dynamics (Callable): function of dynamics with args t, x, u, params
        Us (ArrayLike): Input timeseries shape [Txm]
        params (iLQRParams): Parameters containing x_init, horizon and theta

    Returns:
        Tuple[[Array, Array], float]: A tuple containing the updated state and control trajectory,
            and the total cost of the trajectory.
    """
    ##same as running linear dynamics but replacing the cs with B@us
    Xs = parallel_fwd_integration(params, Us)
    total_cost = jax.vmap(model.cost)(jnp.arange(model.dims.horizon), Xs[:-1], Us, params.theta).sum() + model.costf(Xs[-1], params.theta)
    return (Xs, Us), total_cost

def get_delta_u(Ks, x, v, c):
    Kx, Kv, Kc = Ks
    delta_U = -Kx@x + Kv@v - Kc@c
    return delta_U

def pilqr_forward_pass(
    parallel_dynamics_update: Callable,
    pilqr_simulate: Callable,
    model: System,
    params: iLQRParams,
    values: Tuple[Array, Array],
    Xs: Array,
    Us: Array,
    alpha: float = 1.0,
) -> Tuple[Tuple[Array, Array], float]:
    etas, Js = values
    Fs, cs, Ks = parallel_dynamics_update(model, etas, Js, alpha)
    delta_Xs = cs
    delta_Us = jax.vmap(get_delta_u)(Ks, delta_Xs, etas, cs)
    new_Us = Us + delta_Us
    (new_Xs, _), _ = pilqr_simulate(model, new_Us, params)
    total_cost = jax.vmap(model.cost)(jnp.arange(model.dims.horizon), new_Xs[:-1], new_Us, params.theta).sum() + model.costf(new_Xs[-1], params.theta)
    return (new_Xs, new_Us), total_cost  


def pilqr_solver(
    model: System,
    params: iLQRParams,
    Us_init: Array,
    max_iter: int = 40,
    convergence_thresh: float = 1e-6,
    alpha_init: float = 1.0,
    verbose: bool = False,
    use_linesearch: bool = True,
    parallel_dynamics_update: Callable = parallel_lin_dyn_scan,
    parallel_fwd_integration: Callable = parallel_forward_lin_integration,
    **linesearch_kwargs,
) -> Tuple[Tuple[Array, Array, Array], float, Array]:
    pilqr_simulate = partial(make_pilqr_simulate, parallel_fwd_integration) 
    (Xs_init, _), c_init = pilqr_simulate(model, Us_init, params) ### need to make some parallel v of that ###
    initial_carry = (Xs_init, Us_init, c_init, 0, True)
    prollout = partial(pilqr_forward_pass, pilqr_simulate, model, params)
    def plqr_iter(carry_tuple: Tuple[Array, Array, float, int, bool]):
        """lqr iteration update function"""
        # unravel carry
        old_Xs, old_Us, old_cost, n_iter, carry_on = carry_tuple
        lqr = approx_lqr(model, old_Xs, old_Us, params)
        lqr_params = LQRParams(params.x0, lqr)
        etas, Js, exp_dJ = parallel_riccati_scan(lqr_params) ##need to make a parallel v of that
        values = (etas, Js)
        def linesearch_wrapped(*args): 
            values, Xs_init, Us_init, alpha_init = args
            return ilqr.linesearch(
                prollout,
                values, ###the linesearch should be done with the Ks, not the etas and Js
                Xs_init,
                Us_init,
                alpha_init,
                cost_init=old_cost,
                expected_dJ=exp_dJ,
                **linesearch_kwargs,
            ) ## TO CHECK : for the linesearch I believe we can use the exact same function as in ilqr, as long as we pass a different rollout - is that riht?

        # if no line search: α = 1.0; else use dynamic line search
        (new_Xs, new_Us), new_total_cost = lax.cond(
            use_linesearch,
            linesearch_wrapped,
            prollout,
            values,
            old_Xs,
            old_Us,
            alpha_init,
        )

        # calc change in dold_cost w.r.t old dold_cost
        z = (old_cost - new_total_cost) / old_cost

        # determine cond: Δold_cost > threshold
        carry_on = jnp.abs(z) > convergence_thresh  # n_iter < 70 #
        return (new_Xs, new_Us, new_total_cost, n_iter + 1, carry_on)

    def loop_fun(carry_tuple: Tuple[Array, Array, float, int, bool], _):
        """if cond false return existing carry else run another iteration of lqr_iter"""
        updated_carry = lax.cond(carry_tuple[-1], plqr_iter, lambda x: x, carry_tuple)
        return updated_carry, updated_carry[2]

    # scan through with max iterations
    (Xs_star, Us_star, total_cost, n_iters, _), costs = lax.scan(
        loop_fun, initial_carry, None, length=max_iter
    )
    if verbose:
        jax.debug.print(f"Converged in {n_iters}/{max_iter} iterations")
        jax.debug.print(f"old_cost: {total_cost}")
    lqr_params_stars = approx_lqr(model, Xs_star, Us_star, params)
    Lambs_star = lqr.lqr_adjoint_pass(
        Xs_star, Us_star, LQRParams(Xs_star[0], lqr_params_stars)
    ) #do we need a parallel v of this? yes bc it requires a scan... 
    return (Xs_star, Us_star, Lambs_star), total_cost, costs


