"""iterative LQR solver"""

from typing import Callable, Tuple, Optional, Any
from functools import partial
from jax import Array
import jax
from jax import lax
import jax.numpy as jnp

from diffilqrax.lqr import lqr_adjoint_pass
from diffilqrax.plqr import (
    parallel_lin_dyn_scan,
    parallel_riccati_scan,
    build_fwd_lin_dyn_elements,
    get_dJs,
    dynamic_operator
)
from diffilqrax.ilqr import approx_lqr, linesearch, approx_lqr_dyn
from diffilqrax.typs import (
    iLQRParams,
    System,
    LQR,
    Gains,
    CostToGo,
    LQRParams,
)

jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update("jax_disable_jit", True)  # uncomment for debugging purposes

def make_pilqr_simulate(
    parallel_fwd_integration: Callable, model: System, Us: Array, params: LQRParams
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
    #lqr_params = LQRParams(x0=params.x0, lqr=approx_lqr(model, params.x0, Us, params)
    Xs = parallel_fwd_integration(model, params, Us)
    total_cost = jnp.sum(jax.vmap(model.cost, in_axes = (0,0,0,None))(jnp.arange(model.dims.horizon), Xs[:-1], Us, params.theta))
    total_cost += model.costf(Xs[-1], params.theta)
    return (Xs, Us), total_cost

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
    lqr_model = LQRParams(x0 = jnp.zeros_like(params.x0), lqr = approx_lqr(model, Xs, Us, params)) #this is the model from delta_x, so delta_x0 = 0
    Fs, cs, Ks, offsets = parallel_dynamics_update(lqr_model, etas, Js, alpha) ##not sure why it fails at this linerization...
    Kx = Ks[0]
    delta_Xs = jnp.r_[jnp.zeros_like(params.x0)[None], cs]
    delta_Us = Ks[-1] + offsets + lqr_model.lqr.a - jax.vmap(lambda a, b, c : jnp.linalg.pinv(c)@(a@b), in_axes = (0,0,0))(Kx, delta_Xs[:-1], lqr_model.lqr.B)
    new_Us = Us + delta_Us
    # delta_Xs = cs #- Xs[1:]
    # delta_Xs = jnp.r_[jnp.zeros((1, delta_Xs.shape[1])), delta_Xs]
    # delta_Us = jax.vmap(get_delta_u)(Ks, delta_Xs[:-1], etas[1:], lqr_model.lqr.a) + offsets
    # #print(Us[:10], offsets[:10], "cattt")
    # new_Us = Us + delta_Us
    (new_Xs, _), total_cost = pilqr_simulate(model, new_Us, params)
    return (new_Xs, new_Us), total_cost 


def parallel_forward_lin_integration_ilqr(
    model: System, params: iLQRParams, Us_init: Array
) -> Array:
    """Associative scan for forward linear dynamics

    Args:
        lqr_params (LQRParams): LQR parameters and initial state
        Us_init (Array): input sequence

    Returns:
        Array: state trajectory
    """
    x0 = params.x0
    lqr_params = approx_lqr(model, jnp.r_[x0[None,...], jnp.zeros((Us_init.shape[0],x0.shape[0]))], Us_init, params)
    dyn_elements = build_fwd_lin_dyn_elements(LQRParams(x0, lqr_params), Us_init)
    c_as, c_bs = jax.lax.associative_scan(dynamic_operator, dyn_elements)
    return c_bs
    
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
    parallel_fwd_integration: Callable = parallel_forward_lin_integration_ilqr,
    **linesearch_kwargs,
) -> Tuple[Tuple[Array, Array, Array], float, Array]:
    pilqr_simulate = partial(make_pilqr_simulate, parallel_fwd_integration) 
    (Xs_init, _), c_init = pilqr_simulate(model, Us_init, params) ### need to make some     changes to the simulate function to make it work with the parallel dynamics 
    initial_carry = (Xs_init, Us_init, c_init, 0, True)
    prollout = partial(pilqr_forward_pass, parallel_dynamics_update, pilqr_simulate, model, params)
    def plqr_iter(carry_tuple: Tuple[Array, Array, float, int, bool]):
        """lqr iteration update function"""
        # unravel carry
        old_Xs, old_Us, old_cost, n_iter, carry_on = carry_tuple
        lqr = approx_lqr(model, old_Xs, old_Us, params)
        lqr_params = LQRParams(params.x0, lqr)
        etas, Js = parallel_riccati_scan(lqr_params) ##need to make a parallel v of that
        exp_dJ = get_dJs(lqr_params, etas, Js)
        values = (etas, Js)
        def linesearch_wrapped(*args): 
            values, Xs_init, Us_init, alpha_init = args
            return linesearch(
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
        z = (old_cost - new_total_cost) / jnp.abs(old_cost)

        # determine cond: Δold_cost > threshold
        carry_on = z > convergence_thresh  # n_iter < 70 #
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
    Lambs_star = lqr_adjoint_pass(
        Xs_star, Us_star, LQRParams(Xs_star[0], lqr_params_stars)
    ) #TODO : write parallel version
    return (Xs_star, Us_star, Lambs_star), total_cost, costs


