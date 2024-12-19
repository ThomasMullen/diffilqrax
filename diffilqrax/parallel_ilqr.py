"""iterative LQR solver"""

from typing import Callable, Tuple, Optional, Any
from functools import partial
from jax import Array
from diffilqrax.lqr import bmm
import jax
from jax import lax
import jax.numpy as jnp

from diffilqrax.lqr import lqr_adjoint_pass
from diffilqrax.plqr import (
    associative_opt_traj_scan,
    associative_riccati_scan,
    build_fwd_lin_dyn_elements,
    get_dcosts,
    dynamic_operator,
)
from diffilqrax.ilqr import approx_lqr, linesearch, approx_lqr_dyn
from diffilqrax.typs import (
    iLQRParams,
    System,
    LQR,
    Gains,
    CostToGo,
    LQRParams,
    ParallelSystem,
)

jax.config.update("jax_enable_x64", True)  # double precision


def parallel_forward_lin_integration_ilqr(
    model: System, params: iLQRParams, Us_init: Array, a_term: Array
) -> Array:
    """Associative scan for forward linear dynamics

    Args:
        lqr_params (LQRParams): LQR parameters and initial state
        Us_init (Array): input sequence

    Returns:
        Array: state trajectory
    """
    x0 = params.x0
    lqr_params = approx_lqr(
        model,
        jnp.r_[x0[None, ...], jnp.zeros((Us_init.shape[0], x0.shape[0]))],
        Us_init,
        params,
    )
    dyn_elements = build_fwd_lin_dyn_elements(
        LQRParams(x0, lqr_params), Us_init, a_term
    )
    c_as, c_bs = jax.lax.associative_scan(dynamic_operator, dyn_elements)
    return c_bs


def parallel_feedback_lin_dyn_ilqr(
    model: System, params: iLQRParams, Us_init: Array, a_term: Array, Kx: Array
) -> Array:
    """Associative scan for forward linear dynamics
    Function to include feedback + edit the dynamcis to have (A - K)x

    Args:
        lqr_params (LQRParams): LQR parameters and initial state
        Us_init (Array): input sequence

    Returns:
        Array: state trajectory
    """

    lqr_params = approx_lqr(
        model,
        jnp.r_[params.x0[None], jnp.zeros((model.dims.horizon, model.dims.n))],
        Us_init,
        params,
    )
    lqr_params = lqr_params._replace(A=lqr_params.A - bmm(lqr_params.B, Kx))
    dyn_elements = build_fwd_lin_dyn_elements(
        LQRParams(params.x0, lqr_params), Us_init, a_term
    )
    c_as, c_bs = jax.lax.associative_scan(dynamic_operator, dyn_elements)
    return c_bs


def pilqr_forward_pass(
    parallel_model: ParallelSystem,  ##same as system but additionally taking in the parallel dynamics (could be None in which case we would default to linear w a warning)
    params: iLQRParams,
    values: CostToGo,
    Xs: Array,
    Us: Array,
    alpha: float = 1.0,
) -> Tuple[Tuple[Array, Array], float]:
    model = parallel_model.model

    lqr_mats = approx_lqr_dyn(model, Xs, Us, params)
    dyn_bias = lqr_mats.a
    # this is the model from delta_x, so delta_x0 = 0
    lqr_model = LQRParams(
        x0=jnp.zeros_like(params.x0), lqr=lqr_mats._replace(a=jnp.zeros_like(dyn_bias))
    )
    # this is a parallel lin scan anyway b
    _, cs, (Ks, _, _, ks), offsets = associative_opt_traj_scan(
        lqr_model, values.v, values.V, alpha
    )

    # dyn with Ks edit to linear dyn, u + k + K@x as constant term
    # can define function to include feedback + edit the dynamcis to have (A - K)x
    delta_Xs = jnp.r_[jnp.zeros_like(params.x0)[None], cs]
    # Potentially we could return it as output of the parallel scan
    # δu_= B @ Kv @ (v - V@c) - Kx@x
    # where u = u_ + offset (eq 64 in https://arxiv.org/abs/2104.03186)
    delta_Us = ks - bmm(Ks, delta_Xs[:-1]) + offsets
    Kxxs = bmm(Ks, Xs[:-1]) + ks + offsets
    # NOTE: in the case of a linear system, equivalent to `parallel_feedback_lin_dyn_ilqr`
    new_Xs = parallel_model.parallel_dynamics_feedback(
        model, params, Us + Kxxs, dyn_bias, Ks#, Xs
    )
    # this should define the dynamics incorporating the feedback term that says how to handle delta_X (current state - initial traj)
    # we assume Ks@initial_traj is already passed as input so only care about the current state and the parallel_dynamics_feedback
    # function should define how to handle that
    new_Us = Us + delta_Us
    # new_Xs = parallel_model.parallel_dynamics(model, params,  new_Us,  lqr_model_with_a.lqr.a)
    total_cost = jnp.sum(
        jax.vmap(model.cost, in_axes=(0, 0, 0, None))(
            jnp.arange(model.dims.horizon), new_Xs[:-1], new_Us, params.theta
        )
    )
    total_cost += model.costf(new_Xs[-1], params.theta)
    return (new_Xs, new_Us), total_cost


def pilqr_solver(
    parallel_model: ParallelSystem,
    params: iLQRParams,
    Us_init: Array,
    max_iter: int = 40,
    convergence_thresh: float = 1e-6,
    alpha_init: float = 1.0,
    verbose: bool = False,
    use_linesearch: bool = True,
    **linesearch_kwargs,
) -> Tuple[Tuple[Array, Array, Array], float, Array]:
    model = parallel_model.model
    Xs_init = parallel_model.parallel_dynamics(
        model, params, Us_init, jnp.zeros_like(Us_init[..., 0])
    )
    
    a_term = approx_lqr_dyn(parallel_model.model, Xs_init, Us_init, params).a
    Xs_init = parallel_model.parallel_dynamics(model, params, Us_init, a_term)
    c_init = jnp.sum(
        jax.vmap(model.cost, in_axes=(0, 0, 0, None))(
            jnp.arange(model.dims.horizon), Xs_init[:-1], Us_init, params.theta
        )
    )
    c_init += model.costf(Xs_init[-1], params.theta)
    initial_carry = (Xs_init, Us_init, c_init, 0, True)
    prollout = partial(pilqr_forward_pass, parallel_model, params)

    def plqr_iter(carry_tuple: Tuple[Array, Array, float, int, bool]):
        """lqr iteration update function"""
        # unravel carry
        old_Xs, old_Us, old_cost, n_iter, carry_on = carry_tuple
        lqr = approx_lqr(model, old_Xs, old_Us, params)
        lqr_params = LQRParams(params.x0, lqr)
        etas, Js = associative_riccati_scan(
            lqr_params
        )  ##need to make a parallel v of that
        exp_dJ = get_dcosts(lqr_params, etas, Js)

        def linesearch_wrapped(*args):
            value_fns, Xs_init, Us_init, alpha_init = args
            return linesearch(
                prollout,
                value_fns,  ###the linesearch should be done with the Ks, not the etas and Js
                Xs_init,
                Us_init,
                alpha_init,
                cost_init=old_cost,
                expected_dJ=exp_dJ,
                **linesearch_kwargs,
            )  ## TO CHECK : for the linesearch I believe we can use the exact same function as in ilqr, as long as we pass a different rollout - is that riht?

        # if no line search: α = 1.0; else use dynamic line search
        (new_Xs, new_Us), new_total_cost = lax.cond(
            use_linesearch,
            linesearch_wrapped,
            prollout,
            CostToGo(Js, etas),
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
    # TODO : Not sure if this is needed - otherwise should solve with Vs and vs
    Lambs_star = lqr_adjoint_pass(
        Xs_star, Us_star, LQRParams(Xs_star[0], lqr_params_stars)
    )
    return (Xs_star, Us_star, Lambs_star), total_cost, costs
