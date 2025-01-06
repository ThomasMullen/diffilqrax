"""Solve iLQR with parallel dynamics using associative scans"""

from typing import Tuple, Any
from functools import partial
import jax
from jax import Array
from jax import lax
import jax.numpy as jnp

from diffilqrax.lqr import lqr_adjoint_pass
from diffilqrax.lqr import bmm
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
    CostToGo,
    LQRParams,
    ParallelSystem,
)

jax.config.update("jax_enable_x64", True)  # double precision


def parallel_forward_lin_integration_ilqr(
    model: System, params: iLQRParams, Us_init: Array, a_term: Array
) -> Array:
    """
    Perform associative scan for forward linear dynamics.

    Parameters
    ----------
    model : System
        The system model.
    params : iLQRParams
        The iLQR parameters.
    Us_init : Array
        The initial control sequence.
    a_term : Array
        The affine term for the dynamics.

    Returns
    -------
    Array
        The state trajectory.
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
    """
    Perform associative scan for forward linear dynamics with feedback.

    Parameters
    ----------
    model : System
        The system model.
    params : iLQRParams
        The iLQR parameters.
    Us_init : Array
        The initial control sequence.
    a_term : Array
        The affine term for the dynamics.
    Kx : Array
        The feedback gain matrix.

    Returns
    -------
    Array
        The state trajectory.
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
    parallel_model: ParallelSystem | Any,
    params: iLQRParams,
    values: CostToGo,
    Xs: Array,
    Us: Array,
    alpha: float = 1.0,
) -> Tuple[Tuple[Array, Array], float]:
    """
    Perform a forward pass of the parallel iLQR algorithm.

    Parameters
    ----------
    parallel_model : ParallelSystem | Any
        The parallel system model.
    params : iLQRParams
        The iLQR parameters.
    values : CostToGo
        The cost-to-go values.
    Xs : Array
        The state trajectory.
    Us : Array
        The control trajectory.
    alpha : float, optional
        The linesearch parameter, by default 1.0.

    Returns
    -------
    Tuple[Tuple[Array, Array], float]
        A tuple containing the updated state trajectory and control trajectory, and the total cost
        of the trajectory.
    """
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
        model, params, Us + Kxxs, dyn_bias, Ks
    )
    # this should define the dynamics incorporating the feedback term that says how to handle
    # delta_X (current state - initial traj) we assume Ks@initial_traj is already passed as
    # input so only care about the current state and the parallel_dynamics_feedback function
    # should define how to handle that
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
    """
    Solve the parallel iterative Linear Quadratic Regulator (iLQR) problem.

    Parameters
    ----------
    parallel_model : ParallelSystem
        The parallel system model.
    params : iLQRParams
        The iLQR parameters.
    Us_init : Array
        The initial control trajectory.
    max_iter : int, optional
        The maximum number of iterations, by default 40.
    convergence_thresh : float, optional
        The convergence threshold, by default 1e-6.
    alpha_init : float, optional
        The initial step size for the forward pass, by default 1.0.
    verbose : bool, optional
        Whether to print debug information, by default False.
    use_linesearch : bool, optional
        Whether to use line search for the forward pass, by default True.
    **linesearch_kwargs
        Additional keyword arguments for the line search.

    Returns
    -------
    Tuple[Tuple[Array, Array, Array], float, Array]
        A tuple containing the final state trajectory, control trajectory, and the adjoint
        variables. Also returns the total cost of the trajectory and the cost history.
    """
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
        """
        Perform one iteration of the parallel iLQR algorithm.

        Parameters
        ----------
        carry_tuple : Tuple[Array, Array, float, int, bool]
            The carry tuple containing the state trajectory, control trajectory, total cost,
            iteration number, and a boolean indicating whether to continue.

        Returns
        -------
        Tuple[Array, Array, float, int, bool]
            The updated carry tuple.
        """
        old_Xs, old_Us, old_cost, n_iter, carry_on = carry_tuple
        lqr = approx_lqr(model, old_Xs, old_Us, params)
        lqr_params = LQRParams(params.x0, lqr)
        etas, Js = associative_riccati_scan(lqr_params)
        exp_dJ = get_dcosts(lqr_params, etas, Js)

        def linesearch_wrapped(*args):
            value_fns, Xs_init, Us_init, alpha_init = args
            return linesearch(
                prollout,
                value_fns,
                Xs_init,
                Us_init,
                alpha_init,
                cost_init=old_cost,
                expected_dJ=exp_dJ,
                **linesearch_kwargs,
            )
            # NOTE : for the linesearch I believe we can use the exact same function
            # as in ilqr, as long as we pass a different rollout - is that riht?

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

        z = (old_cost - new_total_cost) / jnp.abs(old_cost)
        carry_on = z > convergence_thresh
        return (new_Xs, new_Us, new_total_cost, n_iter + 1, carry_on)

    def loop_fun(carry_tuple: Tuple[Array, Array, float, int, bool], _):
        """
        Loop function for the parallel iLQR algorithm.

        Parameters
        ----------
        carry_tuple : Tuple[Array, Array, float, int, bool]
            The carry tuple containing the state trajectory, control trajectory, total cost,
            iteration number, and a boolean indicating whether to continue.
        _ : Any
            Unused parameter.

        Returns
        -------
        Tuple[Tuple[Array, Array, float, int, bool], float]
            The updated carry tuple and the total cost.
        """
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
