"""
Module contains functions for solving the differential linear quadratic regulator (DLQR) problem.
"""

from typing import Tuple
from functools import partial
from jax import Array, custom_vjp
import jax.numpy as jnp
from jax.numpy import matmul as mm

from diffilqrax.lqr import (
    solve_lqr,
    solve_lqr_swap_x0,
    bmm,
)
from diffilqrax.typs import (
    LQRParams,
    ModelDims,
    LQR,
    symmetrise_matrix,
    symmetrise_tensor,
)

# v_outer = jax.vmap(jnp.outer) # vectorized outer product through time i.e. 'ij,ik->ijk'


def offset_lqr(lqr: LQR, x_stars: Array, u_stars: Array) -> LQR:
    """
    Adjust linear terms of LQR cost along nominal trajectory.

    Parameters
    ----------
    lqr : LQR
        LQR parameters.
    x_stars : Array
        Nominal state trajectory.
    u_stars : Array
        Nominal control trajectory.

    Returns
    -------
    LQR
        Adjusted LQR parameters.
    """
    return LQR(
        A=lqr.A,
        B=lqr.B,
        a=lqr.a,
        Q=lqr.Q,
        q=lqr.q - bmm(lqr.Q, x_stars[:-1]) - bmm(lqr.S, u_stars),
        R=lqr.R,
        r=lqr.r - bmm(lqr.R, u_stars) - bmm(lqr.S.transpose(0, 2, 1), x_stars[:-1]),
        S=lqr.S,
        Qf=lqr.Qf,
        qf=lqr.qf - mm(lqr.Qf, x_stars[-1]),
    )


def _get_qra_bar(
    dims: ModelDims, params: LQRParams, tau_bar: Array, tau_bar_f: Array
) -> Tuple[Array, Array, Array]:
    """
    Helper function to get gradients wrt to q, r, a. Variables q_bar, r_bar, a_bar from solving the
    rev LQR problem where q_rev = x_bar, r_rev = u_bar, a_rev = lambda_bar (set to 0 here).

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the model.
    tau_bar : Array
        The tau_bar array.
    tau_bar_f : Array
        The tau_bar_f array.

    Returns
    -------
    Tuple[Array, Array, Array]
        The q_bar, r_bar, and a_bar arrays.
    """
    lqr = params.lqr
    n = dims.n
    x_bar, u_bar = tau_bar[:, :n], tau_bar[:, n:]
    swapped_lqr = LQR(
        A=lqr.A,
        B=lqr.B,
        a=jnp.zeros_like(lqr.a),
        Q=lqr.Q,
        q=x_bar,
        Qf=lqr.Qf,
        qf=tau_bar_f[:n],
        R=lqr.R,
        r=u_bar,
        S=lqr.S,
    )
    swapped_params = LQRParams(params.x0, swapped_lqr)
    q_bar, r_bar, a_bar = solve_lqr_swap_x0(swapped_params)
    return (
        q_bar,
        jnp.r_[
            r_bar,
            jnp.zeros(
                (
                    1,
                    dims.m,
                )
            ),
        ],
        a_bar,
    )


def build_ajoint_lqr(
    dims: ModelDims, params: LQRParams, tau_star: Array, lambs: Array, tau_bar: Array
) -> Array:
    """
    Helper function to build LQR problem with reverse gradients.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the model.
    tau_star : Array
        The optimal state-control trajectory.
    lambs : Array
        The adjoint variables.
    tau_bar : Array
        The gradients with respect to tau.

    Returns
    -------
    LQRParams
        The LQR parameters with reverse gradients.
    """
    q_bar, r_bar, a_bar = _get_qra_bar(dims, params, tau_bar[:-1], tau_bar[-1])
    c_bar = jnp.concatenate([q_bar, r_bar], axis=1)
    F_bar = jnp.einsum("ij,ik->ijk", a_bar[1:], tau_star[:-1]) + jnp.einsum(
        "ij,ik->ijk", lambs[1:], c_bar[:-1]
    )
    C_bar = symmetrise_tensor(
        jnp.einsum("ij,ik->ijk", c_bar, tau_star)
    )  # factor of 2 included in symmetrization
    Q_bar, R_bar = C_bar[:, : dims.n, : dims.n], C_bar[:, dims.n :, dims.n :]
    S_bar = 2*C_bar[:, : dims.n, dims.n :]# + C_bar[:, dims.n :, : dims.n].transpose(0, 2, 1))
    A_bar, B_bar = F_bar[..., : dims.n], F_bar[..., dims.n :]
    LQR_bar = LQR(
        A=A_bar,
        B=B_bar,
        a=a_bar[1:],
        Q=Q_bar[:-1],
        q=q_bar[:-1],
        Qf=Q_bar[-1],
        qf=q_bar[-1],
        R=R_bar[:-1],
        r=r_bar[:-1],
        S=S_bar[:-1],
    )
    return LQRParams(x0=a_bar[0], lqr=LQR_bar)


@partial(custom_vjp, nondiff_argnums=(0,))
def dlqr(dims: ModelDims, params: LQRParams, tau_guess: Array) -> Array:
    """
    Solves the differential linear quadratic regulator (DLQR) problem. Custom VJP function for DLQR.
    Reverse mode uses an LQR solver to solve the reverse LQR problem of the gradients on state and
    input trajectory gradients.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the model.
    tau_guess : Array
        The initial guess for the optimal control sequence.

    Returns
    -------
    Array
        Concatenated optimal state and control sequence along axis=1.
    """
    sol = solve_lqr(params)  #  tau_guess)
    Xs_star, Us_star, _ = sol
    tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    return tau_star


def fwd_dlqr(
    dims: ModelDims, params: LQRParams, tau_guess: Array
) -> Tuple[Array, Tuple[LQRParams, Tuple[Array, Array, Array]]]:
    """
    Solves the forward differential linear quadratic regulator (DLQR) problem.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the DLQR problem.
    tau_guess : Array
        The initial guess for the state-control trajectory.

    Returns
    -------
    Tuple[Array, Tuple[LQRParams, Tuple[Array, Array, Array]]]
        A tuple containing the optimal state-control trajectory and the updated parameters
        and solution.
    """
    lqr = params.lqr
    sol = solve_lqr(params)
    Xs_star, Us_star, _ = sol
    tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    new_lqr = offset_lqr(lqr, Xs_star, Us_star)
    new_params = LQRParams(params.x0, new_lqr)
    return tau_star, (new_params, sol)  # check whether params or new_params


def rev_dlqr(dims: ModelDims, res, tau_bar) -> LQRParams:
    r"""
    Reverse mode for DLQR. Find the reverse gradient by solving LQR problem on the optimal
    trajectory and the gradients wrt to tau_star.
    
    Notes
    -----
    :math:`\overline{q}`, :math:`\overline{r}`, :math:`\overline{a}` from solving the reverse LQR 
    problem where :math:`q_{\text{rev}} = \overline{x}`, :math:`r_{\text{rev}} = \overline{u}`, 
    :math:`a_{\text{rev}} = \overline{\lambda}` which is set to 0.
    
    Use :func:`build_ajoint_lqr` to build the reverse LQR problem with gradients. Where 
    :math:`\overline{c} = [\overline{q}, \overline{r}]`, :math:`\overline{F}`, 
    where :math:`F = [A, B]`, and :math:`\overline{C}` (where :math:`C = [Q, R]`) as:
        :math:`\overline{C} = 0.5 \left( \overline{c} \tau_{\star}^T + \tau_{\star} 
        \overline{c}^T \right)`
        :math:`\overline{F}_t = \lambda_{\star_{t+1}} \overline{c}^T + f_{t+1} 
        \tau_{\star_t}^T`


    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    res : Tuple[LQRParams, Tuple[Array, Array, Array]]
        The result from the forward pass, that is, the tuple of LQR gradients and optimal tau 
        gradients.
    tau_bar : Array
        The gradients with respect to tau.

    Returns
    -------
    LQRParams
        The LQR parameters with reverse gradients.
    """
    params, sol = res
    Xs_star, Us_star, Lambs = sol
    tau_star = jnp.c_[Xs_star, jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    lqr_bar_problem = build_ajoint_lqr(dims, params, tau_star, Lambs, tau_bar)

    return lqr_bar_problem, None


dlqr.defvjp(fwd_dlqr, rev_dlqr)


@partial(custom_vjp, nondiff_argnums=(0,))
def dllqr(dims: ModelDims, params: LQRParams, tau_star: Array) -> Array:
    """
    Solves the differential linear quadratic regulator (DLQR) problem. Custom VJP function for 
    DLQR. Reverse mode uses an LQR solver to solve the reverse LQR problem of the gradients on 
    state and input trajectory gradients.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the model.
    tau_star : Array
        The optimal state-control trajectory.

    Returns
    -------
    Array
        Concatenated optimal state and control sequence along axis=1.
    """
    # sol = solve_lqr(params, dims)  #  tau_guess)
    # Xs_star, Us_star, _ = sol
    # tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    return tau_star  # jnp.nan_to_num(tau_star)*(1 - jnp.isnan(jnp.sum(tau_star)))
    # return tau_star


def fwd_dllqr(
    dims: ModelDims, params: LQRParams, tau_star: Array
) -> Tuple[Array, Tuple[LQRParams, Tuple[Array, Array, Array]]]:
    """
    Solves the forward differential linear quadratic regulator (DLQR) problem.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    params : LQRParams
        The parameters of the DLQR problem.
    tau_star : Array
        The optimal state-control trajectory.

    Returns
    -------
    Tuple[Array, Tuple[LQRParams, Tuple[Array, Array, Array]]]
        A tuple containing the optimal state-control trajectory and the updated parameters
        and solution.
    """
    lqr = params.lqr
    sol = solve_lqr(params)
    Xs_star, Us_star, Lambs = sol
    tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    new_lqr = offset_lqr(lqr, Xs_star, Us_star)
    new_params = LQRParams(params.x0, new_lqr)
    return tau_star, (new_params, sol)  # check whether params or new_params


def rev_dllqr(dims: ModelDims, res, tau_bar) -> LQRParams:
    """
    Reverse mode for DLQR.

    Parameters
    ----------
    dims : ModelDims
        The dimensions of the model.
    res : Tuple[LQRParams, Tuple[Array, Array, Array]]
        The result from the forward pass.
    tau_bar : Array
        The gradients with respect to tau.

    Returns
    -------
    LQRParams
        The LQR parameters with reverse gradients.
    """
    params, sol = res
    Xs_star, Us_star, Lambs = sol
    # isnotnan = 1 - jnp.isnan(jnp.sum(tau_bar))
    tau_star = jnp.c_[Xs_star, jnp.r_[Us_star, jnp.zeros(shape=(1, dims.m))]]
    lqr_bar_problem = build_ajoint_lqr(dims, params, tau_star, Lambs, tau_bar)

    return lqr_bar_problem, None


dllqr.defvjp(fwd_dllqr, rev_dllqr)
