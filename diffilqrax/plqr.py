"""
LQR solver using associative parallel scan

Implementation of the Parallel Linear Quadratic Regulator (PLQR) algorithm
-----------------------------------------------------------------------
1. Initialisation: compute elements :math:`a=\{A, b, C, η, J\}`
   do for all in parallel i.e. :code:`vmap`;
2. Parallel backward scan: initialise with all elements & apply associative operator
   note association operator should be vmap. Scan will return :math:`V_{k}(x_{k})=\{V, v\}`;
3. Compute optimal control: :math:`u_k = -K_kx_k + K^{v}_{k} v_{k+1} - K_k^{c} c_{k}`.
   :math:`K`s have closed form solutions, so calculate :math:`u_k` in parallel :code:`vmap`.
"""

from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax import Array, vmap
from jax.lax import associative_scan

from diffilqrax.typs import (
    symmetrise_matrix,
    symmetrise_tensor,
    LQRParams,
    CostToGo,
    LQR,
)

jax.config.update("jax_enable_x64", True)  # double precision

# helper functions - pop first and last element from pytree structures
_pop_first = partial(jax.tree.map, lambda x: x[1:])
_pull_first = partial(jax.tree.map, lambda x: x[0])
_pop_last = partial(jax.tree.map, lambda x: x[:-1])


# build associative riccati elements
def build_associative_riccati_elements(
    model: LQRParams,
) -> Tuple[Tuple[Array, Array, Array, Array, Array]]:
    """
    Join set of elements for associative scan.
    NOTE: This is a special case where reference r_T=0 and readout C=I.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters.

    Returns
    -------
    Tuple
        Tuple of elements A, b, C, η, J.
    """

    def _last(model: LQRParams):
        """Define last element of Riccati recursion.

        Args:
            model (LQRParams): _description_

        Returns:
            Tuple: Elements of conditional value function (A, b, C, η, J)
        """
        n_dims = model.lqr.Q.shape[1]
        A = jnp.zeros((n_dims, n_dims), dtype=float)
        b = jnp.zeros((n_dims,), dtype=float)
        C = jnp.zeros((n_dims, n_dims), dtype=float)
        η = -model.lqr.qf
        J = model.lqr.Qf
        return A, b, symmetrise_matrix(C), η, symmetrise_matrix(J)

    def _generic(model: LQRParams):
        """Generate generic Riccati element.

        Args:
            model (LQRParams): LQR problem

        Returns:
            Tuple: A, b, C, η, J
        """
        m_dims = model.lqr.R.shape[1]
        A = model.lqr.A
        R_invs = vmap(jsc.linalg.inv)(model.lqr.R + 1e-7 * jnp.eye(m_dims))
        C = jnp.einsum("ijk,ikl,iml->ijm", model.lqr.B, R_invs, model.lqr.B)
        η = -model.lqr.q
        J = model.lqr.Q
        r = model.lqr.r
        B = model.lqr.B
        # redefines the offset term in the dynamics to include the linear term in input
        # b = model.lqr.a - jax.vmap(jnp.matmul)(B, jnp.einsum("ijk,ik->ij", R_invs, r))
        b = model.lqr.a - jnp.einsum("bij,bjk,bk->bi", B, R_invs, r)
        return A, b, symmetrise_tensor(C), η, symmetrise_tensor(J)

    generic_elems = _generic(model)
    last_elem = _last(model)

    return tuple(
        jnp.concatenate([gen_es, jnp.expand_dims(last_e, 0)])
        for gen_es, last_e in zip(generic_elems, last_elem)
    )


def associative_riccati_scan(model: LQRParams) -> Tuple[Array, Array]:
    """
    Obtain value function through associative riccati scan.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters.

    Returns
    -------
    Tuple[Array, Array]
        Linear and quadratic value functions.
    """
    lqr_elements = build_associative_riccati_elements(model)
    _, _, _, etas, Js = associative_scan(riccati_operator, lqr_elements, reverse=True)
    return etas, symmetrise_tensor(Js)


def get_dcosts(model: LQRParams, etas: Array, Js: Array, alpha: float = 1.0) -> CostToGo:
    """
    Calculate expected change in cost-to-go. Can change alpha to relevant backtrack step size.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters.
    etas : Array
        Eta values through time.
    Js : Array
        J values through time.
    alpha : float, optional
        Linesearch alpha parameter, by default 1.0.

    Returns
    -------
    CostToGo
        Total change in cost-to-go.
    """

    @partial(vmap, in_axes=(LQR(0, 0, 0, 0, 0, 0, 0, 0, None, None), 0, 0))
    def get_dcost(lqr, eta, J):
        c = lqr.a
        B = lqr.B
        R = lqr.R
        A = lqr.A
        r = lqr.r
        P = B.T @ J @ B + R
        pinv = jsc.linalg.inv(P + 1e-7 * jnp.eye(P.shape[0]))  # quu_inv
        qu = B.T @ eta + r  # - B.T@Kc@c
        hu = B.T @ (-eta + J @ c)
        Huu = symmetrise_matrix(R + B.T @ J @ B)
        k = -pinv @ qu
        dj = k.T @ hu
        dJ = 0.5 * (k.T @ Huu @ k).squeeze()  # 0.5*qu.T@pinv@qu
        return CostToGo(dJ, -dj)  ##this needs to be a function of alpha

    dJs = get_dcost(model.lqr, etas[1:], Js[1:])
    # dj, dJ = dJs.v, dJs.V
    return CostToGo(
        V=jnp.sum(dJs.V * alpha**2), v=jnp.sum(dJs.v * alpha)
    )  # this needs to be a function of alpha


def build_associative_lin_dyn_elements(
    model: LQRParams, etas: Array, Js: Array, alpha: float
) -> Tuple[
    Tuple[Tuple[Array, Array], Tuple[Array, Array]],
    Tuple[Tuple[Array, Array, Array, Array], Tuple[Array, Array, Array, Array]],
    Array,
]:
    """
    Join set of elements for associative scan.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters.
    etas : Array
        Eta values through time.
    Js : Array
        J values through time.
    alpha : float
        Linesearch step parameter.

    Returns
    -------
    Tuple
        Tuple of elements Fs, Cs.
    """

    def _first(
        model: LQRParams, eta0: Array, J0: Array, alpha: float
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array, Array, Array], Array]:
        """
        Build first associative element for optimal trajectory.
        Offset correspond to the coordinate transformation back into the original space.
        Ks returns to recover the optimal control.

        Parameters
        ----------
        model : LQRParams
            LQR problem
        eta0 : Array
            linear value fn time point k+1
        J0 : Array
            quadratic value fn time point k+1
        alpha : float
            lineasearch step parameter. Multiply Kv and Kc terms as they correspond
            to k term in δu = Kx + αk

        Returns
        -------
        Tuple[Tuple[Array, Array], Tuple[Array, Array, Array, Array], Array]
            (Efective state dynamics initialised with 0,
            effective bias initialised with initial state), Ks, offset
        """

        lqr_init = _pull_first(model.lqr)
        m_dim = lqr_init.R.shape[0]
        mu = 1e-7 * jnp.eye(m_dim)

        # TODO: change to dynamic regularisation
        Rinv = jsc.linalg.inv(lqr_init.R + mu)
        offset = -Rinv @ lqr_init.r
        c = lqr_init.a + lqr_init.B @ offset
        # TODO: change to dynamic regularisation
        Huu = symmetrise_matrix(lqr_init.B.T @ J0 @ lqr_init.B + lqr_init.R)
        min_eval = jnp.min(jnp.linalg.eigh(Huu)[0])
        mu = jnp.maximum(1e-12, 1e-12 - min_eval)
        pinv = jsc.linalg.inv(Huu + mu)
        Kv = pinv @ lqr_init.B.T
        Kc = Kv @ J0
        Kx = Kc @ lqr_init.A
        F0 = lqr_init.A - lqr_init.B @ Kx
        c0 = c + alpha * (lqr_init.B @ (Kv @ eta0 - Kc @ c))
        # TODO : should the offset be multiplied by alpha?
        return (
            (jnp.zeros_like(J0), F0 @ model.x0 + c0),
            (Kx, alpha * Kv, alpha * Kc, alpha * (Kv @ eta0 - Kc @ c)),
            offset,
        )

    first_dyn_elem, Ks0, offset0 = _first(model, etas[1], Js[1], alpha)

    @partial(vmap, in_axes=(LQR(0, 0, 0, 0, 0, 0, 0, 0, None, None), 0, 0, None))
    def _generic(lqr: LQR, eta: Array, J: Array, alpha: float):
        """
        Build generic associative element for optimal trajectory.

        Parameters
        ----------
        lqr : LQR
            LQR parameters
        eta : Array
            linear value fn time point k+1
        J : Array
            quadratic value fn time point k+1
        alpha : float
            lineasearch step parameter.

        Returns
        -------
        Tuple[Tuple[Array, Array], Tuple[Array, Array, Array, Array], Array]
            (Efective state dynamics,
            effective bias), Ks, offset
        """
        m_dim = lqr.R.shape[0]
        mu = 1e-7 * jnp.eye(m_dim)
        # ̃n=cn+LnUn Mnrn+Lnsn
        # 0.5(u - s)U(u-s) = 0.ruUu - sUu + 0.5sUs -> r = -sU
        # c_tilde = c - r@U^{-1}
        Rinv = jsc.linalg.inv(lqr.R + mu)

        P = symmetrise_matrix(lqr.B.T @ J @ lqr.B + lqr.R)
        min_eval = jnp.min(jnp.linalg.eigh(P)[0])
        mu = jnp.maximum(1e-12, 1e-12 - min_eval)
        pinv = jsc.linalg.inv(P + mu)
        Kv = pinv @ lqr.B.T
        # Kv_eta for including eta
        Kc = Kv @ J
        Kx = Kc @ lqr.A
        Ft = lqr.A - lqr.B @ Kx
        offset = -Rinv @ lqr.r
        c = lqr.a + lqr.B @ offset
        ct = c + alpha * (lqr.B @ Kv @ eta - lqr.B @ Kc @ c)
        return (
            (Ft, ct),
            (Kx, alpha * Kv, alpha * Kc, alpha * (Kv @ eta - Kc @ c)),
            offset,
        )

    generic_dyn_elems, Ks, offsets = _generic(
        _pop_first(model.lqr), etas[2:], Js[2:], alpha
    )

    Ks = tuple(jnp.r_[jnp.expand_dims(first_k, 0), kk] for first_k, kk in zip(Ks0, Ks))

    associative_dyn_elems = tuple(
        jnp.r_[jnp.expand_dims(first_e, 0), gen_es]
        for first_e, gen_es in zip(first_dyn_elem, generic_dyn_elems)
    )

    offsets = jnp.r_[jnp.expand_dims(offset0, 0), offsets]

    return associative_dyn_elems, Ks, offsets


# parallellised riccati scan
def associative_opt_traj_scan(
    model: LQRParams, etas: Array, Js: Array, alpha: float = 1.0
) -> Tuple[Array, Array, Tuple[Array, Array, Array, Array], Array]:
    """
    Obtain effective state dynamic and bias through associative scan.
    Ks returned (Kx, Kv, Kc, delta) to recover optimal control: state feedback gain,
    linear val fn gain, dynamics offset gain, delta of linear gains.

    Parameters
    ----------
    model : LQRParams
        LQR problem (initial state, LQR parameters).
    etas : Array
        Linear value functions.
    Js : Array
        Quadratic value functions.
    alpha : float, optional
        Linesearch step parameter, by default 1.0.

    Returns
    -------
    Tuple[Array, Array, Tuple[Array, Array, Array, Array], Array]
        Effective state dynamics, effective bias, Ks, offset.
    """
    # need to add vmaps
    associative_dyn_elems, Ks, offsets = build_associative_lin_dyn_elements(
        model, etas, Js, alpha
    )
    final_Fs, final_cs = associative_scan(dynamic_operator, associative_dyn_elems)

    return final_Fs, final_cs, Ks, offsets


def get_delta_u(
    Ks: Tuple[Array, Array, Array, Array], x: Array, v: Array, c: Array
) -> Array:
    """
    Obtain δu from optimal control gains.

    Parameters
    ----------
    Ks : Tuple[Array, Array, Array, Array]
        Optimal control gains.
    x : Array
        State trajectory.
    v : Array
        Linear value function.
    c : Array
        Effective bias.

    Returns
    -------
    Array
        Control input δu.
    """
    _, _, _, ddelta = Ks
    delta_us = ddelta  # -Kx@x +  #Kv@v - Kc@c #+ ddelta #- c#Kv@v - Kc@c #
    return delta_us


@jax.jit
def solve_plqr(model: LQRParams) -> Tuple[Array, Array, Array]:
    """
    Run backward forward sweep to find optimal control.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters.

    Returns
    -------
    Tuple[Array, Array, Array]
        State trajectory, control inputs, and adjoint variables.
    """
    # backward
    vs, Vs = associative_riccati_scan(model)
    # NOTE: cs is already finding updated Xs -> jnp.r_[model.x0[None],cs] == new_xs
    Fs, cs, (Ks, _, _, ks), offsets = associative_opt_traj_scan(model, vs, Vs)
    Xs = jnp.r_[model.x0[None], cs]
    Us = ks - jnp.einsum("bij,bj->bi", Ks, Xs[:-1]) + offsets
    Lambdas = jnp.einsum("bij,bj->bi", Vs, Xs) - vs
    return (Xs, Us, Lambdas)


def solve_plqr_swap_x0(model: LQRParams):
    """
    Run backward forward sweep to find optimal control freeze x0 to zero.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters.

    Returns
    -------
    Tuple[Array, Array, Array]
        State trajectory, control inputs, and adjoint variables.
    """
    model = model._replace(x0=jnp.zeros_like(model.x0))
    vs, Vs = associative_riccati_scan(model)
    Fs, cs, (Ks, _, _, ks), offsets = associative_opt_traj_scan(model, vs, Vs)
    Xs = jnp.r_[model.x0[None], cs]
    Us = ks - jnp.einsum("bij,bj->bi", Ks, Xs[:-1]) + offsets
    Lambdas = jnp.einsum("bij,bj->bi", Vs, Xs) - vs
    return (Xs, Us, Lambdas)


# --------
# Parallel forward integration
# --------
def build_fwd_lin_dyn_elements(
    lqr_params: LQRParams, Us_init: Array, a_term: Array = None
) -> Tuple[Array, Array]:
    """
    Generate sequence of elements {c} for forward integration.

    Parameters
    ----------
    lqr_params : LQRParams
        LQR parameters and initial state.
    Us_init : Array
        Input sequence.
    a_term : Array, optional
        Offset term, by default None.

    Returns
    -------
    Tuple[Array, Array]
        Set of elements {c} for associative scan.
    """

    initial_element = (
        jnp.zeros_like(jnp.diag(lqr_params.x0)),
        lqr_params.x0,
        jnp.zeros_like(a_term[0]),
    )
    # print(initial_element[0].shape, initial_element[1].shape)

    @partial(vmap, in_axes=(0, 0, 0, 0))
    def _generic(a_mat: Array, b_mat: Array, u: Array, a: Array) -> Tuple[Array, Array]:
        """Generate tuple (c_i,a, c_i,b) to parallelise"""
        return a_mat, b_mat @ u + a

    generic_elements = _generic(lqr_params.lqr.A, lqr_params.lqr.B, Us_init, a_term)

    # print(generic_elements[0].shape, generic_elements[1].shape)
    return tuple(
        jnp.concatenate([jnp.expand_dims(first_e, 0), gen_es])
        for first_e, gen_es in zip(initial_element, generic_elements)
    )


def parallel_forward_lin_integration(
    lqr_params: LQRParams, Us_init: Array, a_term: Array
) -> Array:
    """
    Associative scan for forward linear dynamics.

    Parameters
    ----------
    lqr_params : LQRParams
        LQR parameters and initial state.
    Us_init : Array
        Input sequence.
    a_term : Array
        Offset term.

    Returns
    -------
    Array
        State trajectory.
    """
    # delta_us = compute_offset_us(lqr_params)
    dyn_elements = build_fwd_lin_dyn_elements(lqr_params, Us_init, a_term)
    c_as, c_bs = associative_scan(dynamic_operator, dyn_elements)
    return c_bs


# parallel adjoint integration
def build_rev_lin_dyn_elements(
    lqr_params: LQRParams,
    xs_traj: Array,
    us_traj: Array,
) -> Tuple[Array, Array]:
    """
    Generate sequence of elements {c} for reverse integration of adjoints.

    Parameters
    ----------
    lqr_params : LQRParams
        LQR parameters and initial state.
    xs_traj : Array
        State trajectory.
    us_traj : Array
        Control inputs.

    Returns
    -------
    Tuple[Array, Array]
        Set of elements {c} for associative scan.
    """

    lambda_f = lqr_params.lqr.Qf @ xs_traj[-1] + lqr_params.lqr.qf

    last_element = (jnp.diag(lambda_f), lambda_f)
    print(last_element[0].shape, last_element[1].shape)

    @vmap
    def _generic(
        a_trsp_mat: Array, s_mat: Array, q_mat: Array, q_vec: Array, x: Array, u: Array
    ) -> Tuple[Array, Array]:
        """Generate tuple (c_i,a, c_i,b) to parallelise"""
        b_coef = s_mat @ u + q_vec + q_mat @ x
        return a_trsp_mat, b_coef

    generic_elements = _generic(
        lqr_params.lqr.A.transpose(0, 2, 1),
        lqr_params.lqr.S,
        lqr_params.lqr.Q,
        lqr_params.lqr.q,
        xs_traj[:-1],
        us_traj,
    )

    # print(generic_elements[0].shape, generic_elements[1].shape)
    return tuple(
        jnp.concatenate([gen_, jnp.expand_dims(last_, 0)])
        for gen_, last_ in zip(generic_elements, last_element)
    )


def parallel_reverse_lin_integration(
    lqr_params: LQRParams, xs_traj: Array, us_traj: Array
) -> Array:
    """
    Associative scan for reverse linear dynamics.

    Parameters
    ----------
    lqr_params : LQRParams
        LQR parameters and initial state.
    xs_traj : Array
        State trajectory.
    us_traj : Array
        Control inputs.

    Returns
    -------
    Array
        Adjoint variables.
    """
    # delta_us = compute_offset_us(lqr_params)
    dyn_elements = build_rev_lin_dyn_elements(lqr_params, xs_traj, us_traj)
    c_as, c_bs = associative_scan(dynamic_operator, dyn_elements, reverse=True)
    return c_bs


# forward dynamics
@vmap
def dynamic_operator(elem1, elem2):
    """
    Associative operator for forward linear dynamics.

    Parameters
    ----------
    elem1 : Tuple[Array, Array]
        Previous effective state dynamic and effective bias.
    elem2 : Tuple[Array, Array]
        Next effective state dynamic and effective bias.

    Returns
    -------
    Tuple[Array, Array]
        Updated state and control.
    """
    F1, c1 = elem1
    F2, c2 = elem2
    F = F2 @ F1
    c = F2 @ c1 + c2
    return F, c


# riccati recursion
@vmap
def riccati_operator(elem2, elem1):
    """
    Associative operator for Riccati recursion.

    Parameters
    ----------
    elem1 : Tuple[Array, Array, Array, Array, Array]
        Previous Riccati elements.
    elem2 : Tuple[Array, Array, Array, Array, Array]
        Next Riccati elements.

    Returns
    -------
    Tuple[Array, Array, Array, Array, Array]
        Updated Riccati elements.
    """
    A1, b1, C1, η1, J1 = elem1
    A2, b2, C2, η2, J2 = elem2

    dim = A1.shape[0]
    I = jnp.eye(dim)  # note the jnp

    I_C1J2 = I + C1 @ J2
    temp = jsc.linalg.solve(I_C1J2.T, A2.T).T
    A = temp @ A1
    b = temp @ (b1 + C1 @ η2) + b2
    C = temp @ C1 @ A2.T + C2

    I_J2C1 = I + J2 @ C1
    temp = jsc.linalg.solve(I_J2C1.T, A1).T
    η = temp @ (η2 - J2 @ b1) + η1
    J = temp @ J2 @ A1 + J1
    return A, b, C, η, J
