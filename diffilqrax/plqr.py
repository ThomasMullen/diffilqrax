"""LQR solver using associative parallel scan"""

from typing import Callable, Tuple
from functools import partial
from jax.typing import ArrayLike
from jax import Array
import jax
from jax import vmap
from jax import lax
from jax.lax import scan, associative_scan
import jax.numpy as jnp
from jax.scipy.linalg import solve
import jax.scipy as jsc

from diffilqrax.typs import (
    System,
    symmetrise_matrix,
    symmetrise_tensor,
    ModelDims,
    LQRParams,
    Gains,
    CostToGo,
    LQR,
    RiccatiStepParams,
    CostToGo,
)

jax.config.update("jax_enable_x64", True)  # double precision
from jax.lib import xla_bridge

# helper functions - pop first and last element from namedtuple
pop_first = partial(jax.tree_map, lambda x: x[1:])
pop_last = partial(jax.tree_map, lambda x: x[:-1])

"""
Implementation:
---
1. Initialisation: compute elements a={A, b, C, η, J}
   do for all in parallel i.e. vmap
2. Parallel backward scan: initialise with all elements & apply associative operator
   note association operator should be vmap. Scan will return V_k(x_k)={V, v}
3. Compute optimal control: u_k = -K_kx_k + K^{v}_{k} v_{k+1} - K_k^{c} c_{k}
   Ks have closed form sols - so calc u_k in parallel vmap

"""


# build associative riccati elements
def build_associative_riccati_elements(
    model: LQRParams,
) -> Tuple[Tuple[Array, Array, Array, Array, Array]]:
    """Join set of elements for associative scan.
    NOTE: This is a special case where reference r_T=0 and readout C=I.

    Args:
        model (LQRParams)

    Returns:
        Tuple: return tuple of elements A, b, C, η, J
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
        return A, b, C, η, J

    def _generic(model: LQRParams):
        """Generate generic Riccati element.

        Args:
            model (LQRParams): _description_

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
        b = model.lqr.a - jax.vmap(jnp.matmul)(B, jnp.einsum("ijk,ik->ij", R_invs, r))
        return A, b, C, η, J

    generic_elems = _generic(model)
    last_elem = _last(model)

    return tuple(
        jnp.concatenate([gen_es, jnp.expand_dims(last_e, 0)])
        for gen_es, last_e in zip(generic_elems, last_elem)
    )


# parallellised riccati scan
def parallel_riccati_scan(model: LQRParams):
    first_elements = build_associative_riccati_elements(model)

    final_elements = associative_scan(riccati_operator, first_elements, reverse=True)
    etas, Js = final_elements[
        -2:
    ]  # dJ and dj we are supposed to add Quu@k and k@QuuQk.T
    return etas, Js


def get_dJs(model: LQRParams, etas: Array, Js: Array, alpha: float = 1.0) -> CostToGo:
    """Calculate expected change in cost-to-go. Can change alpha to relevant backtrack
    step size.

    Args:
        model (LQRParams): LQR model parameters
        etas (Array): eta values through time
        Js (Array): J values through time
        alpha (float, optional): linesearch alpha parameter. Defaults to 1..

    Returns:
        CostToGo: Return total change in cost-to-go
    """

    @partial(vmap, in_axes=(LQR(0, 0, 0, 0, 0, 0, 0, 0, None, None), 0, 0))
    def get_dJ(lqr, eta, J):
        c = lqr.a
        B = lqr.B
        R = lqr.R
        A = lqr.A
        r = lqr.r
        P = B.T @ J @ B + R
        pinv = jsc.linalg.inv(P + 1e-7 * jnp.eye(P.shape[0]))  # quu_inv
        qu = B.T @ eta  + r #- B.T@Kc@c
        hu = B.T @ (-eta + J @ c)
        Huu = symmetrise_matrix(R + B.T @ J @ B)
        k = -pinv @ qu
        dj = k.T @ hu
        dJ = 0.5 * (k.T @ Huu @ k).squeeze()  # 0.5*qu.T@pinv@qu
        return CostToGo(dJ, -dj)  ##this needs to be a function of alpha

    dJs = get_dJ(model.lqr, etas[1:], Js[1:])
    # dj, dJ = dJs.v, dJs.V
    return CostToGo(
        V=jnp.sum(dJs.V * alpha**2), v=jnp.sum(dJs.v * alpha)
    )  # this needs to be a function of alpha


# jnp.flip(final_elements[-2], axis = 0), jnp.flip(final_elements[-1], axis = 0) #jnp.r_[final_elements[-2][1:], model.lqr.qf[None]], jnp.r_[final_elements[-1][1:], -model.lqr.Qf[None]] #final_elements[-2], final_elements[-1]
# jnp.r_[final_elements[-2][0:], model.lqr.qf[None]], jnp.r_[final_elements[-1][0:], -model.lqr.Qf[None]] #this only returns J, eta, which are the only things we need to compute
# Vk : Sk = Jk_{T+1}, vk = eta_k_{T+1}
##or is it? how do we have all k and k+1 accessible?


def build_associative_lin_dyn_elements(
    model: LQRParams, etas, Js, alpha
) -> Tuple[Tuple[Array, Array, Array, Array, Array]]:
    """Join set of elements for associative scan.
    Args:
        model (LQRParams)

    Returns:
        Tuple: return tuple of elements Fs, Cs
    """

    def _first(model, eta0, J0, alpha):
        S0, v0 = J0, eta0  # this needs to be at k+1 so T = 1
        B = model.lqr.B[0]
        R = model.lqr.R[0]
        A = model.lqr.A[0]
        r = model.lqr.r[0]
        Rinv = jsc.linalg.inv(R + 1e-7 * jnp.eye(R.shape[0]))  
        offset = - Rinv @ r 
        c = model.lqr.a[0] + B@offset
        pinv = jsc.linalg.inv(B.T @ S0 @ B + R + 1e-7 * jnp.eye(R.shape[0]))  # quu_inv
        Kv = pinv @ B.T
        Kc = Kv @ S0
        Kx = Kc @ A
        F0 = A - B @ Kx
        c0 = c   + alpha * (B @ Kv @ v0 - B @ Kc @ c) #- B@offset
        return (jnp.zeros_like(J0), F0 @ model.x0 + c0), (B@Kx, alpha * Kv, alpha * Kc, (Kv @ v0 - Kc @ c)), offset

    first_elem, Ks0, offset0 = _first(model, etas[1], Js[1], alpha)  # this is at k+1
    @partial(vmap, in_axes=(LQR(0, 0, 0, 0, 0, 0, 0, 0, None, None), 0, 0, None))
    def _generic(lqr: LQR, eta: Array, J: Array, alpha: float):
        S, v = J, eta
        c = lqr.a  
        B = lqr.B
        R = lqr.R
        A = lqr.A
        r = lqr.r
        # ̃n=cn+LnUn Mnrn+Lnsn
        #0.5(u - s)U(u-s) = 0.ruUu - sUu + 0.5sUs -> r = -sU
        #c_tilde = c - r@U^{-1}
        P = B.T @ S @ B + R
        Rinv = jsc.linalg.inv(R + 1e-7 * jnp.eye(R.shape[0]))  # quu_inv
        pinv = jsc.linalg.inv(P + 1e-7 * jnp.eye(P.shape[0]))  # quu_inv
        Kv = pinv @ B.T
        # Kv_eta    for including eta
        Kc = Kv @ S
        Kx = Kc @ A
        Ft = A - B @ Kx
        offset = -Rinv @ r  
        c += B@offset #+ B@Rinv@r
        ct = c  + alpha * (B @ Kv @ v - B @ Kc @ c)#+ B@offset#-  B @ Rinv @ r
        return (Ft, ct), (B@Kx, alpha * Kv, alpha * Kc, (Kv @ v - Kc @ c)), offset
    generic_elems, Ks, offsets = _generic(pop_first(model.lqr), etas[2:], Js[2:], alpha)
    Ks = tuple(jnp.r_[jnp.expand_dims(first_k, 0), kk] for first_k, kk in zip(Ks0, Ks))
    associative_elems = tuple(
        jnp.r_[jnp.expand_dims(first_e, 0), gen_es]
        for first_e, gen_es in zip(first_elem, generic_elems)
    )
    offsets = jnp.r_[jnp.expand_dims(offset0, 0), offsets]
    return associative_elems, Ks, offsets


# parallellised riccati scan
def parallel_lin_dyn_scan(model: LQRParams, etas, Js, alpha=1.0):
    # need to add vmaps
    final_elements, Ks, offsets = build_associative_lin_dyn_elements(model, etas, Js, alpha)
    final_Fs, final_cs = associative_scan(dynamic_operator, final_elements)

    return final_Fs, final_cs, Ks, offsets

def get_delta_u(Ks, x, v, c):
    Kx, Kv, Kc, ddelta = Ks
    delta_U = ddelta #-Kx@x +  #Kv@v - Kc@c #+ ddelta #- c#Kv@v - Kc@c #
    return delta_U

@jax.jit
def solve_plqr(model: LQRParams)->Tuple[Array,Array,Array]:
    "run backward forward sweep to find optimal control"
    # backward
    etas, Js = parallel_riccati_scan(model)
    # NOTE: cs is already finding updated Xs -> jnp.r_[model.x0[None],cs] == new_xs
    Fs, cs, Ks, offsets = parallel_lin_dyn_scan(model, etas, Js)
    new_Xs = jnp.r_[model.x0, cs]
    
    Kx = Ks[0]
    # NOTE: new_xs already found new_model redundant - could be useful for testing though
    # new_model = LQRParams(
    #     model.x0, LQR(
    #                 model.lqr.A - Kx, model.lqr.B, 0*model.lqr.a, 
    #                 model.lqr.Q, model.lqr.q, 
    #                 model.lqr.R, model.lqr.r,
    #                 model.lqr.S, 
    #                 model.lqr.Qf, model.lqr.qf
    #                 ))
    # new_Us = Ks[-1] + offsets + model.lqr.a ##not entirely sure if this is the right way to handle a -- it seems to work and think it makes sense to offset what we pass in the parallel_lin_scan, but need to double check
    # new_Xs = parallel_forward_lin_integration(new_model, new_Us)
    
    # NOTE: Not sure about the `a` - this generally would project to the correct space
    # NOTE: Why does this solution not work: vmatmul = vmap(jnp.matmul); us_ = - vmatmul(Kx, xs_[:-1]) + vmatmul(Kv, etas[1:]) + vmatmul(Kc, params.lqr.a)
    new_Us = Ks[-1] + offsets + model.lqr.a ##not entirely sure if this is the right way to handle a -- it seems to work and think it makes sense to offset what we pass in the parallel_lin_scan, but need to double check
    
    updated_Us = new_Us - jax.vmap(lambda a, b, c, d : jnp.linalg.pinv(c)@(a@b + d), in_axes = (0,0,0,0))(Kx, new_Xs[:-1], model.lqr.B, model.lqr.a)
    # NOTE: alternative to additional scan and just vmap
    # new_Lambdas = parallel_reverse_lin_integration(model, new_Xs, updated_Us)
    vmatmul = vmap(jnp.matmul)
    new_Lambdas = vmatmul(Js, new_Xs) - etas #TODO: move operation in the asssociative forward scan with eta and Js

    return (new_Xs, 
            updated_Us,
            new_Lambdas)


# --------
# Parallel forward integration
# --------
def build_fwd_lin_dyn_elements(
    lqr_params: LQRParams, Us_init: Array
) -> Tuple[Array, Array]:
    """Generate sequence of elements {c} for forward integration

    Args:
        lqr_params (LQRParams): LQR parameters and initial state
        Us_init (Array): Input sequence

    Returns:
        Tuple[Array, Array]: set of elements {c} for associative scan
    """

    initial_element = (jnp.zeros_like(jnp.diag(lqr_params.x0)), lqr_params.x0)
    # print(initial_element[0].shape, initial_element[1].shape)

    @partial(vmap, in_axes=(0, 0, 0))
    def _generic(a_mat: Array, b_mat: Array, u: Array) -> Tuple[Array, Array]:
        """Generate tuple (c_i,a, c_i,b) to parallelise"""
        return a_mat, b_mat @ u

    generic_elements = _generic(lqr_params.lqr.A, lqr_params.lqr.B, Us_init)

    # print(generic_elements[0].shape, generic_elements[1].shape)
    return tuple(
        jnp.concatenate([jnp.expand_dims(first_e, 0), gen_es])
        for first_e, gen_es in zip(initial_element, generic_elements)
    )


def parallel_forward_lin_integration(lqr_params: LQRParams, Us_init: Array) -> Array:
    """Associative scan for forward linear dynamics

    Args:
        lqr_params (LQRParams): LQR parameters and initial state
        Us_init (Array): input sequence

    Returns:
        Array: state trajectory
    """
    #delta_us = compute_offset_us(lqr_params)
    dyn_elements = build_fwd_lin_dyn_elements(lqr_params, Us_init)
    c_as, c_bs = associative_scan(dynamic_operator, dyn_elements)
    return c_bs


# parallel adjoint integration
def build_rev_lin_dyn_elements(
    lqr_params: LQRParams, xs_traj: Array, us_traj: Array,
) -> Tuple[Array, Array]:
    """Generate sequence of elements {c} for reverse integration of adjoints

    Args:
        lqr_params (LQRParams): LQR parameters and initial state
        Us_init (Array): Input sequence

    Returns:
        Tuple[Array, Array]: set of elements {c} for associative scan
    """

    lambda_f = lqr_params.lqr.Qf@xs_traj[-1] + lqr_params.lqr.qf
    
    last_element = (jnp.diag(lambda_f), lambda_f)
    print(last_element[0].shape, last_element[1].shape)

    @vmap
    def _generic(aT_mat: Array, 
                 s_mat: Array,
                 q_mat: Array, 
                 q_vec: Array, 
                 x: Array,
                 u: Array
                 ) -> Tuple[Array, Array]:
        """Generate tuple (c_i,a, c_i,b) to parallelise"""
        b_coef = s_mat@u + q_vec + q_mat@x
        return aT_mat, b_coef

    generic_elements = _generic(
        lqr_params.lqr.A.transpose(0,2,1),
        lqr_params.lqr.S,
        lqr_params.lqr.Q,
        lqr_params.lqr.q,
        xs_traj[:-1],
        us_traj)

    # print(generic_elements[0].shape, generic_elements[1].shape)
    return tuple(
        jnp.concatenate([gen_, jnp.expand_dims(last_, 0)])
        for gen_, last_ in zip(generic_elements, last_element)
    )


def parallel_reverse_lin_integration(lqr_params: LQRParams, xs_traj: Array, us_traj: Array) -> Array:
    """Associative scan for reverse linear dynamics

    Args:
        lqr_params (LQRParams): LQR parameters and initial state
        Us_init (Array): input sequence

    Returns:
        Array: state trajectory
    """
    #delta_us = compute_offset_us(lqr_params)
    dyn_elements = build_rev_lin_dyn_elements(lqr_params, xs_traj, us_traj)
    c_as, c_bs = associative_scan(dynamic_operator, dyn_elements, reverse=True)
    return c_bs


# ----------------------------
# Define associative operators
# ----------------------------


# forward dynamics
@vmap
def dynamic_operator(elem1, elem2):
    """Associative operator for forward linear dynamics

    Args:
        elem1 (Tuple[Array, Array]): Previous effective state dynamic and effective bias
        elem2 (Tuple[Array, Array]): Next effective state dynamic and effective bias

    Returns:
        Tuple[Array, Array]: Updated state and control
    """
    F1, c1 = elem1
    F2, c2 = elem2
    F = F2 @ F1
    c = F2 @ c1 + c2
    return F, c


# riccati recursion
@vmap
def riccati_operator(elem2, elem1):
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

