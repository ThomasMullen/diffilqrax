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
    symmetrise_matrix,
    symmetrise_tensor,
    ModelDims,
    LQRParams,
    Gains,
    CostToGo,
    LQR,
    RiccatiStepParams,
    CostToGo
)

#jax.config.update("jax_enable_x64", True)  # double precision
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

# last riccati element
def last_riccati_element(model: LQRParams):
    """Define last element of Riccati recursion.
    NOTE: This is a special case where reference r_T=0 and readout C=I.

    Args:
        model (LQRParams): _description_

    Returns:
        Tuple: Elements of conditional value function (A, b, C, η, J)
    """
    n_dims = model.lqr.Q.shape[1]
    A = jnp.zeros((n_dims,n_dims), dtype=float)
    b = jnp.zeros((n_dims,), dtype=float)
    C = jnp.zeros((n_dims,n_dims), dtype=float)
    # here: set readout C=I, reference r_T=0
    η = -model.lqr.qf
    # here: set readout C=I
    J = model.lqr.Qf
    # J = model.lqr.Q[-1] #jnp.eye(n_dims).T @ model.lqr.Q[-1] @ jnp.eye(n_dims, dtype=float)
    return A, b, C, η, J


# generic riccati element
def generic_riccati_element(model: LQRParams):
    """Generate generic Riccati element.
    NOTE: This is a special case where reference r_T=0 and readout C=I.

    Args:
        model (LQRParams): _description_

    Returns:
        Tuple: A, b, C, η, J
    """
    n_dims = model.lqr.Q.shape[1]
    A = model.lqr.A
    b = model.lqr.a
    R_invs = vmap(jsc.linalg.inv)(model.lqr.R + 1e-7*jnp.eye(n_dims))
    C = jnp.einsum('ijk,ikl,iml->ijm', model.lqr.B, R_invs, model.lqr.B)
    # lqr of form : 0.5 x^T Q x + q^t x + u^T R u + r^T u
    # if we expand 0.5 * (Hx - r)^TX(Hx - r) = 0.5 x^T H^T X H x - r^T X H x - 0.5 r^T X r
    #  H^T X H = Q, r^TXH = q => if H = np.eye(n) then Q = X and r = -q@Q^{-1} so eta = -q and J = Q I think 
    η = -model.lqr.q #jnp.einsum('ji,kjl,l->ki', jnp.eye(n_dims, dtype=float), model.lqr.Q, jnp.zeros((n_dims), dtype=float))
    # here: set readout C=I
    J = model.lqr.Q #jnp.einsum('ij,kjl,lm->kim', jnp.eye(n_dims, dtype=float), model.lqr.Q, jnp.eye(n_dims, dtype=float))
    return A, b, C, η, J


# build associative riccati elements
def build_associative_riccati_elements(
    model: LQRParams
)-> Tuple[Tuple[Array, Array, Array, Array, Array]]:
    """Join set of elements for associative scan.
    Args:
        model (LQRParams)

    Returns:
        Tuple: return tuple of elements A, b, C, η, J
    """
    last_elem = last_riccati_element(model)
    generic_elems = generic_riccati_element(model)
    return tuple(
        jnp.concatenate([gen_es, jnp.expand_dims(last_e, 0)])
        for gen_es, last_e in zip(generic_elems, last_elem)
    )


# parallellised riccati scan
def parallel_riccati_scan(model: LQRParams):
    first_elements = build_associative_riccati_elements(model)

    # riccati operator
    @vmap
    def assoc_riccati_operator(elem2, elem1):
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
    final_elements = associative_scan(
        assoc_riccati_operator, first_elements, reverse = True
    )
    etas = final_elements[-2]
    Js = final_elements[-1]  #dJ and dj we are supposed to add Quu@k and k@QuuQk.T
    return etas, Js



def get_dJs(model:LQRParams, etas:Array, Js:Array, alpha:float = 1.)->CostToGo:
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
    @vmap(in_axes=(LQR(0,0,0,0,0,0,0,0,None,None), 0, 0))
    def get_dJ(lqr, eta, J):
        c =  lqr.a
        B = lqr.B
        R = lqr.R
        A = lqr.A
        r = lqr.r
        P = B.T@J@B + R
        pinv = jsc.linalg.inv(P + 1e-7*jnp.eye(P.shape[0])) #quu_inv
        qu = B.T@eta #+ r #- B.T@Kc@c
        hu = B.T @ (-eta + J @ c)
        Huu = symmetrise_matrix(R + B.T @ J @ B)
        k = -pinv@qu
        dj = k.T@hu
        dJ = 0.5 * (k.T @ Huu @ k).squeeze()   #0.5*qu.T@pinv@qu
        return CostToGo(dJ, -dj) ##this needs to be a function of alpha
    
    dJs = get_dJ(model.lqr, etas[1:], Js[1:])
    # dj, dJ = dJs.v, dJs.V
    return CostToGo(V = jnp.sum(dJs.V*alpha**2), v = jnp.sum(dJs.v*alpha)) #this needs to be a function of alpha
#jnp.flip(final_elements[-2], axis = 0), jnp.flip(final_elements[-1], axis = 0) #jnp.r_[final_elements[-2][1:], model.lqr.qf[None]], jnp.r_[final_elements[-1][1:], -model.lqr.Qf[None]] #final_elements[-2], final_elements[-1]
#jnp.r_[final_elements[-2][0:], model.lqr.qf[None]], jnp.r_[final_elements[-1][0:], -model.lqr.Qf[None]] #this only returns J, eta, which are the only things we need to compute 
#Vk : Sk = Jk_{T+1}, vk = eta_k_{T+1}
##or is it? how do we have all k and k+1 accessible? 




def generic_lin_dyn_elements(lqr, eta, J, alpha):
    S, v = J, eta 
    c =  lqr.a
    B = lqr.B
    R = lqr.R
    A = lqr.A
    r = lqr.r
    P = B.T@S@B + R
    pinv = jsc.linalg.inv(P + 1e-7*jnp.eye(P.shape[0])) #quu_inv
    Kv = pinv@B.T
    #Kv_eta for including eta
    Kc = Kv@S
    Kx = Kc@A
    Ft = A - B@Kx
    ct = c + (B@Kv@v - B@Kc@c)
    return (Ft, ct), (Kx,  alpha*Kv,  alpha*Kc)




def first_lin_dyn_element(model, eta0, J0, alpha): 
    S0, v0 = J0, eta0 #this needs to be at k+1 so T = 1
    c =  model.lqr.a[0]
    B = model.lqr.B[0]
    R = model.lqr.R[0]
    A = model.lqr.A[0]
    r = model.lqr.r[0]
    pinv = jsc.linalg.inv(B.T@S0@B + R + 1e-7*jnp.eye(R.shape[0])) #quu_inv
    Kv = pinv@B.T
    Kc = Kv@S0
    Kx = Kc@A
    F0 = A - B@Kx
    c0 = c + (B@Kv@v0 - B@Kc@c) #
    return (jnp.zeros_like(J0), F0@model.x0 + c0), (Kx,  alpha*Kv,  alpha*Kc)


def build_associative_lin_dyn_elements(
    model: LQRParams, etas, Js, alpha
)-> Tuple[Tuple[Array, Array, Array, Array, Array]]:
    """Join set of elements for associative scan.
    Args:
        model (LQRParams)

    Returns:
        Tuple: return tuple of elements Fs, Cs
    """
    first_elem, Ks0 = first_lin_dyn_element(model, etas[1], Js[1], alpha) #this is at k+1
    # etas = jnp.concatenate([etas, jnp.zeros_like(etas[0])[None]])
    # Js = jnp.concatenate([Js, jnp.zeros_like(Js[0])[None]])
    generic_elems, Ks = jax.vmap(generic_lin_dyn_elements, in_axes = (LQR(0,0,0,0,0,0,0,0,None,None), 0, 0, None))(pop_first(model.lqr), etas[2:], Js[2:], alpha)
    Ks = tuple(jnp.r_[jnp.expand_dims(first_k, 0), kk] for first_k, kk in zip(Ks0, Ks))
    generic_elems = generic_elems[:2]
    return tuple(jnp.r_[jnp.expand_dims(first_e, 0), gen_es] 
                 for first_e, gen_es in zip(first_elem, generic_elems)), Ks


# parallellised riccati scan
def parallel_lin_dyn_scan(model: LQRParams, etas, Js, alpha = 1.0):
    #need to add vmaps
    final_elements, Ks = build_associative_lin_dyn_elements(model, etas, Js, alpha)

    # riccati operator
    @vmap
    def assoc_dynamics_operator(elem1, elem2):
        F1, c1 = elem1
        F2, c2 = elem2
        F = F2@F1
        c = F2@c1 + c2
        return F, c
    
    final_Fs, final_cs = associative_scan(
        assoc_dynamics_operator, final_elements
    )

    return final_Fs, final_cs, Ks


#@jax.jit
def solve_plqr(model: LQRParams):
    "run backward forward sweep to find optimal control"
    # backward
    etas, Js = parallel_riccati_scan(model)
    Fs, cs, _ = parallel_lin_dyn_scan(model, etas, Js)
    return jnp.concatenate([model.x0[None], cs])#Fs@model.x0 + cs])
    # _, gains = lqr_backward_pass(params.lqr, sys_dims)
    # # forward
    # Xs, Us = lqr_forward_pass(gains, params)
    # # adjoint
    # Lambs = lqr_adjoint_pass(Xs, Us, params)
    # return gains, Xs, Us, Lambs



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

    initial_element = (jnp.diag(lqr_params.x0), lqr_params.x0)
    # print(initial_element[0].shape, initial_element[1].shape)

    def gen_ele(a_mat: Array, b_mat: Array, u: Array) -> Tuple[Array, Array]:
        """Generate tuple (c_i,a, c_i,b) to parallelise"""
        return a_mat, b_mat @ u

    generic_elements = vmap(gen_ele, (0, 0, 0))(
        lqr_params.lqr.A, lqr_params.lqr.B, Us_init
    )
    # print(generic_elements[0].shape, generic_elements[1].shape)
    return tuple(
        jnp.concatenate([jnp.expand_dims(first_e, 0), gen_es])
        for first_e, gen_es in zip(initial_element, generic_elements)
    )


def parallel_forward_lin_integration(
    lqr_params: LQRParams, Us_init: Array
) -> Array:
    """Associative scan for forward linear dynamics

    Args:
        lqr_params (LQRParams): LQR parameters and initial state
        Us_init (Array): input sequence

    Returns:
        Array: state trajectory
    """

    dyn_elements = build_fwd_lin_dyn_elements(lqr_params, Us_init)

    @vmap
    def associative_dyn_op(elem1, elem2):
        a1, b1 = elem1
        a2, b2 = elem2
        return a1 @ a2, a2 @ b1 + b2

    c_as, c_bs = associative_scan(associative_dyn_op, dyn_elements)
    return c_bs



