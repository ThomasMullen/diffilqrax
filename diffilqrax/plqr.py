"""LQR solver using associative parallel scan"""

from typing import Callable, Tuple
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
)

jax.config.update("jax_enable_x64", True)  # double precision

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
    η = -model.lqr.q[-1] #jnp.eye(n_dims).T @ jnp.zeros((n_dims), dtype=float)
    # here: set readout C=I
    J = model.lqr.Q[-1] #jnp.eye(n_dims).T @ model.lqr.Q[-1] @ jnp.eye(n_dims, dtype=float)
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
    R_invs = vmap(jsc.linalg.inv)(model.lqr.R)
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
    initial_elements = build_associative_riccati_elements(model)

    # riccati operator
    @vmap
    def asoc_riccati_operator(elem1, elem2):
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
        print("input1", A1.shape, b1.shape, C1.shape, η1.shape, J1.shape)
        print("input2", A2.shape, b2.shape, C2.shape, η2.shape, J2.shape)
        print("output", A.shape, b.shape, C.shape, η.shape, J.shape)
        return A, b, C, η, J
    final_elements = associative_scan(
        asoc_riccati_operator, initial_elements, reverse=True
    )
    
    return final_elements[-2], final_elements[-1] #this only returns J, eta, which are the only things we need to compute 
#Vk : Sk = Jk_{T+1}, vk = eta_k_{T+1}
##or is it? how do we have all k and k+1 accessible? 




def generic_dynamics_elements(lqr, eta, J):
    S, v = J, eta #this needs to be at k+1 anyway
    c =  lqr.a
    B = lqr.B
    R = lqr.R
    A = lqr.A
    pinv = jsc.linalg.inv(B.T@S@B + R)
    Kv = pinv@B.T
    Kx = Kv@S@A
    Kc = Kv@S
    F = A - B@Kx
    c = c + B@Kv@v - B@Kc@c
    return F, c




def first_dynamics_element(model, eta0, J0): 
    S0, v0 = J0, eta0 #this needs to be at k+1 anyway
    c =  model.lqr.a[0]
    B = model.lqr.B[0]
    R = model.lqr.R[0]
    A = model.lqr.A[0]
    pinv = jsc.linalg.inv(B.T@S0@B + R)
    Kv = pinv@B.T
    Kx = Kv@S0@A
    Kc = Kv@S0
    F0 = A - B@Kx
    c0 = c + B@Kv@v0 - B@Kc@c
    return jnp.zeros_like(J0), F0@model.x0 + c0


def build_associative_dynamics_elements(
    model: LQRParams, etas, Js
)-> Tuple[Tuple[Array, Array, Array, Array, Array]]:
    """Join set of elements for associative scan.
    Args:
        model (LQRParams)

    Returns:
        Tuple: return tuple of elements Fs, Cs
    """
    print(etas.shape, Js.shape)
    first_elem = first_dynamics_element(model, etas[0], Js[0])
    generic_elems = jax.vmap(generic_dynamics_elements, in_axes = (LQR(0,0,0,0,0,0,0,0,None,None), 0, 0))(model.lqr, etas[1:], Js[1:])
    return tuple(jnp.concatenate([jnp.expand_dims(first_e, 0), gen_es]) 
                 for first_e, gen_es in zip(first_elem, generic_elems))


# parallellised riccati scan
def parallel_dynamics_scan(model: LQRParams, etas, Js):
    #need to add vmaps
    initial_elements = build_associative_dynamics_elements(model, etas, Js)

    # riccati operator
    @vmap
    def assoc_dynamics_operator(elem1, elem2):
        F1, c1 = elem1
        F2, c2 = elem2
        F = F2@F1
        c = F2@c1 + c2
        return F, c
    
    final_elements = associative_scan(
        assoc_dynamics_operator, initial_elements, reverse=True
    )

    return final_elements #have to check shapes of this : hope is that it returns F, c for all k and that the c is the x we want? 



def solve_plqr(model: LQRParams):
    "run backward forward sweep to find optimal control"
    # backward
    etas, Js = parallel_riccati_scan(model)
    _, xs = parallel_dynamics_scan(model, etas, Js)
    return xs
    # _, gains = lqr_backward_pass(params.lqr, sys_dims)
    # # forward
    # Xs, Us = lqr_forward_pass(gains, params)
    # # adjoint
    # Lambs = lqr_adjoint_pass(Xs, Us, params)
    # return gains, Xs, Us, Lambs


# def generic_gain_element(model, eta, J):
#     S, v = J, eta
#     # S, v = elems #at k+1
#     c =  model.lqr.a
#     B = model.lqr.B
#     R = model.lqr.R
#     A = model.lqr.A
#     #get the 
#     pinv = jsc.linalg.inv(B.T@S@B + R)
#     Kv = pinv@B.T
#     Kx = Kv@S@A
#     Kc = Kv@S
#     return Kv, Kx, Kc, v, c, A, B


    #Kv, Kx, Kc
    #then unroll with x (roll forward)
    # u = -Kx@x[k] + Kv@v - Kc@c[k]
# def build_associative_dynamics_elements(
# model: LQRParams,  etas, Js
# )-> Tuple[Tuple[Array, Array, Array, Array, Array]]:
#     first_elem = first_gain_element(etas[0], Js[0])
#     generic_elems = vmap(generic_gain_element)(model, etas[1:], Js[1:])
#     return tuple(
#         jnp.concatenate(gen_es, jnp.expand_dims(first_e, 0))
#         for gen_es, first_e in zip(generic_elems, first_elem)
#     )
  
# def unroll(x, gain_elems):
#     Kv, Kx, Kc, v, c, A, B = gain_elems
#     #optimal_x_k = np.linalg.inv(np.eye(n) + csk@s_k)@(a_ks@xs +bsk + csk@vk)
#     u = -Kx@x + Kv@v - Kc@c
#     nx = A@x + B@u
#     return nx, (x, u)

# def build_dynamic_elemnt():
#     S, v = J, eta
#     # S, v = elems #at k+1
#     c =  model.lqr.a
#     B = model.lqr.B
#     R = model.lqr.R
#     A = model.lqr.A
#     #get the 
#     pinv = jsc.linalg.inv(B.T@S@B + R)
#     Kv = pinv@B.T
#     Kx = Kv@S@A
#     Kc = Kv@S
#     f = A - B@K

# def lqr_step(model):
#     initial_elements = make_associative_smoothing_elements(model, filtered_means, filtered_covariances)
#     final_elements = associative_scan(smoothing_operator, initial_elements, reverse=True)  # note the vmap
#     gain_elems =  build_associative_gain_elements(model, filtered_means, filtered_covariances)
#     ##I guess also associative unroll would be needed to avoid the T scaling...
#     return final_elements[1], final_elements[2]


#def run_forward... #standard ilqr run forward function

"""
def kf(model, observations):
    def body(carry, y):
        m, P = carry
        m = model.F @ m
        P = model.F @ P @ model.F.T + model.Q

        obs_mean = model.H @ m
        S = model.H @ P @ model.H.T + model.R

        K = solve(S, model.H @ P, assume_a='pos').T  # notice the jsc here
        m = m + K @ (y - model.H @ m)
        P = P - K @ S @ K.T
        return (m, P), (m, P)

    _, (fms, fPs) = scan(body, (model.m0, model.P0), observations)
    return fms, fPs


def first_filtering_element(model, y):
    S = model.H @ model.Q @ model.H.T + model.R
    CF, low = jsc.linalg.cho_factor(S)  # note the jsc

    m1 = model.F @ model.m0
    P1 = model.F @ model.P0 @ model.F.T + model.Q
    S1 = model.H @ P1 @ model.H.T + model.R
    K1 = jsc.linalg.solve(S1, model.H @ P1, assume_a='pos').T  # note the jsc

    A = jnp.zeros_like(model.F)
    b = m1 + K1 @ (y - model.H @ m1)
    C = P1 - K1 @ S1 @ K1.T

    # note the jsc
    eta = model.F.T @ model.H.T @ jsc.linalg.cho_solve((CF, low), y)
    J = model.F.T @ model.H.T @ jsc.linalg.cho_solve((CF, low), model.H @ model.F)
    return A, b, C, J, eta


def generic_filtering_element(model, y):
    S = model.H @ model.Q @ model.H.T + model.R
    CF, low = jsc.linalg.cho_factor(S)  # note the jsc
    K = jsc.linalg.cho_solve((CF, low), model.H @ model.Q).T  # note the jsc
    A = model.F - K @ model.H @ model.F
    b = K @ y
    C = model.Q - K @ model.H @ model.Q

    # note the jsc
    eta = model.F.T @ model.H.T @ jsc.linalg.cho_solve((CF, low), y)
    J = model.F.T @ model.H.T @ jsc.linalg.cho_solve((CF, low), model.H @ model.F)
    return A, b, C, J, eta


def make_associative_filtering_elements(model, observations):
    first_elems = first_filtering_element(model, observations[0])
    generic_elems = vmap(lambda o: generic_filtering_element(model, o))(observations[1:])
    return tuple(jnp.concatenate([jnp.expand_dims(first_e, 0), gen_es]) 
                 for first_e, gen_es in zip(first_elems, generic_elems))


@vmap
def filtering_operator(elem1, elem2):
    # # note the jsc everywhere
    A1, b1, C1, J1, eta1 = elem1
    A2, b2, C2, J2, eta2 = elem2
    dim = A1.shape[0]
    I = jnp.eye(dim)  # note the jnp

    I_C1J2 = I + C1 @ J2
    temp = jsc.linalg.solve(I_C1J2.T, A2.T).T
    A = temp @ A1
    b = temp @ (b1 + C1 @ eta2) + b2
    C = temp @ C1 @ A2.T + C2

    I_J2C1 = I + J2 @ C1
    temp = jsc.linalg.solve(I_J2C1.T, A1).T

    eta = temp @ (eta2 - J2 @ b1) + eta1
    J = temp @ J2 @ A1 + J1

    return A, b, C, J, eta



def pkf(model, observations):
    initial_elements = make_associative_filtering_elements(model, observations)
    final_elements = associative_scan(filtering_operator, initial_elements)
    return final_elements[1], final_elements[2]



def ks(model, ms, Ps):
    def body(carry, inp):
        m, P = inp
        sm, sP = carry

        pm = model.F @ m
        pP = model.F @ P @ model.F.T + model.Q

        C = solve(pP, model.F @ P, assume_a='pos').T  # notice the jsc here
        
        sm = m + C @ (sm - pm)
        sP = P + C @ (sP - pP) @ C.T
        return (sm, sP), (sm, sP)

    _, (sms, sPs) = scan(body, (ms[-1], Ps[-1]), (ms[:-1], Ps[:-1]), reverse=True)
    sms = jnp.append(sms, jnp.expand_dims(ms[-1], 0), 0)
    sPs = jnp.append(sPs, jnp.expand_dims(Ps[-1], 0), 0)
    return sms, sPs


def last_smoothing_element(m, P):
    return jnp.zeros_like(P), m, P

def generic_smoothing_element(model, m, P):
    Pp = model.F @ P @ model.F.T + model.Q

    E  = jsc.linalg.solve(Pp, model.F @ P, assume_a='pos').T
    g  = m - E @ model.F @ m
    L  = P - E @ Pp @ E.T
    return E, g, L

def make_associative_smoothing_elements(model, filtering_means, filtering_covariances):
    last_elems = last_smoothing_element(filtering_means[-1], filtering_covariances[-1])
    generic_elems = vmap(lambda m, P: generic_smoothing_element(model, m, P))(filtering_means[:-1], filtering_covariances[:-1])
    return tuple(jnp.append(gen_es, jnp.expand_dims(last_e, 0), axis=0) 
                 for gen_es, last_e in zip(generic_elems, last_elems))


@vmap
def smoothing_operator(elem1, elem2):
    E1, g1, L1 = elem1
    E2, g2, L2 = elem2

    E = E2 @ E1
    g = E2 @ g1 + g2
    L = E2 @ L1 @ E2.T + L2

    return E, g, L


def pks(model, filtered_means, filtered_covariances):
    initial_elements = make_associative_smoothing_elements(model, filtered_means, filtered_covariances)
    final_elements = associative_scan(smoothing_operator, initial_elements, reverse=True)  # note the vmap
    return final_elements[1], final_elements[2]
"""



# @vmap
# def lqr_operator(elem1, elem2):
    # E1, g1, L1 = elem1
    # E2, g2, L2 = elem2

    # E = E2 @ E1
    # g = E2 @ g1 + g2
    # L = E2 @ L1 @ E2.T + L2

    # return E, g, L

# def generic_smoothing_element(model, m, P):
#     Pp = model.F @ P @ model.F.T + model.Q

#     E  = jsc.linalg.solve(Pp, model.F @ P, assume_a='pos').T
#     g  = m - E @ model.F @ m
#     L  = P - E @ Pp @ E.T
#     return E, g, L