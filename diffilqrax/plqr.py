"""
LQR solver using associative parallel scan [Särkkä & García-Fernández, 2023, arXiv:2104.03186].

The value functions are found with an associative scan over conditional value function elements,
and the optimal trajectory with an associative scan over the closed-loop affine dynamics. The
cross term S and the linear input cost r are removed first by the change of variables
u = ũ - G x - g, with G = R⁻¹Sᵀ and g = R⁻¹r, which leaves an equivalent problem with S = 0, r = 0.
"""

from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax import Array, vmap
from jax.lax import associative_scan

from diffilqrax.typs import (
    symmetrise_tensor,
    LQRParams,
    CostToGo,
    Gains,
    LQR,
)

jax.config.update("jax_enable_x64", True)  # double precision


def _bmv(mat: Array, vec: Array) -> Array:
    """Batched matrix-vector product over the leading time axis"""
    return (mat @ vec[..., None])[..., 0]


def remove_cross_term(lqr: LQR) -> Tuple[LQR, Array, Array]:
    """
    Change of input variables u = ũ - G x - g, with G = R⁻¹Sᵀ and g = R⁻¹r, which removes the cross
    term and the linear input cost. The value functions and the adjoints are unchanged.

    Parameters
    ----------
    lqr : LQR
        LQR parameters, with R invertible.

    Returns
    -------
    Tuple[LQR, Array, Array]
        Equivalent LQR with S = 0 and r = 0, the state gain G [Txmxn] and offset g [Txm].
    """
    rhs = jnp.concatenate([lqr.S.transpose(0, 2, 1), lqr.r[..., None]], axis=-1)
    Gg = jnp.linalg.solve(lqr.R, rhs)
    G, g = Gg[..., :-1], Gg[..., -1]
    new_lqr = lqr._replace(
        A=lqr.A - lqr.B @ G,
        a=lqr.a - _bmv(lqr.B, g),
        Q=symmetrise_tensor(lqr.Q - lqr.S @ G),
        q=lqr.q - _bmv(lqr.S, g),
        r=jnp.zeros_like(lqr.r),
        S=jnp.zeros_like(lqr.S),
    )
    return new_lqr, G, g


def build_associative_riccati_elements(model: LQRParams) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Conditional value function elements (A, b, C, η, J) for the associative Riccati scan, with the
    value function V_t(x) = ½ xᵀ J_t x - η_tᵀ x.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters.

    Returns
    -------
    Tuple
        Elements A, b, C, η, J for t = 0..T, the last of which is the final cost.
    """
    lqr, _, _ = remove_cross_term(model.lqr)
    n = lqr.A.shape[-1]
    C = symmetrise_tensor(lqr.B @ jnp.linalg.solve(lqr.R, lqr.B.transpose(0, 2, 1)))
    generic = (lqr.A, lqr.a, C, -lqr.q, symmetrise_tensor(lqr.Q))
    zeros = jnp.zeros((n, n), dtype=lqr.A.dtype)
    last = (zeros, jnp.zeros_like(lqr.qf), zeros, -lqr.qf, (lqr.Qf + lqr.Qf.T) / 2)
    return tuple(jnp.concatenate([gen, last_e[None]]) for gen, last_e in zip(generic, last))


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
        Linear (η) and quadratic (J) value functions for t = 0..T, with V_t(x) = ½ xᵀ J_t x - η_tᵀ x.
    """
    lqr_elements = build_associative_riccati_elements(model)
    _, _, _, etas, Js = associative_scan(riccati_operator, lqr_elements, reverse=True)
    return etas, symmetrise_tensor(Js)


def optimal_gains(model: LQRParams, etas: Array, Js: Array) -> Tuple[Gains, Array]:
    """
    Optimal feedback u_t = K_t x_t + k_t given the value functions, in the original input variables.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters.
    etas : Array
        Linear value functions for t = 0..T.
    Js : Array
        Quadratic value functions for t = 0..T.

    Returns
    -------
    Tuple[Gains, Array]
        Gains (K, k) and the input Hessians Huu = R + Bᵀ J_{t+1} B.
    """
    lqr, G, g = remove_cross_term(model.lqr)
    BT = lqr.B.transpose(0, 2, 1)
    Huu = symmetrise_tensor(lqr.R + BT @ Js[1:] @ lqr.B)
    rhs = jnp.concatenate([BT @ Js[1:] @ lqr.A, _bmv(BT, etas[1:] - _bmv(Js[1:], lqr.a))[..., None]], axis=-1)
    Kk = jnp.linalg.solve(Huu, rhs)
    # transformed policy ũ = -K̃ x + k̃, so u = -(K̃ + G) x + k̃ - g
    return Gains(-(Kk[..., :-1] + G), Kk[..., -1] - g), Huu


def get_dcosts(model: LQRParams, etas: Array, Js: Array, alpha: float = 1.0) -> CostToGo:
    """
    Expected change in cost terms, as returned by the sequential backward pass: V = Σ ½ kᵀ Huu k and
    v = Σ kᵀ hu, with hu = -Huu k. Scale with alpha for a backtracking step.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters.
    etas : Array
        Linear value functions.
    Js : Array
        Quadratic value functions.
    alpha : float, optional
        Linesearch alpha parameter, by default 1.0.

    Returns
    -------
    CostToGo
        Total change in cost-to-go.
    """
    gains, Huu = optimal_gains(model, etas, Js)
    kHk = jnp.sum(gains.k * _bmv(Huu, gains.k))
    return CostToGo(V=0.5 * kHk * alpha**2, v=-kHk * alpha)


def regularise_lqr(lqr: LQR, mu: float) -> LQR:
    """
    Levenberg-Marquardt regularisation in input space, R -> R + μI, i.e. a proximal cost ½μ‖u‖² on
    the input step. Unlike a regulariser applied to Huu during the Riccati recursion, it does not
    depend on the value function, so the regularised problem is still an LQR and the associative
    scan solves it exactly. In the Riccati recursion it adds μI to every Huu.

    Parameters
    ----------
    lqr : LQR
        LQR parameters.
    mu : float
        Regularisation strength.

    Returns
    -------
    LQR
        Regularised LQR parameters.
    """
    return lqr._replace(R=lqr.R + mu * jnp.eye(lqr.R.shape[-1], dtype=lqr.R.dtype))


def project_psd(mat: Array, delta: float = 0.0) -> Array:
    """Project symmetric matrices [...xnxn] onto the PSD cone, clipping eigenvalues at ``delta``."""
    evals, evecs = jnp.linalg.eigh(0.5 * (mat + jnp.swapaxes(mat, -1, -2)))
    proj = (evecs * jnp.maximum(evals, delta)[..., None, :]) @ jnp.swapaxes(evecs, -1, -2)
    return 0.5 * (proj + jnp.swapaxes(proj, -1, -2))


def project_psd_lqr(lqr: LQR, delta: float = 1e-6, which: str = "full") -> LQR:
    """
    Convexify the local LQR stage by stage, as in ``primal_dual_ilqr.optimizers.regularize``. R is
    projected to eigenvalues ≥ δ and, if ``which == "full"``, the Schur complement Q - S R⁻¹Sᵀ and Qf
    too. The fully projected problem is a strictly convex LQR, so every element of the associative scan
    is valid and every Huu is positive definite without retries. Linear terms are untouched, so the
    gradient of the local problem is unchanged, but it is no longer a Newton model where the projection
    is active. With ``which == "R"`` only negative input curvature is removed: the projection is
    inactive wherever R ≻ δI, where the Newton step is kept.

    Parameters
    ----------
    lqr : LQR
        LQR parameters.
    delta : float
        Minimum eigenvalue after projection.
    which : str
        "R" to project R only, "full" to also project Q - S R⁻¹Sᵀ and Qf.

    Returns
    -------
    LQR
        Convexified LQR parameters.
    """
    assert which in ("R", "full"), f"unknown projection {which}"
    R = project_psd(lqr.R, delta)
    if which == "R":
        return lqr._replace(R=R)
    SRS = symmetrise_tensor(lqr.S @ jnp.linalg.solve(R, lqr.S.transpose(0, 2, 1)))
    Q = project_psd(lqr.Q - SRS, delta) + SRS
    return lqr._replace(R=R, Q=Q, Qf=project_psd(lqr.Qf, delta))


def plqr_backward_pass(model: LQRParams, mu: float = 0.0, pd_thresh: float = 1e-8) -> Tuple[CostToGo, Gains, Array]:
    """
    Parallel counterpart of :func:`diffilqrax.lqr.regularised_backward_pass`: associative Riccati scan
    of the problem regularised with :func:`regularise_lqr`, and its optimal gains.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters.
    mu : float, optional
        Regularisation strength, by default 0.0 (exact LQR solution).
    pd_thresh : float, optional
        Minimum eigenvalue of the regularised Huu for it to count as positive definite, by default 1e-8.

    Returns
    -------
    Tuple[CostToGo, Gains, Array]
        Expected change in cost terms (V: Σ ½ kᵀ Huu k with the unregularised Huu, v: Σ kᵀ hu), the
        gains (u = Kx + k), and a flag that is True if the regularised Huu is positive definite at
        every time step and the gains are finite.
    """
    reg = LQRParams(model.x0, regularise_lqr(model.lqr, mu))
    etas, Js = associative_riccati_scan(reg)
    gains, Huu_reg = optimal_gains(reg, etas, Js)
    min_eval = jnp.min(jnp.linalg.eigvalsh(jax.lax.stop_gradient(Huu_reg)))
    finite = jnp.logical_and(jnp.all(jnp.isfinite(gains.K)), jnp.all(jnp.isfinite(gains.k)))
    is_pd = jnp.logical_and(min_eval > pd_thresh, finite)
    # hu = -Huu_reg k, as in the sequential pass
    kHk_reg = jnp.sum(gains.k * _bmv(Huu_reg, gains.k))
    kHk = kHk_reg - mu * jnp.sum(gains.k**2)
    return CostToGo(V=0.5 * kHk, v=-kHk_reg), gains, is_pd


def build_associative_lin_dyn_elements(model: LQRParams, gains: Gains, alpha: float = 1.0) -> Tuple[Array, Array]:
    """
    Closed-loop affine dynamics elements x_{t+1} = F_t x_t + c_t under u_t = K_t x_t + α k_t. The first
    element absorbs the initial state.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters (only the dynamics and x0 are used).
    gains : Gains
        Feedback gains (K, k).
    alpha : float, optional
        Linesearch step on the feedforward term, by default 1.0.

    Returns
    -------
    Tuple[Array, Array]
        Elements (F, c).
    """
    lqr = model.lqr
    Fs = lqr.A + lqr.B @ gains.K
    cs = lqr.a + alpha * _bmv(lqr.B, gains.k)
    cs = cs.at[0].add(Fs[0] @ model.x0)
    Fs = Fs.at[0].set(jnp.zeros_like(Fs[0]))
    return Fs, cs


def closed_loop_scan(model: LQRParams, gains: Gains, alpha: float = 1.0) -> Array:
    """
    States x_1..x_T of the linear dynamics under u_t = K_t x_t + α k_t, by an associative scan.

    Parameters
    ----------
    model : LQRParams
        LQR model parameters (only the dynamics and x0 are used).
    gains : Gains
        Feedback gains (K, k).
    alpha : float, optional
        Linesearch step on the feedforward term, by default 1.0.

    Returns
    -------
    Array
        States x_1..x_T.
    """
    _, cs = associative_scan(dynamic_operator, build_associative_lin_dyn_elements(model, gains, alpha))
    return cs


def associative_opt_traj_scan(model: LQRParams, etas: Array, Js: Array, alpha: float = 1.0) -> Tuple[Array, Gains]:
    """
    Obtain the optimal state trajectory through an associative scan of the closed-loop dynamics.

    Parameters
    ----------
    model : LQRParams
        LQR problem (initial state, LQR parameters).
    etas : Array
        Linear value functions.
    Js : Array
        Quadratic value functions.
    alpha : float, optional
        Linesearch step on the feedforward term, by default 1.0.

    Returns
    -------
    Tuple[Array, Gains]
        States x_1..x_T, and the gains with u_t = K_t x_t + α k_t.
    """
    gains, _ = optimal_gains(model, etas, Js)
    return closed_loop_scan(model, gains, alpha), gains


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
    etas, Js = associative_riccati_scan(model)
    cs, gains = associative_opt_traj_scan(model, etas, Js)
    Xs = jnp.r_[model.x0[None], cs]
    Us = _bmv(gains.K, Xs[:-1]) + gains.k
    # adjoints are the value function gradients λ_t = J_t x_t - η_t
    Lambdas = _bmv(Js, Xs) - etas
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
    return solve_plqr(model._replace(x0=jnp.zeros_like(model.x0)))


# --------
# Parallel forward integration
# --------
def build_fwd_lin_dyn_elements(lqr_params: LQRParams, Us_init: Array, a_term: Array = None) -> Tuple[Array, Array]:
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
        jnp.zeros((lqr_params.x0.shape[0], lqr_params.x0.shape[0])),
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
        jnp.concatenate([jnp.expand_dims(first_e, 0), gen_es]) for first_e, gen_es in zip(initial_element, generic_elements)
    )


def parallel_forward_lin_integration(lqr_params: LQRParams, Us_init: Array, a_term: Array) -> Array:
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
    _, c_bs = associative_scan(dynamic_operator, dyn_elements)
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

    last_element = (jnp.zeros((lambda_f.shape[0], lambda_f.shape[0])), lambda_f)

    @vmap
    def _generic(a_trsp_mat: Array, s_mat: Array, q_mat: Array, q_vec: Array, x: Array, u: Array) -> Tuple[Array, Array]:
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
    return tuple(jnp.concatenate([gen_, jnp.expand_dims(last_, 0)]) for gen_, last_ in zip(generic_elements, last_element))


def parallel_reverse_lin_integration(lqr_params: LQRParams, xs_traj: Array, us_traj: Array) -> Array:
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
    _, c_bs = associative_scan(dynamic_operator, dyn_elements, reverse=True)
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
    I = jnp.eye(dim, dtype=A1.dtype)

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
