"""Solve LQR problem via dynamic programming"""

from typing import Callable, Tuple, Any
from jax.typing import ArrayLike
import jax
from jax import lax, Array
import jax.numpy as jnp
from jax.scipy.linalg import solve


from diffilqrax.typs import (
    symmetrise_matrix,
    ModelDims,
    LQRParams,
    Gains,
    CostToGo,
    LQR,
    RiccatiStepParams,
)

jax.config.update("jax_enable_x64", True)  # double precision


def bmm(arr1: Array, arr2: Array) -> Array:
    """Batch matrix multiplication"""
    return jax.vmap(jnp.matmul)(arr1, arr2)


# LQR struct
LQRBackParams = Tuple[
    ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike
]


def simulate_trajectory(
    dynamics: Callable[[Array, Array, Array, Any], Array],
    Us: ArrayLike,
    params: LQRParams,
    dims: ModelDims,
) -> Array:
    """
    Simulate forward pass with LQR params.

    Parameters
    ----------
    dynamics : Callable
        Function of dynamics with args t, x, u, params.
    Us : ArrayLike
        Input timeseries shape [Txm].
    params : LQRParams
        Parameters containing x_init, horizon and theta.
    dims : ModelDims
        Parameters containing shape of system n, m, horizon and dt.

    Returns
    -------
    Array
        State trajectory [(T+1)xn].
    """
    horizon = dims.horizon
    x0, lqr = params.x0, params[1]

    def step(x, inputs):
        t, u = inputs
        nx = dynamics(t, x, u, lqr)
        return nx, nx

    Xs = lax.scan(step, x0, (jnp.arange(horizon), Us))[1]

    return jnp.vstack([x0[None], Xs])


def lin_dyn_step(t: int, x: ArrayLike, u: ArrayLike, lqr: LQR) -> Array:
    """State space linear step"""
    nx = lqr.A[t] @ x + lqr.B[t] @ u + lqr.a[t]
    return nx


def lqr_adjoint_pass(Xs: ArrayLike, Us: ArrayLike, params: LQRParams) -> Array:
    """
    Adjoint backward pass with LQR params.

    Parameters
    ----------
    Xs : np.ndarray
        State timeseries shape [(T+1)xn].
    Us : np.ndarray
        Input timeseries shape [Txm].
    params : LQRParams
        LQR state and cost matrices.

    Returns
    -------
    np.ndarray
        Adjoint λs [(T+1)xn].
    """
    lqr = params[1]
    AT = lqr.A.transpose(0, 2, 1)
    lambf = lqr.Qf @ Xs[-1] + lqr.qf

    def adjoint_step(lamb, inputs):
        x, u, aT, Q, q, S = inputs
        nlamb = aT @ lamb + Q @ x + q + S @ u
        return nlamb, nlamb

    lambs = lax.scan(
        adjoint_step, lambf, (Xs[:-1], Us[:], AT, lqr.Q, lqr.q, lqr.S), reverse=True
    )[1]
    return jnp.vstack([lambs, lambf[None]])


def lqr_forward_pass(gains: Gains, params: LQRParams) -> Tuple[Array, Array]:
    """
    LQR forward pass using gain state feedback.

    Parameters
    ----------
    gains : Gains
        K matrices.
    params : LQRParams
        LQR state and cost matrices.

    Returns
    -------
    Tuple[Array, Array]
        Updated state [(T+1)xn] and inputs [Txm].
    """
    x0, lqr = params.x0, params.lqr

    def dynamics(x: jnp.array, params: LQRBackParams):
        A, B, a, K, k = params
        u = K @ x + k
        nx = A @ x + B @ u + a
        return nx, (nx, u)

    Xs, Us = lax.scan(dynamics, init=x0, xs=(lqr.A, lqr.B, lqr.a, gains.K, gains.k))[1]

    return jnp.vstack([x0[None], Xs]), Us


def calc_expected_change(dJ: CostToGo, alpha: float = 0.5):
    """
    Expected change in cost [Tassa, 2020].

    Parameters
    ----------
    dJ : CostToGo
        Cost to go.
    alpha : float, optional
        Scaling factor, by default 0.5.

    Returns
    -------
    float
        Expected change in cost.
    """
    return -(dJ.V * alpha**2 + dJ.v * alpha)


def regularised_backward_pass(
    lqr: LQR,
    mu: float = 0.0,
    pd_thresh: float = 1e-8,
) -> Tuple[CostToGo, Gains, Array]:
    """
    LQR backward pass with Levenberg-Marquardt regularisation of the input Hessian.

    Only ``Huu`` is regularised, ``Huu_reg = Huu + mu B^T B``. The value function update uses the full form so it stays
    consistent with the regularised gains, and the expected change in cost uses the
    unregularised ``Huu`` (Tassa et al., 2012). With ``mu=0`` this is the exact LQR solution.

    Parameters
    ----------
    lqr : LQR
        LQR parameters.
    mu : float, optional
        Regularisation strength, by default 0.0.
    pd_thresh : float, optional
        Minimum eigenvalue of ``Huu_reg`` for it to count as positive definite, by default 1e-8.

    Returns
    -------
    Tuple[CostToGo, Gains, Array]
        Expected change in cost terms (V: 0.5 k^T Huu k, v: k^T hu summed over time), the optimal
        feedback gains, and a boolean flag that is True if ``Huu_reg`` is positive definite at
        every time step.
    """
    a_transp, b_transp = lqr.A.transpose(0, 2, 1), lqr.B.transpose(0, 2, 1)
    n_dim, _ = lqr.B[0].shape

    def riccati_step(
        carry: Tuple[CostToGo, CostToGo, Array], inps: RiccatiStepParams
    ) -> Tuple[Tuple[CostToGo, CostToGo, Array], Gains]:
        AT, BT, (A, B, a, Q, q, R, r, S) = inps
        curr_val, cost_step, is_pd = carry
        V, v, dJ, dj = curr_val.V, curr_val.v, cost_step.V, cost_step.v
        Huu = symmetrise_matrix(R + BT @ V @ B)
        Hxx = symmetrise_matrix(Q + AT @ V @ A)
        Hxu = S + AT @ V @ B
        hx = q + AT @ (v + V @ a)
        hu = r + BT @ (v + V @ a)
        Huu_reg = symmetrise_matrix(Huu + mu * BT @ B)
        min_eval = jnp.min(jnp.linalg.eigvalsh(lax.stop_gradient(Huu_reg)))
        is_pd = jnp.logical_and(is_pd, min_eval > pd_thresh)
        K, k = jnp.hsplit(-solve(Huu_reg, jnp.c_[Hxu.T, hu], assume_a="her"), [n_dim])
        k = k.reshape(
            -1,
        )

        # Find value iteration at current time
        V_curr = symmetrise_matrix(Hxx + Hxu @ K + K.T @ Hxu.T + K.T @ Huu @ K)
        v_curr = hx + (K.T @ Huu @ k) + (K.T @ hu) + (Hxu @ k)

        # expected change in cost
        dJ = dJ + 0.5 * (k.T @ Huu @ k).squeeze()
        dj = dj + (k.T @ hu).squeeze()

        return (CostToGo(V_curr, v_curr), CostToGo(dJ, dj), is_pd), Gains(K, k)

    (_, dJ, is_pd), Ks = lax.scan(
        riccati_step,
        init=(CostToGo(lqr.Qf, lqr.qf), CostToGo(0.0, 0.0), jnp.array(True)),
        xs=(a_transp, b_transp, lqr[:-2]),
        reverse=True,
    )

    return dJ, Ks, is_pd


def lqr_backward_pass(
    lqr: LQR,
) -> Tuple[CostToGo, Gains]:
    """
    LQR backward pass learn optimal Gains given LQR cost constraints and dynamics.

    Parameters
    ----------
    lqr : LQR
        LQR parameters.

    Returns
    -------
    Tuple[CostToGo, Gains]
        Expected change in cost terms and optimal feedback gains.
    """
    dJ, Ks, _ = regularised_backward_pass(lqr, mu=0.0)
    return dJ, Ks


def kkt(
    params: LQRParams, Xs: Array, Us: Array, Lambs: Array
) -> Tuple[Array, Array, Array]:
    """
    Define KKT conditions for LQR problem.

    Parameters
    ----------
    params : LQRParams
        LQR parameters.
    Xs : Array
        State trajectory.
    Us : Array
        Control inputs.
    Lambs : Array
        Adjoint variables.

    Returns
    -------
    Tuple[Array, Array, Array]
        Gradients with respect to states, inputs, and adjoint variables.
    """
    AT = params.lqr.A.transpose(0, 2, 1)
    BT = params.lqr.B.transpose(0, 2, 1)
    ST = params.lqr.S.transpose(0, 2, 1)
    dLdXs = (
        bmm(params.lqr.Q, Xs[:-1])
        + bmm(params.lqr.S, Us[:])
        + params.lqr.q
        + bmm(AT, Lambs[1:])
        - Lambs[:-1]
    )
    dLdXf = jnp.matmul(params.lqr.Qf, Xs[-1]) + params.lqr.qf - Lambs[-1]
    dLdXs = jnp.concatenate([dLdXs, dLdXf[None]])
    dLdUs = (
        bmm(ST, Xs[:-1]) + bmm(params.lqr.R, Us[:]) + params.lqr.r + bmm(BT, Lambs[1:])
    )
    dLdLambs = (
        bmm(params.lqr.A, Xs[:-1]) + bmm(params.lqr.B, Us[:]) + params.lqr.a - Xs[1:]
    )
    dLdLamb0 = params.x0 - Xs[0]
    dLdLambs = jnp.concatenate([dLdLamb0[None], dLdLambs])
    return dLdXs, dLdUs, dLdLambs


def solve_lqr(params: LQRParams) -> Tuple[Array, Array, Array]:
    """
    Run backward forward sweep to find optimal control.

    Parameters
    ----------
    params : LQRParams
        LQR parameters.

    Returns
    -------
    Tuple[Array, Array, Array]
        State trajectory, control inputs, and adjoint variables.
    """
    # backward
    gains = lqr_backward_pass(params.lqr)[1]
    # forward
    Xs, Us = lqr_forward_pass(gains, params)
    # adjoint
    Lambs = lqr_adjoint_pass(Xs, Us, params)
    return Xs, Us, Lambs


def solve_lqr_swap_x0(params: LQRParams) -> Tuple[Array, Array, Array]:
    """
    Run backward forward sweep to find optimal control freeze x0 to zero.

    Parameters
    ----------
    params : LQRParams
        LQR parameters.

    Returns
    -------
    Tuple[Array, Array, Array]
        State trajectory, control inputs, and adjoint variables.
    """
    # backward
    gains = lqr_backward_pass(params.lqr)[1]
    new_params = LQRParams(jnp.zeros_like(params.x0), params.lqr)
    Xs, Us = lqr_forward_pass(gains, new_params)
    # adjoint
    Lambs = lqr_adjoint_pass(Xs, Us, new_params)
    return Xs, Us, Lambs
