from typing import Callable, Tuple, Any
import jax
from jax import lax, Array
import jax.numpy as jnp
from jax.scipy.linalg import solve, cho_factor, cho_solve, inv

from diffilqrax.lqr import lqr_forward_pass, lqr_adjoint_pass
from diffilqrax.typs import (
    LQRParams,
    LQR,
    ModelDims,
    CostToGo,
    Gains,
    symmetrise_matrix,
    RiccatiStepParams,
)

jax.config.update("jax_disable_jit", False)  # Disable JIT for debugging


def gen_lqr_problem():
    dims = ModelDims(2, 2, 20, dt=0.1)
    A = jnp.array([[1, dims.dt], [-1 * dims.dt, 1 - 0.5 * dims.dt]])
    B = jnp.array([[0, 0], [1, 0]]) * dims.dt
    Q = jnp.eye(dims.n, dtype=float)
    R = jnp.eye(dims.m, dtype=float)
    lqr = LQR(
        A=jnp.tile(A, (dims.horizon, 1, 1)),
        B=jnp.tile(B, (dims.horizon, 1, 1)),
        a=jnp.zeros((dims.horizon, dims.n), dtype=float),
        Q=jnp.tile(Q, (dims.horizon, 1, 1)),
        q=jnp.zeros((dims.horizon, dims.m), dtype=float),
        R=jnp.tile(R, (dims.horizon, 1, 1)),
        r=jnp.zeros((dims.horizon, dims.n), dtype=float),
        S=jnp.zeros((dims.horizon, dims.m, dims.m), dtype=float),
        Qf=0.1 * Q,
        qf=jnp.ones(dims.n, dtype=float) * 0.01,
    )()
    return lqr, dims


def lqr_cov_backward_pass(
    lqr: LQR
) -> Tuple[Array, Array, Array, Array]:
    """LQR backward pass learn optimal Gains given LQR cost constraints and dynamics

    Args:
        lqr (LQR): LQR parameters
        T (int): parameter time horizon
        expected_change (bool, optional): Estimate expected change in cost [Tassa, 2020].
        Defaults to False.

    Returns:
        Gains: Optimal feedback gains.
        ValueFn: Optimal value function.
    """

    a_transp, b_transp = lqr.A.transpose(0, 2, 1), lqr.B.transpose(0, 2, 1)
    n_dim, m_dim = lqr.B[0].shape

    def riccati_step(
        carry: Tuple[CostToGo, CostToGo], inps: RiccatiStepParams
    ) -> Tuple[CostToGo, Gains]:
        AT, BT, (A, B, a, Q, q, R, r, S) = inps
        curr_val, cost_step = carry
        V, v, dJ, dj = curr_val.V, curr_val.v, cost_step.V, cost_step.v

        Hxx = symmetrise_matrix(Q + AT @ V @ A)
        Huu = symmetrise_matrix(R + BT @ V @ B)
        Hxu = S + AT @ V @ B
        hx = q + AT @ (v + V @ a)
        hu = r + BT @ (v + V @ a)

        # With Levenberg-Marquardt regulisation
        I_mu = 1e-7 * jnp.eye(m_dim)

        # solve gains with cholesky
        c, lower = cho_factor(Huu + I_mu)
        K, k, Huu_inv = jnp.hsplit(
            cho_solve((c, lower), jnp.c_[-Hxu.T, -hu, jnp.eye(m_dim)]),
            [n_dim, n_dim + 1],
        )
        k = k.squeeze()

        # Find value iteration at current time
        V_curr = symmetrise_matrix(Hxx + Hxu @ K + K.T @ Hxu.T + K.T @ Huu @ K)
        v_curr = hx + (K.T @ Huu @ k) + (K.T @ hu) + (Hxu @ k)

        # expected change in cost
        dJ = dJ + 0.5 * (k.T @ Huu @ k).squeeze()
        dj = dj + (k.T @ hu).squeeze()

        return (CostToGo(V_curr, v_curr), CostToGo(dJ, dj)), (
            Gains(K, k),
            CostToGo(V_curr, v_curr),
            Huu_inv,
        )

    (V_0, dJ), (Ks, Vs, Huu_invs) = lax.scan(
        riccati_step,
        init=(CostToGo(lqr.Qf, lqr.qf), (CostToGo(0.0, 0.0))),
        xs=(a_transp, b_transp, lqr[:-2]),
        reverse=True,
    )
    # NOTE: Check about appending last V (Q^{-1}) and Huu_inv (R^{-1})
    return dJ, Ks, Vs, Huu_invs


def lqr_covariance(
    gains: Gains,
    val_funs: CostToGo,
    Huu_invs: Array,
    lqr_params: LQR,
)->Tuple[Array, Array, Array]:
    """Calculate the posterior state and input covariances LQR problem.

    Args:
        gains (Gains): The optimal control gains.
        val_funs (CostToGo): The cost-to-go values.
        Huu_invs (Array): The inverse of the input input covariance matrix.
        lqr_params (LQR): The LQR parameters.

    Returns:
        Tuple: A tuple containing the posterior state and input covariances, and intermediate precision matrices
    """
    # TODO: add the linear term
    # TODO: add the cross-term covariance

    n_dim, m_dim = lqr_params.B[0].shape
    a_transp, b_transp = lqr_params.A.transpose(0, 2, 1), lqr_params.B.transpose(0, 2, 1)
    k_transp = gains.K.transpose(0, 2, 1)
    Vs = val_funs.V
    # initialise P0, V0
    p_init = jnp.zeros((n_dim, n_dim), dtype=float)

    # x_cov = inv(Vs[0])
    # carry_init = (p_init, x_cov)
    def precision_step(carry, inps):
        K, KT, v, Huu_inv, AT, BT, A, B, Q, R = inps
        p = carry
        # calc state cov
        x_cov = inv(p + v)
        # calc input cov
        u_cov = K @ x_cov @ KT + Huu_inv
        # calc nx state precision
        p_x = inv(AT) @ (p + Q) @ A
        # calc nx input precision
        p_u = R + BT @ p_x @ B
        # calc nx precision
        nx_p = p_x - p_x @ B @ inv(p_u) @ BT @ p_x.T

        return (nx_p), (x_cov, u_cov, p)

    x_covs, u_covs, ps = lax.scan(
        precision_step,
        init=p_init,
        xs=(gains.K, k_transp, Vs, Huu_invs, a_transp, b_transp, lqr_params.A, lqr_params.B, lqr_params.Q, lqr_params.R),
    )[1]
    # NOTE: check to append first x_cov, p, u_cov
    return x_covs, u_covs, ps


def solve_lqr(params: LQRParams):
    "run backward forward sweep to find optimal control"
    # backward
    _, gains, val_fns, q_invs = lqr_cov_backward_pass(params.lqr)
    # covariance
    xcvs, ucvs, ps = lqr_covariance(gains, val_fns, q_invs, params.lqr)
    # forward
    Xs, Us = lqr_forward_pass(gains, params)
    # adjoint
    Lambs = lqr_adjoint_pass(Xs, Us, params)
    return gains, Xs, Us, Lambs, (xcvs, ucvs, ps)


if __name__ == "__main__":

    lqr, dims = gen_lqr_problem()
    x_init = jnp.zeros(dims.n)
    lqr_params = LQRParams(x_init, lqr)
    dJ, Ks, Vs, Huu_invs = lqr_cov_backward_pass(lqr_params.lqr)
    xcvs, ucvs, ps = lqr_covariance(Ks, Vs, Huu_invs, lqr_params.lqr)
    
    gains, Xs, Us, Lambs, (xcvs, ucvs, ps) = solve_lqr(lqr_params)
