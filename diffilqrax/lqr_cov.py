from typing import Callable, Tuple
import jax
from jax import lax
import jax.numpy as jnp
from jax.scipy.linalg import solve, cho_factor, cho_solve
# from diffilqrax import lqr
from diffilqrax.typs import (
    LQRParams,
    LQR,
    ModelDims,
    CostToGo,
    Gains,
    symmetrise_matrix,
    RiccatiStepParams,
)

jax.config.update("jax_disable_jit", True) # Disable JIT for debugging

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
        qf=jnp.ones(dims.n, dtype=float)*0.01,
    )()
    return lqr, dims


def lqr_cov_backward_pass(lqr: LQR, sys_dims: ModelDims):
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
        I_mu = 1e-7 * jnp.eye(sys_dims.m)

        # solve gains
        K, k = jnp.hsplit(
            -solve(Huu + I_mu, jnp.c_[Hxu.T, hu], assume_a="her"), [sys_dims.n]
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
        )

    (V_0, dJ), (Ks, Vs) = lax.scan(
        riccati_step,
        init=(CostToGo(lqr.Qf, lqr.qf), (CostToGo(0.0, 0.0))),
        xs=(a_transp, b_transp, lqr[:-2]),
        reverse=True,
    )
    return dJ, Ks, Vs


def lqr_cov_forward_pass(lqr: LQR, sys_dims: ModelDims):
    pass


def lqr_covariance():
    pass


def solve_lqr(params: LQRParams, sys_dims: ModelDims):
    "run backward forward sweep to find optimal control"
    # backward
    _, gains, val_fns = lqr_cov_backward_pass(params.lqr, sys_dims)
    # forward
    Xs, Us = lqr_cov_forward_pass(gains, params)
    # adjoint
    Lambs = lqr_adjoint_pass(Xs, Us, params)
    return gains, Xs, Us, Lambs


lqr, dims = gen_lqr_problem()
dJ, Ks, Vs = lqr_cov_backward_pass(lqr, dims)

# troubleshoot backpass step
sys_dims = dims
a_transp, b_transp = lqr.A.transpose(0, 2, 1), lqr.B.transpose(0, 2, 1)
inps_all=zip(a_transp,b_transp,lqr.A,lqr.B,lqr.a,lqr.Q,lqr.q,lqr.R,lqr.r,lqr.S)
carry0=(CostToGo(lqr.Qf, lqr.qf), (CostToGo(0.0, 0.0)))

AT, BT, A, B, a, Q, q, R, r, S = next(inps_all)
curr_val, cost_step = carry0
V, v, dJ, dj = curr_val.V, curr_val.v, cost_step.V, cost_step.v

Hxx = symmetrise_matrix(Q + AT @ V @ A)
Huu = symmetrise_matrix(R + BT @ V @ B)
Hxu = S + AT @ V @ B
hx = q + AT @ (v + V @ a)
hu = r + BT @ (v + V @ a)

I_mu = 1e-7 * jnp.eye(sys_dims.m)

# solve with cholesky
c, lower = cho_factor(Huu + I_mu)

Huu_inv = cho_solve((c, lower), jnp.eye(sys_dims.m))

K_c = -cho_solve((c, lower), Hxu.T)
K = -solve(Huu + I_mu, Hxu.T, assume_a="her")

K, k = jnp.hsplit(
    -solve(Huu + I_mu, jnp.c_[Hxu.T, hu], assume_a="her"), [sys_dims.n]
)
k = k.squeeze()

V_curr = symmetrise_matrix(Hxx + Hxu @ K + K.T @ Hxu.T + K.T @ Huu @ K)
v_curr = hx + (K.T @ Huu @ k) + (K.T @ hu) + (Hxu @ k)

dJ = dJ + 0.5 * (k.T @ Huu @ k).squeeze()
dj = dj + (k.T @ hu).squeeze()

carry1=(CostToGo(V_curr, v_curr), CostToGo(dJ, dj))
st1=(Gains(K, k), CostToGo(V_curr, v_curr))