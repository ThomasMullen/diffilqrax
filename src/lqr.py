"""LQR solver via dynamic programming"""
from typing import Callable, NamedTuple, Tuple, Union
import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.lax import batch_matmul as bmm
import jax.random as jr
from functools import partial

from utils import keygen, initialise_stable_dynamics

jax.config.update("jax_enable_x64", True)  # double precision

# symmetrise
symmetrise_tensor = lambda x: (x + x.transpose(0, 2, 1)) / 2
symmetrise_matrix = lambda x: (x + x.T) / 2


# LQR struct
LQRBackParams = Tuple[
    jnp.ndarray, jnp.ndarray, jnp.array, jnp.ndarray, jnp.array, jnp.ndarray, jnp.array
]
LQRTrackParams = Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.array,
    jnp.ndarray,
    jnp.array,
    jnp.ndarray,
    jnp.array,
    jnp.array,
    jnp.array,
]


class ModelDims(NamedTuple):
    """Model dimensions"""

    n: int
    m: int
    horizon: int
    dt: float


class LQR(NamedTuple):
    """LQR params

    Args:
        NamedTuple (jnp.ndarray): Dynamics and Cost parameters. Shape [T,X,Y]
    """

    A: jnp.ndarray
    B: jnp.ndarray
    a: jnp.ndarray
    Q: jnp.ndarray
    q: jnp.ndarray
    Qf: jnp.ndarray
    qf: jnp.ndarray
    R: jnp.ndarray
    r: jnp.ndarray
    S: jnp.ndarray

    def __call__(self):
        """Symmetrise quadratic costs"""
        return LQR(
            A=self.A,
            B=self.B,
            a=self.a,
            Q=symmetrise_tensor(self.Q),
            q=self.q,
            Qf=(self.Qf + self.Qf.T) / 2,
            qf=self.qf,
            R=symmetrise_tensor(self.R),
            r=self.r,
            S=self.S,
        )


class Params(NamedTuple):
    """Contains initial states and LQR parameters"""

    x0: jnp.ndarray
    lqr: Union[LQR, Tuple[jnp.ndarray]]


class Gains(NamedTuple):
    """Linear input gains"""

    K: jnp.ndarray
    k: jnp.ndarray


class CostToGo(NamedTuple):
    """Cost-to-go"""

    V: jnp.ndarray
    v: jnp.ndarray


# simulate trajectory
def simulate_trajectory(
    dynamics: Callable, Us: jnp.ndarray, params: Params, dims: ModelDims
) -> jnp.ndarray:
    """Simulate forward pass with LQR params"""
    horizon = dims.horizon
    x0, lqr = params.x0, params[1]

    def step(x, inputs):
        t, u = inputs
        nx = dynamics(t, x, u, lqr)
        return nx, nx

    _, Xs = lax.scan(step, x0, (jnp.arange(horizon), Us))

    return jnp.vstack([x0[None], Xs])


def lin_dyn_step(t: int, x: jnp.array, u: jnp.array, lqr: LQR) -> jnp.array:
    """State space linear step"""
    nx = lqr.A[t] @ x + lqr.B[t] @ u + lqr.a[t]
    return nx


def lqr_adjoint_pass(Xs: jnp.ndarray, Us: jnp.ndarray, params: Params) -> jnp.ndarray:
    """Adjoint backward pass with LQR params"""
    x0, lqr = params.x0, params[1]
    AT = lqr.A.transpose(0, 2, 1)
    lambf = lqr.Qf @ Xs[-1] + lqr.qf

    def adjoint_step(lamb, inputs):
        x, u, aT, Q, q, S = inputs
        nlamb = aT @ lamb + Q @ x + q + S @ u
        return nlamb, nlamb

    _, lambs = lax.scan(
        adjoint_step, lambf, (Xs[:-1], Us[:], AT, lqr.Q, lqr.q, lqr.S), reverse=True
    )
    return jnp.vstack([lambs,lambf[None]])


def lqr_forward_pass(gains: Gains, params: Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """LQR forward pass using gain state feedback"""
    x0, lqr = params.x0, params.lqr

    def dynamics(x: jnp.array, params: LQRBackParams):
        # TODO: add dims.dt to dynamics
        A, B, a, K, k = params
        u = K @ x + k
        nx = A @ x + B @ u + a
        return nx, (nx, u)

    xf, (Xs, Us) = lax.scan(
        dynamics, init=x0, xs=(lqr.A, lqr.B, lqr.a, gains.K, gains.k)
    )

    return jnp.vstack([x0[None], Xs]), Us


def lqr_tracking_forward_pass(
    gains: Gains, params: Params, Xs_star: jnp.ndarray, Us_star: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """LQR forward pass tracking using gain state feedback on state-input deviations"""
    x0, lqr = params.x0, params.lqr
    dx0 = x0 - Xs_star[0]

    def dynamics(x: jnp.array, params: LQRTrackParams):
        A, B, a, K, k, x_star, u_star = params
        delta_x = x - x_star
        delta_u = K @ delta_x + k
        u_hat = u_star + delta_u
        nx = A @ x + B @ u_hat + a
        return nx, (nx, u_hat)

    xf, (Xs, Us) = lax.scan(
        dynamics,
        init=dx0,
        xs=(lqr.A, lqr.B, lqr.a, gains.K, gains.k, Xs_star[1:], Us_star),
    )

    return jnp.vstack([x0[None], Xs]), Us


def calc_expected_change(dJ: CostToGo, alpha: float = 0.5):
    return dJ.V * alpha**2 + dJ.v * alpha


def lqr_backward_pass(
    lqr: LQR,
    dims: ModelDims,
    expected_change: bool = False,
    verbose: bool = False,
) -> Gains:
    AT, BT = lqr.A.transpose(0, 2, 1), lqr.B.transpose(0, 2, 1)

    def riccati_step(
        carry: Tuple[CostToGo, CostToGo], t: int
    ) -> Tuple[CostToGo, Gains]:
        curr_val, cost_step = carry
        V, v, dJ, dj = curr_val.V, curr_val.v, cost_step.V, cost_step.v
        Hxx = symmetrise_matrix(lqr.Q[t] + AT[t] @ V @ lqr.A[t])
        Huu = symmetrise_matrix(lqr.R[t] + BT[t] @ V @ lqr.B[t])
        Hxu = lqr.S[t] + AT[t] @ V @ lqr.B[t]
        hx = lqr.q[t] + AT[t] @ (v + V @ lqr.a[t])
        hu = lqr.r[t] + BT[t] @ (v + V @ lqr.a[t])

        # With Levenberg-Marquardt regulisation
        min_eval = jnp.linalg.eigh(Huu)[0][0]
        I_mu = jnp.maximum(0.0, 1e-6 - min_eval) * jnp.eye(dims.m)

        # solve gains
        K = -jnp.linalg.solve(Huu + I_mu, Hxu.T)
        k = -jnp.linalg.solve(Huu + I_mu, hu)

        if verbose:
            assert I_mu.shape == (dims.m, dims.m)
            assert v.shape == (dims.n,)
            assert V.shape == (dims.n, dims.n)
            assert Hxx.shape == (dims.n, dims.n)
            assert Huu.shape == (dims.m, dims.m)
            assert Hxu.shape == (dims.n, dims.m)
            assert hx.shape == (dims.n,)
            assert hu.shape == (dims.m,)
            assert k.shape == (dims.m,)
            assert K.shape == (dims.m, dims.n)

        # Find value iteration at current time
        V_curr = symmetrise_matrix(Hxx + Hxu @ K + K.T @ Hxu.T + K.T @ Huu @ K)
        v_curr = hx + (K.T @ Huu @ k) + (K.T @ hu) + (Hxu @ k)

        # expected change in cost
        dJ = dJ + 0.5 * (k.T @ Huu @ k).squeeze()
        dj = dj + (k.T @ hu).squeeze()

        return (CostToGo(V_curr, v_curr), CostToGo(dJ, dj)), Gains(K, k)

    (V_0, dJ), Ks = lax.scan(
        riccati_step,
        init=(CostToGo(lqr.Qf, lqr.qf), (CostToGo(0.0, 0.0))),
        xs=jnp.arange(dims.horizon),
        reverse=True,
    )
    
    if verbose:
        assert not jnp.any(jnp.isnan(Ks.K))
        assert not jnp.any(jnp.isnan(Ks.k))
    
    if not expected_change:
        return dJ, Ks

    return (dJ, Ks), calc_expected_change(dJ=dJ)


def kkt(params: Params, Xs: jnp.ndarray, Us: jnp.ndarray, Lambs: jnp.ndarray):
    """Define KKT conditions for LQR problem"""
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
    dLdXf = bmm(params.lqr.Qf, Xs[-1]) + params.lqr.qf - Lambs[-1]
    dLdXs = jnp.concatenate([dLdXs, dLdXf[None]])
    dLdUs = (
        bmm(ST, Xs[:-1])
        + bmm(params.lqr.R, Us[:])
        + params.lqr.r
        + bmm(BT, Lambs[1:])
    )
    dLdLambs = (
        bmm(params.lqr.A, Xs[:-1])
        + bmm(params.lqr.B, Us[:])
        + params.lqr.a
        - Xs[1:]
    )
    dLdLamb0 = params.x0 - Xs[0]
    dLdLambs = jnp.concatenate([dLdLamb0[None], dLdLambs])
    return dLdXs, dLdUs, dLdLambs


def solve_lqr(params: Params, sys_dims: ModelDims):
    "run backward forward sweep to find optimal control"
    # backward
    _, gains = lqr_backward_pass(params.lqr, sys_dims)
    # forward
    Xs, Us = lqr_forward_pass(gains, params)
    # adjoint
    Lambs = lqr_adjoint_pass(Xs, Us, params)
    return gains, Xs, Us, Lambs

def solve_lqr_swap_x0(params: Params, sys_dims: ModelDims):
    "run backward forward sweep to find optimal control"
    # backward
    #print("r", new_params.lqr.r[-10:])
    _, gains = lqr_backward_pass(params.lqr, sys_dims)
    #print("k", gains.k[-10:])
    new_params = Params(jnp.zeros_like(params.x0), params.lqr)
    Xs, Us = lqr_forward_pass(gains, new_params)
    # adjoint
    Lambs = lqr_adjoint_pass(Xs, Us, new_params)
    return gains, Xs, Us, Lambs

def initialise_lqr(sys_dims: ModelDims, spectral_radius: float = 0.6, 
                   pen_weight: dict = {"Q": 1e-0, "R": 1e-3, "Qf": 1e0, "S": 1e-3}):
    """Generate time-invariant LQR parameters"""
    # # generate random seeds
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    # initialise dynamics
    span_time_m=(sys_dims.horizon, 1, 1)
    span_time_v=(sys_dims.horizon, 1)
    A = initialise_stable_dynamics(next(skeys), sys_dims.n, sys_dims.horizon,radii=spectral_radius)
    B = jnp.tile(jr.normal(next(skeys), (sys_dims.n, sys_dims.m)), span_time_m)
    a = jnp.tile(jr.normal(next(skeys), (sys_dims.n,)), span_time_v)
    # define cost matrices
    Q = pen_weight["Q"] * jnp.tile(jnp.eye(sys_dims.n), span_time_m)
    q = 2*1e-1 * jnp.tile(jnp.ones((sys_dims.n,)), span_time_v)
    R = pen_weight["R"] * jnp.tile(jnp.eye(sys_dims.m), span_time_m)
    r = 1e-6 * jnp.tile(jnp.ones((sys_dims.m,)), span_time_v)
    S = pen_weight["S"] * jnp.tile(jnp.ones((sys_dims.n,sys_dims.m)), span_time_m)
    Qf = pen_weight["Q"] * jnp.eye(sys_dims.n)
    qf = 2*1e-1 * jnp.ones((sys_dims.n,))
    # construct LQR
    lqr = LQR(A, B, a, Q, q, Qf, qf, R, r, S)
    # lqr = LQR(None, None, None, None, None, None, None, None, None, None)
    return lqr()


if __name__ == "__main__":
    # generate data
    sys_dims = ModelDims(n=3, m=2, horizon=60, dt=0.1)
    x0 = jnp.array([2.0, 1.0, 1.0])
    lqr = initialise_lqr(sys_dims=sys_dims, spectral_radius=0.6)
    params = Params(x0, lqr)
    Us = jnp.zeros((sys_dims.horizon,sys_dims.m), dtype=float)
    Us = Us.at[2].set(1.0)

    # simulate trajectory
    Xs_sim = simulate_trajectory(dynamics=lin_dyn_step, Us=Us, params=params, dims=sys_dims)
    # generate adjoints
    Lambs = lqr_adjoint_pass(Xs_sim, Us, params)
    # LQR backward pass
    (dJ, Ks), exp_dJ = lqr_backward_pass(
        lqr=params.lqr, dims=sys_dims, expected_change=True, verbose=False
    )
    # LQR forward update
    Xs_lqr, Us_lqr = lqr_forward_pass(gains=Ks, params=params)

    # LQR solver
    gains_lqr, Xs_lqr, Us_lqr, Lambs_lqr = solve_lqr(params, sys_dims)
