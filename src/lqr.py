"""LQR solver via dynamic programming"""
from typing import Callable, NamedTuple, Tuple, Union
import jax
import jax.lax as lax
import jax.numpy as np
import jax.random as jr

# from src import keygen, initialise_stable_dynamics

jax.config.update("jax_enable_x64", True)  # double precision

# symmetrise
symmetrise_tensor = lambda x: (x + x.transpose(0, 2, 1)) / 2
symmetrise_matrix = lambda x: (x + x.T) / 2


# LQR struct
LQRBackParams = Tuple[
    np.ndarray, np.ndarray, np.array, np.ndarray, np.array, np.ndarray, np.array
]
LQRTrackParams = Tuple[
    np.ndarray,
    np.ndarray,
    np.array,
    np.ndarray,
    np.array,
    np.ndarray,
    np.array,
    np.array,
    np.array,
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
        NamedTuple (np.ndarray): Dynamics and Cost parameters. Shape [T,X,Y]
    """

    A: np.ndarray
    B: np.ndarray
    a: np.ndarray
    Q: np.ndarray
    q: np.ndarray
    Qf: np.ndarray
    qf: np.ndarray
    R: np.ndarray
    r: np.ndarray
    S: np.ndarray

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

    x0: np.ndarray
    lqr: Union[LQR, Tuple[np.ndarray]]


class Gains(NamedTuple):
    """Linear input gains"""

    K: np.ndarray
    k: np.ndarray


class CostToGo(NamedTuple):
    """Cost-to-go"""

    V: np.ndarray
    v: np.ndarray


# simulate trajectory
def simulate_trajectory(
    dynamics: Callable, Us: np.ndarray, params: Params, dims: ModelDims
) -> np.ndarray:
    """Simulate forward pass with LQR params"""
    x0, lqr = params.x0, params[-1]

    def step(x, inputs):
        t, u = inputs
        # TODO: add dims.dt to dynamics
        nx = dynamics(t, x, u, lqr)
        return nx, nx

    _, Xs = lax.scan(step, x0, (np.arange(dims.horizon), Us))

    return np.vstack([x0[None], Xs])


def lin_dyn_step(t: int, x: np.array, u: np.array, lqr: LQR) -> np.array:
    """State space linear step"""
    nx = lqr.A[t] @ x + lqr.B[t] @ u + lqr.a[t]
    return nx


def lqr_adjoint_pass(Xs: np.ndarray, Us: np.ndarray, params: Params) -> np.ndarray:
    """Adjoint backward pass with LQR params"""
    x0, lqr = params.x0, params[1]
    AT = lqr.A.transpose(0, 2, 1)
    lambf = lqr.Qf @ Xs[-1]

    def adjoint_step(lamb, inputs):
        x, u, aT, Q, q, S = inputs
        nlamb = aT @ lamb + Q @ x + q + S @ u
        return nlamb, nlamb

    _, lambs = lax.scan(
        adjoint_step, lambf, (Xs[:-1], Us[:], AT, lqr.Q, lqr.q, lqr.S), reverse=True
    )
    return np.vstack([lambs,lambf[None]])


def lqr_forward_pass(gains: Gains, params: Params) -> Tuple[np.ndarray, np.ndarray]:
    """LQR forward pass using gain state feedback"""
    x0, lqr = params.x0, params.lqr

    def dynamics(x: np.array, params: LQRBackParams):
        # TODO: add dims.dt to dynamics
        A, B, a, K, k = params
        u = K @ x + k
        nx = A @ x + B @ u + a
        return nx, (nx, u)

    xf, (Xs, Us) = lax.scan(
        dynamics, init=x0, xs=(lqr.A, lqr.B, lqr.a, gains.K, gains.k)
    )

    return np.vstack([x0[None], Xs]), Us


def lqr_tracking_forward_pass(
    gains: Gains, params: Params, Xs_star: np.ndarray, Us_star: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """LQR forward pass tracking using gain state feedback on state-input deviations"""
    x0, lqr = params.x0, params.lqr
    dx0 = x0 - Xs_star[0]

    def dynamics(x: np.array, params: LQRTrackParams):
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

    return np.vstack([x0[None], Xs]), Us


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
        min_eval = np.linalg.eigh(Huu)[0][0]
        I_mu = np.maximum(0.0, 1e-6 - min_eval) * np.eye(dims.m)

        # solve gains
        K = -np.linalg.solve(Huu + I_mu, Hxu.T)
        k = -np.linalg.solve(Huu + I_mu, hu)

        if verbose:
            assert I_mu.shape == (dims.m, dims.m)
            assert v.shape == (dims.n, 1)
            assert V.shape == (dims.n, dims.n)
            assert Hxx.shape == (dims.n, dims.n)
            assert Huu.shape == (dims.m, dims.m)
            assert Hxu.shape == (dims.n, dims.m)
            assert hx.shape == (dims.n, 1)
            assert hu.shape == (dims.m, 1)
            assert k.shape == (dims.m, 1)
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
        xs=np.arange(dims.horizon),
        reverse=True,
    )
    
    if verbose:
        assert not np.any(np.isnan(Ks.K))
        assert not np.any(np.isnan(Ks.k))
    
    if not expected_change:
        return dJ, Ks

    return (dJ, Ks), calc_expected_change(dJ=dJ)


def kkt(params: Params, Xs: np.ndarray, Us: np.ndarray, Lambs: np.ndarray):
    """Define KKT conditions for LQR problem"""
    AT = params.lqr.A.transpose(0, 2, 1)
    BT = params.lqr.B.transpose(0, 2, 1)
    ST = params.lqr.S.transpose(0, 2, 1)
    dLdXs = (
        np.matmul(params.lqr.Q, Xs[:-1])
        + np.matmul(params.lqr.S, Us[:])
        + params.lqr.q
        + np.matmul(AT, Lambs[1:])
        - Lambs[:-1]
    )
    dLdXf = np.matmul(params.lqr.Qf, Xs[-1]) + params.lqr.qf - Lambs[-1]
    dLdXs = np.concatenate([dLdXs, dLdXf[None]])
    dLdUs = (
        np.matmul(ST, Xs[:-1])
        + np.matmul(params.lqr.R, Us[:])
        + params.lqr.r
        + np.matmul(BT, Lambs[1:])
    )
    dLdLambs = (
        np.matmul(params.lqr.A, Xs[:-1])
        + np.matmul(params.lqr.B, Us[:])
        + params.lqr.a
        - Xs[1:]
    )
    dLdLamb0 = params.x0 - Xs[0]
    dLdLambs = np.concatenate([dLdLamb0[None], dLdLambs])
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


def initialise_lqr(sys_dims: ModelDims, spectral_radius: float = 0.6, 
                   pen_weight: dict = {"Q": 1e-0, "R": 1e-3, "Qf": 1e0, "S": 1e-3}):
    """Generate time-invariant LQR parameters"""
    # # generate random seeds
    # key = jr.PRNGKey(seed=234)
    # key, skeys = keygen(key, 3)
    # # initialise dynamics
    # span_time=(sys_dims.horizon, 1, 1)
    # A = initialise_stable_dynamics(next(skeys), sys_dims.n, sys_dims.horizon,radii=spectral_radius)
    # B = np.tile(jr.normal(next(skeys), (sys_dims.n, sys_dims.m)), span_time)
    # a = np.tile(jr.normal(next(skeys), (sys_dims.n, 1)), span_time)
    # # define cost matrices
    # Q = pen_weight["Q"] * np.tile(np.eye(sys_dims.n), span_time)
    # q = 2*1e-1 * np.tile(np.ones((sys_dims.n,1)), span_time)
    # R = pen_weight["R"] * np.tile(np.eye(sys_dims.m), span_time)
    # r = 1e-6 * np.tile(np.ones((sys_dims.m,1)), span_time)
    # S = pen_weight["S"] * np.tile(np.ones((sys_dims.n,sys_dims.m)), span_time)
    # Qf = pen_weight["Q"] * np.eye(sys_dims.n)
    # qf = 2*1e-1 * np.ones((sys_dims.n,1))
    # # construct LQR
    # lqr = LQR(A, B, a, Q, q, Qf, qf, R, r, S)
    lqr = LQR(None, None, None, None, None, None, None, None, None, None)
    return lqr()


if __name__ == "__main__":
    # generate data
    sys_dims = ModelDims(n=3, m=2, horizon=60, dt=0.1)
    x0 = np.array([[2.0], [1.0], [1.0]])
    lqr = initialise_lqr(sys_dims=sys_dims, spectral_radius=0.6)
    params = Params(x0, lqr)
    Us = np.zeros((sys_dims.horizon,sys_dims.m, 1), dtype=float)
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
