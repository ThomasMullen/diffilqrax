"""LQR solver via dynamic programming"""
from typing import Callable, NamedTuple, Tuple, Union
import jax
import jax.lax as lax
import jax.numpy as np

jax.config.update("jax_enable_x64", True)  # double precision

# symmetrise
symmetrise_tensor = lambda x: (x + x.transpose(0, 2, 1)) / 2
symmetrise_matrix = lambda x: (x + x.T) / 2


# LQR struct
LQRBackParams = Tuple[
    np.ndarray, 
    np.ndarray, 
    np.array, 
    np.ndarray, 
    np.array, 
    np.ndarray, 
    np.array
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


class ValueIter(NamedTuple):
    """Cost-to-go"""

    V: np.ndarray
    v: np.ndarray


# simulate trajectory
def simulate_trajectory(
    dynamics: Callable, Us: np.ndarray, params: Params
) -> np.ndarray:
    """Simulate forward pass with LQR params"""
    x0, lqr = params.x0, params[1]

    def step(x, u):
        nx = dynamics(x, u, lqr)
        return nx, nx

    return np.vstack([x0[None], lax.scan(step, x0, Us)[1]])


def lin_dyn_step(x: np.array, u: np.array, lqr: LQR) -> np.array:
    """State space linear step"""
    nx = lqr.A @ x + lqr.B @ u + lqr.a
    return nx


def lqr_adjoint_pass(Xs: np.ndarray, Us: np.ndarray, params: Params) -> np.ndarray:
    """Adjoint backward pass with LQR params"""
    x0, lqr = params.x0, params[1]
    AT = lqr.A.T
    lambf = lqr.Qf @ Xs[-1]

    def adjoint_step(lamb, inputs):
        x, u, aT = inputs
        nlamb = aT @ lamb + lqr.Q @ x + lqr.q + lqr.S @ u
        return nlamb, nlamb

    _, lambs = lax.scan(adjoint_step, lambf, (Xs[:-1], Us[:-1], AT), reverse=True)
    return np.vstack([lambs, lambf[None]])


def lqr_forward_pass(gains: Gains, params: Params) -> Tuple[np.ndarray, np.ndarray]:
    """LQR forward pass using gain state feedback"""
    x0, lqr = params.x0, params.lqr

    def dynamics(x: np.array, params: LQRBackParams):
        A, B, a, K, k = params
        u = K @ x + k
        nx = A @ x + B @ u + a
        return nx, (nx, u)

    xf, (Xs, Us) = lax.scan(
        dynamics, init=x0, xs=(lqr.A, lqr.B, lqr.a, gains.K, gains.k)
    )


def lqr_tracking_forward_pass(
    gains: Gains, params: Params, Xs_star: np.ndarray, Us_star: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """LQR forward pass tracking using gain state feedback on state-input deviations"""
    x0, lqr = params.x0, params.lqr

    def dynamics(x: np.array, params: LQRTrackParams):
        A, B, a, K, k, x_star, u_star = params
        δx = x - x_star
        δu = K @ δx + k
        u_hat = u_star + δu
        nx = A @ x + B @ u_hat + a
        return nx, (nx, u_hat)

    xf, (Xs, Us) = lax.scan(
        dynamics, init=x0, xs=(lqr.A, lqr.B, lqr.a, gains.K, gains.k, Xs_star, Us_star)
    )


def calc_expected_change(alpha: float, dJ: ValueIter):
    return dJ.V * alpha**2 + dJ.v * alpha


def lqr_backward_pass(
    lqr: LQR,
    T: int,
    expected_change: bool = False,
    verbose: bool = False,
) -> Gains:
    I_mu = np.eye(lqr.R.shape[-1]) * 1e-8
    AT, BT = symmetrise_tensor(lqr.A), symmetrise_tensor(lqr.B)

    def riccati_step(
        carry: Tuple[ValueIter, ValueIter], t: int
    ) -> Tuple[ValueIter, Gains]:
        curr_val, cost_step = carry
        V, v, dJ, dj = curr_val.V, curr_val.v, cost_step.V, cost_step.v
        Hxx = symmetrise_matrix(lqr.Q[t] + AT[t] @ V @ lqr.A[t])
        Huu = symmetrise_matrix(lqr.R[t] + BT[t] @ V @ lqr.B[t])
        Hxu = lqr.S[t] + AT[t] @ V @ lqr.B[t]
        hx = lqr.q[t] + AT[t] @ (v + V @ lqr.a[t])
        hu = lqr.r[t] + BT[t] @ (v + V @ lqr.a[t])

        # solve gains
        # With Levenberg-Marquardt regulisation
        K = -np.linalg.solve(Huu + I_mu, Hxu.T)
        k = -np.linalg.solve(Huu + I_mu, hu)

        if verbose:
            print("I_mu", I_mu.shape, "v", v.shape, "V", V.shape)
            print(
                "Hxx",
                Hxx.shape,
                "Huu",
                Huu.shape,
                "Hxu",
                Hxu.shape,
                "hx",
                hx.shape,
                "hu",
                hu.shape,
            )
            print("k", k.shape, "K", K.shape)

        # Find value iteration at current time
        V_curr = symmetrise_matrix(Hxx + Hxu @ K + K.T @ Hxu.T + K.T @ Huu @ K)
        v_curr = hx + (K.T @ Huu @ k) + (K.T @ hu) + (Hxu @ k)

        # expected change in cost
        dJ = dJ + 0.5 * (k.T @ Huu @ k).squeeze()
        dj = dj + (k.T @ hu).squeeze()

        return (ValueIter(V_curr, v_curr), ValueIter(dJ, dj)), Gains(K, k)

    (V_0, dJ), Ks = lax.scan(
        riccati_step,
        init=(ValueIter(lqr.Qf, lqr.qf), (ValueIter(0.0, 0.0))),
        xs=np.arange(T),
        reverse=True,
    )
    if not expected_change:
        return dJ, Ks

    return (dJ, Ks), calc_expected_change(dJ=dJ)


# lqr solve
def solve_lqr(params: Params, horizon: int):
    "run backward forward sweep to find optimal control"
    # backward
    _, gains = lqr_backward_pass(params.lqr, horizon)
    # forward
    Xs, Us = lqr_forward_pass(gains, params)
    # adjoint
    Lambs = lqr_adjoint_pass(Xs, Us, params)
    return gains, Xs, Us, Lambs
