"""LQR solver via dynamic programming"""
from typing import Callable, NamedTuple, Tuple
import jax
import jax.lax as lax
import jax.numpy as np

# symmetrise
symmetrise = lambda x: (x + x.T) / 2


# LQR struct
class LQR(NamedTuple):
    """LQR params

    Args:
        NamedTuple (np.ndarray): Dynamics and Cost parameters
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
            Q=symmetrise(self.Q),
            q=self.q,
            Qf=symmetrise(self.Qf),
            qf=self.qf,
            R=symmetrise(self.R),
            r=self.r,
            S=self.S,
        )


class Gains(NamedTuple):
    """LQR Gains

    Args:
        NamedTuple (np.ndarray): Linear input gains
    """

    K: np.ndarray
    k: np.ndarray


class ValueIter(NamedTuple):
    """Cost-to-go"""

    V: np.ndarray
    v: np.ndarray


# forward pass
def forward(
    lqr: LQR, gains: Gains, x_init: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward iteration of LDS using gains"""
    A, B, a = lqr.A, lqr.B, lqr.a

    def dynamics(x, params):
        A, B, a, gain = params
        u = gain.K @ x + gain.k
        nx = A @ x + B @ u + a
        return nx, (nx, u)

    xf, (Xs, Us) = lax.scan(dynamics, init=x_init, xs=(A, B, a, gains))
    return Xs, Us


# riccati step
def riccati_step(lqr: LQR, state: ValueIter) -> Tuple[ValueIter, Gains]:
    V, v = state.V, state.v
    AT, BT = lqr.A.T, lqr.B.T
    Hxx = symmetrise(lqr.Q + AT @ V @ lqr.A)
    # NOTE: add noise to Huu to guarentee inverse
    Huu = symmetrise(lqr.R + BT @ V @ lqr.B)
    Hxu = symmetrise(lqr.S + AT @ V @ lqr.B)
    hx = lqr.q + AT @ (v + V @ lqr.a)
    hu = lqr.r + BT @ (v + V @ lqr.a)

    # solve gains
    K = -np.linalg(Huu, Hxu.T)
    k = -np.linalg(Huu, hu)

    # Find value iteration at current time
    # V_curr = symmetrise(Hxx + Hxu@K + K.T@Hxu + K.T@Huu@K)    # for DDP
    # v_curr = hx + Hxu@k + K.T@hu + K.T @ Huu @ k              # for DDP
    V_curr = symmetrise(Hxx + Hxu @ K + K.T @ Hxu + K.T @ Huu @ K)
    v_curr = hx + Hxu @ k

    return ValueIter(V_curr, v_curr), Gains(K, k)


# backward pass
def backward(
    lqr: LQR,
) -> Gains:
    pass


# lqr solve


if __name__ == "__main__":
    pass
