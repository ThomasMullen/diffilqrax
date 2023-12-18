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

class State(NamedTuple):
    """Cost-to-go and current cost """

    V: np.ndarray
    v: np.ndarray
    C: np.ndarray
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
def riccati_step(lqr: LQR, state:State):
    V, v, Ct, c = state.V, state.v, state.C, state.c
    AT, BT = lqr.A.T, lqr.B.T
    Hxx = symmetrise(lqr.Q + AT@V@lqr.A)
    Huu = symmetrise(lqr.R + BT@V@lqr.B)
    # NOTE: add noise to Huu to guarentee inverse
    Hxu = symmetrise(lqr.S + AT@V@lqr.B)
    hx = lqr.q + AT@(v + V@lqr.a)
    hu = lqr.r + BT@(v + V@lqr.a)
    
    # solve gains
    K = -np.linalg(Huu, Hxu.T)
    k = -np.linalg(Huu, hu)
    
    # Find value iteration at current time
    V_curr = Hxx + Hxu@K
    v_curr = hx + Hxu@k
    
    pass


# backward pass
def backward(
    lqr: LQR,
)->Gains:
    pass


# lqr solve


if __name__ == "__main__":
    pass
