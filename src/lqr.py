"""LQR solver via dynamic programming"""
from typing import Callable, NamedTuple, Tuple
import jax
import jax.lax as lax
import jax.numpy as np

# symmetrise
symmetrise = lambda x: (x + x.transpose(0, 2, 1)) / 2


# LQR struct
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
            Q=symmetrise(self.Q),
            q=self.q,
            Qf=(self.Qf+self.Qf.T)/2,
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
    def dynamics(x, params):
        A, B, a, K, k = params
        u = K @ x + k
        nx = A @ x + B @ u + a
        return nx, (nx, u)

    xf, (Xs, Us) = lax.scan(dynamics, init=x_init, xs=(lqr.A, lqr.B, lqr.a, gains.K, gains.k))
    return Xs, Us


# backward pass
def backward(
    lqr: LQR,
    T: int,
) -> Gains:
    def riccati_step(carry: ValueIter, t: int) -> Tuple[ValueIter, Gains]:
        symmetrise = lambda x: (x + x.T) / 2
        V, v = carry.V, carry.v
        AT, BT = lqr.A.transpose(0, 2, 1), lqr.B.transpose(0, 2, 1)
        Hxx = symmetrise(lqr.Q[t] + AT[t] @ V @ lqr.A[t])
        Huu = symmetrise(lqr.R[t] + BT[t] @ V @ lqr.B[t])
        Hxu = lqr.S[t] + AT[t] @ V @ lqr.B[t]
        hx = lqr.q[t] + AT[t] @ (v + V @ lqr.a[t])
        hu = lqr.r[t] + BT[t] @ (v + V @ lqr.a[t])

        # solve gains
        K = -np.linalg.solve(Huu, Hxu.T)
        k = -np.linalg.solve(Huu, hu)

        # Find value iteration at current time
        V_curr = symmetrise(Hxx + Hxu @ K + K.T @ Hxu.T + K.T @ Huu @ K)
        v_curr = hx + Hxu @ k

        return ValueIter(V_curr, v_curr), Gains(K, k)

    V_0, Ks = lax.scan(
        riccati_step, init=ValueIter(lqr.Qf, lqr.qf), xs=np.arange(T), reverse=True
    )
    # return np.flip(Ks)
    return Ks


# lqr solve
def solve_lqr():
    # backward

    # forward
    pass


def init_params():
    k_spring=10
    k_damp=5
    m=10
    tps=20
    A = np.array([[0.,1.], [-k_spring/m, -k_damp/m]])
    B = np.array([[0.],[1.]])
    a = np.array([[0.],[0.]])
    
    Qf = np.eye(2)*10
    qf = np.array([[0.],[0.]])
    Q = np.eye(2)*10
    q = np.array([[0.],[0.]])
    R = np.eye(1)*0.1
    r = np.array([0.])
    S = np.zeros((2,1))
    
    lqr=LQR(
        A=np.tile(A,(tps,1,1)),
        B = np.tile(B,(tps,1,1)),
        a = np.tile(a,(tps,1,1)),
        Q=np.tile(Q,(tps,1,1)),
        q=np.tile(q,(tps,1,1)),
        Qf=Qf,
        qf=qf,
        R=np.tile(R,(tps,1,1)),
        r=np.tile(r,(tps,1)),
        S=np.tile(S,(tps,1,1)),
        )
    return lqr()

if __name__ == "__main__":
    # generate some dynamics
    x_init = np.array([[2.],[1.]])
    lqr = init_params()
    gains = backward(lqr, 20)
    Xs, Us = forward(lqr, gains, x_init)
    pass
