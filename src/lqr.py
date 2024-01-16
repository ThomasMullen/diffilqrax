"""LQR solver via dynamic programming"""
from typing import Callable, NamedTuple, Tuple
import jax
import jax.lax as lax
import jax.numpy as np

jax.config.update("jax_enable_x64", True)

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


class Params(NamedTuple):
    """Contains initial states and LQR parameters"""
    x0: np.ndarray
    lqr: LQR

class Gains(NamedTuple):
    """Linear input gains"""

    K: np.ndarray
    k: np.ndarray


class ValueIter(NamedTuple):
    """Cost-to-go"""

    V: np.ndarray
    v: np.ndarray
    
    
def rollout(dynamics: Callable, Us: np.ndarray, params: Params):
    """Simulate forward pass with LQR params"""
    x0, lqr = params.x0, params[1]
    def step(x, u):
        nx = dynamics(x, u, lqr)
        return nx, nx
    return np.vstack([x0[None], lax.scan(step, x0, Us)[1]])


# # step for forward rollout
# def linear_step(x, params):
#     A, B, a, K, k = params
#     u = K @ x + k
#     nx = A @ x + B @ u + a
#     return nx, (nx, u)

# # step for forward tracking we rollout
# def track_step(x, params):
#     A, B, a, K, k, x_star, u_star = params
#     δx = x - x_star
#     δu = K @ δx + k
#     u_hat = u_star + δu
#     nx = A @ x + B @ u_hat + a
#     return nx, (nx, u_hat)

# forward pass
def forward(gains: Gains, params: Params
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward iteration of LDS using gains"""
    x0, lqr = params.x0, params.lqr
    def dynamics(x, params):
        A, B, a, K, k = params
        u = K @ x + k
        nx = A @ x + B @ u + a
        return nx, (nx, u)

    xf, (Xs, Us) = lax.scan(dynamics, init=x0, xs=(lqr.A, lqr.B, lqr.a, gains.K, gains.k))
    return Xs, Us


# backward pass
def backward(
    lqr: LQR,
    T: int,
    expected_change: bool=False,
    verbose: bool=False,
) -> Gains:
    I_mu = np.eye(lqr.R.shape[-1])*1e-9
    def riccati_step(carry: Tuple[ValueIter, ValueIter], t: int) -> Tuple[ValueIter, Gains]:
        symmetrise = lambda x: (x + x.T) / 2
        curr_val, cost_step = carry
        V, v, dJ, dj = curr_val.V, curr_val.v, cost_step.V, cost_step.v
        AT, BT = lqr.A.transpose(0, 2, 1), lqr.B.transpose(0, 2, 1)
        Hxx = symmetrise(lqr.Q[t] + AT[t] @ V @ lqr.A[t])
        Huu = symmetrise(lqr.R[t] + BT[t] @ V @ lqr.B[t])
        Hxu = lqr.S[t] + AT[t] @ V @ lqr.B[t]
        hx = lqr.q[t] + AT[t] @ (v + V @ lqr.a[t])
        hu = lqr.r[t] + BT[t] @ (v + V @ lqr.a[t])

        # solve gains
        # With Levenberg-Marquardt regulisation
        K = -np.linalg.solve(Huu+I_mu, Hxu.T)
        k = -np.linalg.solve(Huu+I_mu, hu)

        if verbose:
            print("I_mu", I_mu.shape, "v",v.shape, "V",V.shape)
            print("Hxx",Hxx.shape, "Huu",Huu.shape, "Hxu",Hxu.shape, "hx",hx.shape, "hu",hu.shape)
            print("k",k.shape, "K",K.shape)
        
        # Find value iteration at current time
        V_curr = symmetrise(Hxx + Hxu @ K + K.T @ Hxu.T + K.T @ Huu @ K)
        v_curr = hx + (K.T @ Huu @ k) + (K.T @ hu) + (Hxu @ k)
        
        # expected change in cost
        dJ = dJ + 0.5*(k.T @ Huu @ k).squeeze()
        dj = dj + (k.T @ hu).squeeze()

        return (ValueIter(V_curr, v_curr), ValueIter(dJ, dj)), Gains(K, k)

    (V_0, dJ), Ks = lax.scan(
        riccati_step, init=(ValueIter(lqr.Qf, lqr.qf), (ValueIter(0., 0.))), xs=np.arange(T), reverse=True
    )
    if not expected_change:
        return dJ, Ks
        
    return (dJ, Ks), calc_expected_change(dJ=dJ)

def calc_expected_change(alpha:float, dJ: ValueIter):
    return dJ.V*alpha**2 + dJ.v*alpha


# lqr solve
def solve_lqr(params: Params, horizon: int):
    "run backward forward sweep to find optimal control"
    # backward
    _, gains = backward(params.lqr, horizon)
    # forward
    Xs, Us = forward(gains, params)
    return gains, Xs, Us


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
    x0 = np.array([[2.],[1.]])
    lqr = init_params()
    params = Params(x0, lqr)
    _, gains = backward(params.lqr, 20)
    Xs, Us = forward(gains, params)
