"""LQR solver via dynamic programming"""
from typing import Callable, NamedTuple, Tuple
import jax
import jax.lax as lax
import jax.numpy as np

jax.config.update("jax_enable_x64", True)  # double precision

# symmetrise
symmetrise_tensor = lambda x: (x + x.transpose(0, 2, 1)) / 2
symmetrise_matrix = lambda x: (x + x.T) / 2


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
    lqr: LQR


class Gains(NamedTuple):
    """Linear input gains"""

    K: np.ndarray
    k: np.ndarray


class ValueIter(NamedTuple):
    """Cost-to-go"""

    V: np.ndarray
    v: np.ndarray
