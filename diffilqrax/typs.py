"""Define data structures and types"""

from typing import NamedTuple, Callable, Any, Union, Tuple, Optional
from jax import Array
from jax.typing import ArrayLike

def symmetrise_tensor(x: Array) -> Array:
    """Symmetrise tensor"""
    assert x.ndim == 3
    return (x + x.transpose(0, 2, 1)) / 2


def symmetrise_matrix(x: Array) -> Array:
    """Symmetrise matrix"""
    assert x.ndim == 2
    return (x + x.T) / 2


class ModelDims(NamedTuple):
    """Model dimensions"""

    n: int
    m: int
    horizon: int
    dt: float


class Gains(NamedTuple):
    """Linear input gains"""

    K: ArrayLike
    k: ArrayLike


class CostToGo(NamedTuple):
    """Cost-to-go"""

    V: ArrayLike
    v: ArrayLike


class System(NamedTuple):
    """iLQR System

    cost : Callable
        running cost l(t, x, u, params)
    costf : Callable
        final state cost lf(xf, params)
    dynamics : Callable
        dynamical update f(t, x, u, params)
    dims : ModelDims
        ilQR evaluate time horizon, dt, state and input dimension
    """

    cost: Callable[[int, Array, Array, Optional[Any]], Array]
    costf: Callable[[Array, Optional[Any]], Array]
    dynamics: Callable[[int, ArrayLike, ArrayLike, Optional[Any]], Array]
    dims: ModelDims

class ParallelSystem(NamedTuple):
    """iLQR System

    cost : Callable
        running cost l(t, x, u, params)
    costf : Callable
        final state cost lf(xf, params)
    dynamics : Callable
        dynamical update f(t, x, u, params)
    dims : ModelDims
        ilQR evaluate time horizon, dt, state and input dimension
    """

    cost: Callable[[int, Array, Array, Optional[Any]], Array]
    costf: Callable[[Array, Optional[Any]], Array]
    dynamics: Callable[[int, ArrayLike, ArrayLike, Optional[Any]], Array]
    parallel_dynamics: Callable[[ArrayLike, Optional[Any]], Array]
    dims: ModelDims
    
    
class LQR(NamedTuple):
    """LQR params

    Args:
        NamedTuple (jnp.ndarray): Dynamics and Cost parameters. Shape [T,X,Y]
    """

    A: Array
    B: Array
    a: Array
    Q: Array
    q: Array
    R: Array
    r: Array
    S: Array
    Qf: Array
    qf: Array

    def __call__(self):
        """Symmetrise quadratic costs"""
        return LQR(
            A=self.A,
            B=self.B,
            a=self.a,
            Q=symmetrise_tensor(self.Q),
            q=self.q,
            R=symmetrise_tensor(self.R),
            r=self.r,
            S=self.S,
            Qf=(self.Qf + self.Qf.T) / 2,
            qf=self.qf,
        )


RiccatiStepParams = Tuple[
    ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike,
    ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike,
]


class LQRParams(NamedTuple):
    """Contains initial states and LQR parameters"""

    x0: ArrayLike
    lqr: Union[LQR, Tuple[ArrayLike]]


class iLQRParams(NamedTuple):
    """Non-linear parameter struct"""

    x0: ArrayLike
    theta: Any


class Theta(NamedTuple):
    """RNN parameters"""

    Uh: Array
    Wh: Array
    sigma: ArrayLike
    Q: Array


class Thetax0(NamedTuple):
    """RNN parameters and initial state"""

    Uh: Array
    Wh: Array
    sigma: ArrayLike
    Q: Array
    x0: Array


class PendulumParams(NamedTuple):
    """Pendulum parameters"""

    m: float
    l: float
    g: float
