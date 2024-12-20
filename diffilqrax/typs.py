"""Define data structures and types"""

from typing import NamedTuple, Callable, Any, Union, Tuple, Optional
from functools import partial
from jax import lax, Array
from jax.typing import ArrayLike
# from flax import struct
from diffilqrax.utils import linearise, quadratise


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


CostFn = Callable[[int, Array, Array, Optional[Any]], Array]
FCostFn = Callable[[Array, Optional[Any]], Array]
DynFn = Callable[[int, Array, Array, Optional[Any]], Array]
LinDynFn = Callable[[int, Array, Array, Optional[Any]], Tuple[Array, Array]]
LinCostFn = Callable[[int, Array, Array, Optional[Any]], Tuple[Array, Array]]
QuadCostFn = Callable[
    [int, Array, Array, Optional[Any]], Tuple[Tuple[Array, Array], Tuple[Array, Array]]
]


class System:
    """
    iLQR System

    Attributes
    ----------
    cost : Callable
        Running cost l(t, x, u, params).
    costf : Callable
        Final state cost lf(xf, params).
    dynamics : Callable
        Dynamical update f(t, x, u, params).
    dims : ModelDims
        iLQR evaluate time horizon, dt, state and input dimension.
    """

    def __init__(
        self,
        cost: CostFn,
        costf: FCostFn,
        dynamics: DynFn,
        dims: ModelDims,
        lin_dyn: Optional[LinDynFn] = None,
        lin_cost: Optional[LinCostFn] = None,
        quad_cost: Optional[QuadCostFn] = None,
    ):
        """
        System constructor.

        Parameters
        ----------
        cost : CostFn
            Non-linear cost function (t, x, u, params).
        costf : FCostFn
            Non-linear terminal cost function (xf, params).
        dynamics : DynFn
            Non-linear dynamics function (t, x, u, params).
        dims : ModelDims
            System dimensions e.g. n, m, horizon, dt.
        lin_dyn : Optional[LinDynFn], optional
            First order derivative of non-linear dynamics w.r.t. x. Defaults to None explicitly take fwd jac of dynamics.
        lin_cost : Optional[LinCostFn], optional
            First order derivative of non-linear cost fn w.r.t x and u. Defaults to None, take fwd jac of cost.
        quad_cost : Optional[QuadCostFn], optional
            Second order derivative of cost fn w.r.t (xx, xu), (ux, uu). Defaults to None take hessian of cost fn.
        """
        self.cost = cost
        self.costf = costf
        self.dynamics = dynamics
        self.dims = dims
        # self.lin_dyn = lax.cond(lin_dyn is None, lambda _: linearise(self.dynamics), lambda x: x)
        # self.lin_cost = lax.cond(lin_cost is None, lambda _: linearise(self.cost), lambda x: x)
        # self.quad_cost = lax.cond(quad_cost is None, lambda _: quadratise(self.cost), lambda x: x)
        self.lin_dyn = linearise(self.dynamics) if lin_dyn is None else lin_dyn
        self.lin_cost = linearise(self.cost) if lin_cost is None else lin_cost
        self.quad_cost = quadratise(self.cost) if quad_cost is None else quad_cost


class LQR(NamedTuple):
    """
    LQR params

    Attributes
    ----------
    A : Array
        Dynamics matrix.
    B : Array
        Input matrix.
    a : Array
        Offset vector.
    Q : Array
        State cost matrix.
    q : Array
        State cost vector.
    R : Array
        Input cost matrix.
    r : Array
        Input cost vector.
    S : Array
        Cross term matrix.
    Qf : Array
        Final state cost matrix.
    qf : Array
        Final state cost vector.
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
        """
        Symmetrise quadratic costs.

        Returns
        -------
        LQR
            Symmetrised LQR parameters.
        """
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
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
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


class ParallelSystem(NamedTuple):
    """
    Overloaded ilqr system for parallel dynamics.

    Attributes
    ----------
    model : System
        iLQR problem with non-linear cost, costf, dynamics, dims.
    parallel_dynamics : Callable
        Take ilqr System, iLQRParams, Us, Xs and return Xs.
    parallel_dynamics_feedback : Callable
        Take ilqr System, iLQRParams, Us, Xs, Kx and return Xs.
    """

    model: System
    parallel_dynamics: Callable[[System, iLQRParams, Array, Array], Array]
    parallel_dynamics_feedback: Callable[
        [System, iLQRParams, Array, Array, Array, Array], Array
    ]
