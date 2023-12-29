from typing import Callable, Any, Optional, NamedTuple
import jax
import jax.lax as lax
import jax.numpy as np

from . import LQR

# funct that linearises


# funct that quadratises


# struct: dynamics, cost, costf
class Systen:
    """iLQR System

    cost : Callable
        running cost l(t, x, u)
    costf : Callable
        final state cost lf(xf)
    dynamics : Callable
        dynamical update f(t, x, u, params)
    """

    cost: Callable[[int, np.ndarray, np.ndarray, Optional[Any]], np.ndarray]
    costf: Callable[[np.ndarray, Optional[Any]], np.ndarray]
    dynamics: Callable[[int, np.ndarray, np.ndarray, Optional[Any]], np.ndarray]


# recursive linesearch
