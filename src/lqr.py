"""LQR solver via dynamic programming"""
from typing import Callable, NamedTuple
import jax
import jax.lax as lax
import jax.numpy as np

# symmetrise
symmetrise = lambda x: (x + x.T) / 2


# LQR struct
class LQR(NamedTuple):
    pass


# forward pass


# backward pass


# lqr solve


if __name__ == "__main__":
    pass
