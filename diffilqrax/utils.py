"""Includes utility functions for the project. Generic functions to generate data, seeds, etc."""

from typing import Callable, Tuple, Any
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp


def keygen(key, nkeys):
    """Generate randomness that JAX can use by splitting the JAX keys.

    Args:
    key : the random.PRNGKey for JAX
    nkeys : how many keys in key generator

    Returns:
    2-tuple (new key for further generators, key generator)
    """
    keys = jr.split(key, nkeys + 1)
    return keys[0], (k for k in keys[1:])


def initialise_stable_dynamics(
    key: Tuple[int, int], n_dim: int, T: int, radii: float = 0.6
) -> Array:
    """Generate a state matrix with stable dynamics (eigenvalues < 1)

    Args:
        key (Tuple[int,int]): random key
        n_dim (int): state dimensions
        radii (float, optional): spectral radius. Defaults to 0.6.

    Returns:
        Array: matrix A with stable dynamics.
    """
    mat = jr.normal(key, (n_dim, n_dim)) * radii
    mat /= jnp.sqrt(n_dim)
    mat -= jnp.eye(n_dim)
    return jnp.tile(mat, (T, 1, 1))


def initialise_stable_time_varying_dynamics(
    key: Tuple[int, int], n_dim: int, T: int, radii: float = 0.6
) -> Array:
    """Generate a state matrix with stable dynamics (eigenvalues < 1)

    Args:
        key (Tuple[int,int]): random key
        n_dim (int): state dimensions
        radii (float, optional): spectral radius. Defaults to 0.6.

    Returns:
        Array: matrix A with stable dynamics.
    """
    mat = jr.normal(key, (T, n_dim, n_dim)) * radii
    mat /= jnp.sqrt(n_dim)
    mat -= jnp.eye(n_dim)
    return mat



def linearise(fun: Callable) -> Callable:
    """Function that finds jacobian w.r.t to x and u inputs.

    Args:
        fun (Callable): args (t, x, u, params)

    Returns:
        Callable[[Callable], Callable]): Jacobian tuple evaluated at args 1 and 2
    """
    return jax.jacrev(fun, argnums=(1, 2))


def quadratise(fun: Callable) -> Callable:
    """Function that finds Hessian w.r.t to x and u inputs.

    Args:
        fun (Callable): args (t, x, u, params)

    Returns:
        Tuple([NDARRAY, NDARRAY]): Hessian tuple cross evaluated with args 1 and 2
    """
    return jax.jacfwd(jax.jacrev(fun, argnums=(1, 2)), argnums=(1, 2))


def time_map(fun: Callable) -> Callable:
    """Vectorise function in time. Assumes 0th-axis is time for x and u args of fun, the last
    arg (theta) of Callable function assumed to be time-invariant.

    Args:
        fun (Callable): function that takes args (t, x[Txn], u[Txm], theta)

    Returns:
        Callable: vectorised function along args 1 and 2 0th-axis
    """
    return jax.vmap(fun, in_axes=(0, 0, 0, None))
