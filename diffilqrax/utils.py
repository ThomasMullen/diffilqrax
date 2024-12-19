"""Includes utility functions for the project. Generic functions to generate data, seeds, etc."""

from typing import Callable, Tuple, Any
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp


def keygen(key, nkeys):
    """
    Generate randomness that JAX can use by splitting the JAX keys.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The random key for JAX.
    nkeys : int
        Number of keys to generate.

    Returns
    -------
    tuple
        A tuple containing the new key for further generators and the key generator.
    """
    keys = jr.split(key, nkeys + 1)
    return keys[0], (k for k in keys[1:])


def initialise_stable_dynamics(
    key: Tuple[int, int], n_dim: int, T: int, radii: float = 0.6
) -> Array:
    """
    Generate a state matrix with stable dynamics (|eigenvalues| < 1) discrete dynamics.

    Parameters
    ----------
    key : tuple of int
        Random key.
    n_dim : int
        State dimensions.
    T : int
        Time steps.
    radii : float, optional
        Spectral radius, by default 0.6.

    Returns
    -------
    Array
        Matrix A with stable dynamics.
    """
    mat = jr.normal(key, (n_dim, n_dim)) * radii
    mat /= jnp.sqrt(n_dim)
    return jnp.tile(mat, (T, 1, 1))


def initialise_stable_time_varying_dynamics(
    key: Tuple[int, int], n_dim: int, T: int, radii: float = 0.6
) -> Array:
    """
    Generate a state matrix with stable dynamics (|eigenvalues| < 1).

    Parameters
    ----------
    key : tuple of int
        Random key.
    n_dim : int
        State dimensions.
    T : int
        Time steps.
    radii : float, optional
        Spectral radius, by default 0.6.

    Returns
    -------
    Array
        Matrix A with stable dynamics.
    """
    mat = jr.normal(key, (T, n_dim, n_dim)) * radii
    mat /= jnp.sqrt(n_dim)
    # mat -= jnp.eye(n_dim)
    return mat


def linearise(fun: Callable) -> Callable:
    """
    Find Jacobian with respect to x and u inputs.

    Parameters
    ----------
    fun : Callable
        Function with arguments (t, x, u, params).

    Returns
    -------
    Callable
        Jacobian tuple evaluated at args 1 and 2.
    """
    return jax.jacrev(fun, argnums=(1, 2))


def quadratise(fun: Callable) -> Callable:
    """
    Find Hessian with respect to x and u inputs.

    Parameters
    ----------
    fun : Callable
        Function with arguments (t, x, u, params).

    Returns
    -------
    tuple
        Hessian tuple cross evaluated with args 1 and 2.
    """
    return jax.jacfwd(jax.jacrev(fun, argnums=(1, 2)), argnums=(1, 2))


def time_map(fun: Callable) -> Callable:
    """
    Vectorise function in time. Assumes 0th-axis is time for x and u args of fun, the last
    arg (theta) of Callable function assumed to be time-invariant.

    Parameters
    ----------
    fun : Callable
        Function that takes args (t, x[Txn], u[Txm], theta).

    Returns
    -------
    Callable
        Vectorised function along args 1 and 2 0th-axis.
    """
    return jax.vmap(fun, in_axes=(0, 0, 0, None))
