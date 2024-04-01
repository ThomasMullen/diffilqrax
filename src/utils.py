"""Includes utility functions for the project. Generic functions to generate data, seeds, etc."""
from typing import Tuple
import jax.random as jr
import jax.numpy as jnp
from jax import Array


def keygen(key, nkeys):
    """Generate randomness that JAX can use by splitting the JAX keys.

    Args:
    key : the random.PRNGKey for JAX
    nkeys : how many keys in key generator

    Returns:
    2-tuple (new key for further generators, key generator)
    """
    keys = jr.split(key, nkeys+1)
    return keys[0], (k for k in keys[1:])


def initialise_stable_dynamics(key:Tuple[int,int], n_dim:int, T:int, radii:float=0.6)->Array:
    """Generate a state matrix with stable dynamics (eigenvalues < 1)

    Args:
        key (Tuple[int,int]): random key
        n_dim (int): state dimensions
        radii (float, optional): spectral radius. Defaults to 0.6.

    Returns:
        Array: matrix A with stable dynamics.
    """
    mat = jr.normal(key,(n_dim,n_dim))*radii
    mat /= jnp.sqrt(n_dim)
    mat -= jnp.eye(n_dim)
    return jnp.tile(mat,(T,1,1))


def initialise_stable_time_varying_dynamics(key:Tuple[int,int], n_dim:int, T:int, radii:float=0.6)->Array:
    """Generate a state matrix with stable dynamics (eigenvalues < 1)

    Args:
        key (Tuple[int,int]): random key
        n_dim (int): state dimensions
        radii (float, optional): spectral radius. Defaults to 0.6.

    Returns:
        Array: matrix A with stable dynamics.
    """
    mat = jr.normal(key,(T,n_dim,n_dim))*radii
    mat /= jnp.sqrt(n_dim)
    mat -= jnp.eye(n_dim)
    return mat


