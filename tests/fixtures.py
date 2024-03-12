import pytest
import jax
import jax.random as jr
import jax.numpy as np

from ..src import LQR

@pytest.fixture
def sys_seed():
    """Seed system"""
    return jr.PRNGKey(seed=234)


@pytest.fixture
def sys_dims():
    """Generate system dimensions"""
    n = 3
    m = 2
    T = 20
    return n, m, T

@pytest.fixture
def sys_matrices():
    """Generate random LQR matrices"""
    n, m, T = sys_dims()
    subkeys = jr.split(sys_seed(),11)
    
    A = jr.normal(subkeys[0], (T, n, n))
    B = jr.normal(subkeys[1], (T, n, m))
    a = jr.normal(subkeys[2], (T, n))
    Qf = np.eye(n)
    qf = 0.5 * np.ones(n)
    Q = np.tile(np.eye(n), (T, 1, 1))
    q = 0.5 * np.tile(np.ones(n), (T, 1))
    R = np.tile(np.eye(m), (T, 1, 1))
    r = 0.5 * np.tile(np.ones(m), (T, 1))
    S = 0.5 * np.tile(np.ones((n, m)), (T, 1, 1))
    
    return LQR(A, B, a, Q, q, Qf, qf, R, r, S)()