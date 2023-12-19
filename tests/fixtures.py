"""Generate matrix and system dimensions for testing"""
import pytest
import src.lqr as lqr
import jax
import jax.random as jr
import jax.numpy as np

@pytest.fixture
def sys_seed():
    """Seed system"""
    return jr.PRNGkey(10)


@pytest.fixture
def sys_dims():
    """Generate system dimensions"""
    n = 2
    m = 1
    T = 20
    return n, m, T


@pytest.fixture
def sys_matrices():
    """Generate random LQR matrices"""
    n, m, T = sys_dims()
    k, sk = jr.split(sys_seed())
    
    A = jr.normal(sk, (T, n, n))
    k, sk = jr.split(sk)
    B = jr.normal(k, (T, n, m))
    k, sk = jr.split(sk)
    a = jr.normal(k, (T, n))
    Qf = np.eye(n)
    qf = 0.5 * np.ones(n)
    Q = np.tile(np.eye(n), (T, 1, 1))
    q = 0.5 * np.tile(np.ones(n), (T, 1))
    R = np.tile(np.eye(m), (T, 1, 1))
    r = 0.5 * np.tile(np.ones(m), (T, 1))
    S = 0.5 * np.tile(np.ones((n, m)), (T, 1, 1))
    
    lqr.LQR(A, B, a, Q, q, Qf, qf, R, r, S)
    
    pass
