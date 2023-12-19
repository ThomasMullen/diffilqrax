"""Generate matrix and system dimensions for testing"""
import pytest
import src.lqr as lqr
import jax
import jax.random as jr


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
    pass
