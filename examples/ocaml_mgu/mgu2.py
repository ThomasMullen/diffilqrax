"""
MGU2 latent dynamics, Student-t input prior and Gaussian observation model of the OCaml iLQR-VAE.

Conventions follow the OCaml model with column-vector states:

* ``x_0 = 0`` and ``x_{k+1} = dynamics(k, x_k, u_k)`` for ``k = 0 ... T - 1``;
* for ``k < n_beg`` the dynamics only load the initial condition, ``x_{k+1} = x_k + beg_bs[k].T @ u_k``;
* afterwards ``f = sigmoid(bf + uf.T @ x)``, ``h = phi(bh + uh.T @ (x * f)) + B.T @ u`` and
  ``x' = (1 - f) * x + f * h``, with ``B`` the row-normalised ``wh`` scaled by ``sqrt(n / m)``;
* ``phi(a) = requad(a) - 1`` with ``requad(a) = (a + sqrt(a^2 + 4)) / 2``. The run directory is
  suffixed ``_tanh`` but its saved trajectories are only consistent with requad (one-step residual
  1e-3 vs 6 for tanh);
* observation ``j`` (``j = 0 ... T - n_beg``) is read out from ``x_{j + n_beg}``,
  ``o_j ~ N(bias + c @ x_{j + n_beg}, diag(variances))``. The saved latents ``z`` are ``x_1 ... x_T``,
  so observation ``j`` corresponds to ``z[j + n_beg - 1]``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as onp
from jax import lax
from jax.scipy.special import gammaln

from diffilqrax.typs import ModelDims, System
from load_ocaml import OCamlParams


class MGU2Theta(NamedTuple):
    """MGU2 dynamics, prior and likelihood parameters (``iLQRParams.theta``)"""

    B: jnp.ndarray  # (m, n) normalised input weights
    uh: jnp.ndarray  # (n, n)
    uf: jnp.ndarray  # (n, n)
    bh: jnp.ndarray  # (n,)
    bf: jnp.ndarray  # (n,)
    beg_bs: jnp.ndarray  # (n_beg, m, n) initial-condition loading matrices
    c: jnp.ndarray  # (n_out, n)
    bias: jnp.ndarray  # (n_out,)
    variances: jnp.ndarray  # (n_out,)
    sigma: jnp.ndarray  # (m,) effective Student-t scale
    sigma0: jnp.ndarray  # (m,) Gaussian scale of the first n_beg inputs
    nu: jnp.ndarray  # () Student-t degrees of freedom
    obs: jnp.ndarray  # (T - n_beg + 1, n_out) observed outputs of one trial


def beg_matrices(n_beg: int, m: int, n: int) -> onp.ndarray:
    """``beg_bs[k][i, j] = 1`` if ``j // n_beg == k`` and ``j % m == i`` (as in the OCaml wrapper)"""
    k, i, j = onp.meshgrid(onp.arange(n_beg), onp.arange(m), onp.arange(n), indexing="ij")
    return ((j // n_beg == k) & (j % m == i)).astype(onp.float64)


def make_theta(p: OCamlParams, obs: onp.ndarray, n_beg: int) -> MGU2Theta:
    """Build ``MGU2Theta`` from OCaml parameters and the observations of one trial"""
    m, n = p.wh.shape
    B = onp.sqrt(n / m) * p.wh / onp.sqrt(onp.sum(p.wh**2, axis=1, keepdims=True))
    sigma = onp.sqrt((p.nu - 2.0) / p.nu) * p.spatial_stds
    return MGU2Theta(
        B=jnp.asarray(B),
        uh=jnp.asarray(p.uh),
        uf=jnp.asarray(p.uf),
        bh=jnp.asarray(p.bh),
        bf=jnp.asarray(p.bf),
        beg_bs=jnp.asarray(beg_matrices(n_beg, m, n)),
        c=jnp.asarray(p.c),
        bias=jnp.asarray(p.bias),
        variances=jnp.asarray(p.variances),
        sigma=jnp.asarray(sigma),
        sigma0=jnp.asarray(p.first_step),
        nu=jnp.asarray(p.nu),
        obs=jnp.asarray(obs),
    )


def requad(a: jnp.ndarray) -> jnp.ndarray:
    """Smooth rectifier ``(a + sqrt(a^2 + 4)) / 2``"""
    return 0.5 * (a + jnp.sqrt(a**2 + 4.0))


def phi(a: jnp.ndarray) -> jnp.ndarray:
    """MGU2 candidate nonlinearity"""
    return requad(a) - 1.0


def n_beg_of(theta: MGU2Theta) -> int:
    """Number of initial-condition steps (static)"""
    return theta.beg_bs.shape[0]


def dynamics(t: int, x: jnp.ndarray, u: jnp.ndarray, theta: MGU2Theta) -> jnp.ndarray:
    """MGU2 step ``x_{t+1} = F(t, x_t, u_t)``"""
    n_beg = n_beg_of(theta)

    def beg(_):
        return x + theta.beg_bs[jnp.minimum(t, n_beg - 1)].T @ u

    def mgu(_):
        f = jax.nn.sigmoid(theta.bf + theta.uf.T @ x)
        h = phi(theta.bh + theta.uh.T @ (x * f)) + theta.B.T @ u
        return (1.0 - f) * x + f * h

    return lax.cond(t < n_beg, beg, mgu, None)


def rollout(theta: MGU2Theta, us: jnp.ndarray) -> jnp.ndarray:
    """Latents ``x_1 ... x_T`` (T, n) from inputs ``us`` (T, m), starting at ``x_0 = 0``"""
    x0 = jnp.zeros(theta.uh.shape[0], dtype=us.dtype)

    def step(x, tu):
        t, u = tu
        nx = dynamics(t, x, u, theta)
        return nx, nx

    return lax.scan(step, x0, (jnp.arange(us.shape[0]), us))[1]


def readout(theta: MGU2Theta, zs: jnp.ndarray) -> jnp.ndarray:
    """Noiseless outputs ``bias + c @ z`` for latents ``zs`` (..., n)"""
    return theta.bias + zs @ theta.c.T


def prior_cost(t: int, u: jnp.ndarray, theta: MGU2Theta) -> jnp.ndarray:
    """Negative log prior of ``u_t``: Gaussian for ``t < n_beg``, multivariate Student-t afterwards"""
    m = u.shape[0]
    gauss = 0.5 * (m * jnp.log(2 * jnp.pi) + 2 * jnp.sum(jnp.log(theta.sigma0)) + jnp.sum((u / theta.sigma0) ** 2))
    nu = theta.nu
    cst = gammaln(0.5 * nu) - gammaln(0.5 * (nu + m)) + 0.5 * m * jnp.log(jnp.pi * nu) + jnp.sum(jnp.log(theta.sigma))
    student = cst + 0.5 * (nu + m) * jnp.log1p(jnp.sum((u / theta.sigma) ** 2) / nu)
    return jnp.where(t < n_beg_of(theta), gauss, student)


def gaussian_prior_cost(t: int, u: jnp.ndarray, theta: MGU2Theta) -> jnp.ndarray:
    """Quadratic proxy: Gaussian prior with scale ``sigma0`` for ``t < n_beg`` and ``sigma`` afterwards"""
    m = u.shape[0]
    s = jnp.where(t < n_beg_of(theta), theta.sigma0, theta.sigma)
    return 0.5 * (m * jnp.log(2 * jnp.pi) + 2 * jnp.sum(jnp.log(s)) + jnp.sum((u / s) ** 2))


def likelihood_cost(t: int, x: jnp.ndarray, theta: MGU2Theta) -> jnp.ndarray:
    """Gaussian negative log likelihood of observation ``t - n_beg`` given ``x_t`` (zero for ``t < n_beg``)"""
    n_beg = n_beg_of(theta)
    j = jnp.clip(t - n_beg, 0, theta.obs.shape[0] - 1)
    r = theta.obs[j] - (theta.bias + theta.c @ x)
    n_out = r.shape[0]
    nll = 0.5 * (n_out * jnp.log(2 * jnp.pi) + jnp.sum(jnp.log(theta.variances)) + jnp.sum(r**2 / theta.variances))
    return jnp.where(t < n_beg, 0.0, nll)


def make_system(n: int, m: int, horizon: int, proxy: bool = False) -> System:
    """
    iLQR problem whose optimum is the OCaml posterior mean of the inputs.

    ``cost(t, x_t, u_t)`` is the prior on ``u_t`` plus the likelihood of the observation read out from
    ``x_t``, summed over ``t = 0 ... horizon - 1``; the final cost is zero. With ``proxy=True`` the
    Student-t prior is replaced by a Gaussian with the same scale.
    """
    prior = gaussian_prior_cost if proxy else prior_cost

    def cost(t, x, u, theta):
        return prior(t, u, theta) + likelihood_cost(t, x, theta)

    def costf(x, theta):
        return jnp.zeros((), dtype=x.dtype)

    return System(cost, costf, dynamics, ModelDims(n=n, m=m, horizon=horizon, dt=1.0))


def total_cost(theta: MGU2Theta, us: jnp.ndarray, proxy: bool = False) -> jnp.ndarray:
    """Objective of :func:`make_system` evaluated on inputs ``us`` (T, m)"""
    prior = gaussian_prior_cost if proxy else prior_cost
    xs = jnp.vstack([jnp.zeros((1, theta.uh.shape[0]), dtype=us.dtype), rollout(theta, us)[:-1]])
    ts = jnp.arange(us.shape[0])
    return jnp.sum(jax.vmap(lambda t, x, u: prior(t, u, theta) + likelihood_cost(t, x, theta))(ts, xs, us))
