"""
MGU2 latent dynamics, Student-t input prior and Gaussian observation model of the OCaml iLQR-VAE, for the
parity tests against ``fixtures/ocaml_mgu.npz``.

* ``x_0 = 0`` and ``x_{k+1} = dynamics(k, x_k, u_k)`` for ``k = 0 ... T - 1``;
* for ``k < n_beg`` the dynamics only load the initial condition, ``x_{k+1} = x_k + beg_bs[k].T @ u_k``;
* afterwards ``f = sigmoid(bf + uf.T @ x)``, ``h = tanh(bh + uh.T @ (x * f)) + B.T @ u`` and
  ``x' = (1 - f) * x + f * h``, with ``B`` the row-normalised ``wh`` scaled by ``sqrt(n / m)``;
* observation ``j`` (``j = 0 ... T - n_beg``) is read out from ``x_{j + n_beg}``,
  ``o_j ~ N(bias + c @ x_{j + n_beg}, diag(variances))``.
"""

from pathlib import Path
from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from jax import lax
from jax.scipy.special import gammaln

from diffilqrax.typs import ModelDims, System


class OCamlParams(NamedTuple):
    """Trained generative parameters in OCaml (row-vector) orientation"""

    wh: onp.ndarray  # (m, n) input weights, normalised inside the dynamics
    uh: onp.ndarray  # (n, n) candidate recurrent weights
    uf: onp.ndarray  # (n, n) gate recurrent weights
    bh: onp.ndarray  # (n,)
    bf: onp.ndarray  # (n,)
    c: onp.ndarray  # (n_out, n) readout
    bias: onp.ndarray  # (n_out,)
    variances: onp.ndarray  # (n_out,)
    spatial_stds: onp.ndarray  # (m,)
    first_step: onp.ndarray  # (m,)
    nu: float


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


def load_fixture(path: Path) -> Tuple[OCamlParams, Dict[str, onp.ndarray]]:
    """Parameters and reference arrays of ``ocaml_mgu.npz``"""
    data = dict(onp.load(path))
    params = OCamlParams(
        **{k[len("param_") :]: (float(v) if k == "param_nu" else v) for k, v in data.items() if k.startswith("param_")}
    )
    return params, data


def make_theta(p: OCamlParams, obs: onp.ndarray, n_beg: int) -> MGU2Theta:
    """Build ``MGU2Theta`` from OCaml parameters and the observations of one trial"""
    m, n = p.wh.shape
    B = onp.sqrt(n / m) * p.wh / onp.sqrt(onp.sum(p.wh**2, axis=1, keepdims=True))
    # beg_bs[k][i, j] = 1 if j // n_beg == k and j % m == i (as in the OCaml wrapper)
    k, i, j = onp.meshgrid(onp.arange(n_beg), onp.arange(m), onp.arange(n), indexing="ij")
    beg_bs = ((j // n_beg == k) & (j % m == i)).astype(onp.float64)
    return MGU2Theta(
        B=jnp.asarray(B),
        uh=jnp.asarray(p.uh),
        uf=jnp.asarray(p.uf),
        bh=jnp.asarray(p.bh),
        bf=jnp.asarray(p.bf),
        beg_bs=jnp.asarray(beg_bs),
        c=jnp.asarray(p.c),
        bias=jnp.asarray(p.bias),
        variances=jnp.asarray(p.variances),
        sigma=jnp.asarray(onp.sqrt((p.nu - 2.0) / p.nu) * p.spatial_stds),
        sigma0=jnp.asarray(p.first_step),
        nu=jnp.asarray(p.nu),
        obs=jnp.asarray(obs),
    )


def dynamics(t: int, x: jnp.ndarray, u: jnp.ndarray, theta: MGU2Theta) -> jnp.ndarray:
    """MGU2 step ``x_{t+1} = F(t, x_t, u_t)``"""
    n_beg = theta.beg_bs.shape[0]

    def beg(_):
        return x + theta.beg_bs[jnp.minimum(t, n_beg - 1)].T @ u

    def mgu(_):
        f = jax.nn.sigmoid(theta.bf + theta.uf.T @ x)
        h = jnp.tanh(theta.bh + theta.uh.T @ (x * f)) + theta.B.T @ u
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
    return jnp.where(t < theta.beg_bs.shape[0], gauss, student)


def likelihood_cost(t: int, x: jnp.ndarray, theta: MGU2Theta) -> jnp.ndarray:
    """Gaussian negative log likelihood of observation ``t - n_beg`` given ``x_t`` (zero for ``t < n_beg``)"""
    n_beg = theta.beg_bs.shape[0]
    r = theta.obs[jnp.clip(t - n_beg, 0, theta.obs.shape[0] - 1)] - (theta.bias + theta.c @ x)
    nll = 0.5 * (r.shape[0] * jnp.log(2 * jnp.pi) + jnp.sum(jnp.log(theta.variances)) + jnp.sum(r**2 / theta.variances))
    return jnp.where(t < n_beg, 0.0, nll)


def make_system(n: int, m: int, horizon: int) -> System:
    """iLQR problem whose optimum is the OCaml posterior mean of the inputs (zero final cost)"""

    def cost(t, x, u, theta):
        return prior_cost(t, u, theta) + likelihood_cost(t, x, theta)

    def costf(x, theta):
        return jnp.zeros((), dtype=x.dtype)

    return System(cost, costf, dynamics, ModelDims(n=n, m=m, horizon=horizon, dt=1.0))


def total_cost(theta: MGU2Theta, us: jnp.ndarray) -> jnp.ndarray:
    """Objective of :func:`make_system` evaluated on inputs ``us`` (T, m)"""
    xs = jnp.vstack([jnp.zeros((1, theta.uh.shape[0]), dtype=us.dtype), rollout(theta, us)[:-1]])
    ts = jnp.arange(us.shape[0])
    return jnp.sum(jax.vmap(lambda t, x, u: prior_cost(t, u, theta) + likelihood_cost(t, x, theta))(ts, xs, us))
