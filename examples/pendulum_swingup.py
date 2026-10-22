"""
Minimal diffilqrax example: swing a torque-limited pendulum up from hanging down to upright.

Run from the repository root:
    python examples/pendulum_swingup.py

Prints the optimal cost, the number of iterations and the KKT residual, and saves a plot of the
trajectory to fig_dump/<timestamp>/pendulum_swingup.png.
"""

from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from diffilqrax.ilqr import ilqr_solver, approx_lqr, kkt_input_residual  # pylint: disable=wrong-import-position
from diffilqrax.typs import System, ModelDims, iLQRParams  # pylint: disable=wrong-import-position

jax.config.update("jax_enable_x64", True)

DT, GRAVITY, DAMPING, U_MAX, HORIZON = 0.05, 9.8, 0.1, 3.0, 100


def dynamics(t, x, u, theta):
    """Euler step of a unit pendulum; x = [angle, angular velocity], angle 0 is hanging down"""
    acc = -GRAVITY * jnp.sin(x[0]) - DAMPING * x[1] + u[0]
    return x + DT * jnp.stack([x[1], acc])


def cost(t, x, u, theta):
    """Small running cost on angle, velocity and torque, plus a smooth barrier at |u| = U_MAX"""
    barrier = jax.nn.softplus(5.0 * (u - U_MAX)) ** 2 + jax.nn.softplus(5.0 * (-u - U_MAX)) ** 2
    return 0.1 * (1.0 + jnp.cos(x[0])) + 0.01 * x[1] ** 2 + 0.01 * jnp.sum(u**2) + jnp.sum(barrier)


def final_cost(x, theta):
    """Penalise not being upright (angle pi) and still at the end"""
    return 50.0 * (1.0 + jnp.cos(x[0])) + 5.0 * x[1] ** 2


def main():
    model = System(cost, final_cost, dynamics, ModelDims(n=2, m=1, horizon=HORIZON, dt=DT))
    params = iLQRParams(x0=jnp.array([0.1, 0.0]), theta=None)
    Us_init = 0.5 * jr.normal(jr.PRNGKey(0), (HORIZON, 1))

    (Xs, Us, _), total_cost, cost_log = ilqr_solver(model, params, Us_init, max_iter=300)
    n_iters = int(jnp.sum(jnp.diff(cost_log) != 0))
    kkt = float(jnp.max(jnp.abs(kkt_input_residual(approx_lqr(model, Xs, Us, params)))))
    print(f"optimal cost {float(total_cost):.6f} after {n_iters} iterations, max|dL/dU| = {kkt:.1e}")
    print(f"final angle {float(Xs[-1, 0]):.3f} rad (upright is ±{jnp.pi:.3f}), max|u| = {float(jnp.abs(Us).max()):.2f}")

    out_dir = Path("fig_dump") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    time = jnp.arange(HORIZON + 1) * DT
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(time, Xs[:, 0])
    axes[0].axhline(-jnp.pi, ls=":", c="k")
    axes[0].axhline(jnp.pi, ls=":", c="k")
    axes[0].set(xlabel="time (s)", ylabel="angle (rad)")
    axes[1].plot(time[:-1], Us[:, 0])
    axes[1].axhline(U_MAX, ls=":", c="k")
    axes[1].axhline(-U_MAX, ls=":", c="k")
    axes[1].set(xlabel="time (s)", ylabel="torque")
    axes[2].semilogy(cost_log - cost_log.min() + 1e-12)
    axes[2].set(xlabel="iteration", ylabel="J - J*")
    fig.tight_layout()
    fig.savefig(out_dir / "pendulum_swingup.png", dpi=150)
    print(f"saved {out_dir / 'pendulum_swingup.png'}")


if __name__ == "__main__":
    main()
