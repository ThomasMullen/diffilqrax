"""
Cross-check the iLQR solver against reference solutions from the OCaml dilqr package.

The fixtures are produced by ``benchmarks/make_ocaml_fixtures.py`` from ``examples/pendulum_ref.ml``,
``examples/rnn_ref.ml`` and ``examples/rnn_grad_ref.ml`` in the dilqr repository. OCaml stops on a
1e-14 relative change in cost rather than on the KKT conditions, so trajectories are compared at the
accuracy that rule allows (1e-5) while optimal costs agree to round-off.

Implicit gradients are compared in two cases:

1. ``~fixed_trajectory:true`` (patched dilqr): the tape is linearised around the fixed optimal
   trajectory, so OCaml's gradients are the Gauss-Newton implicit gradients and must equal
   ``dilqr(exact_hessian=False)``.
2. dilqr as released: ``g1`` re-rolls out the states with the differentiable theta, so reverse-mode
   AD also differentiates through that rollout. These gradients are neither exact nor Gauss-Newton
   for non-linear dynamics; they are matched by ``ocaml_rollout_loss``, a test-only copy of that
   behaviour.
"""

import os
from pathlib import Path

import numpy as onp
import pytest
import chex
import jax
import jax.numpy as jnp

from jax import lax

from diffilqrax import ilqr
from diffilqrax.diff_ilqr import dilqr
from diffilqrax.diff_lqr import dllqr
from diffilqrax.typs import iLQRParams, LQRParams
from tests.problems import ocaml_pendulum, vanilla_rnn, vanilla_rnn_params

jax.config.update("jax_default_device", jax.devices(os.environ.get("DIFFILQRAX_TEST_PLATFORM", "cpu"))[0])
jax.config.update("jax_enable_x64", True)

FIXTURES = Path(__file__).parent / "fixtures"
GRAD_SIZES = [5, 20, 50, pytest.param(100, marks=pytest.mark.slow)]  # m is stored in each fixture
MAX_ITER = 400


def load_fixture(name: str) -> dict:
    """Arrays of an OCaml fixture, skipping the test if it has not been generated"""
    path = FIXTURES / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"{path.name} missing; generate it with benchmarks/make_ocaml_fixtures.py")
    return dict(onp.load(path))


@pytest.mark.parametrize(
    "fixture,builder,use_linesearch",
    [
        ("ocaml_pendulum", ocaml_pendulum, False),
        ("ocaml_pendulum_linesearch", ocaml_pendulum, True),
        ("ocaml_rnn", vanilla_rnn, True),
    ],
)
def test_matches_ocaml_dilqr(fixture, builder, use_linesearch):
    """Same optimal cost, state and input trajectories as OCaml dilqr"""
    ref = load_fixture(fixture)
    model, params = builder()
    Us_init = jnp.zeros((model.dims.horizon, model.dims.m))
    (Xs, Us, _), cost, _ = ilqr.ilqr_solver(model, params, Us_init, max_iter=400, use_linesearch=use_linesearch)
    chex.assert_trees_all_close(cost, ref["costs"][-1], rtol=1e-12)
    chex.assert_trees_all_close(Xs, ref["Xs"], atol=1e-6)
    chex.assert_trees_all_close(Us, ref["Us"], atol=1e-5)


def test_rnn_fixture_uses_shared_parameters():
    """The OCaml RNN was solved with exactly the parameters of tests/problems.vanilla_rnn"""
    ref = load_fixture("ocaml_rnn")
    for key, value in vanilla_rnn_params().items():
        onp.testing.assert_allclose(ref[key], value, rtol=1e-10, atol=0)


# --------------------------------------------------------------------------------------------
# implicit gradients
# --------------------------------------------------------------------------------------------


def ocaml_rollout_loss(model, params: iLQRParams, Us_init) -> jax.Array:
    """
    sum(tau*^2) with dilqr-as-released gradients (case 2). The local LQR is built around states
    re-rolled out from the optimal inputs with the differentiable theta (x0 held fixed, as OCaml
    passes it primal), so AD also differentiates through the rollout. Gauss-Newton Hessians.
    """
    (Xs, Us, _), _, _ = ilqr.ilqr_solver(model, lax.stop_gradient(params), Us_init, max_iter=MAX_ITER)
    (Xs_roll, _), _ = ilqr.ilqr_simulate(model, Us, iLQRParams(lax.stop_gradient(params.x0), params.theta))
    tau = jnp.c_[Xs, jnp.r_[Us, jnp.zeros((1, model.dims.m))]]
    local_lqr = ilqr.approx_lqr_offset(model, Xs_roll, Us, params)
    return jnp.sum(dllqr(model.dims, LQRParams(lqr=local_lqr, x0=params.x0), tau) ** 2)


def gauss_newton_loss(model, params: iLQRParams, Us_init) -> jax.Array:
    """sum(tau*^2) with Gauss-Newton implicit gradients (case 1)"""
    return jnp.sum(dilqr(model, params, Us_init, exact_hessian=False, max_iter=MAX_ITER) ** 2)


def load_grad_fixture(n: int, linear: bool = False) -> dict:
    """OCaml gradient fixture of the vanilla RNN with n states"""
    return load_fixture(f"ocaml_rnn_grad{'_linear' if linear else ''}_n{n}")


def assert_matches_ocaml_grads(loss_fn, ref: dict, suffix: str, linear: bool = False, rtol: float = 1e-9):
    """Loss and gradients wrt W, B and x0 of loss_fn on vanilla_rnn agree with an OCaml fixture"""
    model, params = vanilla_rnn(int(ref["n"]), int(ref["m"]), linear=linear)
    Us_init = jnp.zeros((model.dims.horizon, model.dims.m))
    value, grads = jax.value_and_grad(loss_fn, argnums=1)(model, params, Us_init)
    chex.assert_trees_all_close(value, ref["loss"], rtol=1e-10)
    for name, grad in [("W", grads.theta.Uh), ("B", grads.theta.Wh), ("x0", grads.x0)]:
        expected = ref[f"grad_{name}{suffix}"]
        rel_err = onp.linalg.norm(grad - expected) / onp.linalg.norm(expected)
        assert rel_err < rtol, f"grad_{name}{suffix}: relative error {rel_err:.2e}"


@pytest.mark.parametrize("n", GRAD_SIZES)
def test_ocaml_fixed_trajectory_matches_gauss_newton(n):
    """Case 1: patched OCaml (fixed trajectory) gradients equal dilqr(exact_hessian=False)"""
    assert_matches_ocaml_grads(gauss_newton_loss, load_grad_fixture(n), "_fixed")


@pytest.mark.parametrize("n", GRAD_SIZES)
def test_ocaml_released_matches_rollout_emulation(n):
    """Case 2: OCaml dilqr as released equals the test-only copy of its rollout differentiation"""
    assert_matches_ocaml_grads(ocaml_rollout_loss, load_grad_fixture(n), "")


@pytest.mark.parametrize("n", GRAD_SIZES)
def test_ocaml_released_differs_from_gauss_newton(n):
    """The rollout terms of case 2 change the theta gradients on tanh dynamics (x0 is unaffected)"""
    ref = load_grad_fixture(n)
    rel = lambda key: onp.linalg.norm(ref[f"grad_{key}"] - ref[f"grad_{key}_fixed"]) / onp.linalg.norm(
        ref[f"grad_{key}_fixed"]
    )
    assert rel("W") > 1e-3 and rel("B") > 1e-3
    assert rel("x0") < 1e-10


def test_ocaml_linear_dynamics_all_agree():
    """With linear dynamics both OCaml modes, Gauss-Newton and exact implicit gradients coincide"""
    ref = load_grad_fixture(5, linear=True)
    assert_matches_ocaml_grads(gauss_newton_loss, ref, "_fixed", linear=True)
    assert_matches_ocaml_grads(gauss_newton_loss, ref, "", linear=True)
    exact_loss = lambda model, params, Us: jnp.sum(dilqr(model, params, Us, max_iter=MAX_ITER) ** 2)
    assert_matches_ocaml_grads(exact_loss, ref, "_fixed", linear=True)
