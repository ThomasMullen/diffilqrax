"""
Parity of the JAX MGU2 iLQR-VAE model (examples/ocaml_mgu) with the OCaml implementation.

The fixture ``tests/fixtures/ocaml_mgu.npz`` is written by ``examples/ocaml_mgu/make_fixture.py`` from
full-precision outputs of ``examples/ocaml_mgu_ref.ml`` (zebrafish repository): trained parameters,
inputs and observations of 3 test and 3 train trials, the OCaml rollout of those inputs, the OCaml
iLQR objective, and a converged OCaml iLQR solve from ``u = 0``.
"""

import sys
from pathlib import Path

import numpy as onp
import pytest
import jax
import jax.numpy as jnp

from diffilqrax import ilqr
from diffilqrax.typs import iLQRParams

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "ocaml_mgu"))
from load_ocaml import OCamlParams  # pylint: disable=wrong-import-position
from mgu2 import make_system, make_theta, readout, rollout, total_cost  # pylint: disable=wrong-import-position

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

FIXTURE = Path(__file__).parent / "fixtures" / "ocaml_mgu.npz"
N_BEG = 12
OCAML_MAX_ITER = 300
GRAD_TOL = 1e-2  # OCaml stops on a 1e-13 relative cost change, not on the gradient
WARM_U_TOL = 1e-5


@pytest.fixture(scope="module")
def ref():
    """Reference arrays and parameters"""
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE.name} missing; generate it with examples/ocaml_mgu/make_fixture.py")
    data = dict(onp.load(FIXTURE))
    params = OCamlParams(
        **{k[len("param_") :]: (float(v) if k == "param_nu" else v) for k, v in data.items() if k.startswith("param_")}
    )
    return params, data


def converged_trials(data) -> list:
    """Trials where the OCaml solver stopped on its tolerance rather than the iteration cap"""
    n_iter = onp.isfinite(data["ilqr_cost_hist"]).sum(axis=1) - 1
    return [i for i in range(len(n_iter)) if n_iter[i] < OCAML_MAX_ITER]


def test_forward_matches_ocaml(ref):
    """Rolling out the same inputs gives the OCaml latents and outputs to round-off"""
    params, data = ref
    for i in range(data["saved_u"].shape[0]):
        theta = make_theta(params, data["saved_tail"][i], N_BEG)
        zs = rollout(theta, jnp.asarray(data["saved_u"][i]))
        onp.testing.assert_allclose(zs, data["fwd_z"][i], rtol=0, atol=1e-12)
        onp.testing.assert_allclose(readout(theta, zs), data["fwd_o"][i], rtol=0, atol=1e-12)


def test_objective_matches_ocaml(ref):
    """The iLQR objective (Student-t prior + Gaussian likelihood) equals OCaml's on two input sets"""
    params, data = ref
    for i in range(data["saved_u"].shape[0]):
        theta = make_theta(params, data["saved_tail"][i], N_BEG)
        onp.testing.assert_allclose(total_cost(theta, jnp.asarray(data["saved_u"][i])), data["cost_saved_u"][i], rtol=1e-12)
        onp.testing.assert_allclose(total_cost(theta, jnp.asarray(data["ilqr_u"][i])), data["ilqr_cost"][i], rtol=1e-12)


def test_ocaml_optimum_is_stationary(ref):
    """Where OCaml converged, its solution is a stationary point of the JAX objective"""
    params, data = ref
    trials = converged_trials(data)
    assert trials, "no converged OCaml solve in the fixture"
    for i in trials:
        theta = make_theta(params, data["saved_tail"][i], N_BEG)
        grad = jax.grad(total_cost, argnums=1)(theta, jnp.asarray(data["ilqr_u"][i]))
        assert float(jnp.abs(grad).max()) < GRAD_TOL


def test_ilqr_warm_start_stays_at_ocaml_optimum(ref):
    """diffilqrax started at a converged OCaml solution does not move away from it"""
    params, data = ref
    i = converged_trials(data)[0]
    theta = make_theta(params, data["saved_tail"][i], N_BEG)
    model = make_system(120, 10, data["ilqr_u"].shape[1])
    (_, us, _), cost, _ = ilqr.ilqr_solver(
        model, iLQRParams(x0=jnp.zeros(120), theta=theta), jnp.asarray(data["ilqr_u"][i]), max_iter=20, alpha_init=2.0
    )
    assert float(cost) <= data["ilqr_cost"][i] + 1e-9 * abs(data["ilqr_cost"][i])
    onp.testing.assert_allclose(us, data["ilqr_u"][i], rtol=0, atol=WARM_U_TOL)
