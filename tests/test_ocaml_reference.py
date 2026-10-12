"""
Cross-check the iLQR solver against reference solutions from the OCaml dilqr package.

The fixtures are produced by ``benchmarks/make_ocaml_fixtures.py`` from ``examples/pendulum_ref.ml``
and ``examples/rnn_ref.ml`` in the dilqr repository. OCaml stops on a 1e-14 relative change in
cost rather than on the KKT conditions, so trajectories are compared at the accuracy that rule
allows (1e-5) while optimal costs agree to round-off.
"""

from pathlib import Path

import numpy as onp
import pytest
import chex
import jax
import jax.numpy as jnp

from diffilqrax import ilqr
from problems import ocaml_pendulum, vanilla_rnn, vanilla_rnn_params

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
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
        onp.testing.assert_array_equal(ref[key], value)
