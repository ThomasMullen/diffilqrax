"""Shared pytest configuration: the ``heavy`` tier and per-module JAX cache clearing.

CI runs ``pytest -m "not slow and not heavy"`` on 4-core, 16 GB runners. ``heavy`` marks the tests that
are too slow or memory-hungry for that budget and whose property is already covered by a cheaper test:
every function in ``HEAVY``, and every case of a function in ``QUICK`` except its listed representatives
(fnmatch patterns on the parametrisation id). Locally ``pytest -m "not slow"`` still runs everything.
"""

from fnmatch import fnmatch

import jax
import pytest

HEAVY = {
    # unrolled reverse mode, sharding subprocess and long horizons
    "test_batch.py::test_batched_gradient_matches_unrolled",
    "test_batch.py::test_batched_return_info",
    "test_batch.py::test_pilqr_while_matches_scan",
    "test_batch.py::test_reverse_mode_through_while_loop_raises",
    "test_batch.py::test_shard_strategy",
    "test_batch.py::test_zebrafish_batched_warm_start",
    "test_dilqr.py::TestDILQR::test_dilqr",
    "test_gradients.py::test_gauss_newton_exact_for_linear_dynamics",
    "test_gradients.py::test_implicit_matches_direct_gradient_control_problems",
    "test_gradients.py::test_warm_start_is_not_differentiated",
    "test_lqr.py::TestLQRSolutionExact::test_lqr_solution",
    "test_ocaml_mgu.py::test_ilqr_warm_start_stays_at_ocaml_optimum",
    "test_ocaml_reference.py::test_ocaml_linear_dynamics_all_agree",
    "test_optimality.py::test_expected_cost_change_is_exact_for_lqr",
    "test_optimality.py::test_ilqr_float32_best_effort",
    "test_optimality.py::test_ilqr_smooth_l1_sparsity",
    "test_parallel_regularisation.py::test_input_projection_inactive_when_R_is_pd",
    "test_parallel_regularisation.py::test_input_projection_speeds_up_indefinite_R",
    "test_parallel_regularisation.py::test_regularised_pilqr_multistart",
    "test_parallel_regularisation.py::test_unregularised_pilqr_stalls_on_indefinite_problem",
    "test_pdilqr.py::test_pdilqr_adjoints_agree",
    "test_pdilqr.py::test_pdilqr_check_grads",
    "test_pdilqr.py::test_pdilqr_us_init_gets_no_gradient",
    "test_pdlqr.py::test_pdlqr_jit_and_long_horizon",
    "test_pilqr.py::test_parallel_rollout_unstable_long_horizon",
    "test_pilqr.py::test_parallel_system_still_accepted",
    "test_pilqr.py::test_pilqr_cost_history",
    "test_pilqr.py::test_pilqr_multistart",
    "test_pilqr.py::test_stopping_test_unstable_long_horizon",
}

QUICK = {
    "test_batch.py::test_batched_gradient_is_sum_of_gradients": ["map"],
    "test_batch.py::test_batched_matches_unbatched": ["rnn-vmap-*"],
    "test_batch.py::test_batched_parallel_solver": ["rnn"],
    "test_batch.py::test_parallel_linesearch_matches_backtracking": ["rnn-False"],
    "test_batch.py::test_while_matches_scan": ["pendulum"],
    "test_dlqr.py::test_dlqr_check_grads": ["5-5"],
    "test_dlqr.py::test_dlqr_matches_direct_gradient": ["5-5"],
    "test_gradients.py::test_implicit_matches_direct_gradient": ["5-3"],
    "test_gradients.py::test_value_independent_of_differentiation": ["10"],
    "test_ocaml_reference.py::test_matches_ocaml_dilqr": ["ocaml_pendulum_linesearch-*"],
    "test_ocaml_reference.py::test_ocaml_fixed_trajectory_matches_gauss_newton": ["5"],
    "test_ocaml_reference.py::test_ocaml_released_matches_rollout_emulation": ["5"],
    "test_optimality.py::test_control_problem_multistart_agreement": ["van_der_pol-*"],
    "test_optimality.py::test_control_problem_optimality": ["pendulum_no_linesearch-*", "van_der_pol-*"],
    "test_optimality.py::test_cost_grows_quadratically_around_optimum": ["pendulum_problem-*"],
    "test_optimality.py::test_ilqr_recovers_lqr_solution": ["20-5"],
    "test_optimality.py::test_ilqr_rnn_kkt_and_second_order": ["5-5"],
    "test_optimality.py::test_ilqr_rnn_matches_lbfgs": ["5"],
    "test_optimality.py::test_ilqr_rnn_multistart_agreement": ["5"],
    "test_optimality.py::test_lqr_kkt_and_exact_solution": ["20-5"],
    "test_parallel_regularisation.py::test_projected_pilqr_reaches_kkt_point": ["track_log-R"],
    "test_parallel_regularisation.py::test_regularised_pilqr_matches_ilqr": ["sharp_log_u"],
    "test_pdilqr.py::test_pdilqr_matches_dilqr": ["sharp_log_u-parallel"],
    "test_pdilqr.py::test_pdilqr_nonlinear_matches_dilqr": ["rnn-parallel-parallel-True"],
    "test_pdilqr.py::test_pdllqr_nonlinear_matches_dilqr": ["rnn-parallel-False"],
    "test_pdlqr.py::test_pdllqr_matches_dllqr": ["sequential"],
    "test_pdlqr.py::test_pdlqr_check_grads": ["5-5-*"],
    "test_pdlqr.py::test_pdlqr_matches_dlqr": ["5-5-*"],
    "test_pilqr.py::test_pilqr_matches_ilqr": ["rnn-parallel"],
    "test_pilqr.py::test_pilqr_matches_primal_dual_ilqr": ["pdilqr_ilqr_pendulum-parallel"],
    "test_pilqr.py::test_pilqr_without_linesearch": ["parallel"],
    "test_pilqr.py::test_return_info": ["False"],
    "test_plqr.py::test_matches_primal_dual_ilqr": ["*-pdilqr_lqr_n5_m3_T50"],
    "test_plqr.py::test_plqr_horizons": ["1", "128"],
    "test_plqr.py::test_plqr_matches_dense_qp": ["5-5"],
    "test_plqr.py::test_plqr_matches_lqr": ["False-*", "True-5-5"],
}


def pytest_collection_modifyitems(items):
    """Mark the ``heavy`` tier described in the module docstring"""
    for item in items:
        name = item.nodeid.split("[")[0].removeprefix("tests/")
        case = item.callspec.id if hasattr(item, "callspec") else None
        keep = QUICK.get(name)
        if name in HEAVY or (keep is not None and not any(fnmatch(case, pattern) for pattern in keep)):
            item.add_marker(pytest.mark.heavy)


@pytest.fixture(autouse=True, scope="module")
def clear_jax_caches():
    """Free the compiled functions of each test module, so peak memory does not grow over the session"""
    yield
    jax.clear_caches()
