import exponax
import jax
import jax.numpy as jnp

import apebench

# import pytest


def test_external_metric_computation_single_seed():
    adv_scenario = apebench.scenarios.difficulty.Advection()

    # Just use the randomly initialized model
    neural_stepper = adv_scenario.get_neural_stepper(
        task_config="predict",
        network_config="mlp;16;4;relu",
        key=jax.random.PRNGKey(0),
    )

    # Manual path: (1) get test ic, (2) rollout model, (3) compute metrics
    test_ic = adv_scenario.get_test_ic_set()
    neural_rollout = jax.vmap(
        exponax.rollout(neural_stepper, adv_scenario.test_temporal_horizon)
    )(test_ic)
    manual_metrics = adv_scenario.perform_tests_on_rollout(neural_rollout)

    # Automated path: Directly use `perform_tests`
    automated_metrics = adv_scenario.perform_tests(
        neural_stepper,
        remove_singleton_axis=True,
    )

    manual_nRMSE = manual_metrics["mean_nRMSE"]
    automated_nRMSE = automated_metrics["mean_nRMSE"]

    assert jnp.array_equal(manual_nRMSE, automated_nRMSE)
    # assert manual_nRMSE == pytest.approx(automated_nRMSE, rel=1e-6, abs=1e-12)
