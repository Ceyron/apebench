import jax.numpy as jnp

import apebench


def test_ic_set_is_equal_to_first_snapshot():
    # Use a scenario with a warmup
    ks_scenario = apebench.scenarios.difficulty.KuramotoSivashinsky()

    # Train data
    train_data = ks_scenario.get_train_data()
    train_ic_set = ks_scenario.get_train_ic_set()

    assert jnp.array_equal(train_data[:, 0], train_ic_set)

    # Test data
    test_data = ks_scenario.get_test_data()
    test_ic_set = ks_scenario.get_test_ic_set()

    assert jnp.array_equal(test_data[:, 0], test_ic_set)

    # Coarse train data
    coarse_train_data = ks_scenario.get_train_data_coarse()
    coarse_train_ic_set = ks_scenario.get_train_ic_set_coarse()

    assert jnp.array_equal(coarse_train_data[:, 0], coarse_train_ic_set)

    # Coarse test data
    coarse_test_data = ks_scenario.get_test_data_coarse()
    coarse_test_ic_set = ks_scenario.get_test_ic_set_coarse()

    assert jnp.array_equal(coarse_test_data[:, 0], coarse_test_ic_set)
