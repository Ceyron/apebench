"""
The default configuration of all defined scenarios shall not produce train or
test trajectories that contain NaNs. This indicates an instability of the
reference solver. It could be solved by lowering the scenario's difficulty or
performing substeps in the reference solver.
"""


import jax.numpy as jnp
import pytest

import apebench


@pytest.mark.parametrize(
    "name",
    list(apebench.scenarios.scenario_dict.keys()),
)
def test_check_nans_1d(name: str):
    try:
        scene = apebench.scenarios.scenario_dict[name](num_spatial_dims=1)
    except ValueError:
        return

    train_data = scene.get_train_data()

    train_num_nans = jnp.sum(jnp.isnan(train_data))
    assert train_num_nans == 0, f"Train data has {train_num_nans} NaNs"

    test_data = scene.get_test_data()

    test_num_nans = jnp.sum(jnp.isnan(test_data))
    assert test_num_nans == 0, f"Test data has {test_num_nans} NaNs"

    try:
        train_data_coarse = scene.get_train_data_coarse()

        train_num_nans_coarse = jnp.sum(jnp.isnan(train_data_coarse))
        assert (
            train_num_nans_coarse == 0
        ), f"Train data coarse has {train_num_nans_coarse} NaNs"

        test_data_coarse = scene.get_test_data_coarse()

        test_num_nans_coarse = jnp.sum(jnp.isnan(test_data_coarse))
        assert (
            test_num_nans_coarse == 0
        ), f"Test data coarse has {test_num_nans_coarse} NaNs"
    except NotImplementedError:
        return


@pytest.mark.parametrize(
    "name",
    list(apebench.scenarios.scenario_dict.keys()),
)
def test_check_nans_2d(name: str):
    try:
        scene = apebench.scenarios.scenario_dict[name](num_spatial_dims=2)
    except ValueError:
        return

    train_data = scene.get_train_data()

    train_num_nans = jnp.sum(jnp.isnan(train_data))
    assert train_num_nans == 0, f"Train data has {train_num_nans} NaNs"

    test_data = scene.get_test_data()

    test_num_nans = jnp.sum(jnp.isnan(test_data))
    assert test_num_nans == 0, f"Test data has {test_num_nans} NaNs"

    try:
        train_data_coarse = scene.get_train_data_coarse()

        train_num_nans_coarse = jnp.sum(jnp.isnan(train_data_coarse))
        assert (
            train_num_nans_coarse == 0
        ), f"Train data coarse has {train_num_nans_coarse} NaNs"

        test_data_coarse = scene.get_test_data_coarse()

        test_num_nans_coarse = jnp.sum(jnp.isnan(test_data_coarse))
        assert (
            test_num_nans_coarse == 0
        ), f"Test data coarse has {test_num_nans_coarse} NaNs"
    except NotImplementedError:
        return


@pytest.mark.parametrize(
    "name",
    list(apebench.scenarios.scenario_dict.keys()),
)
def test_check_nans_3d(name: str):
    # Reduce to 32 points in 3d
    NUM_POINTS_3d = 32
    try:
        scene = apebench.scenarios.scenario_dict[name](
            num_spatial_dims=3, num_points=NUM_POINTS_3d
        )
    except ValueError:
        return

    train_data = scene.get_train_data()

    train_num_nans = jnp.sum(jnp.isnan(train_data))
    assert train_num_nans == 0, f"Train data has {train_num_nans} NaNs"

    test_data = scene.get_test_data()

    test_num_nans = jnp.sum(jnp.isnan(test_data))
    assert test_num_nans == 0, f"Test data has {test_num_nans} NaNs"

    try:
        train_data_coarse = scene.get_train_data_coarse()

        train_num_nans_coarse = jnp.sum(jnp.isnan(train_data_coarse))
        assert (
            train_num_nans_coarse == 0
        ), f"Train data coarse has {train_num_nans_coarse} NaNs"

        test_data_coarse = scene.get_test_data_coarse()

        test_num_nans_coarse = jnp.sum(jnp.isnan(test_data_coarse))
        assert (
            test_num_nans_coarse == 0
        ), f"Test data coarse has {test_num_nans_coarse} NaNs"
    except NotImplementedError:
        return
