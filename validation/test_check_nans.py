"""
The default configuration of all defined scenarios shall not produce train or
test trajectories that contain NaNs. This indicates an instability of the
reference solver. It could be solved by lowering the scenario's difficulty or
performing substeps in the reference solver.
"""

import jax.numpy as jnp
import pytest

import apebench


def compute_num_nans_trjs(trjs):
    def has_nan(trj):
        if jnp.sum(jnp.isnan(trj)) > 0:
            return 1
        else:
            return 0

    mask = [has_nan(trj) for trj in trjs]

    return sum(mask)


def check_all_data(scene: apebench.BaseScenario):
    train_data = scene.get_train_data()

    train_num_nans = compute_num_nans_trjs(train_data)
    assert (
        train_num_nans == 0
    ), f"Train data has {train_num_nans} trajectories with NaNs"

    del train_data

    test_data = scene.get_test_data()

    test_num_nans = compute_num_nans_trjs(test_data)
    assert test_num_nans == 0, f"Test data has {test_num_nans} trajectories with NaNs"

    del test_data

    try:
        # Some scenarios might not support a correction mode
        train_data_coarse = scene.get_train_data_coarse()

        train_num_nans_coarse = compute_num_nans_trjs(train_data_coarse)
        assert (
            train_num_nans_coarse == 0
        ), f"Train data coarse has {train_num_nans_coarse} trajectories with NaNs"

        del train_data_coarse

        test_data_coarse = scene.get_test_data_coarse()

        test_num_nans_coarse = compute_num_nans_trjs(test_data_coarse)
        assert (
            test_num_nans_coarse == 0
        ), f"Test data coarse has {test_num_nans_coarse} trajectories with NaNs"

        del test_data_coarse
    except NotImplementedError:
        return


@pytest.mark.parametrize(
    "name",
    list(apebench.scenarios.scenario_dict.keys()),
)
def test_check_nans_1d(name: str):
    scene_constructor = apebench.scenarios.scenario_dict[name]
    try:
        scene = scene_constructor(num_spatial_dims=1)
    except ValueError:
        return

    check_all_data(scene)


@pytest.mark.parametrize(
    "name, num_spatial_dims",
    [
        (name, num_spatial_dims)
        for name in list(apebench.scenarios.difficulty.scenario_dict.keys())
        for num_spatial_dims in [1, 2, 3]
    ],
)
def test_nans_on_difficulty_scenarios(name: str, num_spatial_dims: int):
    scene_constructor = apebench.scenarios.scenario_dict[name]

    if num_spatial_dims == 3:
        NUM_POINTS = 32
    else:
        NUM_POINTS = 160

    try:
        scene = scene_constructor(
            num_spatial_dims=num_spatial_dims, num_points=NUM_POINTS
        )
    except ValueError:
        return

    check_all_data(scene)


@pytest.mark.parametrize(
    "name",
    list(apebench.scenarios.scenario_dict.keys()),
)
def test_check_nans_2d(name: str):
    scene_constructor = apebench.scenarios.scenario_dict[name]
    try:
        scene = scene_constructor(num_spatial_dims=2)
    except ValueError:
        return

    check_all_data(scene)


@pytest.mark.parametrize(
    "name",
    list(apebench.scenarios.scenario_dict.keys()),
)
def test_check_nans_3d(name: str):
    # Reduce to 32 points in 3d
    NUM_POINTS_3d = 32

    scene_constructor = apebench.scenarios.scenario_dict[name]
    try:
        scene = scene_constructor(num_spatial_dims=3, num_points=NUM_POINTS_3d)
    except ValueError:
        return

    check_all_data(scene)
