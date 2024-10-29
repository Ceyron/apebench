"""
The default configuration of all defined scenarios shall not produce train or
test trajectories that contain NaNs. This indicates an instability of the
reference solver. It could be solved by lowering the scenario's difficulty or
performing substeps in the reference solver.
"""

import pytest

import apebench

RUN_EXTENSIVE = False


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

    apebench.check_for_nan(scene)


@pytest.mark.parametrize(
    "name, num_spatial_dims",
    apebench.scenarios.guaranteed_non_nan,
)
def test_nans_on_guaranteed_scenarios(name: str, num_spatial_dims: int):
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

    apebench.check_for_nan(scene)


if RUN_EXTENSIVE:

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

        apebench.check_for_nan(scene)

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

        apebench.check_for_nan(scene)

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

        apebench.check_for_nan(scene)
