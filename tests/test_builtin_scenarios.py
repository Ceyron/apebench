import pytest

import apebench


def test_simple():
    apebench.scenarios.difficulty.Advection()


@pytest.mark.parametrize(
    "name",
    list(apebench.scenarios.scenario_dict.keys()),
)
def test_builtin_scenarios(name: str):
    # Some scenarios might not work in 1d, (which is the default number of spatial dims)
    try:
        scene = apebench.scenarios.scenario_dict[name]()
    except ValueError:
        return

    ref = scene.get_ref_sample_data()

    del ref


@pytest.mark.parametrize(
    "name,num_spatial_dims",
    [
        (name, num_spatial_dims)
        for name in [
            "phy_adv",
            "norm_adv",
            "diff_adv",
        ]
        for num_spatial_dims in [1, 2, 3]
    ],
)
def test_simple_training(name: str, num_spatial_dims: int):
    NUM_TRAIN_SAMPLES = 5
    NUM_TEST_SAMPLES = 5
    NUM_POINTS = 15
    OPTIM_CONFIG = "adam;10;constant;1e-4"

    # Some scenarios might not work in 1d, (which is the default number of spatial dims)
    try:
        scene = apebench.scenarios.scenario_dict[name](
            num_spatial_dims=num_spatial_dims,
            num_train_samples=NUM_TRAIN_SAMPLES,
            num_test_samples=NUM_TEST_SAMPLES,
            num_points=NUM_POINTS,
            optim_config=OPTIM_CONFIG,
        )
    except ValueError:
        return

    NETWORK_CONFIG = "Conv;10;2;relu"

    data, trained_net = scene(
        network_config=NETWORK_CONFIG,
    )

    del data, trained_net
