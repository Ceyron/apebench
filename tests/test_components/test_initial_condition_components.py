import jax
import pytest

import apebench


@pytest.mark.parametrize(
    "ic_config, num_spatial_dims",
    [
        (ic_config, num_spatial_dims)
        for ic_config in [
            "fourier;4;true;true",
            "diffused;0.1;true;true",
            "grf;2;true;true",
            # Using further modifications
            "clamp;0.2;0.3;fourier;4;true;true",
        ]
        for num_spatial_dims in [1, 2, 3]
    ],
)
def test_initial_condition_components(ic_config: str, num_spatial_dims: int):
    advection_scenario = apebench.scenarios.difficulty.Advection(
        ic_config=ic_config,
        num_spatial_dims=num_spatial_dims,
    )

    ic_generator = advection_scenario.get_ic_generator()

    NUM_POINTS = 40
    ic = ic_generator(NUM_POINTS, key=jax.random.PRNGKey(0))

    assert ic.shape == (1,) + (NUM_POINTS,) * num_spatial_dims
