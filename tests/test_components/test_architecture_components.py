import jax
import pytest

import apebench


@pytest.mark.parametrize(
    "network_config",
    [
        "Conv;34;10;relu",
        "Res;26;6;relu",
        "UNet;16;2;relu",
        "Dil;2;32;2;relu",
        "MLP;64;3;relu",
        "Pure;5;identity",
        "MoRes;26;6;relu",
        "MoUNet;16;2;relu",
    ],
)
def test_architecture_execution(network_config: str):
    advection_scenario = apebench.scenarios.difficulty.Advection()

    neural_emulator = advection_scenario.get_neural_stepper(
        task_config="predict",
        network_config=network_config,
        key=jax.random.PRNGKey(0),
    )

    x = jax.random.uniform(jax.random.PRNGKey(0), (1, advection_scenario.num_points))

    neural_emulator(x)
