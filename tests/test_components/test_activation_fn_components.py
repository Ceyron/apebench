import jax
import pytest

import apebench


@pytest.mark.parametrize(
    "activation_fn_config",
    [
        "relu",
        "sigmoid",
        "tanh",
        "swish",
        "gelu",
    ],
)
def test_activation_fn(activation_fn_config: str):
    advection_scenario = apebench.scenarios.difficulty.Advection()

    activation_fn = advection_scenario.get_activation(activation_fn_config)

    x = jax.random.uniform(jax.random.PRNGKey(0), (20, 3, 10))

    activation_fn(x)
