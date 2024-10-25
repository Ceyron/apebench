import pytest

import apebench


@pytest.mark.parametrize(
    "optimizer_config,lr_scheduler_config",
    [
        (optimizer_config, lr_scheduler_config)
        for optimizer_config in [
            "adam",
        ]
        for lr_scheduler_config in [
            "constant;0.01",
            "exp;0.01;1000;0.9;True",
            "warmup_cosine;0.01;0.1;40",
        ]
    ],
)
def test_optimization_execution(
    optimizer_config: str,
    lr_scheduler_config: str,
):
    NUM_TRAINING_STEPS = 100

    advection_scenario = apebench.scenarios.difficulty.Advection(
        optim_config=optimizer_config
        + ";"
        + str(NUM_TRAINING_STEPS)
        + ";"
        + lr_scheduler_config,
    )

    optimizer = advection_scenario.get_optimizer()

    del optimizer

    # TODO: Check if optimizer can process a dummy state
