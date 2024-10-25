from typing import Callable, Dict

import optax

optimizer_dict: Dict[
    str,  # optimizer_name
    Callable[
        [
            str,
        ],  # optim_config
        Callable[
            [
                optax.Schedule,
            ],  # lr_schedule
            optax.GradientTransformation,  # optimizer
        ],
    ],
] = {"adam": lambda optim_config: optax.adam}

lr_scheduler_dict: Dict[
    str,  # lr_scheduler_name
    Callable[
        [
            str,  # lr_scheduler_config
            int,  # num_training_steps
        ],
        optax.Schedule,  # lr_schedule
    ],
] = {
    "constant": lambda lr_scheduler_config, num_training_steps: optax.constant_schedule(
        float(lr_scheduler_config.split(";")[1]),
    ),
    "exp": lambda lr_scheduler_config, num_training_steps: optax.exponential_decay(
        init_value=float(lr_scheduler_config.split(";")[1]),
        transition_steps=int(lr_scheduler_config.split(";")[2]),
        decay_rate=float(lr_scheduler_config.split(";")[3]),
        staircase=lr_scheduler_config.split(";")[4] == "True",
    ),
    "warmup_cosine": lambda lr_scheduler_config, num_training_steps: optax.warmup_cosine_decay_schedule(
        init_value=float(lr_scheduler_config.split(";")[1]),
        peak_value=float(lr_scheduler_config.split(";")[2]),
        warmup_steps=int(lr_scheduler_config.split(";")[3]),
        decay_steps=num_training_steps,
    ),
}
