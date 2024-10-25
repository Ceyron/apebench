from typing import Callable, Dict

import jax

activation_fn_dict: Dict[
    str,
    Callable[
        [
            str,  # activation_function_config
        ],
        Callable,  # activation_fn
    ],
] = {
    "relu": lambda activation_fn_config: jax.nn.relu,
    "sigmoid": lambda activation_fn_config: jax.nn.sigmoid,
    "tanh": lambda activation_fn_config: jax.nn.tanh,
    "swish": lambda activation_fn_config: jax.nn.swish,
    "gelu": lambda activation_fn_config: jax.nn.gelu,
    "identity": lambda activation_fn_config: lambda x: x,
}
