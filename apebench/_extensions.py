"""
Dicttionaries for extending the scenarios of APEBench, e.g., with your custom
architecture.
"""

import jax
import pdequinox as pdeqx
from jaxtyping import PRNGKeyArray


def conv_net_extension(
    config_str: str,
    num_spatial_dims: int,
    num_channels: int,
    *,
    key: PRNGKeyArray,
):
    config_args = config_str.split(";")

    depth = int(config_args[1])

    return pdeqx.arch.ConvNet(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=42,
        depth=depth,
        activation=jax.nn.relu,
        key=key,
    )


arch_extensions = {
    "ConvCustom": conv_net_extension,
}
"""
Add custom architectures to be used with APEBench scenarios.

Use a key to identify your architecture type (has to be different from the
default architectures), then you must supply a constructor function that takes
this a configuration string, the number of spatial dimensions, the number of
channels, and a PRNG key, and returns the instantiated architecture as an
equinox module.
"""
