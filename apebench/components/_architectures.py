from typing import Callable, Dict

import equinox as eqx
import pdequinox as pdeqx
from jaxtyping import PRNGKeyArray


def dil_constructor(
    architecture_config: str,
    num_spatial_dims: int,
    num_points: int,
    num_channels: int,
    activation_fn: Callable,
    key: PRNGKeyArray,
):
    architecture_args = architecture_config.split(";")
    dilation_depth = int(architecture_args[1])
    hidden_channels = int(architecture_args[2])
    num_blocks = int(architecture_args[3])

    dilation_rates = [2**i for i in range(dilation_depth + 1)]
    dilation_rates = dilation_rates + dilation_rates[::-1][1:]

    return pdeqx.arch.DilatedResNet(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        dilation_rates=dilation_rates,
        activation=activation_fn,
        boundary_mode="periodic",
        key=key,
    )


architecture_dict: Dict[
    str,  # architecture_name
    Callable[
        [
            str,  # architecture_config
            int,  # num_spatial_dims
            int,  # num_points
            int,  # num_channels
            Callable,  # activation_fn
            PRNGKeyArray,  # key
        ],
        eqx.Module,  # architecture
    ],
] = {
    "conv": lambda architecture_config, num_spatial_dims, num_points, num_channels, activation_fn, key: pdeqx.arch.ConvNet(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=int(architecture_config.split(";")[1]),
        depth=int(architecture_config.split(";")[2]),
        activation=activation_fn,
        boundary_mode="periodic",
        key=key,
    ),
    "res": lambda architecture_config, num_spatial_dims, num_points, num_channels, activation_fn, key: pdeqx.arch.ClassicResNet(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=int(architecture_config.split(";")[1]),
        num_blocks=int(architecture_config.split(";")[2]),
        activation=activation_fn,
        boundary_mode="periodic",
        key=key,
    ),
    "unet": lambda architecture_config, num_spatial_dims, num_points, num_channels, activation_fn, key: pdeqx.arch.ClassicUNet(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=int(architecture_config.split(";")[1]),
        num_levels=int(architecture_config.split(";")[2]),
        activation=activation_fn,
        boundary_mode="periodic",
        key=key,
    ),
    "dil": dil_constructor,
    "fno": lambda architecture_config, num_spatial_dims, num_points, num_channels, activation_fn, key: pdeqx.arch.ClassicFNO(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        num_modes=int(architecture_config.split(";")[1]),
        hidden_channels=int(architecture_config.split(";")[2]),
        num_blocks=int(architecture_config.split(";")[3]),
        activation=activation_fn,
        boundary_mode="periodic",
        key=key,
    ),
    "mlp": lambda architecture_config, num_spatial_dims, num_points, num_channels, activation_fn, key: pdeqx.arch.MLP(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        num_points=num_points,
        width_size=int(architecture_config.split(";")[1]),
        depth=int(architecture_config.split(";")[2]),
        activation=activation_fn,
        key=key,
    ),
    "pure": lambda architecture_config, num_spatial_dims, num_points, num_channels, activation_fn, key: pdeqx.conv.PhysicsConv(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        kernel_size=int(architecture_config.split(";")[1]),
        use_bias=False,  # !!! no bias
        boundary_mode="periodic",
        key=key,
    ),
    "mores": lambda architecture_config, num_spatial_dims, num_points, num_channels, activation_fn, key: pdeqx.arch.ModernResNet(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=int(architecture_config.split(";")[1]),
        num_blocks=int(architecture_config.split(";")[2]),
        activation=activation_fn,
        boundary_mode="periodic",
        key=key,
    ),
    "mounet": lambda architecture_config, num_spatial_dims, num_points, num_channels, activation_fn, key: pdeqx.arch.ModernUNet(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=int(architecture_config.split(";")[1]),
        num_levels=int(architecture_config.split(";")[2]),
        activation=activation_fn,
        boundary_mode="periodic",
        key=key,
    ),
}
