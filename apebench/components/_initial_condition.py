from typing import Callable, Dict

import exponax as ex

ic_dict: Dict[
    str,  # ic_name
    Callable[
        [
            str,  # ic_config
            int,  # num_spatial_dims
        ],
        ex.ic.BaseRandomICGenerator,
    ],
] = {
    "fourier": lambda ic_config, num_spatial_dims: ex.ic.RandomTruncatedFourierSeries(
        num_spatial_dims=num_spatial_dims,
        cutoff=int(ic_config.split(";")[1]),
        offset_range=(0.0, 0.0)
        if ic_config.split(";")[2].lower() == "true"
        else (-0.5, 0.5),
        max_one=ic_config.split(";")[3].lower() == "true",
    ),
    "diffused": lambda ic_config, num_spatial_dims: ex.ic.DiffusedNoise(
        num_spatial_dims=num_spatial_dims,
        intensity=float(ic_config.split(";")[1]),
        zero_mean=ic_config.split(";")[2].lower() == "true",
        max_one=ic_config.split(";")[3].lower() == "true",
    ),
    "grf": lambda ic_config, num_spatial_dims: ex.ic.GaussianRandomField(
        num_spatial_dims=num_spatial_dims,
        powerlaw_exponent=float(ic_config.split(";")[1]),
        zero_mean=ic_config.split(";")[2].lower() == "true",
        max_one=ic_config.split(";")[3].lower() == "true",
    ),
}
