from typing import Callable, Dict

import exponax as ex
from jaxtyping import Array, Float


def mean_fourier_nRMSE_constructor(
    metric_config: str, domain_extent: float
) -> Callable[
    [Array[Float, "batch channel ... N"], Array[Float, "batch channel ... N"]], float
]:
    metric_args = metric_config.split(";")
    low = int(metric_args[1])
    high = int(metric_args[2])

    def mean_fourier_nRMSE(pred, ref):
        return ex.metrics.mean_metric(
            ex.metrics.fourier_nRMSE,
            pred,
            ref,
            domain_extent=domain_extent,
            low=low,
            high=high,
        )

    return mean_fourier_nRMSE


metrics_dict: Dict[
    str,
    Callable[
        [
            str,  # metric_config
            float,  # domain_extent
        ],
        Callable[
            [
                Array[Float, "batch channel ... N"],  # batched pred
                Array[Float, "batch channel ... N"],  # batched target
            ],
            float,
        ],
    ],
] = {
    "mean_MSE": lambda metric_config, domain_extent: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.MSE,
        pred,
        ref,
    ),
    "mean_nMSE": lambda metric_config, domain_extent: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.nMSE,
        pred,
        ref,
    ),
    "mean_sMSE": lambda metric_config, domain_extent: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.sMSE,
        pred,
        ref,
    ),
    "mean_RMSE": lambda metric_config, domain_extent: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.RMSE,
        pred,
        ref,
    ),
    "mean_nRMSE": lambda metric_config, domain_extent: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.nRMSE,
        pred,
        ref,
    ),
    "mean_sRMSE": lambda metric_config, domain_extent: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.sRMSE,
        pred,
        ref,
    ),
    "mean_fourier_nRMSE": mean_fourier_nRMSE_constructor,
}
