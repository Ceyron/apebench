from typing import Callable, Dict

import exponax as ex
from jaxtyping import Array, Float

metrics_dict: Dict[
    str,
    Callable[
        [
            str,  # metric_config
        ],
        Callable[
            [
                Float[Array, "batch channel ... N"],  # batched pred
                Float[Array, "batch channel ... N"],  # batched target
            ],
            float,
        ],
    ],
] = {
    "mean_MSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.MSE,
        pred,
        ref,
    ),
    "mean_nMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.nMSE,
        pred,
        ref,
    ),
    "mean_sMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.sMSE,
        pred,
        ref,
    ),
    "mean_RMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.RMSE,
        pred,
        ref,
    ),
    "mean_nRMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.nRMSE,
        pred,
        ref,
    ),
    "mean_sRMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.sRMSE,
        pred,
        ref,
    ),
    "mean_fourier_MSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.fourier_MSE,
        pred,
        ref,
        low=int(metric_config.split(";")[1]),
        high=int(metric_config.split(";")[2]),
        derivative_order=float(metric_config.split(";")[3]),
    ),
    "mean_fourier_nMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.fourier_nMSE,
        pred,
        ref,
        low=int(metric_config.split(";")[1]),
        high=int(metric_config.split(";")[2]),
        derivative_order=float(metric_config.split(";")[3]),
    ),
    "mean_fourier_RMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.fourier_RMSE,
        pred,
        ref,
        low=int(metric_config.split(";")[1]),
        high=int(metric_config.split(";")[2]),
        derivative_order=float(metric_config.split(";")[3]),
    ),
    "mean_fourier_nRMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.fourier_nRMSE,
        pred,
        ref,
        low=int(metric_config.split(";")[1]),
        high=int(metric_config.split(";")[2]),
        derivative_order=float(metric_config.split(";")[3]),
    ),
}
