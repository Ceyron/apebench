from typing import Callable, Dict

import exponax as ex
from jaxtyping import Array, Float

metric_dict: Dict[
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
    "mean_MAE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.MAE,
        pred,
        ref,
    ),
    "mean_nMAE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.nMAE,
        pred,
        ref,
    ),
    "mean_sMAE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.sMAE,
        pred,
        ref,
    ),
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
    "mean_fourier_MAE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.fourier_MAE,
        pred,
        ref,
        low=int(metric_config.split(";")[1]),
        high=int(metric_config.split(";")[2]),
        derivative_order=float(metric_config.split(";")[3]),
    ),
    "mean_fourier_nMAE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.fourier_nMAE,
        pred,
        ref,
        low=int(metric_config.split(";")[1]),
        high=int(metric_config.split(";")[2]),
        derivative_order=float(metric_config.split(";")[3]),
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
    "mean_H1_MAE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.H1_MAE,
        pred,
        ref,
    ),
    "mean_H1_nMAE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.H1_nMAE,
        pred,
        ref,
    ),
    "mean_H1_MSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.H1_MSE,
        pred,
        ref,
    ),
    "mean_H1_nMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.H1_nMSE,
        pred,
        ref,
    ),
    "mean_H1_RMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.H1_RMSE,
        pred,
        ref,
    ),
    "mean_H1_nRMSE": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.H1_nRMSE,
        pred,
        ref,
    ),
    "mean_correlation": lambda metric_config: lambda pred, ref: ex.metrics.mean_metric(
        ex.metrics.correlation,
        pred,
        ref,
    ),
}
