import jax
import pytest

import apebench


@pytest.mark.parametrize(
    "metric_config",
    [
        "mean_MAE",
        "mean_nMAE",
        "mean_sMAE",
        "mean_MSE",
        "mean_nMSE",
        "mean_sMSE",
        "mean_RMSE",
        "mean_nRMSE",
        "mean_sRMSE",
        "mean_fourier_MAE;0;4;0",
        "mean_fourier_nMAE;0;4;0",
        "mean_fourier_MSE;0;4;0",
        "mean_fourier_nMSE;0;4;0",
        "mean_fourier_RMSE;0;4;0",
        "mean_fourier_nRMSE;0;4;0",
        "mean_H1_MAE",
        "mean_H1_nMAE",
        "mean_H1_MSE",
        "mean_H1_nMSE",
        "mean_H1_RMSE",
        "mean_H1_nRMSE",
        "mean_correlation",
    ],
)
def test_metric_execution(metric_config: str):
    pred = jax.random.uniform(jax.random.PRNGKey(0), (20, 3, 10))
    ref = jax.random.uniform(jax.random.PRNGKey(1), (20, 3, 10))

    metric_name = metric_config.split(";")[0]
    metric_fn = apebench.components.metric_dict[metric_name](metric_config)
    metric_result = metric_fn(pred, ref)

    # Must be a float type in form of a shape-less JAX array
    assert metric_result.shape == ()
