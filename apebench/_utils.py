from typing import Union

import jax
import jax.numpy as jnp
import pandas as pd
from scipy.stats import gmean

from ._base_scenario import BaseScenario

BASE_NAMES = [
    "seed",
    "scenario",
    "task",
    "net",
    "train",
    "scenario_kwargs",
]
BASE_NAMES_NO_TRAIN = [
    "seed",
    "scenario",
    "task",
    "net",
    "scenario_kwargs",
]


def melt_data(
    wide_data: pd.DataFrame,
    quantity_name: Union[str, list[str]],
    uniquifier_name: str,
    *,
    base_columns: list[str] = BASE_NAMES,
) -> pd.DataFrame:
    """
    Melt a wide APEBench result DataFrame into a long format suitable for
    visualization (e.g. with seaborn or plotly).

    **Arguments:**

    * `wide_data`: The wide DataFrame to melt, must contain `quantity_name` and
        `base_columns` as columns.
    * `quantity_name`: The name of the column(s) to melt.
    * `uniquifier_name`: The name of the column that will be used to uniquely
        identify the melted rows.
    * `base_columns`: The columns that should be kept as is in the melted
        DataFrame.

    **Returns:**

    * A long DataFrame with the same columns as `base_columns` and the melted
        `quantity_name`.
    """
    if isinstance(quantity_name, str):
        quantity_name = [
            quantity_name,
        ]
    data_melted = pd.wide_to_long(
        wide_data,
        stubnames=quantity_name,
        i=base_columns,
        j=uniquifier_name,
        sep="_",
    )
    data_melted = data_melted[quantity_name]
    data_melted = data_melted.reset_index()

    return data_melted


def melt_metrics(
    wide_data: pd.DataFrame,
    metric_name: Union[str, list[str]] = "mean_nRMSE",
) -> pd.DataFrame:
    """
    Melt the metrics from a wide DataFrame.
    """
    return melt_data(
        wide_data,
        quantity_name=metric_name,
        uniquifier_name="time_step",
    )


def melt_loss(wide_data: pd.DataFrame, loss_name: str = "train_loss") -> pd.DataFrame:
    """
    Melt the loss from a wide DataFrame.
    """
    return melt_data(
        wide_data,
        quantity_name=loss_name,
        uniquifier_name="update_step",
    )


def melt_sample_rollouts(
    wide_data: pd.DataFrame,
    sample_rollout_name: str = "sample_rollout",
) -> pd.DataFrame:
    """
    Melt the sample rollouts from a wide DataFrame.
    """
    return melt_data(
        wide_data,
        quantity_name=sample_rollout_name,
        uniquifier_name="sample_index",
    )


def split_train(
    metric_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decode the `train` column into `category`, `type`, and `rollout` columns.
    """
    metric_data["category"] = metric_data["train"].apply(lambda x: x.split(";")[0])
    metric_data["type"] = metric_data["category"].apply(
        lambda x: "sup" if x in ["one", "sup"] else "div"
    )
    metric_data["rollout"] = metric_data["train"].apply(
        lambda x: int(x.split(";")[1]) if x != "one" else 1
    )

    return metric_data


def aggregate_gmean(
    metric_data: pd.DataFrame,
    *,
    up_to: int = 100,
    grouping_cols: list[str] = BASE_NAMES,
) -> pd.DataFrame:
    """
    Aggregate an error rollout over time via the geometric mean.

    Args:

    * `metric_data`: The DataFrame to aggregate, must contain `grouping_cols`
        and `mean_nRMSE` as columns. When grouped by `grouping_cols`, the groups
        shall only contain values at different time steps.
    * `up_to`: The time step up to which to aggregate. (inclusive)
    * `grouping_cols`: The columns to group by.

    Returns:

    * A DataFrame with the new column `gmean_mean_nRMSE` containing the
        geometric mean of the `mean_nRMSE` values up to `up_to` for each group.
    """
    return (
        metric_data.query(f"time_step <= {up_to}")
        .groupby(grouping_cols)
        .agg(gmean_mean_nRMSE=("mean_nRMSE", gmean))
        .reset_index()
    )


def relative_by_config(
    data: pd.DataFrame,
    *,
    grouping_cols: list[str] = BASE_NAMES_NO_TRAIN,
    norm_query: str = "train == 'one'",
    value_col: str = "mean_nRMSE",
    suffix: str = "_rel",
) -> pd.DataFrame:
    def relativate_fn(sub_df):
        rel = sub_df.query(norm_query)[value_col]
        if len(rel) != 1:
            raise ValueError(
                f"Expected exactly one row to match {norm_query}, got {len(rel)}"
            )
        col = sub_df[value_col] / rel.values[0]
        sub_df[f"{value_col}{suffix}"] = col
        return sub_df

    return data.groupby(grouping_cols).apply(relativate_fn).reset_index(drop=True)


def read_in_kwargs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse the `scenario_kwargs` column of a DataFrame and add the parsed entries
    as new columns.

    Requires that the dictionary in `scenario_kwargs` has the same keys for all
    rows.
    """
    col = df["scenario_kwargs"].apply(eval)
    entries = list(col[0].keys())
    for entry in entries:
        df[entry] = col.apply(lambda x: x[entry])
    return df


def count_nan_trjs(trjs: jax.Array) -> int:
    """
    Computes the number of trajectories that contain at least one NaN value.
    """

    def has_nan(trj):
        if jnp.sum(jnp.isnan(trj)) > 0:
            return 1
        else:
            return 0

    mask = [has_nan(trj) for trj in trjs]

    return sum(mask)


def check_for_nan(scene: BaseScenario):
    """
    Check for NaNs in the train and test data of a scenario. Also checks the
    train and test data set produced by the coarse stepper if the scenario
    supports a correction mode. Raises an AssertionError if NaNs are found.
    """
    train_data = scene.get_train_data()

    train_num_nans = count_nan_trjs(train_data)
    assert (
        train_num_nans == 0
    ), f"Train data has {train_num_nans} trajectories with NaNs"

    del train_data

    test_data = scene.get_test_data()

    test_num_nans = count_nan_trjs(test_data)
    assert test_num_nans == 0, f"Test data has {test_num_nans} trajectories with NaNs"

    del test_data

    try:
        # Some scenarios might not support a correction mode
        train_data_coarse = scene.get_train_data_coarse()

        train_num_nans_coarse = count_nan_trjs(train_data_coarse)
        assert (
            train_num_nans_coarse == 0
        ), f"Train data coarse has {train_num_nans_coarse} trajectories with NaNs"

        del train_data_coarse

        test_data_coarse = scene.get_test_data_coarse()

        test_num_nans_coarse = count_nan_trjs(test_data_coarse)
        assert (
            test_num_nans_coarse == 0
        ), f"Test data coarse has {test_num_nans_coarse} trajectories with NaNs"

        del test_data_coarse
    except NotImplementedError:
        return
