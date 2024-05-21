from typing import Union

import pandas as pd
from scipy.stats import gmean

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
):
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
):
    return melt_data(
        wide_data,
        quantity_name=metric_name,
        uniquifier_name="time_step",
    )


def melt_loss(wide_data: pd.DataFrame, loss_name: str = "train_loss"):
    return melt_data(
        wide_data,
        quantity_name=loss_name,
        uniquifier_name="update_step",
    )


def melt_sample_rollouts(
    wide_data: pd.DataFrame,
    sample_rollout_name: str = "sample_rollout",
):
    return melt_data(
        wide_data,
        quantity_name=sample_rollout_name,
        uniquifier_name="sample_index",
    )


def split_train(
    metric_data: pd.DataFrame,
):
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
    grouping_cols: list[str] = BASE_NAMES,
):
    return (
        metric_data.groupby(grouping_cols)
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
):
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
):
    col = df["scenario_kwargs"].apply(eval)
    entries = list(col[0].keys())
    for entry in entries:
        df[entry] = col.apply(lambda x: x[entry])
    return df
