import os
import pathlib
from typing import Optional, Union

import equinox as eqx
import pandas as pd
from tqdm.autonotebook import tqdm

from ._utils import melt_loss, melt_metrics, melt_sample_rollouts, read_in_kwargs
from .scenarios import scenario_dict


def run_one_entry(
    *,
    scenario: str,
    task: str,
    net: str,
    train: str,
    start_seed: int,
    num_seeds: int,
    **scenario_kwargs,
) -> tuple[pd.DataFrame, eqx.Module]:
    scenario = scenario_dict[scenario](**scenario_kwargs)

    data, trained_neural_stepper_s = scenario(
        task_config=task,
        network_config=net,
        train_config=train,
        start_seed=start_seed,
        num_seeds=num_seeds,
    )

    if len(scenario_kwargs) == 0:
        data["scenario_kwargs"] = "{}"
    else:
        data["scenario_kwargs"] = str(scenario_kwargs)

    return data, trained_neural_stepper_s


def get_experiment_name(
    *,
    scenario: str,
    task: str,
    net: str,
    train: str,
    start_seed: int,
    num_seeds: int,
    **scenario_kwargs,
):
    # to consider anything beyond the default arguments
    additional_infos = []
    for key, value in scenario_kwargs.items():
        additional_infos.append(f"{key}={value}")
    if len(additional_infos) > 0:
        additional_infos = ",".join(additional_infos)
        additional_infos = f"__{additional_infos}__"
    else:
        additional_infos = "__"

    end_seed = start_seed + num_seeds
    experiment_name = f"{scenario}{additional_infos}{task}__{net}__{train}__{start_seed}-{end_seed - 1}"
    return experiment_name


def run_experiment(
    configs: list[dict],
    base_path: str,
    *,
    overwrite: bool = False,
):
    """
    Configs must contain keys: 'scenario', .. todo
    """
    raw_file_list = []
    network_weights_list = []

    for config in configs:
        experiment_name = get_experiment_name(**config)

        print("Considering")
        print(experiment_name)

        raw_data_folder = base_path / pathlib.Path("raw")
        os.makedirs(raw_data_folder, exist_ok=True)
        raw_data_path = raw_data_folder / pathlib.Path(f"{experiment_name}.csv")

        network_weights_folder = base_path / pathlib.Path("network_weights")
        os.makedirs(network_weights_folder, exist_ok=True)
        network_weights_path = network_weights_folder / pathlib.Path(
            f"{experiment_name}.eqx"
        )

        raw_file_list.append(raw_data_path)
        network_weights_list.append(network_weights_path)

        if (
            os.path.exists(raw_data_path)
            and os.path.exists(network_weights_path)
            and not overwrite
        ):
            print("Skipping, already trained ...")
            print()
            continue

        data, trained_neural_stepper_s = run_one_entry(**config)

        data.to_csv(raw_data_path)
        eqx.tree_serialise_leaves(
            network_weights_path,
            trained_neural_stepper_s,
        )

        print("Finished training!")
        print()

    return raw_file_list, network_weights_list


def melt_concat_metrics_from_list(
    raw_file_list: list[pathlib.Path],
    *,
    metric_name: Union[str, list[str]] = "mean_nRMSE",
):
    metric_df_s = []
    for file_name in tqdm(
        raw_file_list,
        desc="Melt and Concat metrics",
    ):
        data = pd.read_csv(file_name)
        data = melt_metrics(data, metric_name=metric_name)
        metric_df_s.append(data)

    metric_df = pd.concat(metric_df_s)

    return metric_df


def melt_concat_loss_from_list(
    raw_file_list: list[pathlib.Path],
):
    loss_df_s = []
    for file_name in tqdm(
        raw_file_list,
        desc="Melt and Concat loss",
    ):
        data = pd.read_csv(file_name)
        data = melt_loss(data)
        loss_df_s.append(data)

    loss_df = pd.concat(loss_df_s)

    return loss_df


def melt_concat_sample_rollouts_from_list(
    raw_file_list: list[pathlib.Path],
):
    sample_rollout_df_s = []
    for file_name in tqdm(
        raw_file_list,
        desc="Melt and Concat sample rollouts",
    ):
        data = pd.read_csv(file_name)
        data = melt_sample_rollouts(data)
        sample_rollout_df_s.append(data)

    sample_rollout_df = pd.concat(sample_rollout_df_s)

    return sample_rollout_df


def melt_concat_from_list(
    raw_file_list: list[pathlib.Path],
    base_path: str,
    *,
    metric_name: Union[str, list[str]] = "mean_nRMSE",
    metric_file_name: str = "metrics",
    loss_file_name: str = "train_loss",
    sample_rollout_file_name: str = "sample_rollout",
    do_metrics: bool = True,
    do_loss: bool = False,
    do_sample_rollouts: bool = False,
):
    """
    And save to file
    """
    if do_metrics:
        metric_df = melt_concat_metrics_from_list(
            raw_file_list,
            metric_name=metric_name,
        )
        metric_df.to_csv(
            base_path / pathlib.Path(f"{metric_file_name}.csv"),
            index=False,
        )

    if do_loss:
        loss_df = melt_concat_loss_from_list(raw_file_list)
        loss_df.to_csv(
            base_path / pathlib.Path(f"{loss_file_name}.csv"),
            index=False,
        )

    if do_sample_rollouts:
        sample_rollout_df = melt_concat_sample_rollouts_from_list(raw_file_list)
        sample_rollout_df.to_csv(
            base_path / pathlib.Path(f"{sample_rollout_file_name}.csv"),
            index=False,
        )


def run_experiment_convenience(
    configs: list[dict],
    base_path: Optional[str] = None,
    *,
    overwrite: bool = False,
    metric_name: Union[str, list[str]] = "mean_nRMSE",
    do_metrics: bool = True,
    do_loss: bool = False,
    do_sample_rollouts: bool = False,
    parse_kwargs: bool = True,
):
    if base_path is None:
        config_hash = hash(str(configs))
        base_path = pathlib.Path(f"_results_{config_hash}")

    raw_file_list, network_weights_list = run_experiment(
        configs,
        base_path,
        overwrite=overwrite,
    )

    melt_concat_from_list(
        raw_file_list,
        base_path,
        metric_name=metric_name,
        do_metrics=do_metrics,
        do_loss=do_loss,
        do_sample_rollouts=do_sample_rollouts,
    )

    if do_metrics:
        metric_df = pd.read_csv(base_path / pathlib.Path("metrics.csv"))
        if parse_kwargs:
            metric_df = read_in_kwargs(metric_df)
    else:
        metric_df = pd.DataFrame()

    if do_loss:
        loss_df = pd.read_csv(base_path / pathlib.Path("train_loss.csv"))
        if parse_kwargs:
            loss_df = read_in_kwargs(loss_df)
    else:
        loss_df = pd.DataFrame()

    if do_sample_rollouts:
        sample_rollout_df = pd.read_csv(base_path / pathlib.Path("sample_rollout.csv"))
        if parse_kwargs:
            sample_rollout_df = read_in_kwargs(sample_rollout_df)
    else:
        sample_rollout_df = pd.DataFrame()

    return metric_df, loss_df, sample_rollout_df, network_weights_list
