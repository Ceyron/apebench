import os
import pathlib
from typing import Optional, Union

import equinox as eqx
import pandas as pd
from tqdm.autonotebook import tqdm

from ._utils import melt_loss, melt_metrics, melt_sample_rollouts, read_in_kwargs
from .scenarios import scenario_dict


def run_experiment(
    *,
    scenario: str,
    task: str,
    net: str,
    train: str,
    start_seed: int,
    num_seeds: int,
    **scenario_kwargs,
) -> tuple[pd.DataFrame, eqx.Module]:
    """
    Execute a single experiment.

    Only accepts keyword arguments, requires some main arguments and additional
    arguments that can depend on the chosen scenario.

    **Arguments:**

    * `scenario`: The scenario to run, must be a key in
        `apebench.scenarios.scenario_dict`.
    * `task`: The task to run, can be `"predict"` or `"correct;XX"`
        where `"XX"` is the correction mode, e.g., `"correct;sequential"`.
    * `net`: The network to use, must be a compatible descriptor of a
        network architecture, e.g., `"Conv;34;10;relu"`.
    * `train`: The training methodology to use, i.e., how reference
        solver and emulator interact during training. One-step supervised
        training is achieved by `"one"`.
    * `start_seed`: The integer at which the list of seeds starts from.
    * `num_seeds`: The number of seeds to run (in parallel). For many 1D
        scenarios at realistic resolutions (`num_points` ~ 200), doing ten seeds
        in parallel is usually fine for modern GPUs. For scenarios in 2D and 3D
        at realistic resolutions, this likely has to be set to 1 and seed
        processing must be done sequentially via `run_study`.

    **Returns:**

    * `data`: The `pandas.DataFrame` containing the results of the
        experiment. Will contain the columns `seed`, `scenario`, `task`, `net`,
        `train`, `scenario_kwargs`, the metrics, losses and sample rollouts. Can
        be further post-processed, e.g., via `apebench.melt_metrics`.
    * `trained_neural_stepper_s`: Equinox modules containing the
        trained neural emulators. Note that if `num_seeds` is greater than 1,
        weight arrays have a leading (seed-)batch axis.
    """
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
) -> str:
    """
    Produce a unique name for an experiment.
    """
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


def run_study(
    configs: list[dict],
    base_path: str,
    *,
    overwrite: bool = False,
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """
    Execute a study with multiple experiments.

    By default skips experiments that have already been conducted.

    **Arguments:**

    * `configs`: A list of dictionaries, each containing the
        keyword arguments for [`apebench.run_experiment`][].
    * `base_path`: The base path to store the results in.
    * `overwrite`: Whether to overwrite existing results.

    **Returns:**

    * `raw_file_list`: A list of paths to the raw data files.
    * `network_weights_list`: A list of paths to the
        network weights.
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

        data, trained_neural_stepper_s = run_experiment(**config)

        data.to_csv(raw_data_path)
        eqx.tree_serialise_leaves(
            network_weights_path,
            trained_neural_stepper_s,
        )

        del data
        del trained_neural_stepper_s

        print("Finished training!")
        print()

    return raw_file_list, network_weights_list


def melt_concat_metrics_from_list(
    raw_file_list: list[pathlib.Path],
    *,
    metric_name: Union[str, list[str]] = "mean_nRMSE",
) -> pd.DataFrame:
    """
    Melt and concatenate metrics from a list of raw files. Use this function on
    the results of [`apebench.run_study`][].

    **Arguments:**

    * `raw_file_list`: A list of paths to the raw data
      files.
    * `metric_name`: The name of the metric to melt.

    **Returns:**

    * `metric_df`: The DataFrame containing the metrics.
    """
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
) -> pd.DataFrame:
    """
    Melt and concatenate loss from a list of raw files. Use this function on the
    results of [`apebench.run_study`][].

    **Arguments:**

    * `raw_file_list`: A list of paths to the raw data files.

    **Returns:**

    * `loss_df`: The DataFrame containing the loss.
    """
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
) -> pd.DataFrame:
    """
    Melt and concatenate sample rollouts from a list of raw files. Use this
    function on the results of [`apebench.run_study`][].

    **Arguments:**

    * `raw_file_list`: A list of paths to the raw data files.

    **Returns:**

    * `sample_rollout_df`: The DataFrame containing the sample rollouts.
    """
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
) -> tuple[Optional[pathlib.Path], Optional[pathlib.Path], Optional[pathlib.Path]]:
    """
    Melt and concatenate metrics, loss and sample rollouts from a list of raw
    files and save the resulting DataFrames to disk as CSV files. Use this
    function on the results of [`apebench.run_study`][].

    **Arguments:**

    * `raw_file_list`: A list of paths to the raw data files.
    * `base_path`: The base path to store the results in.
    * `metric_name`: The name of the metric to melt.
    * `metric_file_name`: The name of the file to save the metrics to.
    * `loss_file_name`: The name of the file to save the loss to.
    * `sample_rollout_file_name`: The name of the file to save the sample
        rollouts to.
    * `do_metrics`: Whether to melt and save the metrics.
    * `do_loss`: Whether to melt and save the loss.
    * `do_sample_rollouts`: Whether to melt and save the sample rollouts.

    **Returns:**

    * `metric_df_file_name`: The path to the metrics file.
    * `loss_df_file_name`: The path to the loss file.
    * `sample_rollout_df_file_name`: The path to the sample rollouts file.
    """
    if do_metrics:
        metric_df = melt_concat_metrics_from_list(
            raw_file_list,
            metric_name=metric_name,
        )
        metric_df_file_name = base_path / pathlib.Path(f"{metric_file_name}.csv")
        metric_df.to_csv(
            metric_df_file_name,
            index=False,
        )
    else:
        metric_df_file_name = None

    if do_loss:
        loss_df = melt_concat_loss_from_list(raw_file_list)
        loss_df_file_name = base_path / pathlib.Path(f"{loss_file_name}.csv")
        loss_df.to_csv(
            loss_df_file_name,
            index=False,
        )
    else:
        loss_df_file_name = None

    if do_sample_rollouts:
        sample_rollout_df = melt_concat_sample_rollouts_from_list(raw_file_list)
        sample_rollout_df_file_name = base_path / pathlib.Path(
            f"{sample_rollout_file_name}.csv"
        )
        sample_rollout_df.to_csv(
            sample_rollout_df_file_name,
            index=False,
        )
    else:
        sample_rollout_df_file_name = None

    return metric_df_file_name, loss_df_file_name, sample_rollout_df_file_name


def run_study_convenience(
    configs: list[dict],
    base_path: Optional[str] = None,
    *,
    overwrite: bool = False,
    metric_name: Union[str, list[str]] = "mean_nRMSE",
    do_metrics: bool = True,
    do_loss: bool = False,
    do_sample_rollouts: bool = False,
    parse_kwargs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pathlib.Path]]:
    """
    Run a study with multiple experiments and melt and concatenate the results.

    **Arguments:**

    * `configs`: A list of dictionaries, each containing the
        keyword arguments for `run_experiment`.
    * `base_path`: The base path to store the results in. If
        `None`, a path is generated based on the hash of the `configs`.
    * `overwrite`: Whether to overwrite existing results.
    * `metric_name`: The name of the metric to melt.
    * `do_metrics`: Whether to melt and save the metrics.
    * `do_loss`: Whether to melt and save the loss.
    * `do_sample_rollouts`: Whether to melt and save the sample rollouts.
    * `parse_kwargs`: Whether to parse the scenario kwargs.

    **Returns:**

    * `metric_df`: The DataFrame containing the metrics.
    * `loss_df`: The DataFrame containing the loss.
    * `sample_rollout_df`: The DataFrame containing the sample rollouts.
    * `network_weights_list`: A list of paths to the network weights.
    """
    if base_path is None:
        config_hash = hash(str(configs))
        base_path = pathlib.Path(f"_results_{config_hash}")

    raw_file_list, network_weights_list = run_study(
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
