import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

from ._run import melt_concat_from_list, run_study


def app():
    parser = argparse.ArgumentParser(
        description="Run a study by pointing to its config list"
    )
    parser.add_argument("study", type=str, help="The file containig the config list")
    parser.add_argument(
        "--gpu", type=int, help="The GPU to use for the experiment", default=0
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        help="The seed to start the experiment from (helpful for seed-statistics across multiple GPUs)",
        default=None,
    )
    parser.add_argument(
        "--dont_melt_metrics",
        action="store_true",
        help="Do not melt the metrics across the experiments in the study",
        default=False,
    )
    parser.add_argument(
        "--dont_melt_loss",
        action="store_true",
        help="Do not melt the train loss across the experiments in the study",
        default=False,
    )
    parser.add_argument(
        "--melt_sample_rollouts",
        action="store_true",
        help="Melt the sample rollouts across the experiments in the study",
        default=False,
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        help="The name(s) of the metric(s) to melt. For multiple use a comma separated list, e.g. 'mean_nRMSE,mean_correlation'",
        default="mean_nRMSE",
    )
    args = parser.parse_args()

    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # TODO: change hard coded version
    apebench_version = "0.0.1"

    LOG_DIR = "logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    LOG_SUBDIR = LOG_DIR + str(datetime.datetime.now()) + "/"

    os.makedirs(LOG_SUBDIR, exist_ok=True)

    with open(LOG_SUBDIR + "apebench_version.txt", "w") as f:
        f.write(apebench_version)

    # # Save the environment file
    # environment_file = subprocess.run(
    #     ["conda", "env", "export"], capture_output=True, text=True
    # ).stdout
    # with open(LOG_SUBDIR + "environment.yml", "w") as f:
    #     f.write(environment_file)

    study_path = Path(args.study)

    if not os.path.exists(study_path):
        print(f"Error: The file '{study_path}' does not exist.")
        sys.exit(1)

    # Copy the config file to the log directory
    shutil.copy(study_path, LOG_SUBDIR)

    # Load the variable CONFIGS from the python file in experiment_path
    study_dir = os.path.dirname(study_path)
    sys.path.append(study_dir)
    study = os.path.basename(study_path)
    study_name = study.split(".")[0]
    module = __import__(study_name)
    CONFIGS = module.CONFIGS

    if args.start_seed is not None:
        print("Overwriting all start seeds to", args.start_seed)
        new_configs = []
        for config in CONFIGS:
            config["start_seed"] = args.start_seed
            new_configs.append(config)
        CONFIGS = new_configs

    BASE_PATH = "results/"

    os.makedirs(BASE_PATH, exist_ok=True)

    raw_file_list, network_weights_list = run_study(
        CONFIGS,
        BASE_PATH,
    )

    MELTED_DIR = "melted/"

    os.makedirs(MELTED_DIR, exist_ok=True)

    MELTED_SUBDIR = MELTED_DIR + study_name + "/"

    os.makedirs(MELTED_SUBDIR, exist_ok=True)

    with open(MELTED_SUBDIR + "network_weights_list.json", "w") as f:
        dump = json.dumps([str(r) for r in network_weights_list])
        f.write(dump)

    with open(MELTED_SUBDIR + "raw_file_list.json", "w") as f:
        dump = json.dumps([str(r) for r in raw_file_list])
        f.write(dump)

    melt_concat_from_list(
        raw_file_list,
        MELTED_SUBDIR,
        metric_name=args.metric_name.split(","),
        do_metrics=not args.dont_melt_metrics,
        do_loss=not args.dont_melt_loss,
        do_sample_rollouts=args.melt_sample_rollouts,
    )
