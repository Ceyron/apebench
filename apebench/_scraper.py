"""
Utilities to scrape APEBench datasets into numpy arrays and save them to disk.
"""

import json
import logging
from dataclasses import asdict

import jax.numpy as jnp

from .scenarios import scenario_dict


def scrape_data_and_metadata(
    folder: str = None,
    *,
    scenario: str,
    name: str = "auto",
    **scenario_kwargs,
):
    """
    Produce train data, test data, and metadata for a given scenario. Optionally
    write them to disk.

    **Arguments:**

    - `folder` (str, optional): Folder to save the data and metadata to. If
        None, returns the data and metadata as jax arrays and a dictionary,
        respectively.
    - `scenario` (str): Name of the scenario to produce data for. Must be one of
        `apebench.scenarios.scenario_dict`.
    - `name` (str, optional): Name of the scenario. If "auto", the name is
        automatically generated based on the scenario and its additional
        arguments.
    - `**scenario_kwargs`: Additional arguments to pass to the scenario. All
        attributes of a scenario can be modified by passing them as keyword
        arguments.
    """
    scenario = scenario_dict[scenario](**scenario_kwargs)
    if name == "auto":
        name = scenario.get_scenario_name()

        additional_infos = []
        for key, value in scenario_kwargs.items():
            additional_infos.append(f"{key}={value}")
        if len(additional_infos) > 0:
            additional_infos = ", ".join(additional_infos)
            additional_infos = "__" + additional_infos
        else:
            additional_infos = ""

        name += additional_infos

    logging.info(f"Producing train data for {name}")
    train_data = scenario.get_train_data()
    train_num_nans = jnp.sum(jnp.isnan(train_data))
    if train_num_nans > 0:
        logging.warning(f"Train data contains {train_num_nans} NaNs")

    logging.info(f"Producing test data for {name}")
    test_data = scenario.get_test_data()
    test_num_nans = jnp.sum(jnp.isnan(test_data))
    if test_num_nans > 0:
        logging.warning(f"Test data contains {test_num_nans} NaNs")

    info = asdict(scenario)

    metadata = {
        "name": name,
        "info": info,
    }

    if folder is not None:
        with open(f"{folder}/{name}.json", "w") as f:
            json.dump(metadata, f)
        jnp.save(f"{folder}/{name}_train.npy", train_data)
        jnp.save(f"{folder}/{name}_test.npy", test_data)

        del train_data, test_data
    else:
        return train_data, test_data, metadata


CURATION_APEBENCH_V1 = [
    # 1D - Linear
    {
        "scenario": "diff_adv",
    },
    {
        "scenario": "diff_diff",
    },
    {
        "scenario": "diff_adv_diff",
    },
    {
        "scenario": "diff_disp",
    },
    {
        "scenario": "diff_hyp_diff",
    },
    # 1D - Nonlinear
    {"scenario": "diff_burgers"},
    {"scenario": "diff_kdv"},
    {"scenario": "diff_ks"},
    {"scenario": "diff_ks_cons"},
    # 1D - Reaction-Diffusion
    {"scenario": "diff_fisher"},
    # 2D - Linear
    {"scenario": "diff_adv", "num_spatial_dims": 2},
    {"scenario": "diff_diff", "num_spatial_dims": 2},
    {"scenario": "diff_adv_diff", "num_spatial_dims": 2},
    {"scenario": "diff_disp", "num_spatial_dims": 2},
    {"scenario": "diff_hyp_diff", "num_spatial_dims": 2},
    # 2D - Linear Special
    {
        "scenario": "phy_unbal_adv",
        "num_spatial_dims": 2,
        "advection_coef_vector": (0.01, -0.04),
    },
    {"scenario": "phy_diag_diff", "num_spatial_dims": 2},
    {"scenario": "phy_aniso_diff", "num_spatial_dims": 2},
    {"scenario": "phy_mix_disp", "num_spatial_dims": 2},
    {"scenario": "phy_mix_hyp", "num_spatial_dims": 2},
    # 2D - Nonlinear
    {"scenario": "diff_burgers", "num_spatial_dims": 2},
    {"scenario": "diff_burgers_sc", "num_spatial_dims": 2},
    {"scenario": "diff_kdv", "num_spatial_dims": 2},
    {"scenario": "diff_ks", "num_spatial_dims": 2},
    {"scenario": "phy_decay_turb", "num_spatial_dims": 2},
    {"scenario": "phy_kolm_flow", "num_spatial_dims": 2},
    # 2D - Reaction-Diffusion
    {"scenario": "diff_fisher", "num_spatial_dims": 2},
    {"scenario": "phy_gs_type", "num_spatial_dims": 2},
    {"scenario": "phy_sh", "num_spatial_dims": 2},
    # 3D - Linear
    {
        "scenario": "diff_adv",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "diff_diff",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "diff_adv_diff",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "diff_disp",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "diff_hyp_diff",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    # 3D - Linear Special
    {
        "scenario": "phy_unbal_adv",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "phy_diag_diff",
        "num_spatial_dims": 3,
        "num_points": 32,
        "diffusion_coef_vector": (0.001, 0.002, 0.0004),
        "num_test_samples": 10,
    },
    {
        "scenario": "phy_aniso_diff",
        "num_spatial_dims": 3,
        "num_points": 32,
        "diffusion_coef_matrix": (
            (0.001, 0.0005, 0.0003),
            (0.0005, 0.002, 0.0002),
            (0.0003, 0.0002, 0.0004),
        ),
        "num_test_samples": 10,
    },
    {
        "scenario": "phy_mix_disp",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "phy_mix_hyp",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    # 3D - Nonlinear
    {
        "scenario": "diff_burgers",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "diff_burgers_sc",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "diff_kdv",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "diff_ks",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    # 3D - Reaction-Diffusion
    {
        "scenario": "diff_fisher",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "phy_gs_type",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
    {
        "scenario": "phy_sh",
        "num_spatial_dims": 3,
        "num_points": 32,
        "num_test_samples": 10,
    },
]
"""
Collection of default scenarios as used in the original APEBench paper
"""
