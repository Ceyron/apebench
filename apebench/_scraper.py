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
    If `folder` is not None, saves the data and metadata to the folder as
    `folder/{name}.npy` and `folder/{name}.json`. Otherwise, returns the data
    and metadata as jax arrays and a dictionary, respectively.
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

    logging.info(f"Producing data for {name}")
    data = scenario.get_train_data()
    num_nans = jnp.sum(jnp.isnan(data))
    if num_nans > 0:
        logging.warning(f"Data contains {num_nans} NaNs")

    logging.info(f"Writing data for {name}")

    jnp.save(f"{name}.npy", data)

    info = asdict(scenario)

    metadata = {
        "name": name,
        "info": info,
    }

    if folder is not None:
        with open(f"{folder}/{name}.json", "w") as f:
            json.dump(metadata, f)
        jnp.save(f"{folder}/{name}.npy", data)
    else:
        return data, metadata
