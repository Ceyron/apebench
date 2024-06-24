"""
Utilities to scrape APEBench datasets into numpy arrays and save them to disk.
"""

import logging
from dataclasses import asdict

import jax.numpy as jnp

from .scenarios import scenario_dict


def scrape_data_and_metadata(
    *,
    scenario: str,
    name: str = "auto",
    **scenario_kwargs,
):
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

    return data, metadata
