# Utilities for Scraping Datasets from APEBench

Use these functions if you want to procedurally scrape datasets from APEBench to
then use outside of the APEBench ecosystem, e.g., for training/testing
supervised models in PyTorch or in JAX with other deep learning frameworks than
[Equinox](https://docs.kidger.site/equinox/).

APEBench is designed to procedurally generate its data with fixed random seeds
by relying on [JAX' explicit treatment of randomness](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers).

However, this determinism can only be relied on if the code is executed with the same JAX version number and on the same backend (likely also using the same driver version). Beyond that, some low-level routines within CUDA experience some non-determinism (for performance reasons) which [can be deactivated](https://github.com/google/jax/discussions/10674).

::: apebench.scraper.scrape_data_and_metadata