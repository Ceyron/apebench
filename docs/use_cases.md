# Use-Cases of APEBench

I want to:

1. **Benchmark my neural architecture**:
    1. Is your architecture implemented in
       [Equinox](https://github.com/patrick-kidger/equinox)? Then you can
       directly [register it with APEBench](extending_apebench.md) and use the entire
       pipeline.
    2. Your architecture is not implemented in Equinox. Then you can *not* use
       the full pipeline of APEBench (no differentiable physics, for example).
       But you can still [scrape datasets](examples/scrape_datasets.ipynb) for
       purely data-driven training in different ecosystems and you can evaluate
       your architectures (guides for
       [PyTorch](examples/benchmark_pytorch_models.ipynb), [Flax with `linen`
       API](examples/benchmark_flax_models_with_linen.ipynb), and [Flax with
       `nnx` API](examples/benchmark_flax_models_with_nnx.ipynb)) by rolling
       them out in your ecosystem of choice and casting the results back to JAX
       arrays and using APEBench's metric system.
2. **Create datasets** Consider using the [scraping API](examples/scrape_datasets.ipynb).
3. **Conduct studies** Learn about the [typical workflow of
   APEBench](typical_workflow.md) and consider [using the command line interface
   (CLI)](using_cli.md).
4. **Extend APEBench**:
    1. With new architectures. If this is just for your experiments, then you
       can register them in the global `architecture_dict`. If you want to
       expand the built-in architectures, consider opening a pull request to
       [PDEquinox](https://github.com/Ceyron/pdequinox).
    2. With new scenarios. You can subclass from `BaseScenario` (or other child
       classes) and register them in the `scenario_dict`.
    3. New activation functions, initial condition distributions, optimization
       and learning rate scheduling setups, etc. can be registered in their
       respective components dictionary.
5. **Reproduce the experiments of the APEBench paper**: All experimental scripts
   (and also the results) are available
   [here](https://huggingface.co/thuerey-group/apebench-paper) and the results
   of the ablational experiments are
   [here](https://huggingface.co/thuerey-group/apebench-paper-ablations).
