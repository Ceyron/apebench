# Tips & Tricks

* **Avoid excessive storage usage for experiments**:
    * Reduce the number of exported trajectories. This is 1 by default, but in
        2d this can cause excessive memory usage
    * Reduce the logging frequency with `record_loss_every`.
* **Check for NaNs in the trajectories**:
    * `APEBench` only guarantees that the scenarios in difficulty mode work
        across all three spatial dimensions. For the normalized/physical
        scenarios, they will only work in their primary dimension.
    * If you you are modifying a scenario, also always check for instabilities
        in the trajectories. Those can be caused by various actors, often a quick
        hack to avoid them is to increase the diffusivity of the scenario or add
        substeps.
    * All linear scenarios should be free of NaNs (if started at a bandlimited
        initial condition) due to Fourier spectral ETDRK method being able to
        solve them exactly.
    