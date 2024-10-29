# Tips & Tricks

* **Avoid excessive storage usage for experiments**:
    * Reduce the number of exported trajectories. This is 1 by default, but in
        2d this can cause excessive memory usage
    * Reduce the logging frequency with `record_loss_every`.