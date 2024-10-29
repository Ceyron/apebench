# FAQ

will be extended upon future releases

## Why `JAX` and `Equinox`?

* **Single-Batch by Design**: All emulators have a call signature that does not
    require arrays to have a leading batch axis. Vectorized operation is
    achieved with the `jax.vmap` transformation. We believe that this
    design more closely resembles how classical simulators are usually set up.
* **Seed-Parallel Training**: With only a little additional modification, the
    automatic vectorization of `jax.vmap` (or precisely with `eqx.filter_vmap`)
    can also be used to run multiple initialization seeds in parallel. This
    approach is especially helpful when a training run of one network does not
    fully utilize an entire GPU, like in all 1D scenarios. Essentially, this
    allows for free seed statistics.
* **Similar Python Structure for Neural Network and ETDRK Solver**: Both
    simulator and emulator are implemented as `equinox.Module` allowing
    them to operate seamlessly with one another.