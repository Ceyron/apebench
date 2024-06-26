# Extending APEBench

## Defining your own Architecture

You can have experiments run with your architectures. For this, you have to
register them in the `apebench.arch_extensions` dictionary.

```python
import apebench

def conv_net_extension(
    config_str: str,
    num_spatial_dims: int,
    num_channels: int,
    *,
    key: PRNGKeyArray,
):
    config_args = config_str.split(";")

    depth = int(config_args[1])

    return pdeqx.arch.ConvNet(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=42,
        depth=depth,
        activation=jax.nn.relu,
        key=key,
    )

apebench.arch_extensions.update(
    {"MyConvNet": conv_net_extension}
)
```

Then you can use the `Conv` architecture in the `net` configuration string.

```python
apebench.run_experiment(
    scenario="diff_adv",
    task="predict",
    net="MyConvNet;42",
    train="one",
    start_seed=0,
    num_seeds=10,
)
```


## Defining your own Scenario

### Modify an existing Scenario

When instantiating a scenario, use keyword based arguments to change some of the
attributes. For example, this uses less initial conditions for training the
standard advection scenario.

```python
import apebench

modified_adv_scene = apebench.scenarios.difficulty.Advection(
    num_train_samples=10,
)
```

Or if you use the string based interface, you can add additional keyword arguments that match the attribute names of the scenario.

```python
import apebench

apebench.run_experiment(
    scenario="diff_adv",
    task="predict",
    net="Conv;26;10;relu",
    train="one",
    start_seed=0,
    num_seeds=10,
    # Below are the additional keyword arguments
    num_train_samples=10,
)
```

Or if you run entire study, you can also add additional keyword arguments that match the attribute names of the scenario.

```python
CONFIGS = [
    {
        "scenario": "diff_adv",
        "task": "predict",
        "net": net,
        "train": "one",
        "start_seed": 0,
        "num_seeds": 10,
        # Below are the additional keyword arguments
        "num_train_samples": 10,
    }
    for net in ["Conv;26;10;relu", "FNO;12;8;4;gelu"]
]
```

### Your truly own Scenario

If you decide to implement your own scenario, you have to subclass `BaseScenario` and implement the following methods:

1. `get_ref_stepper()`
2. `get_coarse_stepper()` (or implement a raise Notimplemented error if your
    scenario does not support a correction mode)
3. `get_scenario_name()`

Of course, feel free to overwrite some of the other methods if you are unhappy
witht the options, for example to support more network architectures or training
methodologies.