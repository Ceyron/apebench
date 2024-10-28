# Using the CLI

Once installed via pip, `apebench` offers a command line interface (CLI) to run
benchmarks. You can access it via

```bash
apebench --help
```
```
usage: apebench [-h] [--gpu GPU] [--start_seed START_SEED] [--dont_melt_metrics] [--dont_melt_loss] [--melt_sample_rollouts] experiment

Run a study by pointing to its config list

positional arguments:
  experiment            The file containig the config list

options:
  -h, --help            show this help message and exit
  --gpu GPU             The GPU to use for the experiment
  --start_seed START_SEED
                        The seed to start the experiment from (helpful for seed-statistics across multiple GPUs)
  --dont_melt_metrics   Do not melt the metrics across the runs in the experiment
  --dont_melt_loss      Do not melt the train loss across the runs in the experiment
  --melt_sample_rollouts
                        Melt the sample rollouts across the runs in the experiment
```

## Quick Example

Copy the following content into a file named `test_study.py`:
```python
"""
Trains nonlinear emulators for the (linear) 1D advection equation under varying
difficulty in terms of the `advction_gamma` (=CFL).
"""

CONFIGS = [
    {
        "scenario": "diff_adv",
        "task": "predict",
        "net": net,
        "train": "one",
        "start_seed": 0,
        "num_seeds": 10,
        "advection_gamma": advection_gamma,
    }
    for net in [
        "Conv;34;10;relu",  # 31'757 params, 11 receptive field per direction
        "UNet;12;2;relu",  # 27'193 params, 29 receptive field per direction
        "Res;26;8;relu",  # 32'943 params, 16 receptive field per direction
        "FNO;12;18;4;gelu",  # 32'527 params, inf receptive field per direction
        "Dil;2;32;2;relu",  # 31'777 params, 20 receptive field per direction
    ]
    for advection_gamma in [
        0.5,
        2.5,
        10.5,
    ]
]
```

Then run the following command in the same directory as the file:
```bash
apebench test_study.py
```

This creates the following directory structure

``` { .bash .annotate }
📁 logs/ # (1)
📁 melted/ # (2)
    📄 metrics.csv
    📄 network_weights_list.json
    📄 raw_file_list.json
    📄 train_loss.csv
📁 results/ # (3)
    📁 network_weights/
    📁 raw/
```

1. Logging information, e.g., the APEBench version number and a copy of the used configuration file
2. Concatenated and melted data, ready for postprocessing with Pandas and seaborn
3. Raw files, e.g., the network weights and the raw data

For example in a Jupyter notebook you could use the following cell to display the metric rollout:

```python
import apebench
import pandas as pd
import seaborn as sns

train_df = pd.read_csv("melted/train_loss.csv")
train_df = apebench.read_in_kwargs(train_df)
facet = sns.relplot(
    data=train_df,
    x="time_step",
    y="mean_nRMSE",
    hue="net",
    kind="line",
    col="advection_gamma",
    errorbar=("pi", 50),
    estimator="median",
)
for ax in facet.axes.flat:
    ax.set_ylim(-0.05, 1.05)
```

## Using the CLI together with custom APEBench extensions

### Architectures

Similar to [here](extending_apebench.md#defining-your-own-architecture), you can
define your own architecture. You have to register it right before the config
file, like this
```python
import apebench
import pdequinox as pdeqx

def conv_net_extension(
    config_str: str,
    num_spatial_dims: int,
    num_points: int,
    num_channels: int,
    activation_fn,
    key,
):
    config_args = config_str.split(";")

    depth = int(config_args[1])

    return pdeqx.arch.ConvNet(
        num_spatial_dims=num_spatial_dims,
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=42,
        depth=depth,
        activation=activation_fn,
        key=key,
    )

apebench.components.architecture_dict.update(
    # Ensure that the key is in lower case
    {"myconvnet": conv_net_extension}
)

CONFIGS = [
    {
        "scenario": "diff_adv",
        "task": "predict",
        "net": "MyConvNet;42",
        "train": "one",
        "start_seed": 0,
        "num_seeds": 10,
    }
]
```

!!! warning

    The spelling of the key in the arch_extensions has to be lower case

### Scenarios

Similar to [here](extending_apebench.md#your-truly-own-scenario), a custom
scenario can be registered in the scenario dictionary. Let's assume, it is
defined in a file `my_scenario.py` as `MyScenario`:
```python
import apebench

from my_scenario import MyScenario

apebench.scenarios.scenario_dict.update(
    {"my_scenario": MyScenario}
)

CONFIGS = [
    {
        "scenario": "my_scenario",
        "task": "predict",
        "net": "Conv;34;10;relu",
        "train": "one",
        "start_seed": 0,
        "num_seeds": 10,
        "my_custom_arg": 42,  # Assuming MyScenario has this attribute
    }
]
```

Then you can run the experiment with
```bash
apebench test_study.py
```