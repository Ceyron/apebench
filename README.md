 ⚠️ ⚠️ ⚠️ This is a pre-release version of the package to test the PyPI workflow. Proper release **with breaking API changes** and extended documentation will be by end of October. ⚠️ ⚠️ ⚠️
<h4 align="center">A benchmark for Autoregressive PDE Emulators in <a href="https://github.com/google/jax" target="_blank">JAX</a>.</h4>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
    <a href="#documentation">Documentation</a> •
    <a href="#background">Background</a> •
    <a href="#acknowledgements">Acknowledgements</a>
</p>

<h1 align="center">
  <img src="https://github.com/user-attachments/assets/c6b88756-bc35-4e9a-8662-798a16f8302b" width="150">
</h1>


## Installation

```bash
pip install apebench
```

Requires Python 3.10+ and JAX 0.4.12+ 👉 [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).

Quick instruction with fresh Conda environment and JAX CUDA 12 on Linux.

```bash
conda create -n apebench python=3.12 -y
conda activate apebench
pip install -U "jax[cuda12]"
pip install apebench
```

## Quickstart

Train a ConvNet to emulate 1D advection, display train loss, test error metric
rollout, and a sample rollout.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SeCuoYaSfIH2J0IdNeFlDrkCypxtvRie?usp=sharing)

```python
import apebench
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

advection_scenario = apebench.scenarios.difficulty.Advection()

data, trained_nets = advection_scenario(
    task_config="predict",
    network_config="Conv;26;10;relu",
    train_config="one",
    num_seeds=3,
)

data_loss = apebench.melt_loss(data)
data_metrics = apebench.melt_metrics(data)
data_sample_rollout = apebench.melt_sample_rollouts(data)

fig, axs = plt.subplots(1, 3, figsize=(13, 3))

sns.lineplot(data_loss, x="update_step", y="train_loss", ax=axs[0])
axs[0].set_yscale("log")
axs[0].set_title("Training loss")

sns.lineplot(data_metrics, x="time_step", y="mean_nRMSE", ax=axs[1])
axs[1].set_ylim(-0.05, 1.05)
axs[1].set_title("Metric rollout")

axs[2].imshow(
    np.array(data_sample_rollout["sample_rollout"][0])[:, 0, :].T,
    origin="lower",
    aspect="auto",
    vmin=-1,
    vmax=1,
    cmap="RdBu_r",
)
axs[2].set_xlabel("time")
axs[2].set_ylabel("space")
axs[2].set_title("Sample rollout")

plt.show()
```

![](https://github.com/user-attachments/assets/10f968f4-2b30-4972-8753-22b7fad208ed)

You can explore the apebench scenarios using an interactive streamlit notebook
by running

```bash
streamlit run explore_sample_data_streamlit.py
```

## Documentation

Documentation is a available at
[fkoehler.site/apebench/](https://fkoehler.site/apebench/).

## Background

Autoregressive neural emulators can be used to efficiently forecast transient
phenomena, often associated with differential equations. Denote by
$\mathcal{P}_h$ a reference numerical simulator (e.g., the [FTCS
scheme](https://en.wikipedia.org/wiki/FTCS_scheme) for the heat equation). It
advances a state $u_h$ by

$$
u_h^{[t+1]} = \mathcal{P}_h(u_h^{[t]}).
$$

An autoregressive neural emulator $f_\theta$ is trained to mimic $\mathcal{P}_h$, i.e., $f_\theta \approx \mathcal{P}_h$. Doing so requires the following choices:

1. What is the reference simulator $\mathcal{P}_h$?
    1. What is its corresponding continuous transient partial differential
        equation? (advection, diffusion, Burgers, Kuramoto-Sivashinsky,
        Navier-Stokes, etc.)
    2. What consistent numerical scheme is used to discretize the continuous
        transient partial differential equation?
2. What is the architecture of the autoregressive neural emulator $f_\theta$?
3. How do $f_\theta$ and $\mathcal{P}_h$ interact during training (=optimization
    of $\theta$)?
    1. For how many steps are their predictions unrolled and compared?
    2. What is the time-level loss function?
    3. How large is the batch size?
    4. What is the opimizer and its learning rate scheduler?
    5. For how many steps is the training run?
4. Additional training and evaluation related choices:
    1. What is the initial condition distribution?
    2. How long is the time horizon seen during training?
    3. What is the evaluation metric? If it is related to an error rollout, for
        how many steps is the rollout?
    4. How many random seeds are used to draw conclusions?

APEBench is a framework to holistically assess all four ingredients. Component
(1), the discrete reference simulator $\mathcal{P}_h$, is provided by `Exponax`.
This is a suite of
[ETDRK](https://www.sciencedirect.com/science/article/abs/pii/S0021999102969950)-based
methods for semi-linear partial differential equations on periodic domains. This
covers a wide range of dynamics. For the most common scenarios, a unique
interface using normalized (non-dimensionalized) coefficients or a
difficulty-based interface (as described in the APEBench paper) can be used. The
second (2) component is given by `PDEquinox`. This library uses `Equinox`, a
JAX-based deep-learning framework, to implement many commonly found
architectures like convolutional ResNets, U-Nets, and FNOs. The third (3)
component is `Trainax`, an abstract implementation of "trainers" that provide
supervised rollout training and many other features. The fourth (4) component is
to wrap up the former three and is given by this repository.

### About APEBench

APEBench encapsulates the entire pipeline of training and evaluating an
autoregressive neural emulator in a scenario. A scenario is a callable
dataclass.

## Acknowledgements

### Citation

This package was developed as part of the `APEBench paper` (accepted at Neurips 2024), we will soon add the citation here.

### Funding

The main author (Felix Koehler) is a PhD student in the group of [Prof. Thuerey at TUM](https://ge.in.tum.de/) and his research is funded by the [Munich Center for Machine Learning](https://mcml.ai/).

### License

MIT, see [here](https://github.com/Ceyron/apebench/blob/main/LICENSE.txt)

---

> [fkoehler.site](https://fkoehler.site/) &nbsp;&middot;&nbsp;
> GitHub [@ceyron](https://github.com/ceyron) &nbsp;&middot;&nbsp;
> X [@felix_m_koehler](https://twitter.com/felix_m_koehler) &nbsp;&middot;&nbsp;
> LinkedIn [Felix Köhler](https://www.linkedin.com/in/felix-koehler)