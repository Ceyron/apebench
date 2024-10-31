from abc import ABC, abstractmethod
from typing import Callable

import equinox as eqx
import exponax as ex
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import pdequinox as pdeqx
import trainax as tx
from exponax import BaseStepper
from exponax.ic import BaseRandomICGenerator
from jaxtyping import Array, Float, PRNGKeyArray

from ._corrected_stepper import CorrectedStepper
from .components import (
    activation_fn_dict,
    architecture_dict,
    ic_dict,
    lr_scheduler_dict,
    metric_dict,
    optimizer_dict,
)


class BaseScenario(eqx.Module, ABC):
    # Setting up the discretization
    num_spatial_dims: int = 1
    num_points: int = 160

    # Abstract information about the problem
    num_channels: int = 1

    # Settings for both training and testing
    ic_config: str = "fourier;5;true;true"
    num_warmup_steps: int = 0

    # Setting up the training
    num_train_samples: int = 50
    train_temporal_horizon: int = 50
    train_seed: int = 0

    # For testing
    num_test_samples: int = 30
    test_temporal_horizon: int = 200
    test_seed: int = 773

    # For the training configuration
    optim_config: str = "adam;10_000;warmup_cosine;0.0;1e-3;2_000"
    batch_size: int = 20

    # Information for inspection
    num_trjs_returned: int = 1
    record_loss_every: int = 100
    vlim: tuple[float, float] = (-1.0, 1.0)
    report_metrics: str = "mean_nRMSE"  # separate by commas ","
    callbacks: str = ""  # separate by commas ","

    def get_ic_generator(self) -> BaseRandomICGenerator:
        """
        Overwrite for custom initial condition generation.

        Uses the `ic_config` attribute to determine the type of initial
        condition generation.

        Allows for the following options:

        - `fourier;CUTOFF;ZERO_MEAN;MAX_ONE` for a truncated Fourier series
            with CUTOFF (int) number of modes, ZERO_MEAN (bool) for zero mean,
            and MAX_ONE (bool) for having the initial condition being at max in
            (-1, 1) but not clamped to it
        - `diffused;INTENSITY;ZERO_MEAN;MAX_ONE` for a diffused noise with
            INTENSITY (float) for the intensity, ZERO_MEAN (bool) for zero mean,
            and MAX_ONE (bool) for having the initial condition being at max in
            (-1, 1) but not clamped to it
        - `grf;POWERLAW_EXPONENT;ZERO_MEAN;MAX_ONE` for a Gaussian random
            field with POWERLAW_EXPONENT (float) for the powerlaw exponent,
            ZERO_MEAN (bool) for zero mean, and MAX_ONE (bool) for having the
            initial condition being at max in (-1, 1) but not clamped to it
        - `clamp;LOWER_BOUND;UPPER_BOUND;CONFIG` for clamping the
            configuration to the range of LOWER_BOUND (float) to UPPER_BOUND
            (float) and then using the configuration CONFIG for the generation
            of the initial condition
        """

        def _get_single_channel(config):
            ic_name = config.split(";")[0]
            ic_gen = ic_dict[ic_name.lower()](config, self.num_spatial_dims)
            return ic_gen

        ic_args = self.ic_config.split(";")
        if ic_args[0].lower() == "clamp":
            lower_bound = float(ic_args[1])
            upper_bound = float(ic_args[2])

            ic_gen = _get_single_channel(";".join(ic_args[3:]))
            ic_gen = ex.ic.ClampingICGenerator(
                ic_gen,
                limits=(lower_bound, upper_bound),
            )
        else:
            ic_gen = _get_single_channel(self.ic_config)

        multi_channel_ic_gen = ex.ic.RandomMultiChannelICGenerator(
            [
                ic_gen,
            ]
            * self.num_channels
        )

        return multi_channel_ic_gen

    @abstractmethod
    def get_ref_stepper(self) -> BaseStepper:
        """
        Produces the reference stepper for the scenario.
        """
        pass

    @abstractmethod
    def get_coarse_stepper(self) -> BaseStepper:
        """
        Produces the coarse stepper for the scenario.
        """
        pass

    @abstractmethod
    def get_scenario_name(self) -> str:
        """
        Produces a unique identifier for this scenario
        """
        pass

    @property
    def num_training_steps(self):
        optim_args = self.optim_config.split(";")
        return int(optim_args[1])

    def get_optimizer(self) -> optax.GradientTransformation:
        """
        Returns the optimizer used in the scenario.
        """
        optim_args = self.optim_config.split(";")
        optimizer_name = optim_args[0]
        num_training_steps = int(optim_args[1])
        scheduler_args = optim_args[2:]
        scheduler_name = scheduler_args[0]

        lr_scheduler = lr_scheduler_dict[scheduler_name.lower()](
            ";".join(scheduler_args), num_training_steps
        )
        optimizer = optimizer_dict[optimizer_name.lower()](self.optim_config)(
            lr_scheduler
        )

        return optimizer

    def produce_data(
        self,
        *,
        stepper: BaseStepper,
        num_samples: int,
        num_warmup_steps: int,
        temporal_horizon: int,
        key: PRNGKeyArray,
    ) -> Float[Array, "num_samples temporal_horizon+1 num_channels *num_points"]:
        """
        Default generation of data:

        1. Instantiate the intial condition distribution
        2. Generate the number of initial conditions as samples requested and
           discretize them on the grid
        3. Warmup the initial conditions if necessary
        4. Rollout these initial conditions for as many time steps as in the
           configuration

        The returned array has the shape:

        (num_train_samples, train_temporal_horizon+1, num_channels,) +
        (num_points, ) * num_spatial_dims

        the last axes are as many (num_points,) axis as there are spatial
        dimensions.
        """

        ic_distribution = self.get_ic_generator()
        ic_set = ex.build_ic_set(
            ic_distribution,
            num_points=self.num_points,
            num_samples=num_samples,
            key=key,
        )
        warmed_up_ic_set = jax.vmap(
            ex.repeat(
                stepper,
                num_warmup_steps,
            )
        )(ic_set)
        trj_set = jax.vmap(
            ex.rollout(
                stepper,
                temporal_horizon,
                include_init=True,
            )
        )(warmed_up_ic_set)

        return trj_set

    def get_train_data(
        self,
    ) -> Float[
        Array, "num_train_samples train_temporal_horizon+1 num_channels *num_points"
    ]:
        """
        Use the attributes to produce the reference training data.
        """
        return self.produce_data(
            stepper=self.get_ref_stepper(),
            num_samples=self.num_train_samples,
            num_warmup_steps=self.num_warmup_steps,
            temporal_horizon=self.train_temporal_horizon,
            key=jax.random.PRNGKey(self.train_seed),
        )

    def get_train_data_coarse(
        self,
    ) -> Float[
        Array, "num_train_samples train_temporal_horizon+1 num_channels *num_points"
    ]:
        """
        Use the attributes to produce training data with the coarse stepper instead.
        """
        return self.produce_data(
            stepper=self.get_coarse_stepper(),
            num_samples=self.num_train_samples,
            num_warmup_steps=self.num_warmup_steps,
            temporal_horizon=self.train_temporal_horizon,
            key=jax.random.PRNGKey(self.train_seed),
        )

    def get_test_data(
        self,
    ) -> Float[
        Array, "num_test_samples test_temporal_horizon+1 num_channels *num_points"
    ]:
        """
        Use the attributes to produce the reference testing data.
        """
        return self.produce_data(
            stepper=self.get_ref_stepper(),
            num_samples=self.num_test_samples,
            num_warmup_steps=self.num_warmup_steps,
            temporal_horizon=self.test_temporal_horizon,
            key=jax.random.PRNGKey(self.test_seed),
        )

    def get_test_data_coarse(
        self,
    ) -> Float[
        Array, "num_test_samples test_temporal_horizon+1 num_channels *num_points"
    ]:
        """
        Use the attributes to produce testing data with the coarse stepper instead.
        """
        return self.produce_data(
            stepper=self.get_coarse_stepper(),
            num_samples=self.num_test_samples,
            num_warmup_steps=self.num_warmup_steps,
            temporal_horizon=self.test_temporal_horizon,
            key=jax.random.PRNGKey(self.test_seed),
        )

    def get_ref_sample_data(
        self,
    ) -> Float[
        Array, "num_trjs_returned test_temporal_horizon+1 num_channels *num_points"
    ]:
        """
        Return a subset of the testing data, the number of samples is defined by
        the attribute `num_trjs_returned`
        """
        test_trj_set = self.get_test_data()
        test_trj_subset = test_trj_set[: self.num_trjs_returned]
        return test_trj_subset

    def get_callback_fn(self) -> tx.callback.BaseCallback:
        """
        Parse the `callbacks` attribute to a list of callable functions.
        """
        callback_configurations = self.callbacks.split(",")

        callback_fns = []

        for callback in callback_configurations:
            callback_args = callback.split(";")
            if callback_args[0] == "net":
                every = int(callback_args[1])
                callback_fns.append(tx.callback.GetNetwork(every=every, name="net"))
            elif callback_args[0] == "weight_norm":
                every = int(callback_args[1])
                callback_fns.append(
                    tx.callback.WeightNorm(every=every, name="weight_norm")
                )
            elif callback_args[0] == "metrics":
                every = int(callback_args[1])

                def metrics_callback_fn(update_i, model, data):
                    if update_i % every == 0:
                        metrics = self.perform_tests(model, remove_singleton_axis=True)
                        return {"metrics": metrics}
                    else:
                        return {"metrics": None}

                callback_fns.append(metrics_callback_fn)
            elif callback_args[0] == "sample_rollout":
                every = int(callback_args[1])

                def sample_rollout_callback_fn(update_i, model, data):
                    if update_i % every == 0:
                        sample_rollout = self.sample_trjs(model)
                        return {"sample_rollout": sample_rollout}
                    else:
                        return {"sample_rollout": None}

                callback_fns.append(sample_rollout_callback_fn)
            elif callback_args[0] == "":
                continue
            else:
                raise ValueError(f"Unknown callback '{callback}'")

        callback_fn = tx.callback.CompositeCallback(callback_fns)

        return callback_fn

    def get_trainer(self, *, train_config: str) -> tx.GeneralTrainer:
        """
        Expects a str of the defined interface for study. In the default
        configuration, it could for instance accept:

        'sup-03' for supervised rollout trainig with three rollout steps.

        Currently, the three major categories are available:

        - 'one' for one step supervised training
        - 'sup-XX' for supervised training with XX rollout steps
        - 'div-XX' for diverted chain training with XX rollout steps
        """
        train_trjs = self.get_train_data()

        # Needed for diverted chain training
        ref_stepper = self.get_ref_stepper()
        train_args = train_config.split(";")

        optimizer = self.get_optimizer()

        callback_fn = self.get_callback_fn()

        if train_args[0].lower() == "one":
            trainer = tx.trainer.SupervisedTrainer(
                train_trjs,
                optimizer=optimizer,
                num_training_steps=self.num_training_steps,
                batch_size=self.batch_size,
                num_rollout_steps=1,
                cut_bptt=False,
                time_level_weights=None,
                callback_fn=callback_fn,
            )
        elif train_args[0].lower() == "sup":
            num_rollout_steps = int(train_args[1])
            if len(train_args) > 2:
                cut_bptt = train_args[2].lower() == "true"
            else:
                cut_bptt = False
            trainer = tx.trainer.SupervisedTrainer(
                train_trjs,
                optimizer=optimizer,
                num_training_steps=self.num_training_steps,
                batch_size=self.batch_size,
                num_rollout_steps=num_rollout_steps,
                cut_bptt=cut_bptt,
                time_level_weights=None,
                callback_fn=callback_fn,
            )
        elif train_args[0].lower() == "div":
            num_rollout_steps = int(train_args[1])
            if len(train_args) > 2:
                cut_bptt = train_args[2].lower() == "true"
            else:
                cut_bptt = False
            if len(train_args) > 3:
                cut_div_chain = train_args[3].lower() == "true"
            else:
                cut_div_chain = False
            trainer = tx.trainer.DivertedChainBranchOneTrainer(
                train_trjs,
                ref_stepper=ref_stepper,
                optimizer=optimizer,
                num_training_steps=self.num_training_steps,
                batch_size=self.batch_size,
                num_rollout_steps=num_rollout_steps,
                cut_bptt=cut_bptt,
                cut_div_chain=cut_div_chain,
                time_level_weights=None,
                callback_fn=callback_fn,
            )
        else:
            raise ValueError(f"Unknown training argument '{train_config}'")

        return trainer

    def get_activation(
        self,
        activation: str,
    ) -> Callable:
        """
        Parse a string to a callable activation function.
        """
        activation_fn_name = activation.split(";")[0]
        activation_fn = activation_fn_dict[activation_fn_name.lower()](activation)
        return activation_fn

    def get_network(
        self,
        network_config: str,
        key: PRNGKeyArray,
    ) -> eqx.Module:
        """
        Parse the `network_config` to the corresponding neural architectue and
        instantiate it, use the `key` to initialize the parameters.

        "Conv;34;10;relu" for a feedforward convolutional network with 34 hidden
        channels, 10 hidden layers, and the ReLU activation function.

        Currently, the following constructors are available:

        - `Conv;HIDDEN_CHANNELS;DEPTH;ACTIVATION`: A feedforward
            convolutional network with `DEPTH` hidden layers of `WIDTH` size.
            Each layer transition except for the last uses `ACTIVATION`. The
            effective receptive field is `DEPTH + 1`
        - `Res;WIDTH;BLOCKS;ACTIVATION`: A classical/legacy ResNet with
            post-activation and no normalization scheme. Each residual block has
            two convolutions and operates at `WIDTH` channel size. The
            `ACTIVATION` follows each of the convolutions in the residual block.
            There are `BLOCKS` number of residual blocks. Lifting and projection
            are point-wise linear convolutions (=1x1 convs).
        - `UNet;WIDTH;LEVELS;ACTIVATION`: A classical UNet using double
            convolution blocks with group activation in-between (number of
            groups is set to one). `WIDTH` describes the hidden layer's size on
            the highest resolution level. `LEVELS` indicates the number of times
            the spatial resolution is halved by a factor of two while the
            channel count doubles. Skip connections exist between the encoder
            and decoder part of the network.
        - `Dil;DIL-FACTOR;WIDTH;BLOCKS;ACTIVATION`: Similar to the
            classical post-activation ResNet but uses a series of stacked
            convolutions of different dilation rates. Each convolution is
            followed by a group normalization (number of groups is set to one)
            and the `ACTIVATION`. `DIL-FACTOR` of 1 refers to one convolution of
            dilation rate 1. If it is set to 2, this refers to three
            convolutions of rates [1, 2, 1]. If it is 3, then this is [1, 2, 4,
            2, 1], etc.
        - `FNO;MODES;WIDTH;BLOCKS;ACTIVATION`: A vanilla FNO using spectral
            convolutions with `MODES` equally across all spatial dimensions.
            Each block operates at `WIDTH` channel size and has one spectral
            convolution with a point-wise linear bypass. The activation is
            applied to the sum of spectral convolution and bypass result. There
            are `BLOCKS` total blocks. Lifting and projection are point-wise
            linear (=1x1) convolutions.
        - `MLP;WIDTH;DEPTH;ACTIVATION`: A multi-layer perceptron with `DEPTH`
            hidden layers of `WIDTH` size. Each layer transition except for the
            last uses `ACTIVATION`. Channel and spatial axes are flattened into
            one feature axis. Hence, the MLP is hard-coded to one specific
            resolution.
        - `Pure;KERNEL_SIZE`: A purely linear convolution (with no bias) with
            kernel size `KERNEL_SIZE`. Use this to learn finite difference
            stencils. It has as many learnable parameters as the kernel size.
        - `MoRes;WIDTH;BLOCKS;ACTIVATION`: A modern ResNet using pre-activation
            and group normalization. Each residual block has two convolutions
            and operates at `WIDTH` channel size. The `ACTIVATION` follows each
            of the convolutions in the residual block. There are `BLOCKS` number
            of residual blocks. Lifting and projection are point-wise linear
            convolutions (=1x1 convs).
        - `MoUNet;WIDTH;LEVELS;ACTIVATION`: A modern UNet using two resnet
            blocks per level. `WIDTH` describes the hidden layer's size on the
            highest resolution level. `LEVELS` indicates the number of times the
            spatial resolution is halved by a factor of two while the channel
            count doubles. Skip connections exist between the encoder and
            decoder part of the network.

        The `key` is used to initialize the parameters of the neural network.

        To registor your custom architecture use the `arch_extensions`
        dictionary.

        Returns:

        - `network`: eqx.Module, the neural architecture
        """
        network_args = network_config.split(";")

        network_name = network_args[0]
        activation_fn_config = network_args[-1]
        activation_fn = self.get_activation(activation_fn_config)

        network = architecture_dict[network_name.lower()](
            network_config,
            self.num_spatial_dims,
            self.num_points,
            self.num_channels,
            activation_fn,
            key,
        )

        return network

    def get_neural_stepper(
        self, *, task_config: str, network_config: str, key: PRNGKeyArray
    ) -> eqx.Module:
        """
        Use the `network_config` to instantiate the neural architecture with
        `key` for the initial parameters. Then use the `task_config` to
        determine the wrapper around the neural architecture.

        If the `task_config` is 'predict', the neural architecture is returned
        directly.

        If the `task_config` is 'correct;XX', the neural architecture is wrapped
        in a `CorrectedStepper` with `XX` as the mode. Supported modes are:

        - `sequential`
        - `parallel`
        - `sequential_with_bypass`
        """
        network = self.get_network(network_config, key)

        task_args = task_config.split(";")
        if task_args[0] == "predict":
            neural_stepper = network
        elif task_args[0] == "correct":
            coarse_stepper = self.get_coarse_stepper()
            neural_stepper = CorrectedStepper(
                coarse_stepper=coarse_stepper,
                network=network,
                mode=task_args[1],
            )
        else:
            raise ValueError("Unknown task argument")

        return neural_stepper

    def get_parameter_count(
        self,
        network_config: str,
    ) -> int:
        """
        Count the number of parameters associated with `network_config` str.

        Note that this depends on `self.num_spatial_dims`, `self.num_channels,
        and in some cases (so far only the MLP) on `self.num_points`.
        """
        neural_stepper = self.get_neural_stepper(
            task_config="predict",  # Gives pure network without any arrays in the coarse stepper mistakingly considered as parameters
            network_config=network_config,
            key=jax.random.PRNGKey(0),  # Does not matter
        )
        return pdeqx.count_parameters(neural_stepper)

    def get_receptive_field(
        self,
        *,
        network_config: str,
        task_config: str,
    ) -> tuple[tuple[int, int], ...]:
        """
        Return the receptive field of the neural architecture for the given
        configuration.
        """
        neural_stepper = self.get_neural_stepper(
            task_config=task_config,
            network_config=network_config,
            key=jax.random.PRNGKey(0),  # Does not matter
        )
        return neural_stepper.receptive_field

    def load_model(
        self,
        path,
        *,
        num_seeds: int,
        task_config: str,
        network_config: str,
        remove_singleton_axis: bool = True,
    ) -> eqx.Module:
        """
        Load the model from the given path.
        """

        def get_stepper(i):
            return self.get_neural_stepper(
                task_config=task_config,
                network_config=network_config,
                key=jax.random.PRNGKey(i),  # Does not matter
            )

        if num_seeds == 1 and remove_singleton_axis:
            neural_stepper = get_stepper(0)
        else:
            neural_stepper = eqx.filter_vmap(get_stepper)(jnp.arange(num_seeds))
        neural_stepper = eqx.tree_deserialise_leaves(path, neural_stepper)
        return neural_stepper

    def full_loss(
        self,
        model: eqx.Module,
        *,
        train_config: str,
    ) -> float:
        """
        Computes the loss of the model on the entire training set in the
        configuration denoted by `train_config`.
        """
        trainer = self.get_trainer(train_config=train_config)
        loss = trainer.full_loss(model)
        return loss

    def perform_test_rollout(
        self,
        neural_stepper: eqx.Module,
        mean_error_fn: Callable = lambda pred, ref: ex.metrics.mean_metric(
            ex.metrics.nRMSE,
            pred,
            ref,
        ),
    ) -> Float[Array, "test_temporal_horizon"]:
        """
        Rollout the neural stepper starting from the test initial condition and
        compare it to the reference trajectory.
        """
        test_trjs = self.get_test_data()
        test_ics = test_trjs[:, 0]
        ref_trjs = test_trjs[:, 1:]
        pred_trjs = jax.vmap(
            ex.rollout(
                neural_stepper,
                self.test_temporal_horizon,
                include_init=False,
            )
        )(test_ics)

        error_rollout = jax.vmap(
            mean_error_fn,
            in_axes=1,  # over the temporal axis
        )(pred_trjs, ref_trjs)

        return error_rollout

    def get_metric_fns(
        self,
    ) -> dict[
        str,
        Callable[
            [
                Float[
                    Array,
                    "num_samples num_channels *num_points",
                ],
                Float[Array, "num_samples num_channels *num_points"],
            ],
            float,
        ],
    ]:
        """
        Return a dictionary with all metric functions according to the
        `report_metrics` attribute.
        """
        metric_fn_dict = {}

        for metric_config in self.report_metrics.split(","):
            metric_args = metric_config.split(";")
            metric_name = metric_args[0]
            metric_constructor = metric_dict[metric_name]
            metric_fn = metric_constructor(metric_config)
            metric_fn_dict[metric_config] = metric_fn

        return metric_fn_dict

    def perform_tests(
        self,
        neural_stepper: eqx.Module,
        *,
        remove_singleton_axis: bool = False,
    ) -> dict[str, Float[Array, "test_temporal_horizon"]]:
        """
        Computes all metrics according to the `report_metrics` attribute.
        """
        metric_function_dict = self.get_metric_fns()

        results = {}

        for metric_config, func in metric_function_dict.items():
            exec_func = lambda model: self.perform_test_rollout(model, func)
            if remove_singleton_axis:
                # add singleton axis for compatibility
                results[metric_config] = exec_func(neural_stepper)[None]
            else:
                results[metric_config] = eqx.filter_vmap(exec_func)(neural_stepper)

        return results

    def sample_trjs(
        self, neural_stepper: eqx.Module
    ) -> Float[
        Array, "num_trjs_returned test_temporal_horizon+1 num_channels *num_points"
    ]:
        """
        Use the neural_stepper to produce a sample of trajectories. The initial
        conditions are the ones from the test set.
        """
        test_trjs = self.get_test_data()
        test_ics_subset = test_trjs[: self.num_trjs_returned, 0]
        sample_trj_s = jax.vmap(
            ex.rollout(
                neural_stepper,
                self.test_temporal_horizon,
                include_init=True,
            )
        )(test_ics_subset)
        return sample_trj_s

    def run_raw(
        self,
        *,
        task_config: str = "predict",
        network_config: str = "Conv;26;10;relu",
        train_config: str = "one",
        start_seed: int = 0,
        num_seeds: int = 1,
        remove_singleton_axis: bool = False,
    ):
        """
        For more details see the __call__ method.

        Use this function if you intend to wrap your run in further vmaps.

        **Returns:**

        - `trained_neural_stepper_s`: eqx.Module, the trained neural stepper
            for the scenario. If `num_seeds` is 1, the singleton dimension along
            the batch axis is removed (if `remove_singleton_axis` is True).
        - `loss_history_s`: Array, the loss history of the training. The
            shape is `(num_seeds, num_training_steps//record_loss_every)`
        - `aux_history_s`: Array, the auxiliary history of the training. The
            shape is `(num_seeds, num_training_steps)`
        - `metric_trj_s`: dict, the metrics computed on the test set. The
            keys are the metric names and the values are arrays with the shape
            `(num_seeds, test_temporal_horizon)`
        - `sample_rollout_s`: Array, the sample rollouts produced by the
            trained neural stepper. The shape is `(num_seeds, num_trjs_returned,
            test_temporal_horizon+1, num_channels, *num_points)`
        - `seeds`: Array, the seeds used for the run
        """
        trainer = self.get_trainer(train_config=train_config)

        def produce_result_one_seed(seed):
            key = jax.random.PRNGKey(seed)
            init_key, shuffle_key = jax.random.split(key, 2)
            neural_stepper = self.get_neural_stepper(
                task_config=task_config,
                network_config=network_config,
                key=init_key,
            )
            trained_neural_stepper, loss_history, aux_history = trainer(
                neural_stepper,
                shuffle_key,
                record_loss_every=self.record_loss_every,
            )

            sample_rollout = self.sample_trjs(trained_neural_stepper)

            return (
                trained_neural_stepper,
                loss_history,
                aux_history,
                # mean_nRMSE_rollout,
                sample_rollout,
            )

        seeds = start_seed + jnp.arange(num_seeds)

        # Adds additional batch axis to the output of produce_result_one_seed
        (
            trained_neural_stepper_s,
            loss_history_s,
            aux_history_s,
            # error_trj_s,
            sample_rollout_s,
        ) = eqx.filter_vmap(produce_result_one_seed)(seeds)

        metric_trj_s = self.perform_tests(trained_neural_stepper_s)

        results = (
            trained_neural_stepper_s,
            loss_history_s,
            aux_history_s,
            metric_trj_s,
            sample_rollout_s,
            seeds,
        )

        # If only one seed is considered, remove the singleton axis if requested
        if num_seeds == 1 and remove_singleton_axis:
            results = pdeqx.extract_from_ensemble(results, 0)

        return results

    def __call__(
        self,
        *,
        task_config: str = "predict",
        network_config: str = "Conv;34;10;relu",
        train_config: str = "one",
        start_seed: int = 0,
        num_seeds: int = 1,
        remove_singleton_axis: bool = True,
    ) -> tuple[pd.DataFrame, eqx.Module]:
        """
        Execute the scenario with the given attribute configuration and the
        additional configuration strings.

        **Arguments:**

        - `task_config`: What the trained neural predictor should
            represent. Can be either 'predict' or 'correct;XX' where XX is the
            mode of correction. `predict` refers to a pure neural architecture.
            The neural network will fully replace the numerical timestepper. In
            the case of `correct;XX`, the neural network interacts with a coarse
            stepper. To inference such a system after training, the
            corresponding coarse solver is needed, but is already baked into the
            returning module. Default is 'predict'.
        - `network_config`: The configuration of the neural network.
            This begins with a general architecture type, followed by a
            architecture-dependent length list of parameters. See the method
            `get_network` for the available architectures and their
            configuration. Default is 'Conv;34;10;relu' which is a feed-forward
            convolutional network with 34 hidden channels over 10 hidden layers
            and the ReLU activation function (about 30k parameters for 1D
            problems)
        - `train_config`: The training configuration. This determines
            how neural stepper and reference numerical stepper interact during
            training. See the method `get_trainer` for the available training
            configurations. Default is 'one' which refers to a one-step
            supervised approach in which one batch of samples with a length 2
            window is sampled across all initial conditions and temporal
            horizon.
        - `start_seed`: The starting seed for the random number
            generator of network initialization. Default is 0.
        - `num_seeds`: The number of seeds to use. Default is 1.
        - `remove_singleton_axis`: bool, if True and `num_seeds` is 1, the
            singleton axis resulting from the seed parallel runs is removed
            which allows to directly use the returned neural stepper. Otherwise,
            it must be wrapped in a `eqx.filter_vmap(...)`

        **Returns:**

        - `result_df`: A dataframe with the results of the
            scenario. Each row represents one seed. It contains the following
            columns:
            - 'scenario': str, the name of the scenario, created by the
                method `get_scenario_name`
            - 'task': str, the task configuration (as given in the
                argument)
            - 'train': str, the training configuration (as given in the
                argument)
            - 'net': str, the network configuration (as given in the
                argument)
            - 'seed': int, the seed used for the run (this varies
                between the rows if multiple seeds are used at the same time)
            - 'mean_nRMSE_XXXX': float, the mean nRMSE metric produced
                in an error rollout **after the training has finished**. Each
                temporal entry (staring at 1 all the way to
                `self.test_temporal_horizon`) is represented by a separate
                column.
            - `METRICS_XXXX`: float, additional metrics (e.g., mean
                correlation rollout)
            - 'train_loss_XXXXXX': float, the training loss at each
                training step. Each step is represented by a separate column
                (starting at 0 all the way to `self.num_training_steps - 1`)
            - 'aux_XXXXXX': list, the history of auxiliary information
                produced by callbacks. If there is no callback active, each
                entry is an empty dictionary.
            - 'sample_rollout_XXX': list, a list of lists representing
                the sample rollouts produced by the trained neural stepper. The
                outer list represents the different initial conditions, the
                inner lists represent the different time steps. The length of
                the outer list is given by the attribute `num_trjs_returned`. We
                use list to store (jax.)numpy arrays.
        - `trained_neural_stepper_s`: eqx.Module, the trained neural stepper
            for the scenario. This follows an structure of arrays approach to
            represent the collection of networks trained based on different
            initialization seeds. If `num_seeds` is 1 (it is only intended to
            train one network), use the `remove_singleton_axis` argument to
            remove the singleton dimension (True by default).

        !!! note
            A typical workflow is to use the functions
            `apebench.utils.melt_loss`, `apebench.utils.melt_metrics`, and
            `apebench.utils.melt_sample_rollouts` to melt the returned dataframe
            into a long format that can be used for plotting with seaborn.
        """
        (
            trained_neural_stepper_s,
            loss_history_s,
            aux_history_s,
            metric_trj_s,
            sample_rollout_s,
            seeds,
        ) = self.run_raw(
            task_config=task_config,
            network_config=network_config,
            train_config=train_config,
            start_seed=start_seed,
            num_seeds=num_seeds,
            remove_singleton_axis=False,
        )

        n_training_steps = loss_history_s.shape[-1]
        n_sample_rollouts_returned = sample_rollout_s.shape[1]

        scenario_name = self.get_scenario_name()

        metric_dicts = []
        for metric, metric_trj in metric_trj_s.items():
            metric_dicts.append(
                {
                    f"{metric}_{i+1:04d}": metric_trj[:, i]  # noqa: E226
                    for i in range(self.test_temporal_horizon)
                }
            )

        aux_dicts = []
        for i, entry in enumerate(aux_history_s):
            aux_dicts.append(
                {
                    f"aux_{i:06d}": [
                        pdeqx.extract_from_ensemble(entry, j) for j in range(num_seeds)
                    ]
                }
            )

        result_df = pd.DataFrame(
            dict(
                **{
                    "scenario": scenario_name,
                    "task": task_config,
                    "train": train_config,
                    "net": network_config,
                    "seed": seeds,
                    # Needed for being compliant with multi-experiment interface
                    "scenario_kwargs": "{}",
                },
                **{
                    key: value
                    for sub_dict in metric_dicts
                    for key, value in sub_dict.items()
                },
                **{
                    f"train_loss_{(i * self.record_loss_every):06d}": loss_history_s[
                        :, i
                    ]
                    for i in range(n_training_steps)
                },
                **{
                    key: value
                    for sub_dict in aux_dicts
                    for key, value in sub_dict.items()
                },
                **{
                    f"sample_rollout_{i:03d}": sample_rollout_s[:, i].tolist()
                    for i in range(n_sample_rollouts_returned)
                },
            )
        )

        # If there is only one seed considered, remove the singleton dimension
        # in the weight arrays
        if num_seeds == 1 and remove_singleton_axis:
            trained_neural_stepper_s = pdeqx.extract_from_ensemble(
                trained_neural_stepper_s,
                0,
            )

        return result_df, trained_neural_stepper_s
