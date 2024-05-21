import equinox as eqx
import jax
import jax.numpy as jnp

from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex
from ...pdequinox.pdequinox.conv import PhysicsConv


class FOULearning(BaseScenario):
    adv_difficulty: float = 0.75
    coarse_propotion: float = 0.5

    use_analytical: bool = False

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            raise ValueError("This scenario only supports 1D")

    def get_stencil(self, adv_difficulty: float, *, conv_format: bool = True):
        if adv_difficulty <= 0:
            # Forward diff
            left_coefficient = 0
            center_coefficient = 1 + adv_difficulty
            right_coefficient = -adv_difficulty
        else:
            # Backward diff
            left_coefficient = adv_difficulty
            center_coefficient = 1 - adv_difficulty
            right_coefficient = 0

        stencil = jnp.array([left_coefficient, center_coefficient, right_coefficient])

        if conv_format:
            stencil = jnp.reshape(stencil, (1, 1, -1))

        return stencil

    def build_fou_stepper(self, adv_difficulty: float):
        fou_stepper = PhysicsConv(
            num_spatial_dims=1,
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            use_bias=False,
            key=jax.random.PRNGKey(
                0
            ),  # does not matter because we will overwrite the kernel
            boundary_mode="periodic",
        )
        correct_stencil = self.get_stencil(adv_difficulty)
        fou_stepper = eqx.tree_at(
            lambda leaf: leaf.weight,
            fou_stepper,
            correct_stencil,
        )
        return fou_stepper

    def build_stepper(self, adv_difficulty: float) -> ex.BaseStepper:
        if self.use_analytical:
            return ex.normalized.DifficultyLinearStepper(
                num_spatial_dims=self.num_spatial_dims,
                num_points=self.num_points,
                difficulties=(0.0, -adv_difficulty),
            )
        else:
            return self.build_fou_stepper(adv_difficulty)

    def get_ref_stepper(self) -> ex.BaseStepper:
        return self.build_stepper(self.adv_difficulty)

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return self.build_stepper(self.adv_difficulty * self.coarse_propotion)

    def get_scenario_name(self) -> str:
        f"{self.num_spatial_dims}d_diff_fou"


class FTCSLearning(BaseScenario):
    diff_difficulty: float = 0.75
    coarse_propotion: float = 0.5

    use_analytical: bool = False

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            raise ValueError("This scenario only supports 1D")

    def get_stencil(self, diff_difficulty: float, *, conv_format: bool = True):
        left_coefficient = diff_difficulty / 2
        center_coefficient = 1 - diff_difficulty
        right_coefficient = diff_difficulty / 2

        stencil = jnp.array([left_coefficient, center_coefficient, right_coefficient])

        if conv_format:
            stencil = jnp.reshape(stencil, (1, 1, -1))

        return stencil

    def build_ftcs_stepper(self, diff_difficulty: float):
        ftcs_stepper = PhysicsConv(
            num_spatial_dims=1,
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            use_bias=False,
            key=jax.random.PRNGKey(
                0
            ),  # does not matter because we will overwrite the kernel
            boundary_mode="periodic",
        )
        correct_stencil = self.get_stencil(diff_difficulty)
        ftcs_stepper = eqx.tree_at(
            lambda leaf: leaf.weight,
            ftcs_stepper,
            correct_stencil,
        )
        return ftcs_stepper

    def build_stepper(self, diff_difficulty: float) -> ex.BaseStepper:
        if self.use_analytical:
            return ex.normalized.DifficultyLinearStepper(
                num_spatial_dims=self.num_spatial_dims,
                num_points=self.num_points,
                difficulties=(0.0, 0.0, diff_difficulty),
            )
        else:
            return self.build_ftcs_stepper(diff_difficulty)

    def get_ref_stepper(self) -> ex.BaseStepper:
        return self.build_stepper(self.diff_difficulty)

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return self.build_stepper(self.diff_difficulty * self.coarse_propotion)

    def get_scenario_name(self) -> str:
        f"{self.num_spatial_dims}d_diff_ftcs"
