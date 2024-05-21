import equinox as eqx
import jax
import jax.numpy as jnp

from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex
from ...pdequinox.pdequinox.conv import PhysicsConv


class FOULearning(BaseScenario):
    advectivity: float = 0.01
    coarse_propotion: float = 0.5

    use_analytical: bool = False

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            raise ValueError("This scenario only supports 1D")

    def get_stencil(self, advectivity: float, *, conv_format: bool = True):
        # advectivity = c * dt / L

        # advectivity * N = c * dt / dx

        c_dt_by_dx = advectivity * self.num_points

        if advectivity <= 0:
            # Forward diff
            left_coefficient = 0
            center_coefficient = 1 + c_dt_by_dx
            right_coefficient = -c_dt_by_dx
        else:
            # Backward diff
            left_coefficient = c_dt_by_dx
            center_coefficient = 1 - c_dt_by_dx
            right_coefficient = 0

        stencil = jnp.array([left_coefficient, center_coefficient, right_coefficient])

        if conv_format:
            stencil = jnp.reshape(stencil, (1, 1, -1))

        return stencil

    def build_fou_stepper(self, advectivity: float):
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
        correct_stencil = self.get_stencil(advectivity)
        fou_stepper = eqx.tree_at(
            lambda leaf: leaf.weight,
            fou_stepper,
            correct_stencil,
        )
        return fou_stepper

    def build_stepper(self, advectivity: float):
        if self.use_analytical:
            return ex.normalized.NormalizedLinearStepper(
                num_spatial_dims=self.num_spatial_dims,
                num_points=self.num_points,
                normalized_coefficients=(0.0, -advectivity),
            )
        else:
            return self.build_fou_stepper(advectivity)

    def get_ref_stepper(self):
        return self.build_stepper(self.advectivity)

    def get_coarse_stepper(self):
        return self.build_stepper(self.advectivity * self.coarse_propotion)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_fou"


class FTCSLearning(BaseScenario):
    diffusivity: float = 0.0001
    coarse_propotion: float = 0.5

    use_analytical: bool = False

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            raise ValueError("This scenario only supports 1D")

    def get_stencil(self, diffusivity: float, *, conv_format: bool = True):
        # diffusivity = nu * dt / L^2

        # diffusivity * N^2 = nu * dt / dx^2

        k_dt_by_dx2 = diffusivity * self.num_points**2

        left_coefficient = k_dt_by_dx2
        center_coefficient = 1 - 2 * k_dt_by_dx2
        right_coefficient = k_dt_by_dx2

        stencil = jnp.array([left_coefficient, center_coefficient, right_coefficient])

        if conv_format:
            stencil = jnp.reshape(stencil, (1, 1, -1))

        return stencil

    def build_ftcs_stepper(self, diffusivity: float):
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
        correct_stencil = self.get_stencil(diffusivity)
        ftcs_stepper = eqx.tree_at(
            lambda leaf: leaf.weight,
            ftcs_stepper,
            correct_stencil,
        )
        return ftcs_stepper

    def build_stepper(self, diffusivity: float):
        if self.use_analytical:
            return ex.normalized.NormalizedLinearStepper(
                num_spatial_dims=self.num_spatial_dims,
                num_points=self.num_points,
                normalized_coefficients=(0.0, 0.0, diffusivity),
            )
        else:
            return self.build_ftcs_stepper(diffusivity)

    def get_ref_stepper(self):
        return self.build_stepper(self.diffusivity)

    def get_coarse_stepper(self):
        return self.build_stepper(self.diffusivity * self.coarse_propotion)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_ftcs"
