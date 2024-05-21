import jax.numpy as jnp

from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class DecayingTurbulence(BaseScenario):
    num_spatial_dims: int = 2
    num_points: int = 160
    domain_extent: float = 1.0
    dt: float = 0.1

    diffusivity: float = 1e-4

    coarse_proportion: float = 0.5

    def __post_init__(self):
        if self.num_spatial_dims != 2:
            raise ValueError(
                "Decaying turbulence is only defined for 2 spatial dimensions"
            )

    def get_ref_stepper(self) -> ex.BaseStepper:
        return ex.stepper.NavierStokesVorticity(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            self.dt,
            diffusivity=self.diffusivity,
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.stepper.NavierStokesVorticity(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            self.dt * self.coarse_proportion,
            diffusivity=self.diffusivity,
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_decay_turb"


class KolmogorovFlow(BaseScenario):
    num_spatial_dims: int = 2
    num_points: int = 160
    domain_extent: float = 2 * jnp.pi
    dt: float = 0.01

    diffusivity: float = 1e-2  # Just Re=100 to have it almost scale-resolved

    injection_mode: int = 4
    injection_scale: float = 1.0
    drag: float = -0.1

    coarse_proportion: float = 0.5

    num_warmup_steps: int = 2500

    vlim: tuple[float, float] = (-10.0, 10.0)

    def __post_init__(self):
        if self.num_spatial_dims != 2:
            raise ValueError("Kolmogorov Flow is only defined for 2 spatial dimensions")

    def get_ref_stepper(self) -> ex.BaseStepper:
        return ex.stepper.KolmogorovFlowVorticity(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            self.dt,
            diffusivity=self.diffusivity,
            injection_mode=self.injection_mode,
            injection_scale=self.injection_scale,
            drag=self.drag,
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.stepper.KolmogorovFlowVorticity(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            self.dt * self.coarse_proportion,
            diffusivity=self.diffusivity,
            injection_mode=self.injection_mode,
            injection_scale=self.injection_scale,
            drag=self.drag,
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_kolm_flow"
