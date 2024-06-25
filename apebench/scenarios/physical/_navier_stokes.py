from math import pi

import exponax as ex

from ..._base_scenario import BaseScenario


class DecayingTurbulence(BaseScenario):
    domain_extent: float = 1.0
    dt: float = 0.1

    diffusivity: float = 1e-4

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        if self.num_spatial_dims != 2:
            raise ValueError(
                "Decaying turbulence is only defined for 2 spatial dimensions"
            )

    def _build_stepper(self, dt):
        substepped_stepper = ex.stepper.NavierStokesVorticity(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            dt / self.num_substeps,
            diffusivity=self.diffusivity,
            order=self.order,
            dealiasing_fraction=self.dealiasing_fraction,
            num_circle_points=self.num_circle_points,
            circle_radius=self.circle_radius,
        )

        if self.num_substeps == 1:
            stepper = substepped_stepper
        else:
            stepper = ex.RepeatedStepper(substepped_stepper, self.num_substeps)

        return stepper

    def get_ref_stepper(self) -> ex.BaseStepper:
        return self._build_stepper(self.dt)

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return self._build_stepper(self.dt * self.coarse_proportion)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_decay_turb"


class KolmogorovFlow(BaseScenario):
    domain_extent: float = 2 * pi
    dt: float = 0.1

    diffusivity: float = 1e-2  # Just Re=100 to have it almost scale-resolved

    injection_mode: int = 4
    injection_scale: float = 1.0
    drag: float = -0.1

    num_substeps: int = 20

    coarse_proportion: float = 0.5

    num_warmup_steps: int = 500  # Overwrite

    vlim: tuple[float, float] = (-10.0, 10.0)  # Overwrite

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        if self.num_spatial_dims != 2:
            raise ValueError("Kolmogorov Flow is only defined for 2 spatial dimensions")

    def _build_stepper(self, dt):
        substepped_stepper = ex.stepper.KolmogorovFlowVorticity(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            dt / self.num_substeps,
            diffusivity=self.diffusivity,
            injection_mode=self.injection_mode,
            injection_scale=self.injection_scale,
            drag=self.drag,
            order=self.order,
            dealiasing_fraction=self.dealiasing_fraction,
            num_circle_points=self.num_circle_points,
            circle_radius=self.circle_radius,
        )

        if self.num_substeps == 1:
            stepper = substepped_stepper
        else:
            stepper = ex.RepeatedStepper(substepped_stepper, self.num_substeps)

        return stepper

    def get_ref_stepper(self) -> ex.BaseStepper:
        return self._build_stepper(self.dt)

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return self._build_stepper(self.dt * self.coarse_proportion)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_kolm_flow"
