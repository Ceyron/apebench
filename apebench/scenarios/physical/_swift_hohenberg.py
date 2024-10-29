from math import pi

import exponax as ex

from ..._base_scenario import BaseScenario


class SwiftHohenberg(BaseScenario):
    domain_extent: float = 10.0 * pi
    dt: float = 0.1

    num_substeps: int = 5

    reactivity: float = 0.7
    critical_number: float = 1.0
    polynomial_coefficients: tuple[float, ...] = (0.0, 0.0, 1.0, -1.0)

    coarse_proportion: float = 0.5

    def __post_init__(self):
        if self.num_spatial_dims == 1:
            raise ValueError("Swift-Hohenberg is only supported for 2D and 3D")

    def _build_stepper(self, dt):
        substepped_stepper = ex.stepper.reaction.SwiftHohenberg(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=dt / self.num_substeps,
            reactivity=self.reactivity,
            critical_number=self.critical_number,
            polynomial_coefficients=self.polynomial_coefficients,
        )

        if self.num_substeps == 1:
            stepper = substepped_stepper
        else:
            stepper = ex.RepeatedStepper(substepped_stepper, self.num_substeps)

        return stepper

    def get_ref_stepper(self):
        return self._build_stepper(self.dt)

    def get_coarse_stepper(self):
        return self._build_stepper(self.dt * self.coarse_proportion)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_sh"
