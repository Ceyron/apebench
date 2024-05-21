import jax.numpy as jnp

from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class SwiftHohenberg(BaseScenario):
    num_spatial_dims: int = 2
    domain_extent: float = 20.0 * jnp.pi
    dt: float = 0.5

    num_substeps: int = 5

    reactivity: float = 0.7
    critical_number: float = 1.0
    polynomial_coefficients: tuple[float, ...] = (0.0, 0.0, 1.0, -1.0)

    def __post_init__(self):
        if self.num_spatial_dims != 2:
            raise ValueError("Only 2 spatial dimensions are supported for CahnHilliard")

    def get_ref_stepper(self):
        return ex.RepeatedStepper(
            ex.reaction.SwiftHohenberg(
                num_spatial_dims=self.num_spatial_dims,
                domain_extent=self.domain_extent,
                num_points=self.num_points,
                dt=self.dt / self.num_substeps,
                reactivity=self.reactivity,
                critical_number=self.critical_number,
                polynomial_coefficients=self.polynomial_coefficients,
            ),
            self.num_substeps,
        )

    def get_coarse_stepper(self):
        raise NotImplementedError("Coarse stepper is not implemented for CahnHilliard")

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_sh"
