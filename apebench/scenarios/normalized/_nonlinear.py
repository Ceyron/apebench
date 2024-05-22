from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class Nonlinear(BaseScenario):
    """
    Uses the single channel convection mode to not have channels grow with
    spatial dimensions.
    """

    alphas: tuple[float, ...] = (0.0, 0.0, 0.1 * 0.05 / (1.0**2))
    betas: tuple[float, ...] = (0.0, -0.1 * 0.1 / (1.0**1), 0.0)

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        pass

    def _build_stepper(self, alphas, betas):
        substepped_alphas = tuple(a / self.num_substeps for a in alphas)
        substepped_betas = tuple(b / self.num_substeps for b in betas)

        substepped_stepper = ex.normalized.NormlizedGeneralNonlinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            normalized_coefficients_linear=substepped_alphas,
            normalized_coefficients_nonlinear=substepped_betas,
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

    def get_ref_stepper(self):
        return self._build_stepper(self.alphas, self.betas)

    def get_coarse_stepper(self):
        return self._build_stepper(
            tuple(f * self.coarse_proportion for f in self.alphas),
            tuple(f * self.coarse_proportion for f in self.betas),
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_nonlin"


class FisherKPP(Nonlinear):
    alphas: tuple[float, ...] = (
        0.001 * 10.0 / (10.0**0),
        0.0,
        0.001 * 1.0 / (10.0**2),
    )
    betas: tuple[float, ...] = (-0.001 * 10.0 / (10.0**0), 0.0, 0.0)

    ic_config: str = "clamp;0.0;1.0;fourier;10;false;false"  # Overwrite

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_fisher"
