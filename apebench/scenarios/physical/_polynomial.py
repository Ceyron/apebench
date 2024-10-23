import exponax as ex

from ..._base_scenario import BaseScenario


class Polynomial(BaseScenario):
    domain_extent: float = 1.0
    dt: float = 0.001

    a_coefs: tuple[float, ...] = (0.02, 0.0, 4e-6, 0.0, 0.0)
    poly_coefs: tuple[float, ...] = (0.0, 0.0, -0.02)

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        pass

    def _build_stepper(self, dt):
        substepped_stepper = ex.stepper.generic.GeneralPolynomialStepper(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=dt / self.num_substeps,
            linear_coefficients=self.a_coefs,
            polynomial_coefficients=self.poly_coefs,
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
        return self._build_stepper(self.dt)

    def get_coarse_stepper(self):
        return self._build_stepper(self.dt * self.coarse_proportion)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_poly"


class FisherKPP(Polynomial):
    drag_coef: float = 20.0
    diffusion_coef: float = 0.004
    quadratic_coef: float = -20.0

    ic_config: str = "clamp;0.0;1.0;fourier;5;false;false"  # Overwrite

    def __post_init__(self):
        self.a_coefs = (self.drag_coef, 0.0, self.diffusion_coef, 0.0, 0.0)
        self.poly_coefs = (0.0, 0.0, self.quadratic_coef)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_fisher"
