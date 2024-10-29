import exponax as ex

from ..._base_scenario import BaseScenario


class Nonlinear(BaseScenario):
    """
    Uses the single channel convection mode to not have channels grow with
    spatial dimensions.
    """

    domain_extent: float = 1.0
    dt: float = 0.1

    a_coefs: tuple[float, ...] = (0.0, 0.0, 0.0003, 0.0, 0.0)
    b_coefs: tuple[float, float, float] = (0.0, -0.125, 0.0)

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        pass

    def _build_stepper(self, dt):
        substepped_stepper = ex.stepper.generic.GeneralNonlinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=dt / self.num_substeps,
            linear_coefficients=self.a_coefs,
            nonlinear_coefficients=self.b_coefs,
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
        return f"{self.num_spatial_dims}d_phy_nonlin"


class BurgersSingleChannel(Nonlinear):
    convection_sc_coef: float = -0.125
    diffusion_coef: float = 0.0003

    def __post_init__(self):
        self.a_coefs = (0.0, 0.0, self.diffusion_coef, 0.0, 0.0)
        self.b_coefs = (0.0, self.convection_sc_coef, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_burgers_sc"


class KortewegDeVries(Nonlinear):
    domain_extent: float = 50.0  # Overwrite
    convection_sc_coef: float = -6.0
    dispersion_coef: float = -1.0
    hyp_diffusion_coef: float = -0.125

    def __post_init__(self):
        self.a_coefs = (0.0, 0.0, 0.0, self.dispersion_coef, self.hyp_diffusion_coef)
        self.b_coefs = (0.0, self.convection_sc_coef, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_kdv"


class KuramotoSivashinsky(Nonlinear):
    domain_extent: float = 60.0  # Overwrite
    gradient_norm_coef: float = -1
    diffusion_coef: float = -1.0  # Negative diffusion; producing energy!
    hyp_diffusion_coef: float = -1.0

    num_warmup_steps: int = 500  # Overwrite
    vlim: tuple[float, float] = (-6.5, 6.5)  # Overwrite

    report_metrics: str = "mean_nRMSE,mean_correlation"  # Overwrite

    def __post_init__(self):
        self.a_coefs = (0.0, 0.0, self.diffusion_coef, 0.0, self.hyp_diffusion_coef)
        self.b_coefs = (0.0, 0.0, self.gradient_norm_coef)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_ks"


# FisherKPP can be found in the _polynomial.py file
