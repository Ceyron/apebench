import exponax as ex

from ..._base_scenario import BaseScenario


class Nonlinear(BaseScenario):
    """
    Uses the single channel convection mode to not have channels grow with
    spatial dimensions.
    """

    alphas: tuple[float, ...] = (0.0, 0.0, 0.00003, 0.0, 0.0)
    betas: tuple[float, float, float] = (0.0, -0.0125, 0.0)

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

        substepped_stepper = ex.stepper.generic.NormalizedNonlinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            normalized_linear_coefficients=substepped_alphas,
            normalized_nonlinear_coefficients=substepped_betas,
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


class BurgersSingleChannel(Nonlinear):
    convection_sc_beta: float = -0.0125
    diffusion_alpha: float = 0.00003

    def __post_init__(self):
        self.alphas = (0.0, 0.0, self.diffusion_alpha, 0.0, 0.0)
        self.betas = (0.0, self.convection_sc_beta, 0.0)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_burgers_sc"


class KortewegDeVries(Nonlinear):
    convection_sc_beta: float = -0.0125
    dispersion_alpha: float = -8.5e-7
    hyp_diffusion_alpha: float = -2e-9

    def __post_init__(self):
        self.alphas = (0.0, 0.0, 0.0, self.dispersion_alpha, self.hyp_diffusion_alpha)
        self.betas = (0.0, self.convection_sc_beta, 0.0)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_kdv"


class KuramotoSivashinsky(Nonlinear):
    gradient_norm_beta: float = -0.00025
    diffusion_alpha: float = -0.000025
    hyp_diffusion_alpha: float = -3.0e-9

    num_warmup_steps: int = 500  # Overwrite
    vlim: tuple[float, float] = (-6.5, 6.5)  # Overwrite

    report_metrics: str = "mean_nRMSE,mean_correlation"  # Overwrite

    def __post_init__(self):
        self.alphas = (0.0, 0.0, self.diffusion_alpha, 0.0, self.hyp_diffusion_alpha)
        self.betas = (0.0, 0.0, self.gradient_norm_beta)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_ks"


class FisherKPP(Nonlinear):
    quadratic_beta: float = -0.02
    drag_alpha: float = 0.02
    diffusion_alpha: float = 4e-6

    ic_config: str = "clamp;0.0;1.0;fourier;5;false;false"  # Overwrite

    def __post_init__(self):
        self.alphas = (self.drag_alpha, 0.0, self.diffusion_alpha, 0.0, 0.0)
        self.betas = (self.quadratic_beta, 0.0, 0.0)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_fisher"
