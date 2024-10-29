import exponax as ex

from ..._base_scenario import BaseScenario


class Convection(BaseScenario):
    alphas: tuple[float, ...] = (0.0, 0.0, 3.0e-5, 0.0, 0.0)
    convection_beta: float = -1.25e-2
    conservative: bool = True

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: float = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        self.num_channels = self.num_spatial_dims  # Overwrite

    def _build_stepper(self, convection, alphas):
        substepped_convection = convection / self.num_substeps
        substepped_alphas = tuple(a / self.num_substeps for a in alphas)

        substepped_stepper = ex.stepper.generic.NormalizedConvectionStepper(
            self.num_spatial_dims,
            self.num_points,
            normalized_linear_coefficients=substepped_alphas,
            # Need minus to move the convection to the right hand side
            normalized_convection_scale=-substepped_convection,
            conservative=self.conservative,
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
        return self._build_stepper(self.convection_beta, self.alphas)

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return self._build_stepper(
            self.coarse_proportion * self.convection_beta,
            tuple(f * self.coarse_proportion for f in self.alphas),
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.alphas):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_norm_conv_{'_'.join(str(i) for i in active_indices)}"


class Burgers(Convection):
    convection_beta: float = -1.25e-2  # Overwrite
    diffusion_alpha: float = 3.0e-5

    def __post_init__(self):
        self.alphas = (0.0, 0.0, self.diffusion_alpha, 0.0, 0.0)
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_burgers"


class KuramotoSivashinskyConservative(Convection):
    convection_delta: float = -6.0e-3
    diffusion_alpha: float = -4.0e-5
    hyp_diffusion_alpha: float = -3.0e-8

    num_warmup_steps: int = 500  # Overwrite
    vlim: tuple[float, float] = (-2.5, 2.5)  # Overwrite

    report_metrics: str = "mean_nRMSE,mean_correlation"  # Overwrite

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            raise ValueError(
                "Conservative Kuramoto-Sivashinsky is only defined for 1 spatial dimension. Check out the non-conservative version for 2d."
            )
        self.alphas = (0.0, 0.0, self.diffusion_alpha, 0.0, self.hyp_diffusion_alpha)
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_ks_cons"
