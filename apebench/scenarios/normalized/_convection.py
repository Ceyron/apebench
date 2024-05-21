from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class Convection(BaseScenario):
    convection: float = 1.0 * 0.1 / (1.0**1)
    alphas: tuple[float, ...] = (0.0, 0.0, 0.05 * 0.1 / (1.0**2))

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

        substepped_stepper = ex.normalized.NormalizedConvectionStepper(
            self.num_spatial_dims,
            self.num_points,
            normalized_coefficients=substepped_alphas,
            normalized_convection_scale=substepped_convection,
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
        return self._build_stepper(self.convection, self.alphas)

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return self._build_stepper(
            self.coarse_proportion * self.convection,
            tuple(f * self.coarse_proportion for f in self.alphas),
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.alphas):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_norm_conv_{'_'.join(str(i) for i in active_indices)}"


class Burgers(Convection):
    convection: float = 0.01 * 1.0 / (1.0**1)
    diffusivity: float = 0.01 * 0.01 / (1.0**2)

    def __post_init__(self):
        self.alphas = (0.0, 0.0, self.diffusivity)
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_burgers"


class KortewegDeVries(Convection):
    convection: float = 0.01 * (-6) / (20.0) ** 1
    dipersivity: float = 0.01 * 1 / (20.0) ** 3

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            # TODO: Is there a way to nicely define this in 2d?
            raise ValueError("Korteweg-DeVries is only defined for 1 spatial dimension")
        self.alphas = (0.0, 0.0, 0.0, self.dipersivity)
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_kdv"


class KuramotoSivashinskyConservative(Convection):
    convection: float = 0.1 * 1 / (60.0**1)
    second_order: float = 0.1 * 1 / (60.0**2)
    fourth_order: float = 0.1 * 1 / (60.0**4)

    num_warmup_steps: int = 500  # Overwrite
    vlim: tuple[float, float] = (-2.5, 2.5)  # Overwrite

    report_metrics: str = "mean_nRMSE,mean_correlation"  # Overwrite

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            # TODO: Is there a way to nicely define this in 2d?
            raise ValueError(
                "Conservative Kuramoto-Sivashinsky is only defined for 1 spatial dimension. Check out the non-conservative version for 2d."
            )
        self.alphas = (0.0, 0.0, -self.second_order, 0.0, -self.fourth_order)
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_ks_cons"
