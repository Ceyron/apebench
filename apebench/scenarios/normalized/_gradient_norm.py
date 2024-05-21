from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class GradientNorm(BaseScenario):
    gradient_norm: float = 1.0 * 0.1 / (1.0**2)
    alphas: tuple[float, ...] = (
        0.0,
        0.0,
        1.0 * 0.1 / (1.0**2),
    )

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: int = 1.0

    def __post_init__(self):
        pass

    def _build_stepper(self, gradient_norm, alphas):
        substepped_gradient_norm = gradient_norm / self.num_substeps
        substepped_alphas = tuple(a / self.num_substeps for a in alphas)

        substepped_stepper = ex.normalized.NormalizedGradientNormStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            normalized_coefficients=substepped_alphas,
            normalized_gradient_norm_scale=substepped_gradient_norm,
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
        return self._build_stepper(self.gradient_norm, self.alphas)

    def get_coarse_stepper(self):
        return self._build_stepper(
            self.coarse_proportion * self.gradient_norm,
            tuple(f * self.coarse_proportion for f in self.alphas),
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.alphas):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_norm_grad_norm_{'_'.join(str(i) for i in active_indices)}"


class KuramotoSivashinsky(GradientNorm):
    gradient_norm: float = 0.3 * 1.0 / (60.0**2)
    second_order: float = 0.3 * 1.0 / (60.0**2)
    fourth_order: float = 0.3 * 1.0 / (60.0**4)

    num_substeps: int = 3

    num_warmup_steps: int = 500
    vlim: tuple[float, float] = (-6.5, 6.5)

    report_metrics: str = "mean_nRMSE,mean_correlation"

    def __post_init__(self):
        if self.num_spatial_dims == 2:
            # Domain size of 60.0 would be too complicated in 2d
            self.gradient_norm = 0.3 * 1.0 / (30.0**2)
            self.second_order = 0.3 * 1.0 / (30.0**2)
            self.fourth_order = 0.3 * 1.0 / (30.0**4)

        self.alphas = (0.0, 0.0, -self.second_order, 0.0, -self.fourth_order)
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_ks"
