import jax.numpy as jnp

from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class VorticityConvection(BaseScenario):
    vorticity_convection: float = 0.1 * 1.0 / ((1.0) ** 0)
    alphas: tuple[float, ...] = (0, 0, 0.1 * 0.0003 / (1.0**2))
    injection_scale: float = 0.0
    injection_mode: int = 4

    num_substeps: int = 10

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        if self.num_spatial_dims != 2:
            raise ValueError(
                "Vorticity convection is only defined for 2 spatial dimensions"
            )

    def _build_stepper(self, vorticity_convection, alphas):
        substepped_vorticity_convection = vorticity_convection / self.num_substeps
        substepped_alphas = tuple(a / self.num_substeps for a in alphas)

        if hasattr(self, "injection_scale") and hasattr(self, "injection_mode"):
            substepped_injection_scale = self.injection_scale / self.num_substeps
            substepped_stepper = ex.normalized.NormalizedVorticityConvection(
                self.num_spatial_dims,
                self.num_points,
                normalized_coefficients=substepped_alphas,
                normalized_vorticity_convection_scale=substepped_vorticity_convection,
                injection_mode=self.injection_mode,
                normalized_injection_scale=substepped_injection_scale,
                order=self.order,
                dealiasing_fraction=self.dealiasing_fraction,
                num_circle_points=self.num_circle_points,
                circle_radius=self.circle_radius,
            )
        else:
            substepped_stepper = ex.normalized.NormalizedVorticityConvection(
                self.num_spatial_dims,
                self.num_points,
                normalized_coefficients=substepped_alphas,
                normalized_vorticity_convection_scale=substepped_vorticity_convection,
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
        return self._build_stepper(self.vorticity_convection, self.alphas)

    def get_coarse_stepper(self):
        return self._build_stepper(
            self.coarse_proportion * self.vorticity_convection,
            tuple(f * self.coarse_proportion for f in self.alphas),
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.alphas):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_norm_vort_conv_{'_'.join(str(i) for i in active_indices)}"


class DecayingTurbulence(VorticityConvection):
    vorticity_convection: float = 0.1 * 1.0 / ((1.0) ** 0)
    diffusivity: float = 0.1 * 0.0001 / (1.0**2)

    def __post_init__(self):
        self.alphas = (0, 0, self.diffusivity)
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_decay_turb"


class KolmogorovFlow(VorticityConvection):
    vorticity_convection: float = 0.1 * 1.0 / ((2 * jnp.pi) ** 0)
    diffusivity: float = 0.1 * 0.001 / ((2 * jnp.pi) ** 2)
    drag: float = 0.1 * (-0.1) / ((2 * jnp.pi) ** 0)

    injection_mode: int = 4
    injection_scale: float = 0.1 * 1.0

    num_warmup_steps: int = 500  # Overwrite
    ic_config: str = "diffused;0.001;True;True"  # Overwrite

    vlim: tuple[float, float] = (-10.0, 10.0)  # Overwrite

    report_metrics: str = "mean_nRMSE,mean_correlation"  # Overwrite

    def __post_init__(self):
        self.alphas = (self.drag, 0, self.diffusivity)
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_kolmogorov"
