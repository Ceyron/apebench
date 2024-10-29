"""
Scenarios using the general convection interface (growing channels over spatial
dimensions)
"""
import exponax as ex

from ..._base_scenario import BaseScenario


class Convection(BaseScenario):
    gammas: tuple[float, ...] = (0.0, 0.0, 1.5, 0.0, 0.0)
    convection_delta: float = -1.5
    conservative: bool = True

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: float = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        self.num_channels = self.num_spatial_dims  # Overwrite

    def _build_stepper(self, gammas, delta):
        substepped_gammas = tuple(g / self.num_substeps for g in gammas)
        substepped_delta = delta / self.num_substeps

        substepped_stepper = ex.stepper.generic.DifficultyConvectionStepper(
            self.num_spatial_dims,
            self.num_points,
            linear_difficulties=substepped_gammas,
            # Need minus to move the convection to the right hand side
            convection_difficulty=-substepped_delta,
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
        return self._build_stepper(self.gammas, self.convection_delta)

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return self._build_stepper(
            tuple(f * self.coarse_proportion for f in self.gammas),
            self.coarse_proportion * self.convection_delta,
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.gammas):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_diff_conv_{'_'.join(str(i) for i in active_indices)}"


class Burgers(Convection):
    convection_delta: float = -1.5  # Overwrite
    diffusion_gamma: float = 1.5

    def __post_init__(self):
        self.gammas = (0.0, 0.0, self.diffusion_gamma, 0.0, 0.0)
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_burgers"


class KuramotoSivashinskyConservative(Convection):
    convection_delta: float = -1.0  # Overwrite
    diffusion_gamma: float = -2.0  # Negative diffusion; producing energy
    hyp_diffusion_gamma: float = -18.0

    num_warmup_steps: int = 500  # Overwrite
    vlim: tuple[float, float] = (-2.5, 2.5)  # Overwrite

    report_metrics: str = "mean_nRMSE,mean_correlation"  # Overwrite

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            raise ValueError("KuramotoSivashinskyConservative is only defined for 1D")
        self.gammas = (0.0, 0.0, self.diffusion_gamma, 0.0, self.hyp_diffusion_gamma)
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_ks_cons"
