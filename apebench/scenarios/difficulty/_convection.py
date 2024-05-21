from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class Convection(BaseScenario):
    gammas: tuple[float, ...] = (0.0, 0.0, 0.5)
    delta: float = 1.0

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: float = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        self.num_channels = self.num_spatial_dims

    def _build_stepper(self, gammas, delta):
        substepped_gammas = tuple(g / self.num_substeps for g in gammas)
        substepped_delta = delta / self.num_substeps

        substepped_stepper = ex.normalized.DifficultyConvectionStepper(
            self.num_spatial_dims,
            self.num_points,
            linear_difficulties=substepped_gammas,
            convection_difficulty=substepped_delta,
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
        return self._build_stepper(self.gammas, self.delta)

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return self._build_stepper(
            tuple(f * self.coarse_proportion for f in self.gammas),
            self.coarse_proportion * self.delta,
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.gammas):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_diff_conv_{'_'.join(str(i) for i in active_indices)}"


class Burgers(Convection):
    conv_difficulty: float = 1.0
    diff_difficulty: float = 0.5

    def __post_init__(self):
        self.gammas = (0.0, 0.0, self.diff_difficulty)
        self.delta = self.conv_difficulty
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_burgers"


class KortewegDeVries(Convection):
    conv_difficulty: float = 0.15
    diff_difficulty: float = 0.0  # inviscid by default
    disp_difficulty: float = 0.15

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            raise ValueError("KortewegDeVries is only defined for 1D")
        self.gammas = (0.0, 0.0, self.diff_difficulty, self.disp_difficulty)
        self.delta = self.conv_difficulty

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_kdv"


class KuramotoSivashinskyConservative(Convection):
    conv_difficulty: float = 0.1
    diff_difficulty: float = 0.2
    hyp_difficulty: float = 0.6

    num_warmup_steps: int = 500
    vlim: tuple[float, float] = (-2.5, 2.5)

    report_metrics: str = "mean_nRMSE,mean_correlation"

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            raise ValueError("KuramotoSivashinskyConservative is only defined for 1D")
        self.gammas = (0.0, 0.0, -self.diff_difficulty, 0.0, -self.hyp_difficulty)
        self.delta = self.conv_difficulty
        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_ks_cons"
