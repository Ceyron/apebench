from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class Linear(BaseScenario):
    gammas: tuple[float, ...] = (
        0.0,
        -4.0,
    )
    coarse_proportion: float = 0.5

    def get_ref_stepper(self):
        return ex.normalized.DifficultyLinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            difficulties=self.gammas,
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.normalized.DifficultyLinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            difficulties=tuple(f * self.coarse_proportion for f in self.gammas),
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.gammas):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_diff_linear_{'_'.join(str(i) for i in active_indices)}"


class LinearSimple(Linear):
    diff: float = -1.0
    term_order: int = 1

    def __post_init__(self):
        self.gammas = (0.0,) * self.term_order + (self.diff,)


class Advection(Linear):
    adv_difficulty: float = 4.0

    def __post_init__(self):
        self.gammas = (0.0, -self.adv_difficulty)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_adv"


class Diffusion(Linear):
    diff_difficulty: float = 1.0

    def __post_init__(self):
        self.gammas = (0.0, 0.0, self.diff_difficulty)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_diff"


class AdvectionDiffusion(Linear):
    adv_difficulty: float = 4.0
    diff_difficulty: float = 1.0

    def __post_init__(self):
        self.gammas = (0.0, -self.adv_difficulty, self.diff_difficulty)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_adv_diff"


class Dispersion(Linear):
    disp_difficulty: float = 1.0

    def __post_init__(self):
        self.gammas = (0.0, 0.0, 0.0, self.disp_difficulty)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_disp"


class HyperDiffusion(Linear):
    hyp_difficulty: float = 1.0

    def __post_init__(self):
        self.gammas = (0.0, 0.0, 0.0, 0.0, -self.hyp_difficulty)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_hypdiff"


class FirstFour(Linear):
    adv_difficulty: float = 4.0
    diff_difficulty: float = 1.0
    disp_difficulty: float = 1.0
    hyp_difficulty: float = 1.0

    def __post_init__(self):
        self.gammas = (
            0.0,
            -self.adv_difficulty,
            self.diff_difficulty,
            self.disp_difficulty,
            -self.hyp_difficulty,
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_four"
