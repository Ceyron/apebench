import exponax as ex

from ..._base_scenario import BaseScenario


class Linear(BaseScenario):
    gammas: tuple[float, ...] = (0.0, -4.0, 0.0, 0.0, 0.0)
    coarse_proportion: float = 0.5

    def get_ref_stepper(self):
        return ex.stepper.generic.DifficultyLinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            linear_difficulties=self.gammas,
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.stepper.generic.DifficultyLinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            linear_difficulties=tuple(f * self.coarse_proportion for f in self.gammas),
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.gammas):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_diff_linear_{'_'.join(str(i) for i in active_indices)}"


class LinearSimple(Linear):
    linear_gamma: float = -4.0
    linear_term_order: int = 1

    def __post_init__(self):
        self.gammas = (0.0,) * self.linear_term_order + (self.linear_gamma,)


class Advection(Linear):
    advection_gamma: float = -4.0

    def __post_init__(self):
        self.gammas = (0.0, self.advection_gamma, 0.0, 0.0, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_adv"


class Diffusion(Linear):
    diffusion_gamma: float = 4.0

    def __post_init__(self):
        self.gammas = (0.0, 0.0, self.diffusion_gamma, 0.0, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_diff"


class AdvectionDiffusion(Linear):
    advection_gamma: float = -4.0
    diffusion_gamma: float = 4.0

    def __post_init__(self):
        self.gammas = (0.0, self.advection_gamma, self.diffusion_gamma, 0.0, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_adv_diff"


class Dispersion(Linear):
    dispersion_gamma: float = 4.0

    def __post_init__(self):
        self.gammas = (0.0, 0.0, 0.0, self.dispersion_gamma, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_disp"


class HyperDiffusion(Linear):
    hyp_diffusion_gamma: float = -4.0

    def __post_init__(self):
        self.gammas = (0.0, 0.0, 0.0, 0.0, self.hyp_diffusion_gamma, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_hyp_diff"


class FirstFour(Linear):
    advection_gamma: float = -4.0
    diffusion_gamma: float = 4.0
    dispersion_gamma: float = 4.0
    hyp_diffusion_gamma: float = -4.0

    def __post_init__(self):
        self.gammas = (
            0.0,
            self.advection_gamma,
            self.diffusion_gamma,
            self.dispersion_gamma,
            self.hyp_diffusion_gamma,
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_four"
