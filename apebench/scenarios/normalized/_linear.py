import exponax as ex

from ..._base_scenario import BaseScenario


class Linear(BaseScenario):
    alphas: tuple[float, ...] = (0.0, -0.025, 0.0, 0.0, 0.0)
    coarse_proportion: float = 0.5

    def get_ref_stepper(self):
        return ex.stepper.generic.NormalizedLinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            normalized_linear_coefficients=self.alphas,
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.stepper.generic.NormalizedLinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            normalized_linear_coefficients=tuple(
                f * self.coarse_proportion for f in self.alphas
            ),
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.alphas):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_norm_lin_{'_'.join(str(i) for i in active_indices)}"


class LinearSimple(Linear):
    linear_alpha: float = -0.025
    linear_term_order: int = 1

    def __post_init__(self):
        self.alphas = (0.0,) * self.linear_term_order + (self.linear_alpha,)


class Advection(Linear):
    advection_alpha: float = -0.025

    def __post_init__(self):
        self.alphas = (0.0, self.advection_alpha, 0.0, 0.0, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_adv"


class Diffusion(Linear):
    diffusion_alpha: float = 8e-4

    def __post_init__(self):
        self.alphas = (0.0, 0.0, self.diffusion_alpha, 0.0, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_diff"


class AdvectionDiffusion(Linear):
    advection_alpha: float = -0.025
    diffusion_alpha: float = 8e-4

    def __post_init__(self):
        self.alphas = (0.0, self.advection_alpha, self.diffusion_alpha)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_adv_diff"


class Dispersion(Linear):
    dispersion_alpha: float = 2.5e-7

    def __post_init__(self):
        self.alphas = (0.0, 0.0, 0.0, self.dispersion_alpha, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_disp"


class HyperDiffusion(Linear):
    hyp_diffusion_alpha: float = -7.5e-10

    def __post_init__(self):
        self.alphas = (0.0, 0.0, 0.0, 0.0, self.hyp_diffusion_alpha)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_hyp_diff"


class FirstFour(Linear):
    advection_alpha: float = -0.025
    diffusion_alpha: float = 8e-4
    dispersion_alpha: float = 2.5e-7
    hyp_diffusion_alpha: float = -7.5e-10

    def __post_init__(self):
        self.alphas = (
            0.0,
            self.advection_alpha,
            self.diffusion_alpha,
            self.dispersion_alpha,
            self.hyp_diffusion_alpha,
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_four"
