from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class Linear(BaseScenario):
    alphas: tuple[float, ...] = (-0.1,)
    coarse_proportion: float = 0.5

    def get_ref_stepper(self):
        return ex.normalized.NormalizedLinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            normalized_coefficients=self.alphas,
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.normalized.NormalizedLinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            normalized_coefficients=tuple(
                f * self.coarse_proportion for f in self.alphas
            ),
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.alphas):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_norm_lin_{'_'.join(str(i) for i in active_indices)}"


class Advection(Linear):
    advectivity: float = 0.1

    def __post_init__(self):
        self.alphas = (0.0, -self.advectivity)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_adv"


class Diffusion(Linear):
    diffusivity: float = 0.001

    def __post_init__(self):
        self.alphas = (0.0, 0.0, self.diffusivity)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_diff"


class AdvectionDiffusion(Linear):
    advectivity: float = 0.1
    diffusivity: float = 0.001

    def __post_init__(self):
        self.alphas = (0.0, -self.advectivity, self.diffusivity)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_adv_diff"


class Dispersion(Linear):
    dispersivity: float = 1e-5

    def __post_init__(self):
        self.alphas = (0.0, 0.0, 0.0, self.dispersivity)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_disp"


class HyperDiffusion(Linear):
    hyper_diffusivity: float = 3e-6

    def __post_init__(self):
        self.alphas = (0.0, 0.0, 0.0, 0.0, -self.hyper_diffusivity)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_hypdiff"


class FirstFour(Linear):
    advectivity: float = 0.1
    diffusivity: float = 0.001
    dispersivity: float = 1e-5
    hyper_diffusivity: float = 3e-6

    def __post_init__(self):
        self.alphas = (
            0.0,
            -self.advectivity,
            self.diffusivity,
            self.dispersivity,
            -self.hyper_diffusivity,
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_norm_first_four"
