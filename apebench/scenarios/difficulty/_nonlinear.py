"""
Scenarios using the general nonlinear interface
"""

import exponax as ex

from ..._base_scenario import BaseScenario


class Nonlinear(BaseScenario):
    """
    Uses the single channel convection mode to not have channels grow with
    spatial dimensions.

    By default single-channel Burgers
    """

    gammas: tuple[float, ...] = (0.0, 0.0, 1.5, 0.0, 0.0)
    deltas: tuple[float, float, float] = (0.0, -1.5, 0.0)

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        pass

    def _build_stepper(self, gammas, deltas):
        substepped_gammas = tuple(g / self.num_substeps for g in gammas)
        substepped_deltas = tuple(d / self.num_substeps for d in deltas)

        substepped_stepper = ex.stepper.generic.DifficultyNonlinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            linear_difficulties=substepped_gammas,
            nonlinear_difficulties=substepped_deltas,
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
        return self._build_stepper(self.gammas, self.deltas)

    def get_coarse_stepper(self):
        return self._build_stepper(
            tuple(f * self.coarse_proportion for f in self.gammas),
            tuple(f * self.coarse_proportion for f in self.deltas),
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_nonlin"


class BurgersSingleChannel(Nonlinear):
    convection_sc_delta: float = -1.5
    diffusion_gamma: float = 1.5

    def __post_init__(self):
        self.gammas = (0.0, 0.0, self.diffusion_gamma, 0.0, 0.0)
        self.deltas = (0.0, self.convection_sc_delta, 0.0)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_burgers_sc"


class KortewegDeVries(Nonlinear):
    convection_sc_delta: float = -2.0
    dispersion_gamma: float = -14.0
    hyp_diffusion_gamma: float = -9.0

    def __post_init__(self):
        self.gammas = (0.0, 0.0, 0.0, self.dispersion_gamma, self.hyp_diffusion_gamma)
        self.deltas = (0.0, self.convection_sc_delta, 0.0)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_kdv"


class KuramotoSivashinsky(Nonlinear):
    gradient_norm_delta: float = -6.0
    diffusion_gamma: float = -1.2  # Negative diffusion! producing energy
    hyp_diffusion_gamma: float = -15.0

    num_warmup_steps: int = 500  # Overwrite
    vlim: tuple[float, float] = (-6.5, 6.5)  # Overwrite

    report_metrics: str = "mean_nRMSE,mean_correlation"  # Overwrite

    def __post_init__(self):
        self.gammas = (0.0, 0.0, self.diffusion_gamma, 0.0, self.hyp_diffusion_gamma)
        self.deltas = (0.0, 0.0, self.gradient_norm_delta)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_ks"


class FisherKPP(Nonlinear):
    quadratic_delta: float = -0.02
    drag_gamma: float = 0.02
    diffusion_gamma: float = 0.2

    ic_config: str = "clamp;0.0;1.0;fourier;5;false;false"  # Overwrite

    def __post_init__(self):
        self.gammas = (self.drag_gamma, 0.0, self.diffusion_gamma)
        self.deltas = (self.quadratic_delta, 0.0, 0.0)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_fisher"
