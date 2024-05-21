from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class Nonlinear(BaseScenario):
    """
    Uses the single channel convection mode to not have channels grow with
    spatial dimensions.
    """

    gammas: tuple[float, ...] = (0.0, 0.0, 3)
    deltas: tuple[float, ...] = (0.0, -2.0, 0.0)

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

        substepped_stepper = ex.normalized.DifficultyGeneralNonlinearStepper(
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
