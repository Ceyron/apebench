import exponax as ex

from ..._base_scenario import BaseScenario


class Poisson(BaseScenario):
    domain_extent: float = 10.0
    order: int = 2

    num_warmup_steps: int = 0  # Overwrite
    train_temporal_horizon: int = 1  # Overwrite
    test_temporal_horizon: int = 1  # Overwrite

    def __post_init__(self):
        if self.train_temporal_horizon != 1:
            raise ValueError("Only temporal horizon 1 is supported for Poisson")
        if self.test_temporal_horizon != 1:
            raise ValueError("Only temporal horizon 1 is supported for Poisson")
        if self.num_warmup_steps != 0:
            raise ValueError("Warmup steps are not supported for Poisson")

    def get_ref_stepper(self):
        return ex.poisson.Poisson(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            order=self.order,
        )

    def get_coarse_stepper(self):
        raise NotImplementedError("Coarse stepper is not implemented for Poisson")

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_poisson_order{self.order}"
