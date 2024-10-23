import exponax as ex

from ..._base_scenario import BaseScenario


class Convection(BaseScenario):
    domain_extent: float = 1.0
    dt: float = 0.1

    a_coefs: tuple[float, ...] = (0.0, 0.0, 0.0003, 0.0, 0.0)
    convection_coef: float = -0.125
    conservative: bool = True

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        self.num_channels = self.num_spatial_dims  # Overwrite

    def _build_stepper(self, dt):
        substepped_stepper = ex.stepper.generic.GeneralConvectionStepper(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=dt / self.num_substeps,
            linear_coefficients=self.a_coefs,
            # Need minus to move the convection to the right hand side
            convection_scale=-self.convection_coef,
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
        return self._build_stepper(self.dt)

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return self._build_stepper(self.dt * self.coarse_proportion)

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.a_coefs):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_phy_conv_{'_'.join(str(i) for i in active_indices)}"


class Burgers(Convection):
    convection_coef: float = -0.125  # Overwrite
    diffusion_coef: float = 0.0003

    def __post_init__(self):
        self.a_coefs = (0.0, 0.0, self.diffusion_coef, 0.0, 0.0)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        f"{self.num_spatial_dims}d_phy_burgers"


class KuramotoSivashinskyConservative(Convection):
    domain_extent: float = 60.0  # Overwrite
    convection_coef: float = -3.6  # Overwrite
    diffusion_coef: float = -1.44
    hyp_diffusion_coef: float = -0.4

    num_warmup_steps: int = 500  # Overwrite
    vlim: tuple[float, float] = (-2.5, 2.5)  # Overwrite

    report_metrics: str = "mean_nRMSE,mean_correlation"  # Overwrite

    def __post_init__(self):
        if self.num_spatial_dims != 1:
            raise ValueError(
                "Conservative Kuramoto-Sivashinsky is only defined for 1 spatial dimension. Check out the non-conservative version for 2d."
            )
        self.a_coefs = (0.0, 0.0, self.diffusion_coef, 0.0, self.hyp_diffusion_coef)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_ks_conservative"
