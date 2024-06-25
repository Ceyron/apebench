"""
This special modules is for variations that do easily fit into othet categories
"""
import exponax as ex
import jax.numpy as jnp

from ..._base_scenario import BaseScenario


class UnbalancedAdvection(BaseScenario):
    domain_extent: float = 1.0
    dt: float = 0.1

    advection_coef_vector: tuple[float, ...] = (
        0.01,
        -0.04,
        0.005,
    )  # Needs to be as long as num_spatial_dims

    coarse_proportion: float = 0.5

    def __post_init__(self):
        if self.num_spatial_dims == 1:
            raise ValueError("Unbalanced advection is only supported for 2D and 3D")
        if len(self.advection_coef_vector) != self.num_spatial_dims:
            raise ValueError(
                "Advection coefficient vector needs to be as long as the number of spatial dimensions"
            )

    def get_ref_stepper(self) -> ex.BaseStepper:
        return ex.stepper.Advection(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt,
            velocity=jnp.array(self.advection_coef_vector),
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.stepper.Advection(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt * self.coarse_proportion,
            velocity=jnp.array(self.advection_coef_vector),
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_unbal_adv"


class DiagonalDiffusion(BaseScenario):
    domain_extent: float = 1.0
    dt: float = 0.1

    diffusion_coef_vector: tuple[float, ...] = (
        0.001,
        0.002,
    )  # Needs to be as long as num_spatial_dims

    coarse_proportion: float = 0.5

    def __post_init__(self):
        if self.num_spatial_dims == 1:
            raise ValueError("Diagonal diffusion is only supported for 2D and 3D")
        if len(self.diffusion_coef_vector) != self.num_spatial_dims:
            raise ValueError(
                "Diffusion coefficient vector needs to be as long as the number of spatial dimensions"
            )

    def get_ref_stepper(self) -> ex.BaseStepper:
        return ex.stepper.Diffusion(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt,
            diffusivity=jnp.array(self.diffusion_coef_vector),
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.stepper.Diffusion(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt * self.coarse_proportion,
            diffusivity=jnp.array(self.diffusion_coef_vector),
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_diag_diff"


class AnisotropicDiffusion(BaseScenario):
    domain_extent: float = 1.0
    dt: float = 0.1

    diffusion_coef_matrix: tuple[tuple[float, ...], ...] = (
        (0.001, 0.0005),
        (0.0005, 0.002),
    )
    """
    Needs to be a square matrix with the same size as the number of spatial
    dimensions, given as a tuple of tuples.

    Also has to be symmetric and positive definite.
    """

    coarse_proportion: float = 0.5

    def __post_init__(self):
        if self.num_spatial_dims == 1:
            raise ValueError("Anisotropic diffusion is only supported for 2D and 3D")
        if len(self.diffusion_coef_matrix) != self.num_spatial_dims:
            raise ValueError(
                "Diffusion coefficient matrix needs to be as long as the number of spatial dimensions"
            )
        for row in self.diffusion_coef_matrix:
            if len(row) != self.num_spatial_dims:
                raise ValueError("Diffusion coefficient matrix needs to be square")

        # No check for SPD for now

    def get_ref_stepper(self) -> ex.BaseStepper:
        return ex.stepper.Diffusion(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt,
            diffusivity=jnp.array(self.diffusion_coef_matrix),
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.stepper.Diffusion(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt * self.coarse_proportion,
            diffusivity=jnp.array(self.diffusion_coef_matrix),
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_aniso_diff"


class SpatiallyMixedDispersion(BaseScenario):
    domain_extent: float = 1.0
    dt: float = 0.001
    dispersion_coef: float = 0.00025

    coarse_proportion: float = 0.5

    def __post_init__(self):
        if self.num_spatial_dims == 1:
            raise ValueError(
                "Spatially mixed dispersion is only supported for 2D and 3D"
            )

    def get_ref_stepper(self) -> ex.BaseStepper:
        return ex.stepper.Dispersion(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt,
            dispersivity=self.dispersion_coef,
            advect_on_diffusion=True,
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.stepper.Dispersion(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt * self.coarse_proportion,
            dispersivity=self.dispersion_coef,
            advect_on_diffusion=True,
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_mix_disp"


class SpatiallyMixedHyperDiffusion(BaseScenario):
    domain_extent: float = 1.0
    dt: float = 0.00001
    hyp_diffusion_coef: float = -0.000075

    coarse_proportion: float = 0.5

    def __post_init__(self):
        if self.num_spatial_dims == 1:
            raise ValueError(
                "Spatially mixed hyperdiffusion is only supported for 2D and 3D"
            )

    def get_ref_stepper(self) -> ex.BaseStepper:
        return ex.stepper.HyperDiffusion(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt,
            # Need minus to match the sign of the diffusion coefficient
            hyper_diffusivity=-self.hyp_diffusion_coef,
            diffuse_on_diffuse=True,
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.stepper.HyperDiffusion(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt * self.coarse_proportion,
            # Need minus to match the sign of the diffusion coefficient
            hyper_diffusivity=-self.hyp_diffusion_coef,
            diffuse_on_diffuse=True,
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_mix_hyp"
