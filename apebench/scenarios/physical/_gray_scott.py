from typing import Literal

from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex
from ...exponax.exponax.ic import BaseRandomICGenerator


class GrayScott(BaseScenario):
    domain_extent: float = 1.0
    dt: float = 10.0
    num_channels: int = 2  # Overwrite

    num_substeps: int = 10

    ic_config: str = "gray_scott_blobs"  # Overwrite

    feed_rate: float = 0.04
    kill_rate: float = 0.06
    diffusivity_1: float = 2e-5
    diffusivity_2: float = 1e-5

    def __post_init__(self):
        if self.num_spatial_dims != 2:
            raise ValueError("Only 2 spatial dimensions are supported for GrayScott")
        if self.num_channels != 2:
            raise ValueError("Number of channels must be 2 for GrayScott")

    def get_ic_generator(self) -> BaseRandomICGenerator:
        return ex.ic.RandomMultiChannelICGenerator(
            [
                ex.ic.RandomGaussianBlobs(2, one_complement=True),
                ex.ic.RandomGaussianBlobs(2),
            ]
        )

    def get_ref_stepper(self):
        return ex.RepeatedStepper(
            ex.reaction.GrayScott(
                num_spatial_dims=self.num_spatial_dims,
                domain_extent=self.domain_extent,
                num_points=self.num_points,
                dt=self.dt / self.num_substeps,
                feed_rate=self.feed_rate,
                kill_rate=self.kill_rate,
                diffusivity_1=self.diffusivity_1,
                diffusivity_2=self.diffusivity_2,
            ),
            self.num_substeps,
        )

    def get_coarse_stepper(self):
        raise NotImplementedError("Coarse stepper is not implemented for GrayScott")

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_gs"


# Following scenario follows
# "https://www.ljll.fr/hecht/ftp/ff++/2015-cimpa-IIT/edp-tuto/Pearson.pdf"


class GrayScottType(BaseScenario):
    domain_extent: float = 2.5
    dt: float = 20.0
    num_channels: int = 2  # Overwrite

    coarse_proportion: float = 0.5

    num_substeps: int = 20

    ic_config: str = "gray_scott_blobs"  # Overwrite

    diffusivity_1: float = 2e-5
    diffusivity_2: float = 1e-5
    pattern_type: Literal[
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
    ] = "theta"

    def __post_init__(self):
        if not (self.num_spatial_dims in [2, 3]):
            raise ValueError(
                "Only 2 and 3 spatial dimensions are supported for GrayScott"
            )
        if self.num_channels != 2:
            raise ValueError("Number of channels must be 2 for GrayScott")

    def get_feed_and_kill_rate(
        self,
        pattern_type: Literal[
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
            "iota",
            "kappa",
            "lambda",
            "mu",
        ],
    ):
        if pattern_type == "alpha":
            feed_rate = 0.008
            kill_rate = 0.046
        elif pattern_type == "beta":
            feed_rate = 0.020
            kill_rate = 0.046
        elif pattern_type == "gamma":
            feed_rate = 0.024
            kill_rate = 0.056
        elif pattern_type == "delta":
            feed_rate = 0.028
            kill_rate = 0.056
        elif pattern_type == "epsilon":
            feed_rate = 0.02
            kill_rate = 0.056
        elif pattern_type == "zeta":
            # Does not seem to work
            feed_rate = 0.024
            kill_rate = 0.06
        elif pattern_type == "eta":
            # Does not seem to work
            feed_rate = 0.036
            kill_rate = 0.063
        elif pattern_type == "theta":
            feed_rate = 0.04
            kill_rate = 0.06
        elif pattern_type == "iota":
            feed_rate = 0.05
            kill_rate = 0.0605
        elif pattern_type == "kappa":
            feed_rate = 0.052
            kill_rate = 0.063
        elif pattern_type == "lambda":
            # Does not seem to work
            feed_rate = 0.036
            kill_rate = 0.0655
        elif pattern_type == "mu":
            # Does not seem to work
            feed_rate = 0.044
            kill_rate = 0.066
        else:
            raise ValueError("Invalid pattern type")

        return feed_rate, kill_rate

    def get_ic_generator(self) -> BaseRandomICGenerator:
        return ex.ic.RandomMultiChannelICGenerator(
            [
                ex.ic.RandomGaussianBlobs(self.num_spatial_dims, one_complement=True),
                ex.ic.RandomGaussianBlobs(self.num_spatial_dims),
            ]
        )

    def get_ref_stepper(self) -> ex.BaseStepper:
        feed_rate, kill_rate = self.get_feed_and_kill_rate(self.pattern_type)
        return ex.RepeatedStepper(
            ex.reaction.GrayScott(
                num_spatial_dims=self.num_spatial_dims,
                domain_extent=self.domain_extent,
                num_points=self.num_points,
                dt=self.dt / self.num_substeps,
                feed_rate=feed_rate,
                kill_rate=kill_rate,
                diffusivity_1=self.diffusivity_1,
                diffusivity_2=self.diffusivity_2,
            ),
            self.num_substeps,
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        feed_rate, kill_rate = self.get_feed_and_kill_rate(self.pattern_type)
        return ex.RepeatedStepper(
            ex.reaction.GrayScott(
                num_spatial_dims=self.num_spatial_dims,
                domain_extent=self.domain_extent,
                num_points=self.num_points,
                dt=self.dt / self.num_substeps * self.coarse_proportion,
                feed_rate=feed_rate,
                kill_rate=kill_rate,
                diffusivity_1=self.diffusivity_1,
                diffusivity_2=self.diffusivity_2,
            ),
            self.num_substeps,
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_gs_{self.pattern_type}"
