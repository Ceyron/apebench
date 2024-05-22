from ._convection import Burgers, Convection, KuramotoSivashinskyConservative
from ._gray_scott import GrayScott, GrayScottType
from ._linear import (
    Advection,
    AdvectionDiffusion,
    Diffusion,
    Dispersion,
    FirstFour,
    HyperDiffusion,
    Linear,
    LinearSimple,
)
from ._navier_stokes import DecayingTurbulence, KolmogorovFlow
from ._nonlinear import (
    BurgersSingleChannel,
    KortewegDeVries,
    KuramotoSivashinsky,
    Nonlinear,
)
from ._poisson import Poisson
from ._polynomial import FisherKPP, Polynomial
from ._special_linear import (
    AnisotropicDiffusion,
    DiagonalDiffusion,
    SpatiallyMixedDispersion,
    SpatiallyMixedHyperDiffusion,
    UnbalancedAdvection,
)
from ._swift_hohenberg import SwiftHohenberg

scenario_dict = {
    "phy_poisson": Poisson,
    "phy_sh": SwiftHohenberg,
    "phy_gs": GrayScott,
    "phy_gs_type": GrayScottType,
    "phy_decay_turb": DecayingTurbulence,
    "phy_kolm_flow": KolmogorovFlow,
    "phy_lin": Linear,
    "phy_lin_simple": LinearSimple,
    "phy_adv": Advection,
    "phy_diff": Diffusion,
    "phy_adv_diff": AdvectionDiffusion,
    "phy_disp": Dispersion,
    "phy_hyp_diff": HyperDiffusion,
    "phy_four": FirstFour,
    "phy_nonlin": Nonlinear,
    "phy_burgers_sc": BurgersSingleChannel,
    "phy_kdv": KortewegDeVries,
    "phy_ks": KuramotoSivashinsky,
    "phy_conv": Convection,
    "phy_burgers": Burgers,
    "phy_ks_cons": KuramotoSivashinskyConservative,
    "phy_poly": Polynomial,
    "phy_fisher": FisherKPP,
    "phy_unbal_adv": UnbalancedAdvection,
    "phy_diag_diff": DiagonalDiffusion,
    "phy_aniso_diff": AnisotropicDiffusion,
    "phy_mix_disp": SpatiallyMixedDispersion,
    "phy_mix_hyp": SpatiallyMixedHyperDiffusion,
}
