from ._convection import Burgers, Convection, KuramotoSivashinskyConservative
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
from ._nonlinear import (
    BurgersSingleChannel,
    FisherKPP,
    KortewegDeVries,
    KuramotoSivashinsky,
    Nonlinear,
)

scenario_dict = {
    "norm_lin": Linear,
    "norm_adv": Advection,
    "norm_diff": Diffusion,
    "norm_adv_diff": AdvectionDiffusion,
    "norm_disp": Dispersion,
    "norm_fisher": FisherKPP,
    "norm_four": FirstFour,
    "norm_hypdiff": HyperDiffusion,
    "norm_nonlin": Nonlinear,
    "norm_conv": Convection,
    "norm_burgers": Burgers,
    "norm_kdv": KortewegDeVries,
    "norm_ks_cons": KuramotoSivashinskyConservative,
    "norm_ks": KuramotoSivashinsky,
    "norm_burgers_sc": BurgersSingleChannel,
    "norm_lin_simple": LinearSimple,
}
