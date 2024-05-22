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
    "diff_lin": Linear,
    "diff_lin_simple": LinearSimple,
    "diff_adv": Advection,
    "diff_diff": Diffusion,
    "diff_adv_diff": AdvectionDiffusion,
    "diff_disp": Dispersion,
    "diff_hyp_diff": HyperDiffusion,
    "diff_four": FirstFour,
    "diff_conv": Convection,
    "diff_burgers": Burgers,
    "diff_kdv": KortewegDeVries,
    "diff_ks_cons": KuramotoSivashinskyConservative,
    "diff_ks": KuramotoSivashinsky,
    "diff_nonlin": Nonlinear,
    "diff_burgers_sc": BurgersSingleChannel,
    "diff_fisher": FisherKPP,
}
