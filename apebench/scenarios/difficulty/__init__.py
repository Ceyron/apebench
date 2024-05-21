from ._convection import (
    Burgers,
    Convection,
    KortewegDeVries,
    KuramotoSivashinskyConservative,
)
from ._gradient_norm import GradientNorm, KuramotoSivashinsky
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
from ._nonlinear import Nonlinear
from ._stencil_learning import FOULearning, FTCSLearning

scenario_dict = {
    "diff_lin": Linear,
    "diff_lin_simple": LinearSimple,
    "diff_adv": Advection,
    "diff_diff": Diffusion,
    "diff_adv_diff": AdvectionDiffusion,
    "diff_disp": Dispersion,
    "diff_hypdiff": HyperDiffusion,
    "diff_four": FirstFour,
    "diff_conv": Convection,
    "diff_burgers": Burgers,
    "diff_kdv": KortewegDeVries,
    "diff_ks_cons": KuramotoSivashinskyConservative,
    "diff_grad_norm": GradientNorm,
    "diff_ks": KuramotoSivashinsky,
    "diff_fou": FOULearning,
    "diff_ftcs": FTCSLearning,
    "diff_nonlin": Nonlinear,
}
