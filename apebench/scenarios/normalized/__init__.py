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
)
from ._nonlinear import FisherKPP, Nonlinear
from ._stencil_learning import FOULearning, FTCSLearning
from ._vorticity_convection import (
    DecayingTurbulence,
    KolmogorovFlow,
    VorticityConvection,
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
    "norm_grad_norm": GradientNorm,
    "norm_ks": KuramotoSivashinsky,
    "norm_fou": FOULearning,
    "norm_ftcs": FTCSLearning,
    "norm_vort_conv": VorticityConvection,
    "norm_decay_turb": DecayingTurbulence,
    "norm_kolmogorov": KolmogorovFlow,
}
