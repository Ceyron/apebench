from . import difficulty, normalized, physical

scenario_dict = {
    **normalized.scenario_dict,
    **physical.scenario_dict,
    **difficulty.scenario_dict,
}

guaranteed_non_nan: list[
    tuple[
        str,  # scenario_name
        int,  # num_spatial_dims
    ]
] = [
    # Normalized Interface
    ("norm_lin", 1),
    ("norm_adv", 1),
    ("norm_diff", 1),
    ("norm_adv_diff", 1),
    ("norm_disp", 1),
    ("norm_fisher", 1),
    ("norm_four", 1),
    ("norm_hypdiff", 1),
    ("norm_nonlin", 1),
    ("norm_conv", 1),
    ("norm_burgers", 1),
    ("norm_kdv", 1),
    ("norm_ks_cons", 1),
    ("norm_ks", 1),
    ("norm_burgers_sc", 1),
    ("norm_lin_simple", 1),
    # Physical Interface
    ("phy_poisson", 1),
    ("phy_sh", 2),
    ("phy_gs", 2),
    ("phy_gs_type", 2),
    ("phy_decay_turb", 2),
    ("phy_kolm_flow", 2),
    ("phy_lin", 1),
    ("phy_lin_simple", 1),
    ("phy_adv", 1),
    ("phy_diff", 1),
    ("phy_adv_diff", 1),
    ("phy_disp", 1),
    ("phy_hyp_diff", 1),
    ("phy_four", 1),
    ("phy_nonlin", 1),
    ("phy_burgers_sc", 1),
    ("phy_kdv", 1),
    ("phy_ks", 1),
    ("phy_conv", 1),
    ("phy_burgers", 1),
    ("phy_ks_cons", 1),
    ("phy_poly", 1),
    ("phy_fisher", 1),
    ("phy_unbal_adv", 3),
    ("phy_diag_diff", 2),
    ("phy_aniso_diff", 2),
    ("phy_mix_disp", 2),
    ("phy_mix_hyp", 2),
    # In the difficulty-based interface all scenarios are guaranteed to be
    # non-nan for 1d, 2d, and 3d
]
