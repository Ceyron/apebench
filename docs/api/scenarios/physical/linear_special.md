# Special Linear Scenarios in physical Mode

::: apebench.scenarios.physical.UnbalancedAdvection
    options:
        members:
            - domain_extent
            - dt
            - advection_coef_vector
            - coarse_proportion
            - get_scenario_name
            - __post_init__

---

::: apebench.scenarios.physical.DiagonalDiffusion
    options:
        members:
            - domain_extent
            - dt
            - diffusion_coef_vector
            - coarse_proportion
            - get_scenario_name
            - __post_init__

---

::: apebench.scenarios.physical.AnisotropicDiffusion
    options:
        members:
            - domain_extent
            - dt
            - diffusion_coef_matrix
            - coarse_proportion
            - get_scenario_name
            - __post_init__

---

::: apebench.scenarios.physical.SpatiallyMixedDispersion
    options:
        members:
            - domain_extent
            - dt
            - dispersion_coef
            - coarse_proportion
            - get_scenario_name
            - __post_init__

---

::: apebench.scenarios.physical.SpatiallyMixedHyperDiffusion
    options:
        members:
            - domain_extent
            - dt
            - hyp_diffusion_coef
            - coarse_proportion
            - get_scenario_name
            - __post_init__