# Generic Scenarios in normalized Mode

::: apebench.scenarios.normalized.Linear
    options:
        members:
            - alphas
            - coarse_proportion
            - get_scenario_name

---

::: apebench.scenarios.normalized.LinearSimple
    options:
        members:
            - linear_alpha
            - linear_term_order
            - get_scenario_name

---

::: apebench.scenarios.normalized.FirstFour
    options:
        members:
            - advection_alpha
            - diffusion_alpha
            - dispersion_alpha
            - hyp_diffusion_alpha
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.normalized.Nonlinear
    options:
        members:
            - alphas
            - betas
            - num_substeps
            - coarse_proportion
            - order
            - dealiasing_fraction
            - num_circle_points
            - circle_radius
            - _build_stepper
            - get_ref_stepper
            - get_coarse_stepper
            - get_scenario_name

---

::: apebench.scenarios.normalized.Convection
    options:
        members:
            - alphas
            - convection_beta
            - num_substeps
            - coarse_proportion
            - order
            - dealiasing_fraction
            - num_circle_points
            - circle_radius
            - __post_init__
            - _build_stepper
            - get_ref_stepper
            - get_coarse_stepper
            - get_scenario_name