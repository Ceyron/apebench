# Generic Scenarios in physical Mode

::: apebench.scenarios.physical.Linear
    options:
        members:
            - a_coefs
            - coarse_proportion
            - get_scenario_name

---

::: apebench.scenarios.physical.LinearSimple
    options:
        members:
            - linear_coef
            - linear_term_order
            - get_scenario_name

---

::: apebench.scenarios.physical.FirstFour
    options:
        members:
            - advection_coef
            - diffusion_coef
            - dispersion_coef
            - hyp_diffusion_coef
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.Nonlinear
    options:
        members:
            - a_coefs
            - b_coefs
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

::: apebench.scenarios.physical.Convection
    options:
        members:
            - domain_extent
            - dt
            - a_coefs
            - convection_coef
            - conservative
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

---

::: apebench.scenarios.physical.Polynomial
    options:
        members:
            - domain_extent
            - dt
            - a_coefs
            - poly_coefs
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
