# Generic Scenarios in Difficulty Mode

::: apebench.scenarios.difficulty.Linear
    options:
        members:
            - gammas
            - coarse_proportion
            - get_scenario_name

---

::: apebench.scenarios.difficulty.LinearSimple
    options:
        members:
            - linear_gamma
            - linear_term_order
            - get_scenario_name

---

::: apebench.scenarios.difficulty.FirstFour
    options:
        members:
            - advection_gamma
            - diffusion_gamma
            - dispersion_gamma
            - hyp_diffusion_gamma
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.difficulty.Nonlinear
    options:
        members:
            - gammas
            - deltas
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

::: apebench.scenarios.difficulty.Convection
    options:
        members:
            - gammas
            - convection_delta
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