# Convection Scenarios in Difficulty Mode

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

---

::: apebench.scenarios.difficulty.Burgers
    options:
        members:
            - convection_delta
            - diffusion_gamma
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.difficulty.KuramotoSivashinskyConservative
    options:
        members:
            - convection_delta
            - diffusion_gamma
            - hyp_diffusion_gamma
            - num_warmup_steps
            - vlim
            - report_metrics
            - __post_init__
            - get_scenario_name