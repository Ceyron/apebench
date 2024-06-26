# Nonlinear Scenarios in Difficulty Mode

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

::: apebench.scenarios.difficulty.BurgersSingleChannel
    options:
        members:
            - convection_sc_delta
            - diffusion_gamma
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.difficulty.KortewegDeVries
    options:
        members:
            - convection_sc_delta
            - dispersion_gamma
            - hyp_diffusion_gamma
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.difficulty.KuramotoSivashinsky
    options:
        members:
            - gradient_norm_delta
            - diffusion_gamma
            - hyp_diffusion_gamma
            - num_warmup_steps
            - vlim
            - report_metrics
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.difficulty.FisherKPP
    options:
        members:
            - quadratic_delta
            - drag_gamma
            - diffusion_gamma
            - ic_config
            - __post_init__
            - get_scenario_name