# Nonlinear Scenarios in Difficulty Mode

::: apebench.scenarios.difficulty.Burgers
    options:
        members:
            - convection_delta
            - diffusion_gamma
            - __post_init__
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