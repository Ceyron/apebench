# Nonlinear Scenarios in normalized Mode

::: apebench.scenarios.normalized.Burgers
    options:
        members:
            - convection_beta
            - diffusion_alpha
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.normalized.BurgersSingleChannel
    options:
        members:
            - convection_sc_beta
            - diffusion_alpha
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.normalized.KortewegDeVries
    options:
        members:
            - convection_sc_beta
            - dispersion_alpha
            - hyp_diffusion_alpha
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.normalized.KuramotoSivashinsky
    options:
        members:
            - gradient_norm_beta
            - diffusion_alpha
            - hyp_diffusion_alpha
            - num_warmup_steps
            - vlim
            - report_metrics
            - __post_init__
            - get_scenario_name


---

::: apebench.scenarios.normalized.KuramotoSivashinskyConservative
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