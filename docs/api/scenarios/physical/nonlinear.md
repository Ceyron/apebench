# Nonlinear Scenarios in physical Mode

::: apebench.scenarios.physical.Burgers
    options:
        members:
            - convection_coef
            - diffusion_coef
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.BurgersSingleChannel
    options:
        members:
            - convection_sc_coef
            - diffusion_coef
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.KortewegDeVries
    options:
        members:
            - convection_sc_coef
            - dispersion_coef
            - hyp_diffusion_coef
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.KuramotoSivashinsky
    options:
        members:
            - gradient_norm_coef
            - diffusion_coef
            - hyp_diffusion_coef
            - num_warmup_steps
            - vlim
            - report_metrics
            - __post_init__
            - get_scenario_name


---

::: apebench.scenarios.physical.KuramotoSivashinskyConservative
    options:
        members:
            - convection_coef
            - diffusion_coef
            - hyp_diffusion_coef
            - num_warmup_steps
            - vlim
            - report_metrics
            - __post_init__
            - get_scenario_name