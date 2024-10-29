# Linear Scenarios in normalized Mode


::: apebench.scenarios.normalized.Advection
    options:
        members:
            - advection_alpha
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.normalized.Diffusion
    options:
        members:
            - diffusion_alpha
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.normalized.AdvectionDiffusion
    options:
        members:
            - advection_alpha
            - diffusion_alpha
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.normalized.Dispersion
    options:
        members:
            - dispersion_alpha
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.normalized.HyperDiffusion
    options:
        members:
            - hyp_diffusion_alpha
            - __post_init__
            - get_scenario_name


