# Linear Scenarios in physical Mode


::: apebench.scenarios.physical.Advection
    options:
        members:
            - advection_coef
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.Diffusion
    options:
        members:
            - diffusion_coef
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.AdvectionDiffusion
    options:
        members:
            - advection_coef
            - diffusion_coef
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.Dispersion
    options:
        members:
            - dispersion_coef
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.HyperDiffusion
    options:
        members:
            - hyp_diffusion_coef
            - __post_init__
            - get_scenario_name


---

