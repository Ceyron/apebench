# Linear Scenarios in Difficulty Mode

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

::: apebench.scenarios.difficulty.Advection
    options:
        members:
            - advection_gamma
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.difficulty.Diffusion
    options:
        members:
            - diffusion_gamma
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.difficulty.AdvectionDiffusion
    options:
        members:
            - advection_gamma
            - diffusion_gamma
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.difficulty.Dispersion
    options:
        members:
            - dispersion_gamma
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.difficulty.HyperDiffusion
    options:
        members:
            - hyp_diffusion_gamma
            - __post_init__
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