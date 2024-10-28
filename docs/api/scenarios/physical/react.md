# Reaction-Diffusion Scenarios in physical Mode

::: apebench.scenarios.physical.FisherKPP
    options:
        members:
            - quadratic_coef
            - drag_coef
            - diffusion_coef
            - ic_config
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.GrayScott
    options:
        members:
            - domain_extent
            - dt
            - num_channels
            - num_substeps
            - feed_rate
            - kill_rate
            - diffusivity_1
            - diffusivity_2
            - coarse_proportion
            - ic_config
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.GrayScottType
    options:
        members:
            - domain_extent
            - dt
            - num_channels
            - num_substeps
            - pattern_type
            - diffusivity_1
            - diffusivity_2
            - coarse_proportion
            - ic_config
            - __post_init__
            - get_scenario_name

---

::: apebench.scenarios.physical.SwiftHohenberg
    options:
        members:
            - domain_extent
            - dt
            - num_substeps
            - reactivity
            - critical_number
            - polynomial_coefficents
            - coarse_proportion
            - ic_config
            - __post_init__
            - get_scenario_name