# Navier-Stokes Scenarios in physical Mode

::: apebench.scenarios.physical.DecayingTurbulence
    options:
        members:
            - domain_extent
            - dt
            - diffusivity
            - num_substeps
            - coarse_proportion
            - order
            - dealiasing_fraction
            - num_circle_points
            - circle_radius
            - get_scenario_name
            - __post_init__

---

::: apebench.scenarios.physical.KolmogorovFlow
    options:
        members:
            - domain_extent
            - dt
            - diffusivity
            - injection_mode
            - injection_scale
            - drag
            - num_substeps
            - coarse_proportion
            - num_warmup_steps
            - vlim
            - order
            - dealiasing_fraction
            - num_circle_points
            - circle_radius
            - get_scenario_name
            - __post_init__