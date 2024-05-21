from . import difficulty, normalized, physical

scenario_dict = {
    **normalized.scenario_dict,
    **physical.scenario_dict,
    **difficulty.scenario_dict,
}
