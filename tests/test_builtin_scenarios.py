import pytest

import apebench


@pytest.mark.parametrize(
    "name",
    list(apebench.scenarios.scenario_dict.keys()),
)
def test_builtin_scenarios(name: str):
    scene = apebench.scenarios.scenario_dict[name]()

    scene.get_ref_sample_data()
