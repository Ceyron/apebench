# import pytest

import apebench


def test_simple():
    apebench.scenarios.difficulty.Advection()


# @pytest.mark.parametrize(
#     "name",
#     list(apebench.scenarios.scenario_dict.keys()),
# )
# def test_builtin_scenarios(name: str):
#     # Some scenarios might not work in 1d, (which is the default number of spatial dims)
#     try:
#         scene = apebench.scenarios.scenario_dict[name]()
#     except ValueError:
#         return

#     ref = scene.get_ref_sample_data()

#     del ref
