from ._gray_scott import GrayScott, GrayScottType
from ._navier_stokes import DecayingTurbulence, KolmogorovFlow
from ._poisson import Poisson
from ._swift_hohenberg import SwiftHohenberg

scenario_dict = {
    "phy_poisson": Poisson,
    "phy_sh": SwiftHohenberg,
    "phy_gs": GrayScott,
    "phy_gs_type": GrayScottType,
    "phy_decay_turb": DecayingTurbulence,
    "phy_kolm_flow": KolmogorovFlow,
}
