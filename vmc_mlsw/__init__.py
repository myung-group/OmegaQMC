"""VMC-MLSW"""

from vmc_mlsw.vmc_gto import get_vmc_func
#from vmc_mlsw import psi_gto
#from vmc_mlsw import psi_gto_cusp
#from vmc_mlsw import vmc_gto

__version__ = "0.1.0"
# __all__ = ["run"]

# Constants for Jastrow factor (electron-electron correlation)
# These are placeholders if Jastrow factor is not being used
JASTROW_EE_L_CUT = 7.5  # Cutoff distance for electron-electron Jastrow
JASTROW_EE_M_POWER = 3  # Power parameter in Jastrow factor
EE_CUSP_VALUE = 0.25  # Electron-electron cusp condition value
