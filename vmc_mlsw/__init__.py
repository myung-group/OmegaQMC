"""VMC-MLSW"""

from .core import *

__version__ = "0.1.0"
# __all__ = ["run"]

# Constants for Jastrow factor (electron-electron correlation)
# These are placeholders if Jastrow factor is not being used
JASTROW_EE_L_CUT = 7.5  # Cutoff distance for electron-electron Jastrow
JASTROW_EE_M_POWER = 3  # Power parameter in Jastrow factor
EE_CUSP_VALUE = 0.25  # Electron-electron cusp condition value
