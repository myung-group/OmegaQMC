"""VMC-MLSW"""

from vmc_mlsw.vmc_gto import get_vmc_func
#from vmc_mlsw import psi_gto
#from vmc_mlsw import psi_gto_cusp
#from vmc_mlsw import vmc_gto

__version__ = "0.1.0"
# __all__ = ["run"]

# Re-export constants from dedicated module
from vmc_mlsw.constants import (
    JASTROW_EE_L_CUT,
    JASTROW_EE_M_POWER,
    EE_CUSP_VALUE,
)
