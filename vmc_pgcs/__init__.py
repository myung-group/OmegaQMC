"""VMC-MLSW"""

import vmc_pgcs.config  # noqa: F401  (activates jax_enable_x64)

from vmc_pgcs.vmc_gto import generate_molecular_orbitals, get_vmc_func
from vmc_pgcs.vmcopt_gto import get_vmcopt_func
#from vmc_pgcs import psi_gto
#from vmc_pgcs import psi_gto_cusp
#from vmc_pgcs import vmc_gto

__version__ = "0.1.0"
# __all__ = ["run"]

# Re-export constants from dedicated module
from vmc_pgcs.constants import (
    CHEMICAL_ACCURACY,
    JASTROW_EE_L_CUT,
    JASTROW_EE_M_POWER,
    EE_CUSP_VALUE,
)
