"""VMC-PGCS"""

import OmegaQMC.config

from OmegaQMC.vmc_gto import generate_molecular_orbitals, get_vmc_func
from OmegaQMC.vmc_nn import get_vmc_nn_func
from OmegaQMC.vmcopt_gto_linear import get_vmcopt_func
from OmegaQMC.vmcopt_nn import get_vmcopt_nn_func
from OmegaQMC.afqmc_gto import get_afqmc_func, extract_casscf_trial
from OmegaQMC.qed_afqmc_gto import get_qed_afqmc_func
# from OmegaQMC import psi_gto
# from OmegaQMC import psi_gto_cusp
# from OmegaQMC import vmc_gto

__version__ = "0.1.0"
# __all__ = ["run"]

# Re-export constants from dedicated module
from OmegaQMC.constants import (
    CHEMICAL_ACCURACY,
    JASTROW_EE_L_CUT,
    JASTROW_EE_M_POWER,
    EE_CUSP_VALUE,
)
