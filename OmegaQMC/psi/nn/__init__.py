"""Neural network trial wavefunctions.

Ports the DeepQMC architecture (PauliNet, FermiNet,
DeepErwin, PsiFormer) into OmegaQMC using Flax NNX.

Public API
----------
get_nn_psi_fun : Build NN trial and return
    VMCTrialState + initial parameters.
NNAnsatzConfig : Dataclass of architecture
    hyperparameters.
load_nn_config : Load config from YAML.
MoleculeInfo   : Lightweight molecular specification.
"""

from .config import NNAnsatzConfig, load_nn_config
from .wf import MoleculeInfo
from .adapter import make_nn_log_psi

__all__ = [
    'NNAnsatzConfig',
    'load_nn_config',
    'MoleculeInfo',
    'make_nn_log_psi',
]
