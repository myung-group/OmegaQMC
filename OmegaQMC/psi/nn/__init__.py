"""Neural network trial wavefunctions.

Ports the DeepQMC architecture (PauliNet, FermiNet,
DeepErwin, PsiFormer) into OmegaQMC using Flax NNX.

Public API
----------
NNAnsatzConfig : Dataclass of architecture
    hyperparameters.
load_nn_config : Load config from YAML.
make_nn_log_psi : Build NN trial and return
    ``(log_psi, init_params, graphdef)``.
"""

from .config import NNAnsatzConfig, load_nn_config
from .adapter import make_nn_log_psi

__all__ = [
    'NNAnsatzConfig',
    'load_nn_config',
    'make_nn_log_psi',
]
