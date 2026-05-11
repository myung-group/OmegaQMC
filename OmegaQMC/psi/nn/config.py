"""Configuration system for NN trial wavefunctions.

Loads architecture hyperparameters from YAML files and
maps them to constructor arguments for the NNX modules.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


_CONF_DIR = os.path.join(os.path.dirname(__file__), 'conf')

_BUILTIN_CONFIGS = {
    'paulinet', 'ferminet', 'deeperwin', 'psiformer',
}


@dataclass
class NNAnsatzConfig:
    """Architecture hyperparameters for the NN ansatz.

    See the YAML files in ``OmegaQMC/psi/nn/conf/``
    for preset configurations (PauliNet, FermiNet,
    DeepErwin, PsiFormer).
    """

    # --- Top-level wavefunction ---
    n_determinants: int = 16
    full_determinant: bool = True
    backflow_transform: str = 'mult'

    # --- Envelope ---
    envelope_isotropic: bool = True
    envelope_per_shell: bool = False
    envelope_per_orbital_exponent: bool = True
    envelope_spin_restricted: bool = False
    envelope_init_to_ones: bool = True
    envelope_softplus_zeta: bool = False

    # --- Electronic cusp ---
    cusp_electrons: bool = True
    cusp_electrons_type: str = 'deepqmc'
    cusp_same_scale: float = 0.25
    cusp_anti_scale: float = 0.5
    cusp_alpha: float = 10.0
    cusp_trainable_alpha: bool = False
    cusp_nuclei: bool = False

    # --- OmniNet / GNN ---
    embedding_dim: int = 128
    use_jastrow: bool = True
    use_backflow: bool = True
    n_interactions: int = 3
    two_particle_stream_dim: int = 32
    self_interaction: bool = False
    deep_features: str = 'shared'
    update_rule: str = 'concatenate'

    # --- MLP defaults ---
    mlp_hidden_layers: Optional[list] = None
    mlp_bias: bool = True
    mlp_activation: str = 'tanh'
    mlp_init: str = 'default'

    # --- Backflow MLP ---
    bf_mlp_hidden_layers: Optional[list] = None
    bf_mlp_bias: bool = False
    bf_mlp_last_linear: bool = True
    bf_mlp_activation: Optional[str] = None
    bf_mlp_init: str = 'default'

    # --- Jastrow MLP ---
    jas_mlp_hidden_layers: Optional[list] = None
    jas_mlp_bias: bool = False
    jas_mlp_last_linear: bool = True
    jas_mlp_activation: Optional[str] = None
    jas_mlp_init: str = 'default'
    jas_sum_first: bool = True

    # --- G subnet MLP ---
    g_mlp_hidden_layers: Optional[list] = None
    g_mlp_bias: bool = False
    g_mlp_last_linear: bool = False
    g_mlp_activation: str = 'tanh'
    g_mlp_init: str = 'default'

    # --- Conf coeff ---
    conf_coeff: str = 'linear'

    # --- Update features ---
    update_features: Optional[list] = None

    # --- Edge features ---
    edge_types: Optional[list] = None

    # --- Embeddings ---
    use_spin_embedding: bool = False
    project_to_embedding_dim: bool = False

    # --- Ne positional embedding ---
    ne_powers: Optional[list] = None
    ne_log_rescale: bool = False

    # --- Residual connections ---
    electron_residual_normalize: bool = True
    two_particle_residual_normalize: bool = True
    nucleus_residual: bool = False

    # --- QED extension ---
    # When set, the electron embedding concatenates a
    # per-electron one-hot encoding of the photon Fock
    # index n (size nph_max + 1) before projection.
    # Tang 2025 (2503.15644) Sec. II.C.
    qed_nph_max: Optional[int] = None

    # --- Complex-valued wavefunction (chiral cavity QED) ---
    # When True, the wavefunction is built as a complex
    # Slater determinant: orbital matrices become complex
    # (Re + i*Im) via a parallel imaginary-part pathway,
    # log_psi becomes complex, and Psi.sign is a complex
    # unit (phase factor exp(i*phi)) instead of +-1.
    #
    # Required for time-reversal-broken Hamiltonians such
    # as a circularly polarized (chiral) cavity, external
    # magnetic field, or non-equilibrium current-carrying
    # states. Default False reproduces existing real-Psi
    # behaviour exactly.
    #
    # Downstream consumers (qed_physics local-energy
    # estimator, SR optimizer) must take Re(E_loc) and
    # Re(<O^dag O>) appropriately when this flag is set.
    complex_psi: bool = False

    def __post_init__(self):
        """Fill in ``None`` fields with sensible defaults."""
        if self.mlp_hidden_layers is None:
            self.mlp_hidden_layers = ['log', 2]
        if self.bf_mlp_hidden_layers is None:
            self.bf_mlp_hidden_layers = ['log', 1]
        if self.jas_mlp_hidden_layers is None:
            self.jas_mlp_hidden_layers = ['log', 1]
        if self.g_mlp_hidden_layers is None:
            self.g_mlp_hidden_layers = ['log', 1]
        if self.update_features is None:
            self.update_features = [
                {'type': 'residual'},
                {
                    'type': 'node_sum',
                    'node_types': ['up', 'down'],
                    'normalize': True,
                },
                {
                    'type': 'convolution',
                    'edge_types': ['same', 'anti'],
                    'normalize': False,
                },
            ]
        if self.edge_types is None:
            self.edge_types = ['same', 'anti']
        if self.ne_powers is None:
            self.ne_powers = [1]


def load_nn_config(name_or_path: str) -> NNAnsatzConfig:
    """Load an ansatz config by name or YAML path.

    Built-in names: ``paulinet``, ``ferminet``,
    ``deeperwin``, ``psiformer``.

    Args:
        name_or_path: A built-in config name, or a
            path to a custom YAML file.

    Returns:
        :class:`NNAnsatzConfig` instance.
    """
    if name_or_path in _BUILTIN_CONFIGS:
        path = os.path.join(
            _CONF_DIR, f'{name_or_path}.yaml',
        )
    else:
        path = name_or_path
    with open(path) as f:
        data = yaml.safe_load(f)
    return NNAnsatzConfig(**data)
