"""Build NN wavefunction from config + molecule info.

Maps :class:`NNAnsatzConfig` fields to the NNX module
constructors for envelope, GNN, backflow, Jastrow,
cusp, and the top-level wavefunction.
"""

from flax import nnx

from .config import NNAnsatzConfig
from .cusp import (
    DeepQMCCusp,
    ElectronicCuspAsymptotic,
    NuclearCuspAsymptotic,
    PsiformerCusp,
)
from .env import ExponentialEnvelopes
from .gnn.edge_features import (
    CombinedEdgeFeature,
    DifferenceEdgeFeature,
    DistancePowerEdgeFeature,
)
from .layers import (
    MLP,
    Identity,
    ResidualConnection,
    SumPool,
)
from .omni import Backflow, Jastrow, OmniNet
from .wf import (
    BackflowOp,
    NeuralNetworkWaveFunction,
)


def _make_ne_embedding(config):
    """Build nucleus-electron positional embedding."""
    features = []
    features.append(
        DistancePowerEdgeFeature(
            powers=config.ne_powers,
            log_rescale=config.ne_log_rescale,
        ),
    )
    features.append(
        DifferenceEdgeFeature(
            log_rescale=config.ne_log_rescale,
        ),
    )
    return CombinedEdgeFeature(features=features)


def _ne_feat_dim(config):
    """Dimension of ne positional embedding."""
    return len(config.ne_powers) + 3


def build_nn_wf(
    config: NNAnsatzConfig,
    mol_info,
    rngs: nnx.Rngs,
) -> NeuralNetworkWaveFunction:
    """Construct the full NN wavefunction from config.

    Args:
        config: Ansatz hyperparameters.
        mol_info: Molecular geometry and electron counts.
        rngs: Flax NNX RNG state.

    Returns:
        :class:`NeuralNetworkWaveFunction` instance.
    """
    n_up = mol_info.n_up
    n_down = mol_info.n_down
    n_elec = n_up + n_down
    n_nuc = len(mol_info.charges)
    n_det = config.n_determinants
    fd = config.full_determinant

    # --- Envelope ---
    envelope = ExponentialEnvelopes(
        mol_info, n_det,
        isotropic=config.envelope_isotropic,
        per_shell=config.envelope_per_shell,
        per_orbital_exponent=(
            config.envelope_per_orbital_exponent
        ),
        spin_restricted=(
            config.envelope_spin_restricted
        ),
        init_to_ones=config.envelope_init_to_ones,
        softplus_zeta=config.envelope_softplus_zeta,
    )

    # --- Cusp electrons ---
    cusp_elec = None
    if config.cusp_electrons:
        cusp_fn = {
            'deepqmc': DeepQMCCusp,
            'psiformer': PsiformerCusp,
        }[config.cusp_electrons_type]()
        cusp_elec = ElectronicCuspAsymptotic(
            same_scale=config.cusp_same_scale,
            anti_scale=config.cusp_anti_scale,
            alpha=config.cusp_alpha,
            cusp_function=cusp_fn,
            trainable_alpha=config.cusp_trainable_alpha,
        )

    # --- Cusp nuclei ---
    cusp_nuc = None
    if config.cusp_nuclei:
        cusp_fn = DeepQMCCusp()
        cusp_nuc = NuclearCuspAsymptotic(
            mol_info.charges,
            cusp_function=cusp_fn,
        )

    # --- Backflow op ---
    backflow_op = BackflowOp()

    # --- Conf coeff ---
    if config.conf_coeff == 'sum_pool':
        conf_coeff = SumPool(1)
    else:
        conf_coeff = nnx.Linear(
            n_det, 1, use_bias=False, rngs=rngs,
        )

    # --- OmniNet (GNN + Jastrow + Backflow) ---
    n_bf = (
        2 if config.backflow_transform == 'both'
        else 1
    )
    if fd:
        n_orb_up = n_orb_down = n_elec
    else:
        n_orb_up, n_orb_down = n_up, n_down

    omni = _build_omni(
        config, mol_info, n_orb_up, n_orb_down,
        n_det, n_bf, rngs,
    )

    return NeuralNetworkWaveFunction(
        mol_info,
        omni=omni,
        envelope=envelope,
        backflow_op=backflow_op,
        n_determinants=n_det,
        full_determinant=fd,
        cusp_electrons=cusp_elec,
        cusp_nuclei=cusp_nuc,
        backflow_transform=config.backflow_transform,
        conf_coeff=conf_coeff,
        complex_psi=config.complex_psi,
        embedding_dim=config.embedding_dim,
        rngs=rngs,
    )


def _build_omni(
    config, mol_info, n_orb_up, n_orb_down,
    n_det, n_bf, rngs,
):
    """Build the OmniNet (GNN + Jastrow + Backflow)."""
    from .gnn.electron_gnn import (
        ElectronEmbedding,
        ElectronGNN,
        ElectronGNNLayer,
    )

    n_up = mol_info.n_up
    n_down = mol_info.n_down
    n_elec = n_up + n_down
    n_nuc = len(mol_info.charges)
    emb_dim = config.embedding_dim
    tp_dim = config.two_particle_stream_dim

    # --- Edge features for GNN ---
    edge_types = config.edge_types or []

    def _ee_edge_feat():
        return CombinedEdgeFeature(features=[
            DistancePowerEdgeFeature(powers=[1]),
            DifferenceEdgeFeature(),
        ])

    edge_features = {}
    for et in edge_types:
        edge_features[et] = _ee_edge_feat()
    ee_feat_dim = 4 if edge_types else 0

    # --- Ne positional embedding ---
    ne_embed = _make_ne_embedding(config)
    ne_dim = _ne_feat_dim(config)

    # --- Electron embedding ---
    elec_embed = ElectronEmbedding(
        n_nuc=n_nuc,
        n_up=n_up,
        n_down=n_down,
        embedding_dim=emb_dim,
        ne_feat_dim=ne_dim,
        positional_embedding_ne=ne_embed,
        use_spin=config.use_spin_embedding,
        project_to_embedding_dim=(
            config.project_to_embedding_dim
        ),
        qed_nph_max=config.qed_nph_max,
        rngs=rngs,
    )

    # --- Initial embedding dimension ---
    # Positional embeddings without projection
    # produce dim != embedding_dim.
    if (
        ne_embed is not None
        and not config.project_to_embedding_dim
    ):
        init_emb = ne_dim * n_nuc + (
            1 if config.use_spin_embedding else 0
        )
    else:
        init_emb = emb_dim

    # --- Build GNN layers ---
    n_layers = config.n_interactions

    def _make_layer(idx):
        # First layer: edges = ee_feat_dim,
        #   nodes = init_emb.
        # Later: edges = tp_dim, nodes = emb_dim.
        ef = (
            ee_feat_dim if idx == 0
            else tp_dim
        )
        nd = init_emb if idx == 0 else emb_dim
        return _build_gnn_layer(
            config, emb_dim, tp_dim,
            edge_types,
            n_up, n_down, idx, ef, nd, rngs,
        )

    layers = [_make_layer(i) for i in range(n_layers)]

    gnn = ElectronGNN(
        mol_info=mol_info,
        embedding_dim=emb_dim,
        two_particle_stream_dim=tp_dim,
        n_interactions=n_layers,
        self_interaction=config.self_interaction,
        edge_types=edge_types,
        electron_embedding=elec_embed,
        layers=layers,
        edge_features=edge_features,
    )

    # --- Jastrow ---
    jastrow = None
    if config.use_jastrow:
        jas_out = MLP(
            emb_dim, 1,
            hidden_layers=config.jas_mlp_hidden_layers,
            bias=config.jas_mlp_bias,
            last_linear=config.jas_mlp_last_linear,
            activation=config.jas_mlp_activation,
            init=config.jas_mlp_init,
            rngs=rngs,
        )
        jastrow = Jastrow(
            net=jas_out,
            sum_first=config.jas_sum_first,
        )

    # --- Backflow ---
    backflow = None
    if config.use_backflow:
        bf_dict = {}
        for spin, n_orb in [
            ('up', n_orb_up), ('down', n_orb_down),
        ]:
            out_dim = n_orb * n_det
            nets = [
                MLP(
                    emb_dim, out_dim,
                    hidden_layers=(
                        config.bf_mlp_hidden_layers
                    ),
                    bias=config.bf_mlp_bias,
                    last_linear=(
                        config.bf_mlp_last_linear
                    ),
                    activation=(
                        config.bf_mlp_activation
                    ),
                    init=config.bf_mlp_init,
                    rngs=rngs,
                )
                for _ in range(n_bf)
            ]
            bf_dict[spin] = Backflow(
                n_orb, n_det, n_bf, spin,
                nets=nets,
            )
        backflow = bf_dict

    return OmniNet(
        n_up=n_up,
        gnn=gnn,
        jastrow=jastrow,
        backflow=backflow,
    )


def _build_gnn_layer(
    config, emb_dim, tp_dim,
    edge_types,
    n_up, n_down, layer_idx,
    edge_feat_dim, node_dim, rngs,
):
    """Build a single ElectronGNNLayer."""
    from .gnn.electron_gnn import ElectronGNNLayer
    from .gnn.update_features import (
        ConvolutionElectronUpdateFeature,
        EdgeSumElectronUpdateFeature,
        NodeAttentionElectronUpdateFeature,
        NodeSumElectronUpdateFeature,
        ResidualElectronUpdateFeature,
    )

    # Main subnet MLP factory
    def _subnet(in_d, out_d):
        return MLP(
            in_d, out_d,
            hidden_layers=config.mlp_hidden_layers,
            bias=config.mlp_bias,
            last_linear=False,
            activation=config.mlp_activation,
            init=config.mlp_init,
            rngs=rngs,
        )

    # G subnet MLP factory
    def _g_subnet(in_d, out_d):
        return MLP(
            in_d, out_d,
            hidden_layers=config.g_mlp_hidden_layers,
            bias=config.g_mlp_bias,
            last_linear=config.g_mlp_last_linear,
            activation=config.g_mlp_activation,
            init=config.g_mlp_init,
            rngs=rngs,
        )

    # Residual connections
    e_res = (
        ResidualConnection(
            normalize=(
                config.electron_residual_normalize
            ),
        )
        if config.electron_residual_normalize
        is not False
        else None
    )
    tp_res = (
        ResidualConnection(
            normalize=(
                config.two_particle_residual_normalize
            ),
        )
        if config.two_particle_residual_normalize
        is not False
        else None
    )

    # Build update features
    # node_dim = actual embedding dim on this layer
    uf_list = []
    for uf_spec in config.update_features:
        uf_type = uf_spec['type']
        if uf_type == 'residual':
            uf_list.append(
                ResidualElectronUpdateFeature(
                    node_dim,
                )
            )
        elif uf_type == 'node_sum':
            uf_list.append(
                NodeSumElectronUpdateFeature(
                    node_dim, n_up, n_down,
                    node_types=uf_spec.get(
                        'node_types',
                        ['up', 'down'],
                    ),
                    normalize=uf_spec.get(
                        'normalize', True,
                    ),
                )
            )
        elif uf_type == 'convolution':
            w_nets = {}
            h_nets = {}
            conv_ets = uf_spec.get(
                'edge_types', edge_types,
            )
            for et in conv_ets:
                w_nets[et] = _subnet(
                    edge_feat_dim, tp_dim,
                )
                h_nets[et] = _subnet(
                    node_dim, tp_dim,
                )
            uf_list.append(
                ConvolutionElectronUpdateFeature(
                    tp_dim, n_up, n_down,
                    edge_types=conv_ets,
                    normalize=uf_spec.get(
                        'normalize', False,
                    ),
                    w_nets=w_nets,
                    h_nets=h_nets,
                )
            )
        elif uf_type == 'edge_sum':
            # edge_sum is a pure aggregation (no transform),
            # so its output dim equals the *actual* incoming
            # edge feature dim on this layer (edge_feat_dim),
            # not the architectural target tp_dim. They differ
            # at layer 0 (raw ee features) before deep_features
            # transforms edges to tp_dim.
            uf_list.append(
                EdgeSumElectronUpdateFeature(
                    edge_feat_dim, n_up, n_down,
                    edge_types=uf_spec.get(
                        'edge_types', edge_types,
                    ),
                    normalize=uf_spec.get(
                        'normalize', True,
                    ),
                )
            )
        elif uf_type == 'node_attention':
            uf_list.append(
                NodeAttentionElectronUpdateFeature(
                    node_dim,
                    num_heads=uf_spec.get(
                        'num_heads', 4,
                    ),
                    mlp_factory=(
                        lambda ind, outd: MLP(
                            ind, outd,
                            hidden_layers=uf_spec.get(
                                'mlp_hidden_layers',
                                ['log', 2],
                            ),
                            bias=uf_spec.get(
                                'mlp_bias', True,
                            ),
                            last_linear=False,
                            activation=uf_spec.get(
                                'mlp_activation',
                                'tanh',
                            ),
                            init=uf_spec.get(
                                'mlp_init', 'ferminet',
                            ),
                            rngs=rngs,
                        )
                    ),
                    attention_residual=(
                        ResidualConnection(
                            normalize=uf_spec.get(
                                'attention_residual'
                                '_normalize', False,
                            ),
                        )
                    ),
                    mlp_residual=(
                        ResidualConnection(
                            normalize=uf_spec.get(
                                'mlp_residual'
                                '_normalize',
                                False,
                            ),
                        )
                    ),
                    rngs=rngs,
                )
            )

    # Compute update input dim
    uf_total_dim = sum(
        uf.output_dim for uf in uf_list
    )

    # Main subnet for concatenation
    subnet = _subnet(uf_total_dim, emb_dim)

    # G subnet for two-particle stream
    g_subnet = _g_subnet(edge_feat_dim, tp_dim)

    return ElectronGNNLayer(
        embedding_dim=emb_dim,
        two_particle_stream_dim=tp_dim,
        edge_types=edge_types,
        subnet=subnet,
        subnet_g=g_subnet,
        electron_residual=e_res,
        nucleus_residual=None,
        two_particle_residual=tp_res,
        deep_features=config.deep_features,
        update_rule=config.update_rule,
        update_features=uf_list,
    )
