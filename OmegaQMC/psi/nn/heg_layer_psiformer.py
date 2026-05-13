"""PsiFormer GNN layer builder for HEG.

Provides :func:`_build_heg_gnn_layer`, which constructs one
``ElectronGNNLayer`` with the PsiFormer update-feature recipe
``[Residual, NodeSum, NodeAttention]``.  Used by
:func:`~.heg_wf_module.build_heg_psiformer_wf` when
``ansatz.backbone == 'psiformer'``.
"""

from flax import nnx

from .gnn.electron_gnn import ElectronGNNLayer
from .gnn.update_features import (
    NodeAttentionElectronUpdateFeature,
    NodeSumElectronUpdateFeature,
    ResidualElectronUpdateFeature,
)
from .layers import MLP, ResidualConnection


def _build_heg_gnn_layer(
    config, *, idx,
    emb_dim, tp_dim,
    edge_types,
    ee_feat_dim,
    n_up, n_down,
    rngs,
):
    """Build one PsiFormer-style layer for the HEG GNN.

    Mirrors the molecular ``_build_gnn_layer`` but with a fixed,
    HEG-specific update-feature set (no nuclear convolution) and
    pre-chosen MLP widths.
    """
    def _subnet(in_d, out_d):
        return MLP(
            in_d, out_d,
            hidden_layers=config.mlp_hidden_layers,
            bias=True,
            last_linear=False,
            activation='tanh',
            init='ferminet',
            rngs=rngs,
        )

    def _g_subnet(in_d, out_d):
        return MLP(
            in_d, out_d,
            hidden_layers=config.g_mlp_hidden_layers,
            bias=False,
            last_linear=False,
            activation='tanh',
            init='ferminet',
            rngs=rngs,
        )

    electron_residual = ResidualConnection(normalize=False)
    tp_residual = ResidualConnection(normalize=False)

    # On layer 0 the incoming node dim is the raw embedding, on
    # later layers it is emb_dim (after subnet projection).
    node_dim = emb_dim
    edge_feat_dim = ee_feat_dim if idx == 0 else tp_dim

    # Update features: residual + per-spin node sum + attention.
    uf_list = [
        ResidualElectronUpdateFeature(node_dim),
        NodeSumElectronUpdateFeature(
            node_dim, n_up, n_down,
            node_types=['up', 'down'],
            normalize=True,
        ),
        NodeAttentionElectronUpdateFeature(
            node_dim,
            num_heads=config.n_attention_heads,
            mlp_factory=(
                lambda ind, outd: MLP(
                    ind, outd,
                    hidden_layers=config.mlp_hidden_layers,
                    bias=True,
                    last_linear=False,
                    activation='tanh',
                    init='ferminet',
                    rngs=rngs,
                )
            ),
            attention_residual=ResidualConnection(normalize=False),
            mlp_residual=ResidualConnection(normalize=False),
            rngs=rngs,
        ),
    ]
    uf_total_dim = sum(uf.output_dim for uf in uf_list)

    subnet = _subnet(uf_total_dim, emb_dim)
    g_subnet = _g_subnet(edge_feat_dim, tp_dim)

    return ElectronGNNLayer(
        embedding_dim=emb_dim,
        two_particle_stream_dim=tp_dim,
        edge_types=edge_types,
        subnet=subnet,
        subnet_g=g_subnet,
        electron_residual=electron_residual,
        nucleus_residual=None,
        two_particle_residual=tp_residual,
        deep_features='shared',
        update_rule='concatenate',
        update_features=uf_list,
    )
