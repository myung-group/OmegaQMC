"""FermiNet GNN layer builder for HEG.

Provides :func:`_build_heg_ferminet_layer`, which constructs one
``ElectronGNNLayer`` with FermiNet's update-feature recipe
``[Residual, NodeSum, EdgeSum]`` (sender-spin edges).  Used by
:func:`~.heg_wf_module.build_heg_psiformer_wf` when
``ansatz.backbone == 'ferminet'``.
"""

from flax import nnx

from .gnn.electron_gnn import ElectronGNNLayer
from .gnn.update_features import (
    EdgeSumElectronUpdateFeature,
    NodeSumElectronUpdateFeature,
    ResidualElectronUpdateFeature,
)
from .layers import MLP, ResidualConnection


def _build_heg_ferminet_layer(
    config, *, idx,
    emb_dim, tp_dim,
    edge_types,
    ee_feat_dim,
    n_up, n_down,
    rngs,
):
    """Build one FermiNet-style layer for the HEG GNN.

    Mirrors :func:`~.heg_layer_psiformer._build_heg_gnn_layer` but
    with FermiNet's update feature recipe ``[Residual, NodeSum,
    EdgeSum]`` (vs PsiFormer's ``[Residual, NodeSum, NodeAttention]``).
    EdgeSum aggregates each sender-spin's edge embeddings — keyed by
    ``edge_types`` from the GNN, which for FermiNet is
    ``['up', 'down']``.
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

    node_dim = emb_dim
    edge_feat_dim = ee_feat_dim if idx == 0 else tp_dim

    # Update features: residual + per-spin node sum + sender-spin
    # edge sum.  Matches conf/ferminet.yaml's update_features list.
    uf_list = [
        ResidualElectronUpdateFeature(node_dim),
        NodeSumElectronUpdateFeature(
            node_dim, n_up, n_down,
            node_types=['up', 'down'],
            normalize=True,
        ),
        EdgeSumElectronUpdateFeature(
            tp_dim, n_up, n_down,
            edge_types=edge_types,
            normalize=True,
        ),
    ]
    # Note: EdgeSum.output_dim is tp_dim*n_edge_types but at layer 0
    # the edges haven't been mapped to tp_dim yet — they still carry
    # the raw ee_feat_dim.  Compute uf_total_dim explicitly so the
    # subnet's input width matches the actual concatenation width.
    edge_sum_actual_dim = edge_feat_dim * len(edge_types)
    uf_total_dim = (
        uf_list[0].output_dim
        + uf_list[1].output_dim
        + edge_sum_actual_dim
    )

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
