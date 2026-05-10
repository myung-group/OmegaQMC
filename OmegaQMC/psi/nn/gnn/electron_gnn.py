"""Electron graph neural network (Flax NNX).

Rewritten from ``deepqmc/gnn/electron_gnn.py``.
Provides :class:`ElectronGNN`, the message-passing
backbone used by all DeepQMC-style architectures.
"""

from itertools import accumulate

import jax
import jax.numpy as jnp
from flax import nnx

from .graph import (
    Graph,
    GraphNodes,
    GraphUpdate,
    MolecularGraphEdgeBuilder,
)
from .utils import NodeEdgeMapping


# ------------------------------------------------------------
# Electron embedding
# ------------------------------------------------------------

class ElectronEmbedding(nnx.Module):
    """Create initial electron embeddings.

    Supports two modes:

    * **Positional** (when *positional_embedding_ne*
      is not ``None``): nucleus-electron edge features
      are flattened into a per-electron vector and
      optionally projected to *embedding_dim*.
    * **Type** (otherwise): a learned embedding per
      electron type (1 if ``n_up == n_down``, else 2).

    Args:
        n_nuc: Number of nuclei.
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        embedding_dim: Target embedding dimension.
        ne_feat_dim: Dimension of the ne edge
            features (only for positional mode).
        positional_embedding_ne: Edge feature callable
            for ne edges, or ``None``.
        use_spin: Concatenate a spin indicator
            (positional mode only).
        project_to_embedding_dim: Project the
            positional features to *embedding_dim*.
        rngs: Flax NNX RNG state.
    """

    def __init__(
        self, *, n_nuc, n_up, n_down,
        embedding_dim, ne_feat_dim,
        positional_embedding_ne,
        use_spin, project_to_embedding_dim,
        rngs,
        qed_nph_max=None,
    ):
        self.n_nuc = n_nuc
        self.n_up = n_up
        self.n_down = n_down
        self.embedding_dim = embedding_dim
        self.ne_embed = positional_embedding_ne
        self.use_spin = use_spin
        # Tang-style per-electron one-hot photon injection.
        # When set, append one_hot(phys_conf.n) of size
        # nph_max+1 to each electron's input feature vector
        # before the projection layer (Tang 2025 Sec. II.C).
        self.qed_nph_max = qed_nph_max

        if positional_embedding_ne is not None:
            in_dim = ne_feat_dim * n_nuc
            if use_spin:
                in_dim += 1
            if qed_nph_max is not None:
                in_dim += qed_nph_max + 1
            if project_to_embedding_dim:
                self.proj = nnx.Linear(
                    in_dim, embedding_dim,
                    use_bias=False, rngs=rngs,
                )
            else:
                self.proj = None
        else:
            n_types = (
                1 if n_up == n_down else 2
            )
            self.type_embed = nnx.Embed(
                n_types, embedding_dim, rngs=rngs,
            )
            self.elec_types = jnp.array(
                n_up * [0]
                + n_down * [int(n_up != n_down)]
            )

    def __call__(self, phys_conf, nucleus_embedding):
        """Build initial per-electron embeddings.

        Args:
            phys_conf: :class:`~..types.PhysicalConfiguration`.
            nucleus_embedding: Nucleus embedding array
                (unused; accepted for interface
                compatibility).

        Returns:
            Electron embedding array ``(n_elec, emb_dim)``
            or ``(n_elec, ne_feat_dim * n_nuc [+ 1])``.
        """
        if self.ne_embed is not None:
            factory = MolecularGraphEdgeBuilder(
                self.n_nuc,
                self.n_up,
                self.n_down,
                ['ne'],
                self_interaction=False,
            )
            ne_edges = factory(phys_conf)['ne']
            feats = self.ne_embed(
                ne_edges.single_array,
            )
            n_el = self.n_up + self.n_down
            x = (
                feats
                .swapaxes(0, 1)
                .reshape(n_el, -1)
            )
            if self.use_spin:
                spins = jnp.concatenate([
                    jnp.ones(self.n_up),
                    -jnp.ones(self.n_down),
                ])[:, None]
                x = jnp.concatenate(
                    [x, spins], axis=1,
                )
            if self.qed_nph_max is not None:
                # Per-electron one-hot photon Fock index.
                n_el = self.n_up + self.n_down
                n_oh = jax.nn.one_hot(
                    phys_conf.n,
                    self.qed_nph_max + 1,
                    dtype=x.dtype,
                )
                n_oh = jnp.broadcast_to(
                    n_oh[None, :],
                    (n_el, self.qed_nph_max + 1),
                )
                x = jnp.concatenate([x, n_oh], axis=1)
            if self.proj is not None:
                x = self.proj(x)
        else:
            x = self.type_embed(self.elec_types)
        return x


# ------------------------------------------------------------
# GNN layer
# ------------------------------------------------------------

class ElectronGNNLayer(nnx.Module):
    """Single message-passing layer of the GNN.

    Args:
        embedding_dim: Electron embedding dimension.
        two_particle_stream_dim: Edge feature dim.
        edge_types: Edge type labels.
        subnet: MLP for combining update features
            into new node embeddings.
        subnet_g: MLP for deep-feature edge updates
            (shared mode) or dict of MLPs (separate).
        electron_residual: Residual connection for
            electron embeddings, or ``None``.
        nucleus_residual: Residual connection for
            nucleus embeddings, or ``None``.
        two_particle_residual: Residual connection
            for edge features, or ``None``.
        deep_features: ``False``, ``'shared'``, or
            ``'separate'``.
        update_rule: ``'concatenate'``, ``'sum'``,
            ``'featurewise'``, or
            ``'featurewise_shared'``.
        update_features: List of
            :class:`~.update_features.UpdateFeature`.
    """

    def __init__(
        self, *, embedding_dim,
        two_particle_stream_dim,
        edge_types, subnet, subnet_g,
        electron_residual, nucleus_residual,
        two_particle_residual,
        deep_features, update_rule,
        update_features,
    ):
        assert update_rule in (
            'concatenate', 'featurewise',
            'featurewise_shared', 'sum',
        )
        assert deep_features in (
            False, 'shared', 'separate',
        )
        self.embedding_dim = embedding_dim
        self.tp_dim = two_particle_stream_dim
        self.edge_types = edge_types
        self.subnet = subnet
        self.subnet_g = subnet_g
        self.electron_residual = electron_residual
        self.nucleus_residual = nucleus_residual
        self.tp_residual = two_particle_residual
        self.deep_features = deep_features
        self.update_rule = update_rule
        self.update_features = nnx.List(
            update_features,
        )

    def __call__(self, graph, last_layer=False):
        """Apply one message-passing step.

        Args:
            graph: :class:`~.graph.Graph` containing
                current node and edge states.
            last_layer: If ``True``, skip the deep-
                feature edge update (not needed on the
                final layer).

        Returns:
            Updated :class:`~.graph.Graph`.
        """
        nodes, edges = graph

        # 1. Aggregate: compute update features
        #    (convolution sees CURRENT edges)
        fs = sum(
            (
                uf(nodes, edges)
                for uf in self.update_features
            ),
            start=[],
        )
        elec_feats = [
            f.electrons for f in fs
            if f.electrons is not None
        ]

        # 2. Update nodes
        updated = self._apply_update(
            elec_feats,
        )
        if self.electron_residual:
            updated = self.electron_residual(
                nodes.electrons, updated,
            )

        # 3. Update edges (deep features) AFTER
        #    node update, not on last layer
        if not last_layer and self.deep_features:
            edges = self._update_edges(edges)

        return Graph(
            GraphNodes(nodes.nuclei, updated),
            edges,
        )

    # ---- edge update (deep features) ----

    def _update_edges(self, edges):
        if self.deep_features == 'shared':
            keys = list(edges.keys())
            objs = [edges[k] for k in keys]
            feats = [
                e.single_array for e in objs
            ]
            sizes = [f.shape[-2] for f in feats]
            idxs = list(accumulate(sizes[:-1]))
            combined = jnp.concatenate(
                feats, axis=-2,
            )
            combined = self.subnet_g(combined)
            parts = jnp.split(
                combined, idxs, axis=-2,
            )
            new_edges = {
                k: e.update_from_single_array(p)
                for k, e, p in zip(
                    keys, objs, parts,
                )
            }
        elif self.deep_features == 'separate':
            new_edges = {
                k: e.update_from_single_array(
                    self.subnet_g[k](
                        e.single_array,
                    ),
                )
                for k, e in edges.items()
            }
        else:
            return edges

        if self.tp_residual:
            new_edges = self.tp_residual(
                edges, new_edges,
            )
        return new_edges

    # ---- update rule ----

    def _apply_update(self, features):
        rule = self.update_rule
        if rule == 'concatenate':
            return self.subnet(
                jnp.concatenate(
                    features, axis=-1,
                ),
            )
        elif rule == 'sum':
            return self.subnet(sum(features))
        elif rule == 'featurewise_shared':
            return jnp.sum(
                self.subnet(
                    jnp.stack(features),
                ),
                axis=0,
            )
        elif rule == 'featurewise':
            return sum(
                self.subnet[name](fi)
                for fi, name in zip(
                    features,
                    self.subnet.keys(),
                )
            )
        raise ValueError(
            f'Unknown update rule: {rule}'
        )


# ------------------------------------------------------------
# Full electron GNN
# ------------------------------------------------------------

class ElectronGNN(nnx.Module):
    """Graph neural network over electrons.

    Produces per-electron embedding vectors by
    repeated message passing on a molecular graph.

    Args:
        mol_info: :class:`~OmegaQMC.utils.Mole_custom`.
        embedding_dim: Electron embedding dimension.
        two_particle_stream_dim: Edge feature dim.
        n_interactions: Number of GNN layers.
        self_interaction: Include self-loop edges.
        edge_types: Edge type labels.
        electron_embedding: :class:`ElectronEmbedding`
            instance.
        layers: List of :class:`ElectronGNNLayer`.
        edge_features: Dict mapping edge type to a
            callable that computes edge features.
    """

    def __init__(
        self, *, mol_info, embedding_dim,
        two_particle_stream_dim,
        n_interactions, self_interaction,
        edge_types, electron_embedding,
        layers, edge_features,
    ):
        n_nuc = len(mol_info.charges)
        self.n_nuc = n_nuc
        self.n_up = mol_info.n_up
        self.n_down = mol_info.n_down
        self.embedding_dim = embedding_dim
        self.self_interaction = self_interaction
        self.edge_types = tuple(edge_types)
        self.electron_embedding = electron_embedding
        self.layers = nnx.List(layers)
        self.edge_features = edge_features

    def _build_edges(self, phys_conf):
        """Build initial edge features."""
        factory = MolecularGraphEdgeBuilder(
            self.n_nuc,
            self.n_up,
            self.n_down,
            self.edge_types,
            self_interaction=self.self_interaction,
        )
        raw = factory(phys_conf)
        return {
            typ: raw[typ].update_from_single_array(
                self.edge_features[typ](
                    raw[typ].single_array,
                ),
            )
            for typ in self.edge_types
        }

    def __call__(self, phys_conf):
        """Run the GNN and return final embeddings.

        Args:
            phys_conf: :class:`PhysicalConfiguration`.

        Returns:
            :class:`GraphNodes` with the final
            electron (and optional nucleus)
            embeddings.
        """
        edges = self._build_edges(phys_conf)
        elec_emb = self.electron_embedding(
            phys_conf, None,
        )
        nodes = GraphNodes(None, elec_emb)
        graph = Graph(nodes, edges)

        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            last = (i == n - 1)
            graph = layer(graph, last_layer=last)

        return graph.nodes
