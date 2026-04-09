"""GNN update feature modules (Flax NNX).

Rewritten from ``deepqmc/gnn/update_features.py``.
Each class computes one or more update features
for node embeddings in an ElectronGNNLayer.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .graph import GraphNodes
from .utils import NodeEdgeMapping


# ------------------------------------------------------------
# Multi-head attention (replaces hk.MultiHeadAttention)
# ------------------------------------------------------------

class _MultiHeadAttention(nnx.Module):
    """Multi-head dot-product self-attention.

    Args:
        num_heads: Number of attention heads.
        emb_dim: Embedding dimension (must be
            divisible by *num_heads*).
        rngs: Flax NNX RNG state.
    """

    def __init__(self, num_heads, emb_dim, *, rngs):
        hd = emb_dim // num_heads
        assert hd * num_heads == emb_dim
        self.num_heads = num_heads
        self.head_dim = hd
        ki = nnx.initializers.variance_scaling(
            1.0, 'fan_in', 'normal',
        )
        self.wq = nnx.Linear(
            emb_dim, emb_dim, use_bias=False,
            kernel_init=ki, rngs=rngs,
        )
        self.wk = nnx.Linear(
            emb_dim, emb_dim, use_bias=False,
            kernel_init=ki, rngs=rngs,
        )
        self.wv = nnx.Linear(
            emb_dim, emb_dim, use_bias=False,
            kernel_init=ki, rngs=rngs,
        )
        self.wo = nnx.Linear(
            emb_dim, emb_dim, use_bias=False,
            kernel_init=ki, rngs=rngs,
        )

    def __call__(self, q, k, v, mask=None):
        nq = q.shape[0]
        nk = k.shape[0]
        nh = self.num_heads
        hd = self.head_dim
        q = self.wq(q).reshape(nq, nh, hd)
        k = self.wk(k).reshape(nk, nh, hd)
        v = self.wv(v).reshape(nk, nh, hd)
        scale = jnp.sqrt(
            jnp.array(hd, dtype=q.dtype),
        )
        attn = (
            jnp.einsum('ihd,jhd->hij', q, k)
            / scale
        )
        if mask is not None:
            attn = jnp.where(
                mask,
                attn,
                jnp.finfo(q.dtype).min,
            )
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum('hij,jhd->ihd', attn, v)
        out = out.reshape(nq, nh * hd)
        return self.wo(out)


# ------------------------------------------------------------
# Update features — all nnx.Module for nnx.List compat
# ------------------------------------------------------------

class ResidualElectronUpdateFeature(nnx.Module):
    """Return unchanged electron embeddings.

    Args:
        emb_dim: Embedding dimension.
    """

    def __init__(self, emb_dim):
        self._emb_dim = emb_dim

    @property
    def names(self):
        return ('residual',)

    @property
    def output_dim(self):
        return self._emb_dim

    def __call__(self, nodes, edges):
        """Return unchanged electron embeddings.

        Args:
            nodes: Current :class:`~.graph.GraphNodes`.
            edges: Current edge dict (unused).

        Returns:
            List containing one
            :class:`~.graph.GraphNodes` with the
            original electron embeddings.
        """
        return [
            GraphNodes(None, nodes.electrons),
        ]


class NodeSumElectronUpdateFeature(nnx.Module):
    """Sum/mean of node embeddings per spin channel.

    Args:
        emb_dim: Embedding dimension.
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        node_types: Spin types to aggregate
            (``'up'``, ``'down'``, or both).
        normalize: Use mean instead of sum.
    """

    def __init__(
        self, emb_dim, n_up, n_down,
        *, node_types, normalize,
    ):
        assert all(
            nt in {'up', 'down'}
            for nt in node_types
        )
        self._emb_dim = emb_dim
        self.n_up = n_up
        self.n_down = n_down
        self.node_types = tuple(node_types)
        self.normalize = normalize

    @property
    def names(self):
        return tuple(
            f'node_{nt}' for nt in self.node_types
        )

    @property
    def output_dim(self):
        return self._emb_dim * len(self.node_types)

    def __call__(self, nodes, edges):
        """Aggregate per-spin node sums.

        Args:
            nodes: Current :class:`~.graph.GraphNodes`.
            edges: Current edge dict (unused).

        Returns:
            List of :class:`~.graph.GraphNodes`, one per
            requested *node_type*, each tiled to all
            electrons.
        """
        idx = {
            'up': slice(None, self.n_up),
            'down': slice(self.n_up, None),
        }
        fn = (
            jnp.mean if self.normalize
            else jnp.sum
        )
        n_el = self.n_up + self.n_down
        return [
            GraphNodes(
                None,
                jnp.tile(
                    fn(
                        nodes.electrons[idx[nt]],
                        axis=0,
                        keepdims=True,
                    ),
                    (n_el, 1),
                ),
            )
            for nt in self.node_types
        ]


class EdgeSumElectronUpdateFeature(nnx.Module):
    """Sum/mean of edge embeddings as update.

    Args:
        tp_dim: Two-particle stream dimension.
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        edge_types: Edge types to aggregate.
        normalize: Normalize by sender count.
    """

    def __init__(
        self, tp_dim, n_up, n_down,
        *, edge_types, normalize,
    ):
        self._tp_dim = tp_dim
        self.n_up = n_up
        self.n_down = n_down
        self.edge_types = tuple(edge_types)
        self.normalize = normalize

    @property
    def names(self):
        return tuple(
            f'edge_{et}'
            for et in self.edge_types
        )

    @property
    def output_dim(self):
        return self._tp_dim * len(self.edge_types)

    def __call__(self, nodes, edges):
        """Aggregate edge sums per edge type.

        Args:
            nodes: Current :class:`~.graph.GraphNodes`
                (unused).
            edges: Current edge dict.

        Returns:
            List of :class:`~.graph.GraphNodes`, one per
            *edge_type*, holding the summed edge features.
        """
        updates = []
        for et in self.edge_types:
            if et == 'ee':
                n = self.n_up + self.n_down
                fac = n if self.normalize else 1.0
                val = (
                    edges['same'].sum_senders(False)
                    + edges['anti'].sum_senders(
                        False,
                    )
                ) / fac
                updates.append(
                    GraphNodes(None, val),
                )
            else:
                updates.append(
                    GraphNodes(
                        None,
                        edges[et].sum_senders(
                            self.normalize,
                        ),
                    ),
                )
        return updates


class ConvolutionElectronUpdateFeature(
    nnx.Module,
):
    """Convolution of node and edge embeddings.

    Args:
        tp_dim: Two-particle stream dimension.
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        edge_types: Edge types to convolve over.
        normalize: Normalize by sender count.
        w_nets: Dict of MLPs for edge features,
            keyed by edge type.
        h_nets: Dict of MLPs for sender node
            embeddings, keyed by edge type.
        node_edge_mapping: Optional pre-built
            :class:`NodeEdgeMapping`.
    """

    def __init__(
        self, tp_dim, n_up, n_down,
        *, edge_types, normalize,
        w_nets, h_nets,
        node_edge_mapping=None,
    ):
        self._tp_dim = tp_dim
        self.n_up = n_up
        self.n_down = n_down
        self.edge_types = tuple(edge_types)
        self.normalize = normalize
        self.w_nets = nnx.Dict(w_nets)
        self.h_nets = nnx.Dict(h_nets)
        if node_edge_mapping is None:
            all_ets = list(edge_types)
            if 'ee' in all_ets:
                all_ets.remove('ee')
                for s in ('same', 'anti'):
                    if s not in all_ets:
                        all_ets.append(s)
            self._mapping = NodeEdgeMapping(all_ets)
        else:
            self._mapping = node_edge_mapping

    @property
    def names(self):
        return tuple(
            f'conv_{et}'
            for et in self.edge_types
        )

    @property
    def output_dim(self):
        return self._tp_dim * len(self.edge_types)

    def _single(self, nodes, edges, et, norm):
        w = self.w_nets[et]
        h = self.h_nets[et]
        we = w(edges[et].single_array)
        sender = self._mapping.sender_data_of(
            et, nodes,
        )
        hx = h(sender)
        if edges[et].single_array.size == 0:
            n_el = self.n_up + self.n_down
            return jnp.zeros(
                (n_el, self._tp_dim),
            )
        return (
            edges[et]
            .update_from_single_array(we)
            .convolve(hx, norm)
        )

    def __call__(self, nodes, edges):
        """Compute convolution-based update features.

        Args:
            nodes: Current :class:`~.graph.GraphNodes`.
            edges: Current edge dict.

        Returns:
            List of :class:`~.graph.GraphNodes`, one per
            *edge_type*, holding the convolved features.
        """
        updates = []
        for et in self.edge_types:
            if et == 'ee':
                ee = sum(
                    self._single(
                        nodes, edges, s, False,
                    )
                    for s in ('same', 'anti')
                )
                n = self.n_up + self.n_down
                fac = (
                    n if self.normalize else 1.0
                )
                updates.append(
                    GraphNodes(None, ee / fac),
                )
            else:
                updates.append(
                    GraphNodes(
                        None,
                        self._single(
                            nodes, edges, et,
                            self.normalize,
                        ),
                    ),
                )
        return updates


class NodeAttentionElectronUpdateFeature(
    nnx.Module,
):
    """Attention-based update feature (PsiFormer).

    Args:
        emb_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_factory: ``(in_d, out_d) -> MLP``
            for the post-attention MLP.
        attention_residual: Residual connection
            after attention, or ``None``.
        mlp_residual: Residual connection after
            MLP, or ``None``.
        rngs: Flax NNX RNG state.
    """

    def __init__(
        self, emb_dim, *, num_heads,
        mlp_factory, attention_residual,
        mlp_residual, rngs,
    ):
        self._emb_dim = emb_dim
        self.attention = _MultiHeadAttention(
            num_heads, emb_dim, rngs=rngs,
        )
        self.mlp = mlp_factory(emb_dim, emb_dim)
        self.attention_residual = attention_residual
        self.mlp_residual = mlp_residual

    @property
    def names(self):
        return ('attention',)

    @property
    def output_dim(self):
        return self._emb_dim

    def __call__(self, nodes, edges):
        """Compute attention-based update feature.

        Args:
            nodes: Current :class:`~.graph.GraphNodes`.
            edges: Current edge dict (unused).

        Returns:
            List containing one
            :class:`~.graph.GraphNodes` with the
            post-attention, post-MLP embedding.
        """
        h = nodes.electrons
        att = self.attention(h, h, h)
        if self.attention_residual:
            att = self.attention_residual(h, att)
        out = self.mlp(att)
        if self.mlp_residual:
            out = self.mlp_residual(att, out)
        return [GraphNodes(None, out)]
