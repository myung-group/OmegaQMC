"""Graph data structures and edge builders.

Rewritten from ``deepqmc/gnn/graph.py``. Replaces
``jax_dataclasses`` with ``register_pytree_node``.
"""

from collections import namedtuple

import jax.numpy as jnp

from ..compat import register_pytree

GraphNodes = namedtuple('GraphNodes', 'nuclei electrons')
Graph = namedtuple('Graph', 'nodes edges')


def _offdiag_sender_idx(n_node):
    """Sender indices for off-diagonal edges."""
    return (
        jnp.arange(n_node)[None, :]
        <= jnp.arange(n_node - 1)[:, None]
    ) + jnp.arange(n_node - 1)[:, None]


def _compute_edges(
    pos_sender, pos_receiver, filter_diagonal,
):
    """Raw edge computation (coordinate diffs)."""
    diffs = (
        pos_receiver[..., None, :, :]
        - pos_sender[..., None, :]
    )
    if filter_diagonal:
        assert (
            pos_sender.shape[-2]
            == pos_receiver.shape[-2]
        )
        n = pos_sender.shape[-2]
        recv_idx = jnp.broadcast_to(
            jnp.arange(n)[None], (n - 1, n),
        )
        send_idx = _offdiag_sender_idx(n)
        diffs = diffs[..., send_idx, recv_idx, :]
    return diffs


# ----------------------------------------------------------------
# Edge builder factories
# ----------------------------------------------------------------

def GraphEdgeBuilder(mask_self):
    """Factory returning an edge-builder function.

    Args:
        mask_self: If True, exclude self-loops.
    """
    def build(pos_sender, pos_receiver):
        assert pos_sender.shape[-1] == 3
        assert pos_receiver.shape[-1] == 3
        assert len(pos_sender.shape) == 2
        assert (
            not mask_self
            or pos_sender.shape[0]
            == pos_receiver.shape[0]
        )
        return _compute_edges(
            pos_sender, pos_receiver, mask_self,
        )
    return build


def MolecularGraphEdgeBuilder(
    n_nuc, n_up, n_down, edge_types,
    *, self_interaction,
):
    """Build many types of molecular graph edges.

    Args:
        n_nuc: Number of nuclei.
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        edge_types: Edge types to build (e.g.
            ``['same', 'anti', 'ne']``).
        self_interaction: Include self-loops.
    """
    _map = {
        'nn': ['nn'], 'ne': ['ne'], 'en': ['en'],
        'same': ['uu', 'dd'], 'anti': ['ud', 'du'],
        'up': ['up'], 'down': ['down'],
    }
    _mask_self = {
        'nn': not self_interaction,
        'ne': False, 'en': False,
        'uu': not self_interaction,
        'dd': not self_interaction,
        'ud': False, 'du': False,
        'up': False, 'down': False,
    }
    builders = {
        bt: GraphEdgeBuilder(mask_self=_mask_self[bt])
        for et in edge_types
        for bt in _map[et]
    }

    def _build_same(pc):
        return SameGraphEdges(
            builders['uu'](
                pc.r[:n_up], pc.r[:n_up],
            ),
            builders['dd'](
                pc.r[n_up:], pc.r[n_up:],
            ),
        )

    def _build_anti(pc):
        return AntiGraphEdges(
            builders['du'](
                pc.r[n_up:], pc.r[:n_up],
            ),
            builders['ud'](
                pc.r[:n_up], pc.r[n_up:],
            ),
        )

    _rules = {
        'nn': lambda pc: SimpleGraphEdges(
            builders['nn'](pc.R, pc.R),
        ),
        'ne': lambda pc: SimpleGraphEdges(
            builders['ne'](pc.R, pc.r),
        ),
        'en': lambda pc: SimpleGraphEdges(
            builders['en'](pc.r, pc.R),
        ),
        'same': _build_same,
        'anti': _build_anti,
        'up': lambda pc: UpGraphEdges(
            builders['up'](pc.r[:n_up], pc.r),
        ),
        'down': lambda pc: DownGraphEdges(
            builders['down'](pc.r[n_up:], pc.r),
        ),
    }

    def build(phys_conf):
        assert phys_conf.r.shape[0] == n_up + n_down
        return {
            et: _rules[et](phys_conf)
            for et in edge_types
        }
    return build


def GraphUpdate(
    aggregate_edges_for_nodes_fn,
    update_nodes_fn=None,
    update_edges_fn=None,
):
    """Graph update step for GNNs.

    Args:
        aggregate_edges_for_nodes_fn: Edge aggregation.
        update_nodes_fn: Optional node update function.
        update_edges_fn: Optional edge update function.
    """
    def update_graph(graph):
        nodes, edges = graph
        if update_nodes_fn:
            agg = aggregate_edges_for_nodes_fn(
                nodes, edges,
            )
            nodes = update_nodes_fn(nodes, agg)
        if update_edges_fn:
            edges = update_edges_fn(edges)
        return Graph(nodes, edges)
    return update_graph


# ----------------------------------------------------------------
# Graph edge pytree classes
# ----------------------------------------------------------------

class GraphEdges:
    """Abstract base for typed graph edges."""

    @property
    def single_array(self):
        raise NotImplementedError

    def update_from_single_array(self, array):
        raise NotImplementedError

    def sum_senders(self, normalize=False):
        raise NotImplementedError

    def convolve(self, nodes, normalize=False):
        raise NotImplementedError


@register_pytree
class SimpleGraphEdges(GraphEdges):
    """Edges stored as a single array."""

    def __init__(self, edges):
        self.edges = edges

    def tree_flatten(self):
        return (self.edges,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])

    @property
    def single_array(self):
        return self.edges

    def update_from_single_array(self, array):
        return self.__class__(array)

    def sum_senders(self, normalize=False):
        fn = jnp.mean if normalize else jnp.sum
        return fn(self.edges, axis=-3)

    def convolve(self, nodes, normalize=False):
        prod = self.edges * nodes[:, None]
        return self.__class__(prod).sum_senders(
            normalize,
        )


@register_pytree
class UpGraphEdges(SimpleGraphEdges):
    """Edges from spin-up to all electrons."""

    def convolve(self, nodes, normalize=False):
        n_up = self.edges.shape[-3]
        prod = self.edges * nodes[:n_up, None]
        return self.__class__(prod).sum_senders(
            normalize,
        )


@register_pytree
class DownGraphEdges(SimpleGraphEdges):
    """Edges from spin-down to all electrons."""

    def convolve(self, nodes, normalize=False):
        n_dn = self.edges.shape[-3]
        prod = self.edges * nodes[-n_dn:, None]
        return self.__class__(prod).sum_senders(
            normalize,
        )


@register_pytree
class SameGraphEdges(GraphEdges):
    """Edges between same-spin electrons (uu + dd)."""

    def __init__(self, uu, dd):
        self.uu = uu
        self.dd = dd

    def tree_flatten(self):
        return (self.uu, self.dd), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], children[1])

    @property
    def single_array(self):
        bd = self.uu.shape[:-3]
        return jnp.concatenate([
            self.uu.reshape(*bd, -1, self.uu.shape[-1]),
            self.dd.reshape(*bd, -1, self.dd.shape[-1]),
        ], axis=-2)

    def update_from_single_array(self, array):
        n_up = self.uu.shape[-2]
        n_sender_up = self.uu.shape[-3]
        n_sender_dn = self.dd.shape[-3]
        uu, dd = jnp.split(
            array, (n_up * n_sender_up,), axis=-2,
        )
        uu = uu.reshape(
            *uu.shape[:-2], n_sender_up, n_up,
            uu.shape[-1],
        )
        dd = dd.reshape(
            *dd.shape[:-2], n_sender_dn,
            self.dd.shape[-2], dd.shape[-1],
        )
        return self.__class__(uu, dd)

    def sum_senders(self, normalize=False):
        def _norm(x):
            return max(x.shape[-3], 1) if normalize else 1
        up = jnp.sum(self.uu, axis=-3) / _norm(self.uu)
        dn = jnp.sum(self.dd, axis=-3) / _norm(self.dd)
        return jnp.concatenate([up, dn], axis=-2)

    def convolve(self, nodes, normalize=False):
        si = (
            self.uu.shape[-3] == self.uu.shape[-2]
        )
        if si:
            up_idx = (slice(None, self.uu.shape[-2]),
                      None)
            dn_idx = (slice(self.uu.shape[-2], None),
                      None)
        else:
            up_idx = _offdiag_sender_idx(
                self.uu.shape[-2],
            )
            dn_idx = (
                self.uu.shape[-2]
                + _offdiag_sender_idx(
                    self.dd.shape[-2],
                )
            )
        uu = self.uu * nodes[up_idx]
        dd = self.dd * nodes[dn_idx]
        return self.__class__(uu, dd).sum_senders(
            normalize,
        )


@register_pytree
class AntiGraphEdges(GraphEdges):
    """Edges between opposite-spin electrons (du + ud)."""

    def __init__(self, du, ud):
        self.du = du
        self.ud = ud

    def tree_flatten(self):
        return (self.du, self.ud), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], children[1])

    @property
    def single_array(self):
        bd = self.du.shape[:-3]
        return jnp.concatenate([
            self.du.reshape(*bd, -1, self.du.shape[-1]),
            self.ud.reshape(*bd, -1, self.ud.shape[-1]),
        ], axis=-2)

    def update_from_single_array(self, array):
        n_up = self.du.shape[-2]
        n_dn = self.ud.shape[-2]
        du, ud = jnp.split(
            array, (n_up * n_dn,),
        )
        du = du.reshape(
            *du.shape[:-2], n_dn, n_up, du.shape[-1],
        )
        ud = ud.reshape(
            *ud.shape[:-2], n_up, n_dn, ud.shape[-1],
        )
        return self.__class__(du, ud)

    def sum_senders(self, normalize=False):
        def _norm(x):
            return max(x.shape[-3], 1) if normalize else 1
        up = jnp.sum(self.du, axis=-3) / _norm(self.du)
        dn = jnp.sum(self.ud, axis=-3) / _norm(self.ud)
        return jnp.concatenate([up, dn], axis=-2)

    def convolve(self, nodes, normalize=False):
        n_up = self.du.shape[-2]
        du = self.du * nodes[n_up:, None]
        ud = self.ud * nodes[:n_up, None]
        return self.__class__(du, ud).sum_senders(
            normalize,
        )
