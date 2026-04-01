"""GNN utility classes for node/edge type mapping.

Rewritten from ``deepqmc/gnn/utils.py``.
"""


def _dict_or_nt_get(data, key):
    """Get item from dict or namedtuple."""
    try:
        return getattr(data, key)
    except AttributeError:
        return data[key]


def _dict_or_nt_keys(container):
    """Get keys/fields of a dict or namedtuple."""
    try:
        return list(container._fields)
    except AttributeError:
        return list(container.keys())


def is_node(label):
    """True if *label* is a node type."""
    return label in {'nuclei', 'electrons'}


def is_edge(label):
    """True if *label* is an edge type."""
    return label in {
        'nn', 'ne', 'en', 'same', 'anti', 'up', 'down',
    }


_RECEIVER = {
    'same': 'electrons',
    'anti': 'electrons',
    'ne': 'electrons',
    'en': 'nuclei',
    'nn': 'nuclei',
    'up': 'electrons',
    'down': 'electrons',
}
_SENDER = {
    'same': 'electrons',
    'anti': 'electrons',
    'ne': 'nuclei',
    'en': 'electrons',
    'nn': 'nuclei',
    'up': 'electrons',
    'down': 'electrons',
}


class NodeEdgeMapping:
    """Mapping between node and edge types in the GNN.

    Provides lookups such as *receiver/sender of an edge*,
    *edges with a given receiver node*, etc.

    Args:
        edges: Sequence of edge-type labels present in
            the graph (e.g. ``['same', 'anti', 'ne']``).
        node_data: Optional dict mapping node names to
            data containers.
    """

    def __init__(self, edges, node_data=None):
        self.edges = list(edges)
        self.nodes = {
            self.receiver_of(e) for e in edges
        }
        self.node_data = node_data

    # ---- receivers / senders ----

    @staticmethod
    def receiver_of(edge):
        return _RECEIVER[edge]

    @staticmethod
    def sender_of(edge):
        return _SENDER[edge]

    # ---- edge queries ----

    def with_receiver(self, node_or_edge):
        if is_edge(node_or_edge):
            return [node_or_edge]
        return [
            e for e in self.edges
            if self.receiver_of(e) == node_or_edge
        ]

    def with_sender(self, node_or_edge):
        if is_edge(node_or_edge):
            return [node_or_edge]
        return [
            e for e in self.edges
            if self.sender_of(e) == node_or_edge
        ]

    # ---- data access ----

    def _get_container(self, data):
        if isinstance(data, str):
            assert self.node_data is not None
            return self.node_data[data]
        return data

    def node_data_of(self, node, data):
        return _dict_or_nt_get(
            self._get_container(data), node,
        )

    def edge_data_of(self, edge, data):
        return _dict_or_nt_get(data, edge)

    def receiver_data_of(self, edge, data):
        return self.node_data_of(
            self.receiver_of(edge), data,
        )

    def sender_data_of(self, edge, data):
        return self.node_data_of(
            self.sender_of(edge), data,
        )

    def data_with_receiver(self, node_or_edge, data):
        return [
            self.edge_data_of(e, data)
            for e in self.with_receiver(node_or_edge)
        ]

    def data_with_sender(self, node_or_edge, data):
        return [
            self.edge_data_of(e, data)
            for e in self.with_sender(node_or_edge)
        ]

    def node_or_receiver_data_of(
        self, node_or_edge, data,
    ):
        fn = (
            self.node_data_of
            if is_node(node_or_edge)
            else self.receiver_data_of
        )
        return fn(node_or_edge, data)

    def node_or_sender_data_of(
        self, node_or_edge, data,
    ):
        fn = (
            self.node_data_of
            if is_node(node_or_edge)
            else self.sender_data_of
        )
        return fn(node_or_edge, data)

    def reduce_to_receiver(
        self, node, data, reduce_fn,
    ):
        container = self._get_container(data)
        keys = _dict_or_nt_keys(container)
        if node in keys:
            return _dict_or_nt_get(container, node)
        return reduce_fn(
            self.data_with_receiver(node, container),
        )
