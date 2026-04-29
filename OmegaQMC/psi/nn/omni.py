"""OmniNet: GNN + Jastrow + Backflow (Flax NNX).

Rewritten from ``deepqmc/wf/omni.py``.
"""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from .compat import param_value
from .layers import GLU
from .utils import unflatten


class Jastrow(nnx.Module):
    """Deep Jastrow factor.

    Args:
        net: MLP mapping embeddings to a scalar.
        sum_first: If True, sum embeddings before the
            MLP; otherwise apply MLP per electron then
            sum outputs.
    """

    def __init__(self, *, net, sum_first):
        self.net = net
        self.sum_first = sum_first

    def __call__(self, xs):
        if self.sum_first:
            xs = self.net(xs.sum(axis=-2))
        else:
            xs = self.net(xs).sum(axis=-2)
        return xs.squeeze(axis=-1)


class Backflow(nnx.Module):
    """Deep backflow factor.

    Args:
        n_orbitals: Orbitals per determinant.
        n_determinants: Number of determinants.
        n_backflows: Independent backflow channels.
        spin: ``'up'`` or ``'down'``.
        multi_head: Separate MLP per backflow channel.
        nets: List of MLP sub-modules (if multi_head)
            or a single MLP (if not).
    """

    def __init__(
        self,
        n_orbitals,
        n_determinants,
        n_backflows,
        spin,
        *,
        nets=None,
        net=None,
        multi_head=True,
    ):
        self.multi_head = multi_head
        self.n_orbitals = n_orbitals
        self.n_determinants = n_determinants
        self.spin = spin
        if multi_head:
            assert nets is not None
            self.nets = nnx.List(nets)
        else:
            assert net is not None
            self.net = net

    def __call__(self, xs):
        if self.multi_head:
            xs = jnp.stack(
                [n(xs) for n in self.nets], axis=-3,
            )
        else:
            xs = self.net(xs)
            xs = unflatten(
                xs, -1,
                (-1, self.n_orbitals
                 * self.n_determinants),
            )
            xs = xs.swapaxes(-2, -3)
        xs = unflatten(
            xs, -1, (-1, self.n_orbitals),
        )
        xs = xs.swapaxes(-2, -3)
        return xs


class OmniNet(nnx.Module):
    """Combine the GNN, Jastrow, and backflow MLPs.

    Args:
        n_up: Number of spin-up electrons.
        gnn: ElectronGNN instance (or None).
        jastrow: Jastrow instance (or None).
        backflow: Dict ``{'up': Backflow, 'down': ...}``
            (or None).
        nuclear_gnn_head: Optional NuclearGNNHead.
    """

    def __init__(
        self, *,
        n_up,
        gnn=None,
        jastrow=None,
        backflow=None,
        nuclear_gnn_head=None,
    ):
        self.n_up = n_up
        self.gnn = gnn
        self.jastrow = jastrow
        self.backflow = (
            nnx.Dict(backflow)
            if backflow is not None
            else None
        )
        self.nuclear_gnn_head = nuclear_gnn_head

    def __call__(self, phys_conf):
        # Returns (jastrow, backflow, nuc_params, embeddings).
        # ``embeddings`` is the per-electron post-GNN feature stream
        # (n_elec, d1) — exposed so downstream wavefunctions can
        # add their own readout heads (e.g. coord-transform
        # backflow x_i = r_i + W·h_i^(T) à la Smith 2024).
        # Pre-existing callers that unpacked 3 values must be
        # updated.
        if self.gnn is None:
            return None, None, None, None
        graph_nodes = self.gnn(phys_conf)
        embeddings = graph_nodes.electrons
        nuc_emb = graph_nodes.nuclei
        nuc_params = (
            self.nuclear_gnn_head(nuc_emb)
            if self.nuclear_gnn_head is not None
            else None
        )
        jastrow = (
            self.jastrow(embeddings)
            if self.jastrow is not None
            else None
        )
        backflow = None
        if self.backflow is not None:
            backflow = (
                self.backflow['up'](
                    embeddings[:self.n_up],
                ),
                self.backflow['down'](
                    embeddings[self.n_up:],
                ),
            )
        return jastrow, backflow, nuc_params, embeddings


class NuclearGNNHead(nnx.Module):
    """Predict parameters from nucleus embeddings.

    Args:
        one_particle_parameters: Dict mapping param
            names to per-nucleus shapes.
        embedding_dim: Nucleus embedding dimension.
        rngs: Flax NNX RNG state.
    """

    def __init__(
        self, *,
        one_particle_parameters,
        embedding_dim,
        rngs,
    ):
        readouts = {}
        biases = {}
        for k, per_nuc_shape in (
            one_particle_parameters.items()
        ):
            out_dim = math.prod(per_nuc_shape)
            for spin in ['up', 'down']:
                label = f'{k}_{spin}'
                readouts[label] = GLU(
                    embedding_dim, out_dim,
                    rngs=rngs,
                )
                biases[label] = nnx.Param(
                    2.0 * jnp.ones(per_nuc_shape),
                )
        self.readouts = nnx.Dict(readouts)
        self.biases = nnx.Dict(biases)
        self._params_spec = one_particle_parameters

    def __call__(self, nucleus_embeddings):
        out = {}
        for k, per_nuc_shape in (
            self._params_spec.items()
        ):
            for spin in ['up', 'down']:
                label = f'{k}_{spin}'
                glu_out = self.readouts[label](
                    nucleus_embeddings,
                    nucleus_embeddings,
                ).reshape(
                    -1, *per_nuc_shape,
                )
                out[label] = (
                    glu_out
                    + param_value(self.biases[label])
                )
        return out
