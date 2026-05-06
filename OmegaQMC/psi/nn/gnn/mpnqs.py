"""Message-passing neural quantum state (MP-NQS) GNN.

Implements the dual-stream message-passing architecture from Pescia
et al. (PRB 110, 035108, 2024) as adapted by Smith et al. (PRL 133,
266504, 2024) for the 2D HEG.  Two persistent streams are maintained
across ``T`` layers:

    h_i^(t)  ∈ R^{d1}   one-body per-electron
    h_ij^(t) ∈ R^{d2}   two-body per-pair

Update equations (Smith supplement eqs. 8-17):

    g_i^(t)  = [v_i, h_i^(t-1)]
    g_ij^(t) = [v_ij, h_ij^(t-1)]
    m_ij^(t) = A_ij^(t) ⊙ F_m^(t)(g_ij^(t))
    h_i^(t)  = F1^(t)(Σ_{j≠i} m_ij^(t), g_i^(t)) + h_i^(t-1)     (residual)
    h_ij^(t) = F2^(t)(m_ij^(t), g_ij^(t))         + h_ij^(t-1)   (residual)

Attention (eq. 15):

    A_ij^(t) = Linear^(t) ∘ GELU( (1/√N) Σ_l q_il^(t) ⊙ k_lj^(t) )

with ``q_ij = W_q g_ij``, ``k_ij = W_k g_ij``.  The ``⊙`` is element-wise;
the sum over the middle index ``l`` couples paths through every "via"
electron, giving the architecture an explicit 4-body inductive bias
that vanilla per-particle attention (PsiFormer) lacks.

Visible features (Smith supplement eq. 13):

    v_i  = ∅
    v_ij = [cos(2π A^{-1} r_ij), sin(2π A^{-1} r_ij),
            ‖sin(π A^{-1} r_ij)‖, s_ij]

with ``s_ij = +1`` for parallel spins, ``-1`` antiparallel.

Returns the final one-body stream wrapped in a
:class:`~OmegaQMC.psi.nn.gnn.graph.GraphNodes` so this module is a
drop-in replacement for :class:`ElectronGNN` in
:func:`build_heg_psiformer_wf`.  The two-body stream is consumed
internally and discarded.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..periodic import PeriodicLattice
from .graph import GraphNodes


def _periodic_pair_features(
    r: jax.Array, A_inv: jax.Array,
) -> jax.Array:
    """Compute Smith eq. 13 periodic pair features.

    Returns a tensor of shape ``(n_elec, n_elec, 3·dim + 1)`` containing
    ``[cos(2π s), sin(2π s), ‖sin(π s)‖]`` where ``s = A_inv · r_ij`` is
    the fractional pair displacement.  The trailing ``+1`` slot is left
    for the spin feature, appended by the caller.

    Args:
        r: Electron positions, shape ``(n_elec, dim)``.
        A_inv: Inverse cell tensor, shape ``(dim, dim)``.

    Returns:
        Pair-feature tensor (without spin), shape
        ``(n_elec, n_elec, 3·dim)``.
    """
    n_elec, dim = r.shape
    r_diff = r[None, :, :] - r[:, None, :]                    # (n, n, dim)
    s = jnp.einsum('ab,ijb->ija', A_inv, r_diff)              # (n, n, dim)
    cos_2pi = jnp.cos(2.0 * jnp.pi * s)                       # (n, n, dim)
    sin_2pi = jnp.sin(2.0 * jnp.pi * s)                       # (n, n, dim)
    abs_sin_pi = jnp.abs(jnp.sin(jnp.pi * s))                 # (n, n, dim)
    return jnp.concatenate([cos_2pi, sin_2pi, abs_sin_pi],
                           axis=-1)                           # (n, n, 3·dim)


def _spin_feature(n_up: int, n_down: int) -> jax.Array:
    """Pairwise spin compatibility ``s_ij = 2·δ_{σi,σj} - 1``.

    Returns:
        ``(n_elec, n_elec, 1)`` with ``+1`` on parallel-spin pairs and
        ``-1`` on antiparallel pairs.
    """
    n_elec = n_up + n_down
    spins = jnp.concatenate(
        [jnp.ones(n_up), -jnp.ones(n_down)],
    )                                                          # (n,)
    s_ij = (spins[:, None] * spins[None, :])                   # (n, n)
    return s_ij[:, :, None]                                    # (n, n, 1)


class _GeluMLP(nnx.Module):
    """``Linear → GELU → Linear`` (Smith eq. 18).

    Single hidden layer with GELU activation — matches the F1, F2, F_m
    blocks in Smith's MP-NQS layer.  Keep flexible: the user can compose
    these into deeper stacks externally if needed (e.g. for the 4-layer
    deep Jastrow MLP, eq. 21).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.lin1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.lin2 = nnx.Linear(hidden, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.lin2(nnx.gelu(self.lin1(x)))


class MPNqsLayer(nnx.Module):
    """One layer of dual-stream message-passing.

    Args:
        n_elec: Number of electrons (used for 1/√N normalisation).
        d1: Dim of the one-body stream ``h_i``.
        d2: Dim of the two-body stream ``h_ij``.
        v_i_dim: Dim of one-body visible feature ``v_i`` (0 in Smith).
        v_ij_dim: Dim of two-body visible feature ``v_ij``.
        hidden: Hidden width of the F1/F2/F_m MLPs.
        use_layer_norm: If True, apply post-norm LayerNorm to both
            streams after the residual sum.  Smith 2024 supplement
            page 7 ("To summarize ... we apply skip connection and
            layer normalization") notes this stabilises training; in
            our setup it bounds h_i^(T) magnitude, which directly
            bounds the coord-backflow displacement W_bf · h_i^(T)
            and prevents pathological coord-collapse.
    """

    def __init__(
        self,
        *,
        n_elec: int,
        d1: int,
        d2: int,
        v_i_dim: int,
        v_ij_dim: int,
        hidden: int,
        use_layer_norm: bool = False,
        layer_norm_mode: str = 'post_each',
        rngs: nnx.Rngs,
    ):
        self.n_elec = int(n_elec)
        self.d1 = int(d1)
        self.d2 = int(d2)
        self.v_i_dim = int(v_i_dim)
        self.v_ij_dim = int(v_ij_dim)
        self.use_layer_norm = bool(use_layer_norm)
        # LN placement modes:
        #   'post_each': LN after residual sum, every layer (most restrictive)
        #   'pre_each':  LN inputs before f1/f2, residual on un-LN'd inputs
        #   'none':      MPNqsLayer applies no LN (caller may apply
        #                'post_final_only' externally)
        if layer_norm_mode not in (
            'post_each', 'pre_each', 'none',
        ):
            raise ValueError(
                f"layer_norm_mode must be 'post_each'/'pre_each'/'none', "
                f"got {layer_norm_mode!r}"
            )
        self.layer_norm_mode = str(layer_norm_mode)

        # F_m: g_ij → message in R^{d2} (eq. 14, second factor).
        self.f_m = _GeluMLP(
            in_dim=v_ij_dim + d2, out_dim=d2,
            hidden=hidden, rngs=rngs,
        )
        # F1: [aggregated_msg, g_i] → R^{d1}.
        self.f1 = _GeluMLP(
            in_dim=d2 + v_i_dim + d1, out_dim=d1,
            hidden=hidden, rngs=rngs,
        )
        # F2: [m_ij, g_ij] → R^{d2}.
        self.f2 = _GeluMLP(
            in_dim=d2 + v_ij_dim + d2, out_dim=d2,
            hidden=hidden, rngs=rngs,
        )

        # Attention projections (eqs. 16-17).
        self.w_q = nnx.Linear(
            v_ij_dim + d2, d2, use_bias=False, rngs=rngs,
        )
        self.w_k = nnx.Linear(
            v_ij_dim + d2, d2, use_bias=False, rngs=rngs,
        )
        # Linear after GELU in eq. 15: maps the d2-dim attention vector
        # back to a d2-dim per-pair gate.
        self.attn_out = nnx.Linear(d2, d2, rngs=rngs)

        # LayerNorm on both streams (Smith stabiliser).  Created when
        # use_layer_norm=true AND layer_norm_mode != 'none'.  The
        # 'none' mode reserves LN for the GNN-level final norm only.
        if self.use_layer_norm and self.layer_norm_mode != 'none':
            self.norm_h_i = nnx.LayerNorm(d1, rngs=rngs)
            self.norm_h_ij = nnx.LayerNorm(d2, rngs=rngs)

    def __call__(
        self,
        h_i: jax.Array,           # (n, d1)
        h_ij: jax.Array,          # (n, n, d2)
        v_i: Optional[jax.Array],  # (n, v_i_dim) or None
        v_ij: jax.Array,          # (n, n, v_ij_dim)
    ) -> tuple[jax.Array, jax.Array]:
        n = self.n_elec

        # Pre-norm: LN inputs to f1/f2 BEFORE concatenation, but
        # residual sum uses the un-LN'd originals.  Pattern from
        # modern transformers (Wang+2019, Xiong+2020) — keeps the
        # residual path "clean" so feature magnitudes can grow
        # across layers, only the MLP inputs are bounded.
        if self.use_layer_norm and self.layer_norm_mode == 'pre_each':
            h_i_in = self.norm_h_i(h_i)
            h_ij_in = self.norm_h_ij(h_ij)
        else:
            h_i_in = h_i
            h_ij_in = h_ij

        # g_i^(t) = [v_i, h_i^(t-1)]
        if v_i is not None and self.v_i_dim > 0:
            g_i = jnp.concatenate([v_i, h_i_in], axis=-1)     # (n, v_i+d1)
        else:
            g_i = h_i_in                                       # (n, d1)

        # g_ij^(t) = [v_ij, h_ij^(t-1)]
        g_ij = jnp.concatenate([v_ij, h_ij_in], axis=-1)      # (n, n, v_ij+d2)

        # Attention: q, k from g_ij; sum element-wise over middle index l.
        q = self.w_q(g_ij)                                     # (n, n, d2)
        k = self.w_k(g_ij)                                     # (n, n, d2)
        # Σ_l q_il ⊙ k_lj  → (n, n, d2)
        # Implemented as a batched matmul over the d2 dim (treating d2
        # as a leading batch axis) so the contraction over l never
        # materialises an O(N^3) intermediate — peak memory is
        # O(N^2 * d2).  The naive ``einsum('ild,ljd->ijd')`` creates a
        # (N, N, N, d2) intermediate that OOMs on N=50 + 1024 walkers
        # even on an A100 80 GB.
        q_t = jnp.transpose(q, (2, 0, 1))                      # (d2, n, n)
        k_t = jnp.transpose(k, (2, 0, 1))                      # (d2, n, n)
        attn_t = jnp.matmul(q_t, k_t) / jnp.sqrt(n)            # (d2, n, n)
        attn = jnp.transpose(attn_t, (1, 2, 0))                # (n, n, d2)
        A_ij = self.attn_out(nnx.gelu(attn))                  # (n, n, d2)

        # Message m_ij = A_ij ⊙ F_m(g_ij).
        m_ij = A_ij * self.f_m(g_ij)                           # (n, n, d2)

        # h_i^(t) update: aggregate messages over j ≠ i, concat with g_i.
        # Mask diagonal so a node doesn't message itself.
        eye = jnp.eye(n, dtype=m_ij.dtype)[..., None]          # (n, n, 1)
        m_ij_off = m_ij * (1.0 - eye)                          # zero diag
        msg_agg = jnp.sum(m_ij_off, axis=1)                    # (n, d2)
        h_i_new = self.f1(jnp.concatenate([msg_agg, g_i], axis=-1))
        h_i_new = h_i_new + h_i                                # residual

        # h_ij^(t) update: F2([m_ij, g_ij]) + residual.
        h_ij_new = self.f2(
            jnp.concatenate([m_ij, g_ij], axis=-1),
        )
        h_ij_new = h_ij_new + h_ij                             # residual

        # Post-norm LayerNorm: bounds activation magnitude after the
        # residual sum so it can't grow unboundedly across layers.
        # In our setup this is the safety belt that prevents the
        # coord-backflow displacement W_bf · h_i^(T) from collapsing
        # electron coordinates and producing a near-singular Slater
        # determinant.  Most restrictive — applied at every layer.
        if self.use_layer_norm and self.layer_norm_mode == 'post_each':
            h_i_new = self.norm_h_i(h_i_new)
            h_ij_new = self.norm_h_ij(h_ij_new)

        return h_i_new, h_ij_new


class MPNqsGnn(nnx.Module):
    """MP-NQS GNN — drop-in replacement for ``ElectronGNN``.

    Args:
        n_up, n_down: Per-spin electron counts.
        lattice: :class:`PeriodicLattice` (for ``A_inv``).
        n_layers: Number of MP-NQS layers ``T``.
        d1, d2: One-body and two-body stream dims.
        hidden: Hidden width of internal MLPs.
        include_spin_in_v_ij: If True, append spin feature ``s_ij`` to
            ``v_ij`` (Smith default).
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        *,
        n_up: int,
        n_down: int,
        lattice: PeriodicLattice,
        n_layers: int = 4,
        d1: int = 32,
        d2: int = 26,
        hidden: int = 32,
        include_spin_in_v_ij: bool = True,
        use_layer_norm: bool = False,
        layer_norm_mode: str = 'post_each',
        rngs: nnx.Rngs,
    ):
        self.n_up = int(n_up)
        self.n_down = int(n_down)
        self.n_elec = self.n_up + self.n_down
        self.A_inv = nnx.data(jnp.asarray(lattice.A_inv))
        self.dim = int(lattice.A_inv.shape[-1])
        self.d1 = int(d1)
        self.d2 = int(d2)
        self.include_spin_in_v_ij = bool(include_spin_in_v_ij)
        self.use_layer_norm = bool(use_layer_norm)
        # LN placement mode (only used when use_layer_norm=True):
        #   'post_each':       LN after residual sum, every layer
        #   'pre_each':        LN inputs to f1/f2 every layer
        #   'post_final_only': LN only after the final layer's outputs
        if layer_norm_mode not in (
            'post_each', 'pre_each', 'post_final_only',
        ):
            raise ValueError(
                f"layer_norm_mode must be 'post_each'/'pre_each'/"
                f"'post_final_only', got {layer_norm_mode!r}"
            )
        self.layer_norm_mode = str(layer_norm_mode)

        # v_i is empty in Smith.
        self.v_i_dim = 0
        # v_ij: 3·dim trig features [+ 1 spin] → 7 for 2D, 10 for 3D.
        self.v_ij_dim = 3 * self.dim + (1 if include_spin_in_v_ij else 0)

        # Learnable seed vectors for h_i^(0) and h_ij^(0).  Following
        # Smith we initialise to zero so the iter-0 wavefunction is
        # determined entirely by the envelope.
        self.h0_i = nnx.Param(jnp.zeros((d1,), dtype=jnp.float64))
        self.h0_ij = nnx.Param(jnp.zeros((d2,), dtype=jnp.float64))

        # Pre-compute spin feature (constant given n_up, n_down).
        if include_spin_in_v_ij:
            self.s_ij = nnx.data(_spin_feature(self.n_up, self.n_down))
        else:
            self.s_ij = None

        # Stack of T layers.  When mode is 'post_final_only', the
        # per-layer LN is disabled and we apply a single LN to the
        # final stream output below.
        per_layer_mode = (
            self.layer_norm_mode
            if self.layer_norm_mode in ('post_each', 'pre_each')
            else 'none'
        )
        self.layers = nnx.List([
            MPNqsLayer(
                n_elec=self.n_elec,
                d1=d1, d2=d2,
                v_i_dim=self.v_i_dim,
                v_ij_dim=self.v_ij_dim,
                hidden=hidden,
                use_layer_norm=self.use_layer_norm,
                layer_norm_mode=per_layer_mode,
                rngs=rngs,
            )
            for _ in range(int(n_layers))
        ])

        # Final-only LN: a single LayerNorm pair applied after the
        # T-layer stack (most surgical placement — bounds h_i^(T) for
        # BF without restricting any intermediate dynamics).
        if self.use_layer_norm and self.layer_norm_mode == 'post_final_only':
            self.final_norm_h_i = nnx.LayerNorm(d1, rngs=rngs)
            self.final_norm_h_ij = nnx.LayerNorm(d2, rngs=rngs)

    def __call__(self, phys_conf):
        r = phys_conf.r                                        # (n, dim)
        n = self.n_elec

        # Pair visible features (periodic + optional spin).
        v_ij_periodic = _periodic_pair_features(r, self.A_inv)  # (n, n, 3·dim)
        if self.include_spin_in_v_ij:
            v_ij = jnp.concatenate([v_ij_periodic, self.s_ij], axis=-1)
        else:
            v_ij = v_ij_periodic

        # Initial streams (broadcast learnable seed across particles/pairs).
        h_i = jnp.broadcast_to(self.h0_i, (n, self.d1))         # (n, d1)
        h_ij = jnp.broadcast_to(self.h0_ij, (n, n, self.d2))    # (n, n, d2)

        v_i = None  # empty per Smith.

        for layer in self.layers:
            h_i, h_ij = layer(h_i, h_ij, v_i, v_ij)

        # Apply final-only LN if requested (bounds the h_i^(T) that
        # feeds BF readout without restricting per-layer dynamics).
        if self.use_layer_norm and self.layer_norm_mode == 'post_final_only':
            h_i = self.final_norm_h_i(h_i)
            h_ij = self.final_norm_h_ij(h_ij)

        # Match ElectronGNN's return: GraphNodes(nuclei, electrons).
        return GraphNodes(nuclei=None, electrons=h_i)
