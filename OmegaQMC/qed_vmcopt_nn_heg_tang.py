"""Level 8 V2 cavity-QED HEG — full Tang-architecture (arxiv 2503.15644v1).

Pure Tang: one-hot(n) is injected into the *electron embedding layer*
of the V11 PsiFormer/FermiNet trunk, so every downstream layer
(GNN attention, backflow, Jastrow head, Slater envelope) sees an
n-conditioned matter representation.  The trunk is run once per
Fock sector n ∈ {0..N_max} — N_max+1 forward passes per ψ_vec.

This module is fully isolated from
``OmegaQMC.qed_vmcopt_nn_heg_l5`` (L7) and
``OmegaQMC.qed_vmcopt_nn_heg_fock`` (L8 V1).  It borrows existing
matter-network primitives from ``OmegaQMC.psi.nn.heg_wf_module`` and
the GNN/embedding modules by **subclassing** them — the parent
classes are not modified.

Subclass hierarchy (each overrides ``__call__`` to thread ``one_hot_n``):

    TangElectronEmbedding(HEGElectronEmbedding)
        + n_cond_proj : Linear (N_max+1 → embedding_dim, zero-init)
        Adds  delta = n_cond_proj(one_hot_n)  to the per-electron
        embedding at the input.  Vacuum init preserved (n_cond_proj
        zero-init → all sectors give matter HF at iter 0).

    TangElectronGNN(ElectronGNN)
        Passes one_hot_n to self.electron_embedding.

    TangOmniNet(OmniNet)
        Passes one_hot_n to self.gnn.

    TangPsiFormerWaveFunction(HEGPsiFormerWaveFunction)
        Passes one_hot_n to self.omni and adds a phase readout MLP
        on the post-GNN per-electron embeddings.  Returns
        (sign_det, log_mag, phase) — composed into complex ψ_n by
        the caller.

Composition (matches V11 stack exactly modulo Tang's n-conditioning):
    plane-wave Slater envelope    (n-blind)
    PsiFormer attention GNN       (n-aware via initial embeddings)
    multiplicative backflow head  (n-aware via post-GNN h_i)
    Smith deep Jastrow            (n-aware via post-GNN h_i + x_i)
    periodic electronic cusp      (n-blind — Kato-asymptotic)
    new: phase readout MLP        (n-aware via post-GNN h_i, summed)

See design/L8_fock_spec.md and inline comments for derivations.
"""
from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.flatten_util import ravel_pytree

# ---- Borrow from existing matter-network code (no modification) ----
from .psi.nn.heg_wf_module import (
    HEGElectronEmbedding,
    HEGPsiFormerWaveFunction,
    SmithDeepJastrow,
    PeriodicElectronicCusp,
    apply_heg_backflow_mult,
    _periodic_pair_distances_full,
    _HEGMolInfoShim,
)
from .psi.nn.heg_wf import (
    HEGPsiFormerConfig,
    PeriodicPairJastrow,
    _make_heg_phys_conf,
)
from .psi.nn.gnn.electron_gnn import ElectronGNN
from .psi.nn.omni import OmniNet, Backflow
from .psi.nn.gnn.edge_features_periodic import (
    PeriodicCombinedEdgeFeature,
    PeriodicSinCosFeature,
    PeriodicDistancePowerEdgeFeature,
)
from .psi.nn.periodic import (
    make_cubic_lattice, make_square_lattice,
    PeriodicLattice, wrap_to_cell,
)
# Numerical-primitive imports (not architecture-specific) — same as L8 V1
# uses these.  Treating them as a shared utility library.
from .observables.ewald_dispatch import build_ewald_tables_dim
from .observables.ewald_2d import ewald_2d_pair_energy
from .qed_vmcopt_nn_heg_l5 import make_l5_sr_smw_solver
from .qed_vmcopt_nn_heg_sr import _adapt_step_size
from .psi.nn.compat import param_value
from .psi.nn.layers import MLP
from .psi.nn.types import Psi
from .psi.nn.gnn.graph import Graph, GraphNodes
from .psi.nn.env_periodic import PlaneWaveEnvelope
from .psi.nn.utils import flatten, triu_flat
from .psi.nn.wf import eval_log_slater
from .psi.nn.heg_layer_ferminet import _build_heg_ferminet_layer
from .psi.nn.heg_layer_psiformer import _build_heg_gnn_layer


# =====================================================================
# 1. Tang-conditioned electron embedding
# =====================================================================

class TangElectronEmbedding(HEGElectronEmbedding):
    """HEGElectronEmbedding + n-conditioning at the embedding layer.

    delta = n_cond_proj(one_hot_n) ∈ ℝ^embedding_dim
    h_i = parent(phys_conf) + delta  (broadcast over electrons)

    Zero-init of n_cond_proj → delta = 0 at iter 0 → identical to
    matter HF for every n.  Vacuum init preserved by the offset_floor
    mechanism applied in the trial-wavefunction builder.
    """

    def __init__(
        self,
        *,
        n_up: int,
        n_down: int,
        embedding_dim: int,
        lattice: Optional[PeriodicLattice] = None,
        use_ghost_atom: bool = False,
        n_max_for_tang: int,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            n_up=n_up, n_down=n_down,
            embedding_dim=embedding_dim,
            lattice=lattice,
            use_ghost_atom=use_ghost_atom,
            rngs=rngs,
        )
        self.n_max_for_tang = int(n_max_for_tang)
        # Zero-init kernel → n-conditioning starts at 0.
        self.n_cond_proj = nnx.Linear(
            in_features=n_max_for_tang + 1,
            out_features=embedding_dim,
            use_bias=False,
            rngs=rngs,
        )
        # Force zero init so all sectors start at matter HF.
        self.n_cond_proj.kernel = nnx.Param(
            jnp.zeros_like(param_value(self.n_cond_proj.kernel)),
        )

    def __call__(self, phys_conf, nucleus_embedding=None, one_hot_n=None):
        h = super().__call__(phys_conf, nucleus_embedding)  # (n_elec, emb_dim)
        if one_hot_n is None:
            return h
        delta = self.n_cond_proj(
            jnp.asarray(one_hot_n, dtype=h.dtype)
        )  # (emb_dim,)
        return h + delta[None, :]                            # broadcast over electrons


# =====================================================================
# 2. Tang-aware GNN (one_hot_n threaded to the embedding step)
# =====================================================================

class TangElectronGNN(ElectronGNN):
    """ElectronGNN subclass that passes one_hot_n to the embedding.

    Mirrors ElectronGNN.__call__ verbatim except the embedding call
    site, which now accepts the Tang conditioning.  All GNN layers
    are reused unchanged — they operate on the n-aware embeddings.
    """

    def __call__(self, phys_conf, one_hot_n=None):
        edges = self._build_edges(phys_conf)
        elec_emb = self.electron_embedding(
            phys_conf, None, one_hot_n=one_hot_n,
        )
        nodes = GraphNodes(None, elec_emb)
        graph = Graph(nodes, edges)
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            last = (i == n - 1)
            graph = layer(graph, last_layer=last)
        return graph.nodes


# =====================================================================
# 3. Tang-aware OmniNet (one_hot_n threaded to the GNN)
# =====================================================================

class TangOmniNet(OmniNet):
    """OmniNet subclass that threads one_hot_n through to the GNN.

    Backflow / Jastrow / nuclear heads are unchanged — they operate
    on the n-aware embeddings the GNN produces.
    """

    def __call__(self, phys_conf, one_hot_n=None):
        if self.gnn is None:
            return None, None, None, None
        graph_nodes = self.gnn(phys_conf, one_hot_n=one_hot_n)
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
            emb_up = embeddings[:self.n_up]
            emb_down = embeddings[self.n_up:]
            bf_up = (
                self.backflow['up'](emb_up)
                if emb_up.shape[0] > 0 else None
            )
            bf_down = (
                self.backflow['down'](emb_down)
                if emb_down.shape[0] > 0 else None
            )
            backflow = (bf_up, bf_down)
        return jastrow, backflow, nuc_params, embeddings


# =====================================================================
# 4. Tang PsiFormer wavefunction with dual readout (log_mag, phase)
# =====================================================================

class TangPsiFormerWaveFunction(HEGPsiFormerWaveFunction):
    """HEGPsiFormerWaveFunction + Tang n-conditioning + phase readout.

    Output: returns (sign_det, log_mag, phase) where the complex
    wavefunction is

        ψ_n(R) = sign_det · exp(log_mag) · exp(i · phase)

    log_mag is the V11 PsiFormer's existing log|ψ| (now n-conditioned
    via the embedding layer).  phase is computed by a small MLP head
    on the post-GNN per-electron embeddings, summed over electrons:

        phase(R, n) = Σ_i  phase_mlp(h_i^(T)(R, n))

    where h_i^(T) is the post-GNN one-body stream.  Same n-conditioning
    pathway as log_mag — the phase head is just a different readout on
    the same n-aware embeddings.
    """

    def __init__(
        self,
        *,
        n_up: int,
        n_down: int,
        n_det: int,
        L: float,
        omni,
        envelope,
        lattice,
        cusp_electrons=None,
        pair_jastrow=None,
        coord_backflow=None,
        smith_deep_jastrow=None,
        full_determinant: bool = False,
        phase_mlp,
    ):
        super().__init__(
            n_up=n_up, n_down=n_down, n_det=n_det, L=L,
            omni=omni, envelope=envelope, lattice=lattice,
            cusp_electrons=cusp_electrons,
            pair_jastrow=pair_jastrow,
            coord_backflow=coord_backflow,
            smith_deep_jastrow=smith_deep_jastrow,
            full_determinant=full_determinant,
        )
        # Per-electron phase MLP, zero-init last layer → phase=0 at init.
        self.phase_mlp = phase_mlp

    def __call__(self, r: jax.Array, one_hot_n=None):
        """Returns (sign, log_mag, phase).  All three are real scalars.

        Mirrors HEGPsiFormerWaveFunction.__call__ but threads
        one_hot_n through the OmniNet and adds a phase readout.
        """
        pc = _make_heg_phys_conf(r)
        jastrow, fs, _nuc_params, emb = self.omni(pc, one_hot_n=one_hot_n)

        if (getattr(self, 'coord_backflow', None) is not None
                and emb is not None):
            r_bf = r + self.coord_backflow(emb)
            pc_bf = _make_heg_phys_conf(r_bf)
        else:
            r_bf = r
            pc_bf = pc

        orb = self.envelope(pc_bf)
        if self.full_determinant:
            orb_up = orb[:, :self.n_up]
            orb_down = orb[:, self.n_up:]
        else:
            orb_up, orb_down = jnp.split(orb, [self.n_up], axis=-1)
            orb_up = orb_up[:, :self.n_up]
            orb_down = orb_down[:, self.n_up:]

        if fs is not None:
            orb_up = apply_heg_backflow_mult(orb_up, fs[0])
            if orb_down.shape[-1] > 0:
                orb_down = apply_heg_backflow_mult(orb_down, fs[1])

        if self.full_determinant:
            orb_full = jnp.concatenate([orb_up, orb_down], axis=-2)
            sign, xs = eval_log_slater(orb_full)
        else:
            sign_u, log_u = eval_log_slater(orb_up)
            sign_d, log_d = eval_log_slater(orb_down)
            sign = sign_u * sign_d
            xs = log_u + log_d

        xs_shift = jnp.max(xs)
        xs_shift = jnp.where(
            jnp.isfinite(xs_shift), xs_shift, jnp.zeros_like(xs_shift),
        )
        xs_exp = sign * jnp.exp(xs - xs_shift)
        psi_sum = jnp.sum(xs_exp)

        log_mag = jnp.log(jnp.abs(psi_sum)) + xs_shift
        sign_psi = jax.lax.stop_gradient(jnp.sign(psi_sum))

        if self.cusp_electrons is not None:
            dists_elec = _periodic_pair_distances_full(r, self.lattice)
            same_dists = jnp.concatenate([
                triu_flat(dists_elec[idx, idx])
                for idx in self._spin_slices
            ], axis=-1)
            anti_dists = flatten(
                dists_elec[:self.n_up, self.n_up:],
            )
            log_mag = log_mag + self.cusp_electrons(
                same_dists, anti_dists,
            )

        if jastrow is not None:
            log_mag = log_mag + jastrow

        if (getattr(self, 'smith_deep_jastrow', None) is not None
                and emb is not None):
            log_mag = log_mag + self.smith_deep_jastrow(emb, r_bf)

        if self.pair_jastrow is not None:
            log_mag = log_mag + self.pair_jastrow(r, self.lattice)

        # ---- New: phase readout (n-aware via embeddings) ----
        # phase_mlp maps each per-electron embedding h_i → scalar.
        # Sum over electrons gives a permutation-invariant total phase.
        # Zero-init last layer of phase_mlp → phase = 0 at iter 0.
        if emb is not None and self.phase_mlp is not None:
            per_elec_phase = self.phase_mlp(emb)   # (n_elec, 1)
            phase = jnp.sum(per_elec_phase)
        else:
            phase = jnp.float64(0.0)

        return sign_psi, log_mag, phase


# =====================================================================
# 5. Helpers — periodic edge feature constructor (mirror of build code)
# =====================================================================

def _periodic_ee_feature(lattice):
    """Same edge feature builder as HEG PsiFormer (log_rescale=True)."""
    return PeriodicCombinedEdgeFeature(features=[
        PeriodicSinCosFeature(lattice=lattice),
        PeriodicDistancePowerEdgeFeature(
            lattice=lattice, powers=[1], log_rescale=True,
        ),
    ])


# =====================================================================
# 6. Build Tang-aware PsiFormer (clone of build_heg_psiformer_wf)
# =====================================================================

def build_tang_psiformer_wf(
    config: HEGPsiFormerConfig,
    n_max_for_tang: int,
    phase_mlp_hidden=(64, 64),
    rngs: Optional[nnx.Rngs] = None,
) -> TangPsiFormerWaveFunction:
    """Assemble a Tang-aware HEG PsiFormer matching V11's stack.

    Mirrors ``OmegaQMC.psi.nn.heg_wf_module.build_heg_psiformer_wf``
    but constructs Tang variants of the embedding / GNN / OmniNet
    and a TangPsiFormerWaveFunction with a phase readout.

    Args:
        config: HEGPsiFormerConfig (same fields as V11).
        n_max_for_tang: Fock truncation; the n-conditioning Linear
            has input dim N_max+1.
        phase_mlp_hidden: hidden widths of the phase readout MLP
            (input = embedding_dim, output = 1, zero-init last).
        rngs: nnx.Rngs.

    Returns:
        TangPsiFormerWaveFunction.  Call as
            sign, log_mag, phase = wf(r, one_hot_n=one_hot(n))
    """
    if rngs is None:
        raise ValueError("rngs is required")

    n_up = config.n_up
    n_down = config.n_down
    n_det = config.n_det
    L = float(config.L)
    L_y_attr = getattr(config, 'L_y', None)
    if L_y_attr is not None:
        L_y_val = float(L_y_attr)
        L_pass = (L, L_y_val)
        cell_area = L * L_y_val
    else:
        L_pass = L
        cell_area = L * L
    n_elec = n_up + n_down
    emb_dim = config.embedding_dim
    tp_dim = config.two_particle_stream_dim
    dim = int(getattr(config, 'dim', 3))

    if dim == 3:
        lattice = make_cubic_lattice(L)
    elif dim == 2:
        lattice = make_square_lattice(L)
    else:
        raise ValueError(f"config.dim must be 2 or 3, got {dim}")

    # --- Envelope (plane-wave; crystal not exposed in Tang v1) ---
    envelope_type = getattr(config, 'envelope_type', 'plane_wave')
    if envelope_type != 'plane_wave':
        raise NotImplementedError(
            f"Tang v1 only supports plane_wave envelope "
            f"(got {envelope_type!r})"
        )
    if config.full_determinant:
        pw_basis_size = (n_up + n_down) + int(config.n_virt_pw)
    else:
        pw_basis_size = max(n_up, n_down, 1) + int(config.n_virt_pw)
    envelope = PlaneWaveEnvelope(
        n_up=n_up, n_down=n_down, n_det=n_det, L=L_pass,
        init_pw_count=pw_basis_size,
        det_jitter=config.det_jitter,
        dim=dim,
        full_determinant=config.full_determinant,
    )

    # --- Tang-conditioned electron embedding ---
    electron_embedding = TangElectronEmbedding(
        n_up=n_up, n_down=n_down,
        embedding_dim=emb_dim,
        lattice=lattice,
        use_ghost_atom=config.use_ghost_atom,
        n_max_for_tang=n_max_for_tang,
        rngs=rngs,
    )

    # --- GNN layers (PsiFormer or FermiNet variant) ---
    backbone = str(getattr(config, 'backbone', 'psiformer')).lower()
    if backbone not in ('psiformer', 'ferminet'):
        raise NotImplementedError(
            f"Tang v1 only supports psiformer/ferminet backbone "
            f"(got {backbone!r})"
        )
    if backbone == 'psiformer':
        edge_types = ['same', 'anti']
    else:
        edge_types = ['up', 'down']
    edge_features = {et: _periodic_ee_feature(lattice) for et in edge_types}
    ee_feat_dim = 2 * dim + 1

    # Reuse the existing layer builders.  These build attention or
    # message-passing layers that take edges + embeddings and don't
    # need n-awareness — the n-info already lives in the embeddings.
    layer_builder = (
        _build_heg_gnn_layer if backbone == 'psiformer'
        else _build_heg_ferminet_layer
    )
    layers = []
    for idx in range(config.n_interactions):
        layer = layer_builder(
            config, idx=idx,
            emb_dim=emb_dim, tp_dim=tp_dim,
            edge_types=edge_types,
            ee_feat_dim=ee_feat_dim,
            n_up=n_up, n_down=n_down,
            rngs=rngs,
        )
        layers.append(layer)

    gnn = TangElectronGNN(
        mol_info=_HEGMolInfoShim(n_up=n_up, n_down=n_down),
        embedding_dim=emb_dim,
        two_particle_stream_dim=tp_dim,
        n_interactions=config.n_interactions,
        self_interaction=False,
        edge_types=edge_types,
        electron_embedding=electron_embedding,
        layers=layers,
        edge_features=edge_features,
    )

    # --- Backflow ---
    backflow = None
    if config.use_backflow:
        n_bf = 1
        bf_dict = {}
        if config.full_determinant:
            n_orb_up = n_orb_down = n_elec
        else:
            n_orb_up, n_orb_down = n_up, n_down
        for spin, n_orb in [('up', n_orb_up), ('down', n_orb_down)]:
            out_dim = n_orb * n_det
            nets = [MLP(
                emb_dim, out_dim,
                hidden_layers=config.bf_mlp_hidden_layers,
                bias=False, last_linear=True,
                activation=None, init='ferminet',
                rngs=rngs,
            )]
            bf_dict[spin] = Backflow(n_orb, n_det, n_bf, spin, nets=nets)
        backflow = bf_dict

    # --- OmniNet (deep Jastrow disabled when Smith DJ is on; same as V11) ---
    use_smith_dj = bool(getattr(config, 'use_smith_deep_jastrow', False))
    deep_jastrow = None
    if config.use_deep_jastrow and not use_smith_dj:
        from .psi.nn.omni import Jastrow as _DeepJastrow
        jas_mlp = MLP(
            emb_dim, 1,
            hidden_layers=config.jas_mlp_hidden_layers,
            bias=config.jas_mlp_bias,
            last_linear=True,
            activation=config.jas_mlp_activation,
            init='ferminet', rngs=rngs,
        )
        if config.jas_mlp_zero_init_last:
            last = jas_mlp.layers[-1]
            last.kernel = nnx.Param(jnp.zeros_like(param_value(last.kernel)))
            if last.use_bias:
                last.bias = nnx.Param(jnp.zeros_like(param_value(last.bias)))
        deep_jastrow = _DeepJastrow(net=jas_mlp, sum_first=False)

    omni = TangOmniNet(
        n_up=n_up, gnn=gnn,
        jastrow=deep_jastrow,
        backflow=backflow,
        nuclear_gnn_head=None,
    )

    # --- Cusp (n-blind, same as V11) ---
    cusp_electrons = None
    if config.use_cusp:
        same_scale = 0.5 if dim == 2 else 0.25
        anti_scale = 1.0 if dim == 2 else 0.5
        cusp_electrons = PeriodicElectronicCusp(
            L=L,
            same_scale=same_scale, anti_scale=anti_scale,
            alpha_init=1.0, r_cut_frac=0.45,
            trainable_alpha=bool(getattr(
                config, 'cusp_trainable_alpha', False,
            )),
            softplus_alpha=True,
        )

    # --- Pair Jastrow (off in V11) ---
    pair_jastrow = None
    if config.use_pair_jastrow:
        pair_jastrow = PeriodicPairJastrow(
            n_up, n_down,
            hidden=config.pair_jastrow_hidden,
            rngs=rngs,
        )

    # --- Smith deep Jastrow (V11 enables this) ---
    smith_deep_jastrow = None
    if use_smith_dj:
        smith_deep_jastrow = SmithDeepJastrow(
            d1=emb_dim, dim=dim,
            hidden=int(getattr(config, 'smith_jastrow_hidden', 32)),
            n_layers=int(getattr(config, 'smith_jastrow_n_layers', 4)),
            zero_init_last=True,
            lattice=lattice,
            rngs=rngs,
        )

    # --- Coord backflow (off in V11) ---
    coord_backflow = None
    if getattr(config, 'use_coord_backflow', False):
        coord_backflow = nnx.Linear(
            in_features=emb_dim, out_features=dim,
            use_bias=False, rngs=rngs,
        )
        if getattr(config, 'coord_bf_zero_init', True):
            coord_backflow.kernel = nnx.Param(
                jnp.zeros_like(param_value(coord_backflow.kernel)),
            )

    # --- Phase readout (Tang-specific) ---
    # MLP from per-electron embedding (emb_dim) → scalar phase
    # contribution.  Sum over electrons for the total phase.
    #
    # Init choice (kernel of last layer):
    #   * Xavier (default, was the bug-fix here): nonzero phase at iter 0.
    #     The bilinear coupling −λ q_c (ε̂·P̂) then has a nonzero gradient
    #     w.r.t. n_cond_proj from iter 1.  Without this, both pathways
    #     start at zero and the cross-derivative (which SR cannot see)
    #     is the only descent direction — vacuum-stuck local-min.
    #   * Bias is always zero-init (only the readout direction matters
    #     for gradient flow; zero bias keeps the per-electron sum centered).
    phase_mlp = MLP(
        emb_dim, 1,
        hidden_layers=tuple(phase_mlp_hidden),
        bias=True,
        last_linear=True,
        activation='tanh',
        init='ferminet',
        rngs=rngs,
    )
    last = phase_mlp.layers[-1]
    if last.use_bias:
        last.bias = nnx.Param(jnp.zeros_like(param_value(last.bias)))

    return TangPsiFormerWaveFunction(
        n_up=n_up, n_down=n_down, n_det=n_det, L=L,
        omni=omni, envelope=envelope, lattice=lattice,
        cusp_electrons=cusp_electrons,
        pair_jastrow=pair_jastrow,
        coord_backflow=coord_backflow,
        smith_deep_jastrow=smith_deep_jastrow,
        full_determinant=config.full_determinant,
        phase_mlp=phase_mlp,
    )


# =====================================================================
# 7. Top-level builder: make_tang_log_psi
# =====================================================================
#
# Mirrors make_heg_psiformer_log_psi but returns a callable
# log_psi(r, params, one_hot_n) → (sign, log_mag, phase).

def make_tang_log_psi(
    config: HEGPsiFormerConfig,
    rng_key: jax.Array,
    *,
    n_max_for_tang: int,
    phase_mlp_hidden=(64, 64),
):
    """Build a Tang-aware HEG log Ψ callable.

    Returns:
        (log_psi, init_params, graphdef) where
          log_psi(r, params, one_hot_n) → (sign, log_mag, phase)
        Use sign · exp(log_mag) · exp(i · phase) for complex ψ_n.
    """
    rngs = nnx.Rngs(rng_key)
    model = build_tang_psiformer_wf(
        config, n_max_for_tang=n_max_for_tang,
        phase_mlp_hidden=phase_mlp_hidden, rngs=rngs,
    )
    graphdef, params, other = nnx.split(model, nnx.Param, ...)

    def log_psi(r, params, one_hot_n):
        mdl = nnx.merge(graphdef, params, other)
        return mdl(r, one_hot_n=one_hot_n)

    return log_psi, params, graphdef


# =====================================================================
# 8. Trial wavefunction builder — top-level psi_vec for L8 V2
# =====================================================================

def build_tang_trial(
    config: HEGPsiFormerConfig,
    init_key: jax.Array,
    *,
    N_max: int,
    phase_mlp_hidden=(64, 64),
    offset_floor: float = -50.0,
):
    """Construct the full L8 V2 (Tang) trial wavefunction.

    Returns a dict with:
      psi_vec(R, p_pytree) → (N_max+1,) complex   trial amplitudes
      psi_vec_flat(r_flat, p_flat) → same
      log_psi_total(R, p_pytree) → real scalar    = log|Ψ(R)| for MCMC
      log_psi_total_flat(r_flat, p_flat) → same
      init_params_flat, init_params_pytree, unravel, n_params, …

    Trial composition for each Fock sector n:
        log|ψ_n(R)| = trunk_log_mag(R, one_hot(n)) + offset[n]
        arg ψ_n(R)  = trunk_phase(R, one_hot(n))
        ψ_n(R)      = sign · exp(log|ψ_n|) · exp(i · arg ψ_n)

    offset = (0, -floor, -floor, …) ⇒ ψ_0 ≈ ψ_HF, ψ_{n>0} ≈ 0 at init
    (since the trunk's n_cond_proj is zero-init, all sectors share
    ψ_HF, and the offset_floor pushes n>0 to negligible amplitude).
    """
    nelec = int(config.n_up) + int(config.n_down)
    dim = int(getattr(config, "dim", 3))
    if dim != 2:
        raise ValueError(
            f"build_tang_trial supports dim=2 only (got dim={dim})"
        )
    if N_max < 0:
        raise ValueError(f"N_max must be ≥ 0 (got {N_max})")

    log_psi, init_params, graphdef = make_tang_log_psi(
        config, init_key,
        n_max_for_tang=N_max,
        phase_mlp_hidden=phase_mlp_hidden,
    )
    init_params_flat, unravel = ravel_pytree(init_params)
    n_params = int(init_params_flat.shape[0])

    # Non-trainable amplitude offset for vacuum init.
    offset = jnp.concatenate([
        jnp.zeros((1,), dtype=jnp.float64),
        jnp.full((N_max,), float(offset_floor), dtype=jnp.float64),
    ])  # (N_max+1,)
    eye = jnp.eye(N_max + 1, dtype=jnp.float64)  # one-hot rows

    def psi_vec(R, p_pytree):
        """Vector amplitude ψ_n(R) for n=0..N_max as (N_max+1,) complex.

        Used by MCMC density (Σ|ψ_n|²) and SR Jacobian (needs all n at once).
        For per-n Laplacians in the local energy, use ``psi_n_only`` instead
        — it avoids the redundant trunk forwards for unused Fock sectors.
        """
        sign_list, log_mag_list, phase_list = [], [], []
        # Python-unrolled loop over n.  JAX traces it once at compile.
        for n in range(N_max + 1):
            sign_n, log_mag_n, phase_n = log_psi(R, p_pytree, eye[n])
            sign_list.append(sign_n)
            log_mag_list.append(log_mag_n + offset[n])
            phase_list.append(phase_n)
        sign = jnp.stack(sign_list)         # (N_max+1,) real
        log_mag = jnp.stack(log_mag_list)   # (N_max+1,) real
        phase = jnp.stack(phase_list)       # (N_max+1,) real
        mag = sign * jnp.exp(log_mag)       # real (signed)
        return mag * (jnp.cos(phase) + 1j * jnp.sin(phase))

    def psi_n_only(R, p_pytree, n_idx):
        """Single-Fock-sector amplitude ψ_n(R) (complex scalar).

        Runs the trunk ONCE with one_hot(n_idx) — does not compute the
        other Fock sectors.  Used in the eloc per-n Laplacian loop to
        avoid the (N_max+1)² trunk-forward blow-up of running psi_vec
        per Laplacian and indexing into it.  n_idx must be a Python
        int (compile-time constant) so the eye[n_idx] indexing
        resolves to a static one-hot vector.
        """
        one_hot_n = eye[n_idx]
        sign_n, log_mag_n, phase_n = log_psi(R, p_pytree, one_hot_n)
        log_mag_n = log_mag_n + offset[n_idx]
        mag = sign_n * jnp.exp(log_mag_n)
        return mag * (jnp.cos(phase_n) + 1j * jnp.sin(phase_n))

    def psi_vec_flat(r_flat, p_flat):
        R = r_flat.reshape(nelec, dim)
        p_pytree = unravel(p_flat)
        return psi_vec(R, p_pytree)

    def log_psi_total(R, p_pytree):
        """log|Ψ(R)| = ½ log Σ_n |ψ_n(R)|².  Used as the MCMC log-density
        target.  Uses log-sum-exp for numerical stability.
        """
        log_mag_list = []
        for n in range(N_max + 1):
            _, log_mag_n, _ = log_psi(R, p_pytree, eye[n])
            log_mag_list.append(log_mag_n + offset[n])
        log_mag_vec = jnp.stack(log_mag_list)
        # log|ψ_n|² = 2 log_mag_n  (sign² = 1, |exp(i phase)|² = 1)
        a = 2.0 * log_mag_vec
        a_max = jnp.max(a)
        return 0.5 * (a_max + jnp.log(jnp.sum(jnp.exp(a - a_max))))

    def log_psi_total_flat(r_flat, p_flat):
        R = r_flat.reshape(nelec, dim)
        p_pytree = unravel(p_flat)
        return log_psi_total(R, p_pytree)

    def mean_n(R, p_pytree):
        """⟨n⟩(R) = Σ_n n·|ψ_n|² / Σ_n |ψ_n|².  Diagnostic for N_max sufficiency.

        If ⟨n⟩ during training climbs near N_max the truncation is biting;
        restart with larger N_max.
        """
        psi = psi_vec(R, p_pytree)
        p_n = jnp.abs(psi) ** 2
        Z = jnp.sum(p_n)
        n_vec = jnp.arange(p_n.shape[0], dtype=jnp.float64)
        return jnp.sum(n_vec * p_n) / Z

    n_electronic = n_params  # all params live in the trunk (no separate heads)

    return {
        "psi_vec":            psi_vec,
        "psi_vec_flat":       psi_vec_flat,
        "psi_n_only":         psi_n_only,
        "log_psi_total":      log_psi_total,
        "log_psi_total_flat": log_psi_total_flat,
        "mean_n":             mean_n,
        "init_params_flat":   init_params_flat,
        "init_params_pytree": init_params,
        "unravel":            unravel,
        "n_params":           n_params,
        "n_electronic":       n_electronic,
        "offset":             offset,
        "N_max":              int(N_max),
        "nelec":              nelec,
        "dim":                dim,
        "trunk_log_psi":      log_psi,
        "trunk_graphdef":     graphdef,
    }


# =====================================================================
# 9. Laplacian primitive (local copy — keeps this module self-contained)
# =====================================================================

def _laplacian_vmap(f, x):
    """O(N) Laplacian + gradient via vmap-of-JVP of grad.

    For a scalar f(x), x ∈ ℝⁿ, returns (∇²f(x), ∇f(x)).
    """
    grad_f = jax.grad(f)
    df = grad_f(x)
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)

    def hvp_diag_i(e_i, i):
        _, hvp_e = jax.jvp(grad_f, (x,), (e_i,))
        return hvp_e[i]

    diag = jax.vmap(hvp_diag_i, in_axes=(0, 0))(eye, jnp.arange(n))
    return jnp.sum(diag), df


# =====================================================================
# 10. Local energy (Fock sum over n; identical formula to L8 V1)
# =====================================================================

def make_tang_eloc_no_vee(
    psi_n_only_fn,
    *,
    eps,
    lam: float,
    omega_eff: float,
    N_max: int,
    nelec: int,
    dim: int,
    coupling_op: str = "P",
):
    """Fock-basis local energy for the L8 V2 (Tang) trial.

    Takes ``psi_n_only_fn(R, p_pytree, n_idx)`` which runs the trunk
    ONCE per Fock sector — avoiding the (N_max+1)² blow-up that the
    earlier ``psi_vec``-indexing version had (each Laplacian re-trace
    ran psi_vec which internally ran the trunk N_max+1 times, and
    we did N_max+1 such Laplacians per eloc → (N_max+1)² × n_dofs
    trunk forwards in the worst case).  With psi_n_only we get
    (N_max+1) × n_dofs trunk forwards — same as L8 V1.

    Formula otherwise identical to L8 V1 eloc.  Caller adds V_ee + V_ext.
    """
    eps_arr = jnp.asarray(eps, dtype=jnp.float64)
    if coupling_op != "P":
        raise NotImplementedError(
            f"L8 V2 supports coupling_op='P' only "
            f"(got {coupling_op!r})"
        )
    n_axis = jnp.arange(N_max + 1, dtype=jnp.float64)
    bilinear_prefactor = jnp.sqrt(1.0 / (2.0 * omega_eff))
    m_idx = jnp.arange(1, N_max + 1, dtype=jnp.float64)
    sqrt_m = jnp.sqrt(m_idx)

    def eloc(R, p_pytree):
        r_flat = R.reshape(-1)

        # Per-n single-sector wrappers — one trunk forward each.
        def re_at_n(rf, n_idx):
            return jnp.real(
                psi_n_only_fn(rf.reshape(nelec, dim), p_pytree, n_idx)
            )

        def im_at_n(rf, n_idx):
            return jnp.imag(
                psi_n_only_fn(rf.reshape(nelec, dim), p_pytree, n_idx)
            )

        psi_list, grad_list, lap_list = [], [], []
        for n in range(N_max + 1):
            re_fn = lambda rf, _n=n: re_at_n(rf, _n)
            im_fn = lambda rf, _n=n: im_at_n(rf, _n)
            lap_re, grad_re = _laplacian_vmap(re_fn, r_flat)
            lap_im, grad_im = _laplacian_vmap(im_fn, r_flat)
            psi_n = re_fn(r_flat) + 1j * im_fn(r_flat)
            grad_n = grad_re + 1j * grad_im
            lap_n = lap_re + 1j * lap_im
            psi_list.append(psi_n)
            grad_list.append(grad_n)
            lap_list.append(lap_n)

        psi = jnp.stack(psi_list)
        grad = jnp.stack(grad_list)
        lap = jnp.stack(lap_list)
        psi_conj = jnp.conj(psi)
        Z = jnp.sum(jnp.abs(psi) ** 2)

        T_e_num = -0.5 * jnp.sum(psi_conj * lap)
        H_phot_num = omega_eff * jnp.sum(
            (n_axis + 0.5) * jnp.abs(psi) ** 2,
        )
        grad_per_elec = grad.reshape(N_max + 1, nelec, dim)
        g_eps = jnp.einsum("nid,d->n", grad_per_elec, eps_arr)
        bilin_sum = jnp.sum(
            sqrt_m * (
                jnp.conj(psi[:-1]) * g_eps[1:]
                + jnp.conj(psi[1:])  * g_eps[:-1]
            )
        )
        bilinear_num = 1j * lam * bilinear_prefactor * bilin_sum

        E_loc = (T_e_num + H_phot_num + bilinear_num) / Z
        return jnp.real(E_loc), jnp.imag(E_loc)

    return eloc


# =====================================================================
# 11. SR primitives (vector log-derivative — same as L8 V1)
# =====================================================================

def make_tang_sr_primitives(psi_vec_flat_fn, *, nelec: int, dim: int):
    """Per-walker (du/dθ, dv/dθ) for the vector log-derivative.

        ∂_θ log Ψ(R) ≡ ( Σ_n ψ_n*·∂_θψ_n ) / Σ_n |ψ_n|²

    Returns dict with same keys as L7/L8 V1 SR primitives so the
    downstream SMW solver is a drop-in.
    """
    def re_im_vec(r_flat, p_flat):
        psi = psi_vec_flat_fn(r_flat, p_flat)
        return jnp.stack([jnp.real(psi), jnp.imag(psi)])

    jac_re_im = jax.jacrev(re_im_vec, argnums=1)

    def per_walker_d_log_psi(r_flat, p_flat):
        ri = re_im_vec(r_flat, p_flat)
        psi = ri[0] + 1j * ri[1]
        jac = jac_re_im(r_flat, p_flat)
        jac_c = jac[0] + 1j * jac[1]
        Z = jnp.sum(jnp.abs(psi) ** 2)
        psi_conj = jnp.conj(psi)
        d_log = jnp.einsum("n,nk->k", psi_conj, jac_c) / Z
        return jnp.real(d_log), jnp.imag(d_log)

    batched_jacobian = jax.jit(
        jax.vmap(per_walker_d_log_psi, in_axes=(0, None))
    )
    return {
        "per_walker_d_log_psi": per_walker_d_log_psi,
        "batched_jacobian":     batched_jacobian,
    }


# =====================================================================
# 12. SR-VMC optimizer (mirror of _QEDFockOptimizer for the Tang trial)
# =====================================================================
#
# This class is the SR-VMC trainer for the Tang trial.  Structure is
# identical to _QEDFockOptimizer in OmegaQMC.qed_vmcopt_nn_heg_fock
# (the underlying SR-VMC loop is architecture-agnostic); only the
# trial / eloc / SR builders differ.  Per-iter compute will be ~(N_max+1)×
# the V11 L7 cost since the trunk is forwarded once per Fock sector.

class _QEDTangOptimizer:
    """SR-VMC optimizer for the L8 V2 (Tang-architecture) cavity-QED HEG."""

    def __init__(
        self,
        config,
        init_key,
        *,
        lr: float = 0.005,
        damping: float = 1e-3,
        n_cg: int = 20,
        ewald_n_real: int = 3,
        ewald_n_recip: int = 6,
        ewald_eta=None,
        ofname_chkpt=None,
        lr_schedule: str = "cosine",
        lr_min: float = 1e-5,
        lr_T_max=None,
        spring_mu: float = 0.0,
        spring_norm_clip: float = 0.0,
        omega: float = 0.1,
        coupling_lambda: float = 0.0,
        coupling_polarization=None,
        coupling_op: str = "P",
        v_ext_amp: float = 0.0,
        v_ext_a=None,
        include_vee: bool = True,
        # L8 V2 architecture knobs (defaults updated post-perf-work)
        N_max: int = 2,          # was 4; (N_max+1)/(4+1) compute reduction
        phase_mlp_hidden=(64, 64),
        offset_floor: float = -5.0,  # was -50; |ψ_{n>0}|/|ψ_0| = exp(-5) ≈ 7e-3
                                     # — easier for optimizer to grow than exp(-50)
    ):
        self.config = config
        L_y_attr = getattr(config, "L_y", None)
        if L_y_attr is not None:
            self.L_x = float(config.L)
            self.L_y = float(L_y_attr)
        else:
            self.L_x = self.L_y = float(config.L)
        self.L = float(config.L)
        self.n_up = int(config.n_up)
        self.n_down = int(config.n_down)
        self.nelec = self.n_up + self.n_down
        self.dim = int(getattr(config, "dim", 3))
        if self.dim != 2:
            raise ValueError(
                f"_QEDTangOptimizer supports dim=2 only (got dim={self.dim})"
            )
        self.lr = float(lr)
        self.damping = float(damping)
        self.n_cg = int(n_cg)
        self.ofname_chkpt = ofname_chkpt

        # Cavity
        self.omega = float(omega)
        self.coupling_lambda = float(coupling_lambda)
        self.coupling_op = str(coupling_op)
        if self.coupling_op != "P":
            raise NotImplementedError(
                "L8 V2 supports coupling_op='P' only "
                f"(got {self.coupling_op!r})"
            )
        self.omega_eff = float(jnp.sqrt(
            self.omega ** 2 + self.nelec * self.coupling_lambda ** 2
        ))
        if coupling_polarization is None:
            eps_list = [1.0] + [0.0] * (self.dim - 1)
        else:
            eps_list = list(coupling_polarization)
        eps_arr = jnp.asarray(eps_list, dtype=jnp.float64)
        eps_arr = eps_arr / jnp.linalg.norm(eps_arr)
        self.eps = eps_arr

        # Build trial (Tang-conditioned trunk + dual readout)
        self.tang = build_tang_trial(
            config, init_key,
            N_max=N_max,
            phase_mlp_hidden=phase_mlp_hidden,
            offset_floor=offset_floor,
        )
        self.params_flat = self.tang["init_params_flat"]
        self.unravel = self.tang["unravel"]
        self.n_params = self.tang["n_params"]
        self.N_max = int(N_max)

        # SR primitives (vector log-derivative)
        self.sr = make_tang_sr_primitives(
            self.tang["psi_vec_flat"],
            nelec=self.nelec, dim=self.dim,
        )

        # Local energy (no V_ee).  Uses psi_n_only to avoid the
        # (N_max+1)² trunk-forward blow-up — each per-n Laplacian
        # runs the trunk once instead of N_max+1 times.
        self.eloc_no_vee_fn = make_tang_eloc_no_vee(
            self.tang["psi_n_only"],
            eps=self.eps,
            lam=self.coupling_lambda,
            omega_eff=self.omega_eff,
            N_max=self.N_max,
            nelec=self.nelec, dim=self.dim,
            coupling_op=self.coupling_op,
        )

        # Lattice + Ewald
        if abs(self.L_x - self.L_y) > 1e-9:
            from .psi.nn.periodic import make_rectangular_lattice
            self.lattice = make_rectangular_lattice(self.L_x, self.L_y)
            ewald_L = (self.L_x, self.L_y)
        else:
            self.lattice = make_square_lattice(self.L_x)
            ewald_L = self.L_x
        self.ewald = build_ewald_tables_dim(
            ewald_L, dim=self.dim, eta=ewald_eta,
            n_real=ewald_n_real, n_recip=ewald_n_recip,
        )
        self._ewald_per_walker = jax.jit(jax.vmap(
            lambda r: ewald_2d_pair_energy(r[None], self.ewald)[0]
        ))
        self.include_vee = bool(include_vee)

        # v_ext (Weber cosine)
        self.v_ext_amp = float(v_ext_amp)
        self.v_ext_a = float(v_ext_a) if v_ext_a is not None else float(self.L_x)
        k_ext = 2.0 * jnp.pi / self.v_ext_a
        amp = self.v_ext_amp

        def _vext_one(R):
            return -amp * jnp.sum(jnp.cos(k_ext * R))
        self._vext_per_walker = jax.jit(jax.vmap(_vext_one))

        # LR schedule + SR solver
        self.lr_schedule = lr_schedule
        self.lr_min = float(lr_min)
        self.lr_T_max = lr_T_max
        self.spring_mu = float(spring_mu)
        self.spring_norm_clip = float(spring_norm_clip)
        self.smw_solver = make_l5_sr_smw_solver()

        # Batched eloc
        def _eloc_one(R_i, p_pytree):
            return self.eloc_no_vee_fn(R_i, p_pytree)
        self._eloc_batched = jax.jit(jax.vmap(
            _eloc_one, in_axes=(0, None),
        ))

        # MCMC step (jitted via vmap; recompiled per walker batch size)
        log_psi_total_flat = self.tang["log_psi_total_flat"]
        lattice_c = self.lattice

        def R_move_one(key, R, step, p_flat):
            kp, ka = jax.random.split(key)
            R_prop = R + step * jax.random.normal(
                kp, R.shape, dtype=jnp.float64,
            )
            R_prop = wrap_to_cell(R_prop, lattice_c)
            lp_old = log_psi_total_flat(R.reshape(-1), p_flat)
            lp_new = log_psi_total_flat(R_prop.reshape(-1), p_flat)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old)
            )
            return jnp.where(accept, R_prop, R), accept

        self._R_move_batch = jax.vmap(
            R_move_one, in_axes=(0, 0, None, None),
        )

        def _mcmc_step(rng_key, R, step_R, p_flat, num_walkers):
            keys = jax.random.split(rng_key, num_walkers)
            R_new, acc = self._R_move_batch(keys, R, step_R, p_flat)
            ar = jnp.mean(acc).astype(jnp.float64)
            new_step = _adapt_step_size(step_R, ar)
            return R_new, new_step, ar
        self._mcmc_step_uncompiled = _mcmc_step

    def _compute_lr(self, it):
        if self.lr_schedule == "cosine":
            T = int(self.lr_T_max) if self.lr_T_max is not None else 500
            if it >= T:
                return self.lr_min
            cos = 0.5 * (1.0 + jnp.cos(jnp.pi * it / T))
            return self.lr_min + (self.lr - self.lr_min) * float(cos)
        return self.lr

    def initialize_walkers(self, rng_key, num_walkers):
        """R walkers — uniform in [0, L_x] × [0, L_y]."""
        L_arr = jnp.asarray([self.L_x, self.L_y], dtype=jnp.float64)
        u = jax.random.uniform(
            rng_key, (num_walkers, self.nelec, self.dim),
            dtype=jnp.float64,
        )
        return u * L_arr

    def _batched_eloc_with_vee(self, R, p_pytree):
        re_no_vee, im = self._eloc_batched(R, p_pytree)
        V_ee = self._ewald_per_walker(R) if self.include_vee else 0.0
        V_ext = (
            self._vext_per_walker(R) if self.v_ext_amp != 0.0 else 0.0
        )
        return re_no_vee + V_ee + V_ext, im

    def train(
        self,
        rng_key,
        num_walkers: int,
        n_iters: int,
        *,
        mcmc_decorr_steps: int = 15,
        mc_timestep_R: float = 0.1,
        equil_steps: int = 50,
        save_every: int = 0,
        verbose: int = 1,
        chkpt_path=None,
        log_file=None,
    ):
        """SR-VMC training loop (Python-orchestrated)."""
        if chkpt_path is None:
            chkpt_path = self.ofname_chkpt

        key_init, key_train = jax.random.split(rng_key)
        R = self.initialize_walkers(key_init, num_walkers)
        step_R = jnp.float64(mc_timestep_R)
        params_flat = self.params_flat
        prev_delta = jnp.zeros_like(params_flat)

        for _ in range(equil_steps):
            key_train, sub = jax.random.split(key_train)
            R, step_R, ar = self._mcmc_step_uncompiled(
                sub, R, step_R, params_flat, num_walkers,
            )
        if verbose >= 1:
            print(
                f"# L8 V2 (Tang) SR-VMC — {self.n_params} params  "
                f"(N_max={self.N_max})",
                file=log_file,
            )
            print(
                f"# Equilibration: ar_R={float(ar):.3f}; "
                f"step_R={float(step_R):.3f}",
                file=log_file,
            )
            print(
                f"# Ω={self.omega:.4f}, λ={self.coupling_lambda:.4f}, "
                f"Ω_eff={self.omega_eff:.4f}",
                file=log_file,
            )
            print(
                "# iter   <Re E>/N        Var(Re E)      <Im E>/N        |g|",
                file=log_file,
            )
            if log_file is not None:
                log_file.flush()

        import time
        damping = jnp.float64(self.damping)
        mu = jnp.float64(self.spring_mu)
        c_clip = jnp.float64(self.spring_norm_clip)

        for it in range(1, n_iters + 1):
            t0 = time.time()
            for _ in range(mcmc_decorr_steps):
                key_train, sub = jax.random.split(key_train)
                R, step_R, _ = self._mcmc_step_uncompiled(
                    sub, R, step_R, params_flat, num_walkers,
                )
            p_pytree = self.unravel(params_flat)
            e_re, e_im = self._batched_eloc_with_vee(R, p_pytree)
            r_flat = R.reshape(num_walkers, -1)
            Jac_u, Jac_v = self.sr["batched_jacobian"](r_flat, params_flat)
            delta, e_mean, e_var, im_mean, g_norm, scale = self.smw_solver(
                Jac_u, Jac_v, e_re, e_im,
                prev_delta, damping, mu, c_clip,
            )
            lr_now = self._compute_lr(it - 1)
            params_flat = params_flat - lr_now * delta
            prev_delta = delta
            dt = time.time() - t0

            if verbose >= 1 and (it <= 10 or it % 10 == 0):
                print(
                    f"{it:5d}  {float(e_mean) / self.nelec:+.6e}  "
                    f"{float(e_var):.4e}  "
                    f"{float(im_mean) / self.nelec:+.3e}  "
                    f"{float(g_norm):.3e}  ({dt:.2f}s)",
                    file=log_file,
                )
                if log_file is not None:
                    log_file.flush()

            if save_every > 0 and chkpt_path is not None and it % save_every == 0:
                np.savez(
                    chkpt_path,
                    params_flat=np.asarray(params_flat),
                    R=np.asarray(R),
                    R_step_size=np.asarray(step_R),
                    E_final_ha=np.asarray(e_mean / self.nelec),
                    n_iters_trained=int(it),
                )
                if verbose >= 1:
                    print(
                        f"# [chkpt] saved at iter {it}, "
                        f"E/N={float(e_mean)/self.nelec:+.6f} Ha → {chkpt_path}",
                        file=log_file,
                    )
                    if log_file is not None:
                        log_file.flush()

        self.params_flat = params_flat
        return params_flat, R

    # ---------------------------------------------------------------
    # Fused JIT train step — mirror of L8 V1's pattern, drops q_c branch
    # ---------------------------------------------------------------
    def _build_fused_train_step(self, num_walkers, mcmc_decorr_steps):
        """One @jax.jit train_step fusing MCMC + eloc + Jac + SMW + update."""
        log_psi_total_flat = self.tang["log_psi_total_flat"]
        lattice = self.lattice
        unravel = self.unravel
        eloc_no_vee_fn = self.eloc_no_vee_fn
        ewald = self.ewald
        damping_c = jnp.float64(self.damping)
        mu_c = jnp.float64(self.spring_mu)
        c_clip_c = jnp.float64(self.spring_norm_clip)
        include_vee = self.include_vee
        v_ext_amp = self.v_ext_amp
        k_ext = 2.0 * jnp.pi / self.v_ext_a
        per_walker_d_log_psi = self.sr["per_walker_d_log_psi"]

        def R_move_one(key, R, step, p_flat):
            kp, ka = jax.random.split(key)
            R_prop = R + step * jax.random.normal(
                kp, R.shape, dtype=jnp.float64,
            )
            R_prop = wrap_to_cell(R_prop, lattice)
            lp_old = log_psi_total_flat(R.reshape(-1), p_flat)
            lp_new = log_psi_total_flat(R_prop.reshape(-1), p_flat)
            accept = jax.random.uniform(ka) < jnp.exp(
                2.0 * (lp_new - lp_old)
            )
            return jnp.where(accept, R_prop, R), accept

        R_move_batch = jax.vmap(R_move_one, in_axes=(0, 0, None, None))
        batched_jac = jax.vmap(per_walker_d_log_psi, in_axes=(0, None))

        def eloc_one(R_i, p_pytree):
            return eloc_no_vee_fn(R_i, p_pytree)
        eloc_batched = jax.vmap(eloc_one, in_axes=(0, None))

        def ewald_batched(R):
            _one = lambda r: ewald_2d_pair_energy(r[None], ewald)[0]
            return jax.vmap(_one)(R)

        def vext_batched(R):
            if v_ext_amp == 0.0:
                return jnp.zeros(R.shape[0], dtype=R.dtype)
            return -v_ext_amp * jnp.sum(jnp.cos(k_ext * R), axis=(1, 2))

        def mcmc_one_step(carry, _):
            rng, R, step_R, p_flat = carry
            rng, sub = jax.random.split(rng)
            keys = jax.random.split(sub, num_walkers)
            R, acc_R = R_move_batch(keys, R, step_R, p_flat)
            ar_R = jnp.mean(acc_R).astype(jnp.float64)
            step_R = _adapt_step_size(step_R, ar_R)
            return (rng, R, step_R, p_flat), ar_R

        @jax.jit
        def fused_train_step(carry, lr):
            (rng, R, step_R, params_flat, prev_delta) = carry

            ar_R_sum = jnp.float64(0.0)
            mcmc_carry = (rng, R, step_R, params_flat)
            for _ in range(mcmc_decorr_steps):
                mcmc_carry, ar_R_i = mcmc_one_step(mcmc_carry, None)
                ar_R_sum = ar_R_sum + ar_R_i
            rng, R, step_R, _ = mcmc_carry
            ar_R = ar_R_sum / mcmc_decorr_steps

            p_pytree = unravel(params_flat)
            re_no_vee, e_im = eloc_batched(R, p_pytree)
            V_ee = (
                ewald_batched(R) if include_vee
                else jnp.zeros(num_walkers)
            )
            V_ext = vext_batched(R)
            e_re = re_no_vee + V_ee + V_ext

            r_flat = R.reshape(num_walkers, -1)
            Jac_u, Jac_v = batched_jac(r_flat, params_flat)

            n_w = num_walkers
            e_re_mean = jnp.mean(e_re)
            e_im_mean = jnp.mean(e_im)
            e_re_var = jnp.var(e_re)
            de_re = e_re - e_re_mean
            de_im = e_im - e_im_mean
            dJu = Jac_u - jnp.mean(Jac_u, axis=0)
            dJv = Jac_v - jnp.mean(Jac_v, axis=0)
            g = 2.0 * (
                (dJu.T @ de_re) / n_w + (dJv.T @ de_im) / n_w
            )
            rhs = g + (mu_c * damping_c) * prev_delta
            do_s = jnp.concatenate([dJu, dJv], axis=0)
            inv_lambda_n = 1.0 / (damping_c * n_w)
            K = (do_s @ do_s.T) * inv_lambda_n
            I_plus_K = K + jnp.eye(K.shape[0], dtype=K.dtype)
            o_rhs = (do_s @ rhs) * inv_lambda_n
            u_smw = jnp.linalg.solve(I_plus_K, o_rhs)
            delta = (rhs - do_s.T @ u_smw) / damping_c

            proj = do_s @ delta
            f_norm_sq = jnp.maximum(jnp.sum(proj * proj) / n_w, 1e-20)
            f_norm = jnp.sqrt(f_norm_sq)
            raw_scale = c_clip_c / (f_norm + 1e-20)
            scale = jnp.where(
                c_clip_c > 0.0,
                jnp.minimum(1.0, raw_scale),
                jnp.ones_like(raw_scale),
            )
            delta = scale * delta
            g_norm = jnp.sqrt(jnp.sum(g * g))

            params_flat = params_flat - lr * delta
            prev_delta = delta

            new_carry = (rng, R, step_R, params_flat, prev_delta)
            metrics = {
                "e_mean": e_re_mean, "e_var": e_re_var,
                "im_mean": e_im_mean, "g_norm": g_norm,
                "scale": scale, "ar_R": ar_R, "step_R": step_R,
            }
            return new_carry, metrics

        return fused_train_step

    def train_fused(
        self,
        rng_key,
        num_walkers: int,
        n_iters: int,
        *,
        mcmc_decorr_steps: int = 15,
        mc_timestep_R: float = 0.1,
        equil_steps: int = 50,
        save_every: int = 0,
        verbose: int = 1,
        chkpt_path=None,
        log_file=None,
    ):
        """Fused-JIT training loop."""
        if chkpt_path is None:
            chkpt_path = self.ofname_chkpt
        key_init, key_train = jax.random.split(rng_key)
        R = self.initialize_walkers(key_init, num_walkers)
        step_R = jnp.float64(mc_timestep_R)
        params_flat = self.params_flat
        prev_delta = jnp.zeros_like(params_flat)

        for _ in range(equil_steps):
            key_train, sub = jax.random.split(key_train)
            R, step_R, ar = self._mcmc_step_uncompiled(
                sub, R, step_R, params_flat, num_walkers,
            )
        if verbose >= 1:
            print(
                f"# L8 V2 (Tang) SR-VMC fused — {self.n_params} params  "
                f"(N_max={self.N_max})",
                file=log_file,
            )
            print(
                f"# Equilibration: ar_R={float(ar):.3f}; "
                f"step_R={float(step_R):.3f}",
                file=log_file,
            )
            print(
                f"# Ω={self.omega:.4f}, λ={self.coupling_lambda:.4f}, "
                f"Ω_eff={self.omega_eff:.4f}",
                file=log_file,
            )
            print(
                "# iter   <Re E>/N        Var(Re E)      <Im E>/N        |g|",
                file=log_file,
            )
            if log_file is not None:
                log_file.flush()

        fused_step = self._build_fused_train_step(
            num_walkers, mcmc_decorr_steps,
        )
        import time
        carry = (key_train, R, step_R, params_flat, prev_delta)
        for it in range(1, n_iters + 1):
            t0 = time.time()
            lr_now = jnp.float64(self._compute_lr(it - 1))
            carry, metrics = fused_step(carry, lr_now)
            metrics["e_mean"].block_until_ready()
            dt = time.time() - t0

            if verbose >= 1 and (it <= 10 or it % 10 == 0):
                print(
                    f"{it:5d}  "
                    f"{float(metrics['e_mean']) / self.nelec:+.6e}  "
                    f"{float(metrics['e_var']):.4e}  "
                    f"{float(metrics['im_mean']) / self.nelec:+.3e}  "
                    f"{float(metrics['g_norm']):.3e}  ({dt:.2f}s)",
                    file=log_file,
                )
                if log_file is not None:
                    log_file.flush()

            if save_every > 0 and chkpt_path is not None and it % save_every == 0:
                _, R_now, step_R_now, params_now, _ = carry
                np.savez(
                    chkpt_path,
                    params_flat=np.asarray(params_now),
                    R=np.asarray(R_now),
                    R_step_size=np.asarray(step_R_now),
                    E_final_ha=np.asarray(metrics["e_mean"] / self.nelec),
                    n_iters_trained=int(it),
                )

        _, R_final, _, params_final, _ = carry
        self.params_flat = params_final
        return params_final, R_final

    def evaluate(
        self,
        rng_key,
        num_walkers: int,
        *,
        num_blocks: int,
        steps_per_block: int = 10,
        equil_blocks: int = 5,
        mc_timestep_R: float = 0.1,
        params_flat=None,
        R_init=None,
        verbose: int = 1,
        log_file=None,
    ):
        """Blocked evaluation of ⟨H⟩/N."""
        if params_flat is None:
            params_flat = self.params_flat
        key_init, key_eval = jax.random.split(rng_key)
        if R_init is None:
            R = self.initialize_walkers(key_init, num_walkers)
        else:
            R = R_init
        step_R = jnp.float64(mc_timestep_R)
        for _ in range(equil_blocks * steps_per_block):
            key_eval, sub = jax.random.split(key_eval)
            R, step_R, _ = self._mcmc_step_uncompiled(
                sub, R, step_R, params_flat, num_walkers,
            )
        block_re, block_im = [], []
        p_pytree = self.unravel(params_flat)
        import time
        t0 = time.time()
        for b in range(num_blocks):
            for _ in range(steps_per_block):
                key_eval, sub = jax.random.split(key_eval)
                R, step_R, _ = self._mcmc_step_uncompiled(
                    sub, R, step_R, params_flat, num_walkers,
                )
            e_re, e_im = self._batched_eloc_with_vee(R, p_pytree)
            block_re.append(float(jnp.mean(e_re)) / self.nelec)
            block_im.append(float(jnp.mean(e_im)) / self.nelec)
        dt = time.time() - t0
        e_re_arr = np.asarray(block_re)
        e_im_arr = np.asarray(block_im)
        mean_re = float(e_re_arr.mean())
        std_re = float(e_re_arr.std(ddof=1)) if len(e_re_arr) > 1 else 0.0
        sem_re = std_re / max(1.0, np.sqrt(len(e_re_arr)))
        if verbose >= 1:
            print(
                f"# Eval: {num_blocks} blocks × {steps_per_block} × {num_walkers}, "
                f"time {dt:.1f}s",
                file=log_file,
            )
            print(
                f"  E/N = {mean_re:+.6e} ± {sem_re:.2e} Ha",
                file=log_file,
            )
        return {
            "E_per_e_ha":  mean_re,
            "E_per_e_sem": sem_re,
            "E_per_e_std": std_re,
            "Im_per_e_ha": float(e_im_arr.mean()),
            "block_re":    e_re_arr.tolist(),
            "block_im":    e_im_arr.tolist(),
            "wall_time_s": dt,
        }
