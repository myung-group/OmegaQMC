"""PsiFormer-style neural-network wavefunction for the 3D HEG.

Adapts the existing molecular PsiFormer stack (``OmegaQMC.psi.nn``)
to the homogeneous electron gas with periodic boundary conditions.
Structurally this ansatz is:

.. code-block:: none

    r  →  ElectronGNN(periodic edge features, no nuclei)
             │
             ▼
    per-electron embeddings  →  Backflow MLP  →  fs (mult)
             │                                    │
             ▼                                    │
    PlaneWaveEnvelope  →  orbital matrix  ──────► orb * (1 + 2·tanh(fs/4))
                                                    │
                                                    ▼
                                           multi-det Slater
                                                    │
                                                    ▼
                                        + electronic cusp (periodic)
                                        + Jastrow (PeriodicPairJastrow)
                                                    │
                                                    ▼
                                              log ψ (real)

Re-used from the molecular stack (unchanged):
  * :class:`~.gnn.electron_gnn.ElectronGNN`,
    :class:`~.gnn.electron_gnn.ElectronGNNLayer` — GNN backbone
  * :class:`~.gnn.update_features.NodeAttentionElectronUpdateFeature`
    — PsiFormer self-attention
  * :class:`~.omni.Backflow` — orbital-producing head
  * :class:`~.wf.BackflowOp`, :func:`~.wf.eval_log_slater` — det math
  * :class:`~.layers.MLP`, :class:`~.layers.ResidualConnection`
    — building blocks

New pieces specific to HEG:
  * :class:`HEGElectronEmbedding` — type-only embedding distinguishing
    up vs down spins (the molecular one collapses them when
    ``n_up == n_down``).
  * :func:`build_heg_psiformer_wf` — assembles the GNN / Backflow /
    envelope / cusp / Jastrow pieces with periodic edge features.
  * :class:`HEGPsiFormerWaveFunction` — top-level module returning
    ``log|ψ|`` for Metropolis and kinetic-energy consumers.
"""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .compat import param_value
from .cusp import (
    ElectronicCuspAsymptotic,
    PsiformerCusp,
    DeepQMCCusp,
)
from .env_periodic import (
    ComplexPlaneWaveEnvelope,
    PlaneWaveEnvelope,
)
from .gnn.edge_features_periodic import (
    PeriodicCombinedEdgeFeature,
    PeriodicDistancePowerEdgeFeature,
    PeriodicSinCosFeature,
)
from .gnn.electron_gnn import ElectronGNN, ElectronGNNLayer
from .gnn.update_features import (
    NodeAttentionElectronUpdateFeature,
    NodeSumElectronUpdateFeature,
    ResidualElectronUpdateFeature,
)
from .heg_wf import PeriodicPairJastrow, _make_heg_phys_conf
from .layers import MLP, ResidualConnection, SumPool
from .omni import Backflow, OmniNet
from .periodic import (
    PeriodicLattice,
    fractional_coords,
    make_cubic_lattice,
    periodic_norm,
)
from .types import Psi
from .utils import flatten, triu_flat
from .wf import eval_log_slater


# =====================================================================
# HEG electron embedding
# =====================================================================

class HEGElectronEmbedding(nnx.Module):
    """Spin-typed initial electron embedding for the HEG.

    Always distinguishes up and down spins (unlike the molecular
    :class:`~.gnn.electron_gnn.ElectronEmbedding` which collapses
    them to a single type when ``n_up == n_down``, relying solely
    on edge-type distinctions to differentiate the spins in the
    later attention layers).

    When ``use_ghost_atom=True`` the embedding is *also* conditioned
    on each electron's position relative to a phantom (Z=0) atom at
    the cell origin — the trick FermiNet uses for HEG (see
    ``ferminet/configs/heg.py``).  Without this, same-spin
    electrons get identical embeddings at initialisation (the GNN
    aggregation over neighbours is nearly-uniform across electrons
    of the same spin when the walker is statistically bulk-like),
    which leaves the backflow head with essentially no signal to
    specialise and the ansatz plateaus near HF.  Adding a ghost-atom
    positional fingerprint breaks this symmetry structurally.

    The ghost-atom features are
    ``(sin 2π s, cos 2π s, |r|_periodic)`` with ``s = A⁻¹ r``, i.e.
    the same periodic features FermiNet's ``pbc/feature_layer.py``
    uses for its ae edge channel.  These are periodic under full
    lattice translations (so wrap-to-cell doesn't change them) but
    *not* invariant under continuous translations — this mild
    symmetry breaking is shared with FermiNet and is the standard
    accommodation for phantom-nucleus HEG ansätze.

    Args:
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        embedding_dim: Per-electron embedding dimension.
        lattice: :class:`PeriodicLattice` — required when
            ``use_ghost_atom=True``, else unused.
        use_ghost_atom: Enable the ghost-atom positional
            fingerprint.  Strongly recommended for HEG; structurally
            breaks the embedding-degeneracy problem described in the
            class docstring.
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        *,
        n_up: int,
        n_down: int,
        embedding_dim: int,
        lattice: Optional[PeriodicLattice] = None,
        use_ghost_atom: bool = False,
        rngs: nnx.Rngs,
    ):
        self.n_up = n_up
        self.n_down = n_down
        self.use_ghost_atom = bool(use_ghost_atom)
        self.type_embed = nnx.Embed(
            num_embeddings=2,
            features=embedding_dim,
            rngs=rngs,
        )
        self.elec_types = nnx.data(jnp.asarray(
            [0] * n_up + [1] * n_down, dtype=jnp.int32,
        ))
        if self.use_ghost_atom:
            if lattice is None:
                raise ValueError(
                    "use_ghost_atom=True requires a lattice.",
                )
            self.lattice = nnx.data(lattice)
            # dim sin + dim cos + 1 periodic-norm = 2*dim+1 features.
            # Inferred from lattice.A.shape (3 for cubic, 2 for square).
            self._dim = int(lattice.A.shape[0])
            ne_feat_dim = 2 * self._dim + 1
            self.ghost_proj = nnx.Linear(
                embedding_dim + ne_feat_dim, embedding_dim,
                use_bias=False, rngs=rngs,
            )
        else:
            self.lattice = None
            self.ghost_proj = None
            self._dim = None

    def __call__(self, phys_conf, nucleus_embedding=None):
        """Return per-electron embeddings.

        Args:
            phys_conf: :class:`~.types.PhysicalConfiguration` — the
                ``.r`` field (electron positions) is used when
                ``use_ghost_atom=True``; otherwise ignored.
            nucleus_embedding: Unused.
        """
        del nucleus_embedding
        spin_emb = self.type_embed(self.elec_types)  # (n_elec, emb_dim)
        if not self.use_ghost_atom:
            return spin_emb

        # Ghost-atom positional fingerprint.  The ghost sits at the
        # cell origin, so ae = r_elec - 0 = r_elec.  Shapes carry the
        # spatial dimension (3 for 3D HEG, 2 for 2D HEG); the
        # downstream feature concatenation has 2*dim + 1 columns.
        r = phys_conf.r                                      # (n_elec, dim)
        s = fractional_coords(r, self.lattice)               # (n_elec, dim)
        two_pi_s = 2.0 * jnp.pi * s
        sin_s = jnp.sin(two_pi_s)                            # (n_elec, dim)
        cos_s = jnp.cos(two_pi_s)                            # (n_elec, dim)
        r_periodic = periodic_norm(r, self.lattice, safe=True)  # (n_elec,)
        ne_feat = jnp.concatenate(
            [sin_s, cos_s, r_periodic[:, None]], axis=-1,
        )                                                    # (n_elec, 2*dim+1)
        return self.ghost_proj(
            jnp.concatenate([spin_emb, ne_feat], axis=-1),
        )


# =====================================================================
# HEG backflow applier (multiplicative-only, no nuclear cutoff)
# =====================================================================

class SmithDeepJastrow(nnx.Module):
    """Smith 2024 deep Jastrow (supplement eqs. 20-21).

        U({r_i}) = Σ_i J(h_i^(T), Linear_pre(x_i))
        J = Linear_L ∘ GELU ∘ ... ∘ Linear_1   (L = 4 layers)

    Takes both the post-GNN one-body stream ``h_i`` AND the
    backflow-shifted positions ``x_i``; concatenates ``[h_i, Linear_pre(x_i)]``
    and feeds through an L-layer GELU MLP.  The final layer is
    zero-initialised so the contribution is exactly zero at iter 0
    (preserves the envelope-only initialisation).

    Differs from :class:`~OmegaQMC.psi.nn.omni.Jastrow` (used in
    PsiFormer/FermiNet path) in that it uses position information
    explicitly — Smith reports this gives "better expressivity" than
    feeding only the embedding stream.

    Args:
        d1: One-body stream dim.
        dim: Spatial dim (2 or 3).
        hidden: Hidden width.
        n_layers: Number of Linear layers ``L`` (default 4 → 3 hidden).
        zero_init_last: Zero-initialise the final layer (default True).
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        *,
        d1: int,
        dim: int,
        hidden: int = 32,
        n_layers: int = 4,
        zero_init_last: bool = True,
        rngs: nnx.Rngs,
    ):
        self.d1 = int(d1)
        self.dim = int(dim)
        # Linear_pre: dim → d1 (so the position projection is on the
        # same scale as h_i before concatenation).
        self.linear_pre = nnx.Linear(dim, d1, rngs=rngs)
        # MLP J: in_dim = d1 + d1 (concat of h_i and Linear_pre(x_i)).
        in_dim = 2 * d1
        layers = []
        for li in range(int(n_layers)):
            out_dim = 1 if li == n_layers - 1 else hidden
            layers.append(
                nnx.Linear(in_dim, out_dim, rngs=rngs),
            )
            in_dim = out_dim
        self.layers = nnx.List(layers)
        if zero_init_last:
            last = self.layers[-1]
            last.kernel = nnx.Param(
                jnp.zeros_like(param_value(last.kernel)),
            )
            if last.use_bias:
                last.bias = nnx.Param(
                    jnp.zeros_like(param_value(last.bias)),
                )

    def __call__(self, h: jax.Array, x: jax.Array) -> jax.Array:
        """Compute scalar log-Jastrow contribution.

        Args:
            h: One-body stream, ``(n_elec, d1)``.
            x: Backflow-shifted positions, ``(n_elec, dim)``.

        Returns:
            Scalar — sum over electrons of per-electron J-output.
        """
        x_proj = self.linear_pre(x)
        z = jnp.concatenate([h, x_proj], axis=-1)
        for li, layer in enumerate(self.layers):
            z = layer(z)
            if li < len(self.layers) - 1:
                z = nnx.gelu(z)
        return z.squeeze(-1).sum()


def apply_heg_backflow_mult(orb, fs):
    """Multiplicative backflow without the molecular nuclear cutoff.

    ``orb`` is reshaped to ``(n_det, n_spin_electrons, n_orb_spin)``
    and multiplied by ``1 + 2·tanh(fs/4)`` element-wise.  The
    nuclear-cutoff path used by :class:`~.wf.BackflowOp` is skipped
    entirely — it preserves nuclear cusps, which do not exist for
    the HEG.

    Args:
        orb: ``(n_det, n_spin, n_orb)`` orbital matrix.
        fs: ``(n_bf=1, n_det, n_spin, n_orb)`` backflow tensor from
            :class:`~.omni.Backflow`.

    Returns:
        Backflow-transformed orbital matrix, same shape as ``orb``.
    """
    fs_mult = fs.squeeze(axis=0)
    return orb * (1.0 + 2.0 * jnp.tanh(fs_mult / 4.0))


# =====================================================================
# HEG PsiFormer wavefunction
# =====================================================================

class HEGPsiFormerWaveFunction(nnx.Module):
    """PsiFormer-style HEG wavefunction at Γ point.

    The top-level module composes the GNN / Backflow / envelope /
    Jastrow pieces and returns ``log|ψ|`` (real).  A complex
    twist-averaged variant can be derived mechanically by swapping
    :class:`PlaneWaveEnvelope` for
    :class:`~.env_periodic.ComplexPlaneWaveEnvelope`; that's a
    separate class to keep the real/Γ-point path pure-real.

    Args:
        n_up, n_down, n_det, L: as :class:`~.heg_wf.HEGSlaterJastrow`.
        omni: :class:`~.omni.OmniNet` instance (GNN + backflow +
            optional Jastrow head).
        envelope: :class:`~.env_periodic.PlaneWaveEnvelope`.
        lattice: :class:`PeriodicLattice` for cusp distances and
            any periodic operations.
        cusp_electrons: Optional
            :class:`~.cusp.ElectronicCuspAsymptotic`.
        pair_jastrow: Optional
            :class:`~.heg_wf.PeriodicPairJastrow` — a flat e-e
            two-body Jastrow on top of the full PsiFormer.  Set to
            ``None`` to let the network's one-body backflow +
            deep Jastrow carry all correlation.
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
    ):
        self.n_up = n_up
        self.n_down = n_down
        self.n_det = n_det
        self.L = float(L)
        self.lattice = nnx.data(lattice)
        self.omni = omni
        self.envelope = envelope
        self.cusp_electrons = cusp_electrons
        self.pair_jastrow = pair_jastrow
        self.coord_backflow = coord_backflow
        self.smith_deep_jastrow = smith_deep_jastrow

    @property
    def _spin_slices(self):
        return (
            slice(None, self.n_up),
            slice(self.n_up, None),
        )

    def orbital_matrices(self, r: jax.Array):
        """Return the post-backflow Slater orbital matrices.

        Used by supervised pre-training: the network's orbital
        output is compared to a Hartree-Fock target without yet
        collapsing it into a Slater determinant.

        Args:
            r: Electron positions ``(n_elec, 3)`` grouped
                up-then-down.

        Returns:
            ``(orb_up, orb_down)`` with shapes
            ``(n_det, n_up, n_up)`` and ``(n_det, n_down, n_down)``.
        """
        pc = _make_heg_phys_conf(r)
        _, fs, _nuc_params, emb = self.omni(pc)
        # Coord-transform backflow (Smith 2024): shift positions
        # before evaluating the orbital basis.
        if (getattr(self, 'coord_backflow', None) is not None
                and emb is not None):
            r_bf = r + self.coord_backflow(emb)
            pc = _make_heg_phys_conf(r_bf)
        orb = self.envelope(pc)
        orb_up, orb_down = jnp.split(orb, [self.n_up], axis=-1)
        orb_up = orb_up[:, :self.n_up]
        orb_down = orb_down[:, self.n_up:]
        if fs is not None:
            orb_up = apply_heg_backflow_mult(orb_up, fs[0])
            if orb_down.shape[-1] > 0:
                orb_down = apply_heg_backflow_mult(orb_down, fs[1])
        return orb_up, orb_down

    def __call__(self, r: jax.Array) -> Psi:
        """Evaluate the wavefunction.

        Args:
            r: Electron positions ``(n_elec, 3)`` in grouped
                (up-then-down) order.

        Returns:
            :class:`Psi` with ``sign`` (real) and ``log`` (real).
        """
        pc = _make_heg_phys_conf(r)

        # GNN + backflow (and optional deep jastrow).
        jastrow, fs, _nuc_params, emb = self.omni(pc)

        # Coord-transform backflow (Smith 2024 PRL 133 266504, eq. 19):
        # x_i = r_i + W_bf · h_i^(T).  When active, the orbital basis
        # is evaluated at the shifted positions.  Cusp + pair-Jastrow
        # always use the *original* r (the cusp condition is a
        # property of the *true* electron coordinates).  ``r_bf`` is
        # the backflow-wrapped quasiparticle coordinate, used by both
        # the envelope and the Smith deep Jastrow (eq. 20).
        if (getattr(self, 'coord_backflow', None) is not None
                and emb is not None):
            r_bf = r + self.coord_backflow(emb)
            pc_bf = _make_heg_phys_conf(r_bf)
        else:
            r_bf = r
            pc_bf = pc

        # Envelope orbitals.
        orb = self.envelope(pc_bf)  # (n_det, n_elec, n_up + n_down)

        # full_determinant=False convention: split columns, slice rows.
        orb_up, orb_down = jnp.split(orb, [self.n_up], axis=-1)
        orb_up = orb_up[:, :self.n_up]        # (n_det, n_up, n_up)
        orb_down = orb_down[:, self.n_up:]    # (n_det, n_dn, n_dn)

        # Apply multiplicative backflow.
        if fs is not None:
            orb_up = apply_heg_backflow_mult(orb_up, fs[0])
            if orb_down.shape[-1] > 0:
                orb_down = apply_heg_backflow_mult(orb_down, fs[1])

        # Multi-det Slater-determinant combination (sum pool).
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

        log_psi = jnp.log(jnp.abs(psi_sum)) + xs_shift
        sign_psi = jax.lax.stop_gradient(jnp.sign(psi_sum))

        # Cusp correction (electronic).  The smooth periodic norm
        # is now in Bohr units (see :func:`~.periodic.periodic_norm`
        # docstring) so the Kato cusp coefficients can be used
        # directly with it — no need for minimum-image Euclidean
        # distances, which had a gradient kink at the cell face.
        # Matches FermiNet PBC convention (pbc/feature_layer.py).
        if self.cusp_electrons is not None:
            dists_elec = _periodic_pair_distances_full(
                r, self.lattice,
            )
            same_dists = jnp.concatenate([
                triu_flat(dists_elec[idx, idx])
                for idx in self._spin_slices
            ], axis=-1)
            anti_dists = flatten(
                dists_elec[:self.n_up, self.n_up:],
            )
            log_psi = log_psi + self.cusp_electrons(
                same_dists, anti_dists,
            )

        # Deep jastrow from the GNN (sum over electrons after MLP).
        if jastrow is not None:
            log_psi = log_psi + jastrow

        # Smith 2024 deep Jastrow (eqs. 20-21): J(h_i^(T), Linear_pre(x_i)).
        # Uses both the one-body GNN stream and the backflow-shifted
        # positions r_bf — strictly more expressive than ``jastrow``
        # above (which only uses h_i).  Mutually exclusive with the
        # OmniNet jastrow at builder level: at most one is configured.
        if (getattr(self, 'smith_deep_jastrow', None) is not None
                and emb is not None):
            log_psi = log_psi + self.smith_deep_jastrow(emb, r_bf)

        # Optional extra pair Jastrow.
        if self.pair_jastrow is not None:
            log_psi = log_psi + self.pair_jastrow(
                r, self.lattice,
            )

        return Psi(sign_psi, log_psi)


class PeriodicElectronicCusp(nnx.Module):
    """Kato e-e cusp for HEG, hardened for periodic boundaries.

    Drop-in replacement for :class:`~.cusp.ElectronicCuspAsymptotic`
    with three structural fixes specific to periodic systems:

    1. **Softplus-parameterised α (bug fix)**.  The molecular cusp
       stores ``same_alpha, anti_alpha`` as raw ``nnx.Param`` values
       with no positivity constraint.  Adam / SR will happily push
       α → 0 (cusp collapses) or α < 0 (singular pole at
       ``r = −α > 0``, NaN).  Here α is always ``softplus(raw_α)``,
       so the effective α is strictly positive; the Kato slope at
       ``r = 0`` is ``scale`` independent of α, so enforcing
       positivity costs nothing variationally.

    2. **Smooth cell-face cutoff (bug fix)**.  The cusp is computed
       on minimum-image Euclidean distances, which have a
       discontinuous first derivative at the cell face
       (fractional coord ``±0.5``).  Walkers crossing a face see a
       jump in the local kinetic energy.  Multiplying the cusp by
       a cosine cutoff ``½(1 + cos(π r / r_cut))`` that vanishes
       at ``r_cut ≈ L/2`` removes the kink.  The cutoff has
       ``c(0) = 1``, ``c'(0) = 0``, so the Kato slope at the
       origin is preserved exactly.

    3. **Unchanged Kato physics**.  The functional form inside the
       cutoff is the PsiFormer cusp
       ``u(r) = −scale · α² / (α + r)`` with
       ``u'(0) = scale``.  Slopes ``same_scale = 0.25``,
       ``anti_scale = 0.5`` match the standard Kato condition.

    Note:
        This class does *not* fix the cusp-vs-Coulomb regularization
        inconsistency at exact coincidence — that fix lives in
        :func:`OmegaQMC.observables.ewald.ewald_pair_potential`,
        where the origin-image distance is given the same
        ``machine-eps`` floor that ``_pair_distances_mi_full`` uses.
        With both fixes in place, the cusp's ``(2/r) u'(0)``
        kinetic divergence is exactly cancelled by the Coulomb
        ``1/r`` divergence at coincidence, restoring variational
        validity of the initial state.
    """

    def __init__(
        self,
        *,
        L,
        same_scale: float = 0.25,
        anti_scale: float = 0.5,
        alpha_init: float = 1.0,
        r_cut_frac: float = 0.45,
        trainable_alpha: bool = True,
        softplus_alpha: bool = True,
    ):
        self.L = float(L)
        self.r_cut = float(r_cut_frac) * float(L)
        self.same_scale = float(same_scale)
        self.anti_scale = float(anti_scale)
        self.trainable_alpha = bool(trainable_alpha)
        self.softplus_alpha = bool(softplus_alpha)
        self._alpha_init = float(alpha_init)

        if self.trainable_alpha:
            if self.softplus_alpha:
                # Pick raw_init so softplus(raw_init) == alpha_init.
                # log(expm1(x)) is the stable inverse-softplus.
                raw_init = float(np.log(np.expm1(alpha_init)))
            else:
                raw_init = float(alpha_init)
            self.same_raw_alpha = nnx.Param(jnp.asarray(raw_init))
            self.anti_raw_alpha = nnx.Param(jnp.asarray(raw_init))

    def _effective_alpha(self, raw):
        if self.softplus_alpha:
            return jax.nn.softplus(raw)
        return raw

    def _u_cutoff(self, r, scale, alpha):
        """PsiformerCusp times smooth cell-face cutoff.

        Kato slope check — for the combined form
        ``u_eff(r) = u(r) · c(r)`` with
        ``u(r) = −scale · α² / (α + r)``,
        ``c(r) = ½(1 + cos(π r / r_cut))``:

            u'(0) = scale,   c(0) = 1,   c'(0) = 0
            u_eff'(0) = u'(0)·c(0) + u(0)·c'(0) = scale

        so the cutoff preserves the Kato condition exactly.
        """
        u = -scale * (alpha * alpha) / (alpha + r)
        cutoff = jnp.where(
            r < self.r_cut,
            0.5 * (1.0 + jnp.cos(jnp.pi * r / self.r_cut)),
            0.0,
        )
        return u * cutoff

    def __call__(self, same_dists, anti_dists):
        """Scalar cusp contribution to ``log|ψ|``.

        Args:
            same_dists: ``(n_pairs_same,)`` parallel-spin e-e
                distances (minimum-image Euclidean).
            anti_dists: ``(n_pairs_anti,)`` antiparallel-spin e-e
                distances.

        Returns:
            Scalar added to ``log|ψ|``.
        """
        cusp = jnp.asarray(0.0)
        if same_dists.size > 0:
            if self.trainable_alpha:
                alpha_s = self._effective_alpha(
                    param_value(self.same_raw_alpha),
                )
            else:
                alpha_s = jnp.asarray(self._alpha_init)
            cusp = cusp + self._u_cutoff(
                same_dists, self.same_scale, alpha_s,
            ).sum()
        if anti_dists.size > 0:
            if self.trainable_alpha:
                alpha_a = self._effective_alpha(
                    param_value(self.anti_raw_alpha),
                )
            else:
                alpha_a = jnp.asarray(self._alpha_init)
            cusp = cusp + self._u_cutoff(
                anti_dists, self.anti_scale, alpha_a,
            ).sum()
        return cusp


def _periodic_pair_distances_full(r, lattice):
    """Full ``(n_elec, n_elec)`` matrix of smooth periodic distances.

    Uses the Cassella smooth norm, which is ``~2π|r|`` at short
    range and smooth across the cell face — appropriate for GNN
    edge features where smoothness matters.

    Diagonals are zero (self-distance), as in
    :func:`~.physics.pairwise_self_distance(full=True)`.
    """
    n_elec = r.shape[-2]
    i, j = jnp.triu_indices(n_elec, k=1)
    diff = r[..., i, :] - r[..., j, :]
    d_up = periodic_norm(diff, lattice, safe=True)
    mat = (
        jnp.zeros((n_elec, n_elec), dtype=d_up.dtype)
        .at[i, j].set(d_up)
        .at[j, i].set(d_up)
    )
    return mat


def _pair_distances_mi_full(r, lattice):
    """Full ``(n_elec, n_elec)`` matrix of minimum-image Euclidean
    distances.

    Unlike :func:`_periodic_pair_distances_full` which uses the
    Cassella smooth norm (≈ 2π|r| at short range), this returns
    the physical Euclidean distance of the minimum-image pair
    vector.  Required by the Kato cusp correction, whose
    coefficients assume Bohr-scaled r: using the 2π-inflated
    smooth norm would shrink the useful cusp range by 2π and
    effectively disable the correction for HEG at rs ≳ 0.5.
    """
    from .periodic import minimum_image_diff as _mi
    n_elec = r.shape[-2]
    i, j = jnp.triu_indices(n_elec, k=1)
    diff = r[..., i, :] - r[..., j, :]
    diff_mi = _mi(diff, lattice)
    d = jnp.sqrt(
        jnp.sum(diff_mi ** 2, axis=-1)
        + jnp.finfo(diff_mi.dtype).eps
    )
    mat = (
        jnp.zeros((n_elec, n_elec), dtype=d.dtype)
        .at[i, j].set(d)
        .at[j, i].set(d)
    )
    return mat


# =====================================================================
# Builder: assemble a PsiFormer-style HEG WF from a config
# =====================================================================

def build_heg_psiformer_wf(
    config,
    rngs: nnx.Rngs,
) -> HEGPsiFormerWaveFunction:
    """Assemble an HEG PsiFormer from a :class:`HEGPsiFormerConfig`.

    Composes:
      * :class:`~.env_periodic.PlaneWaveEnvelope` (plane-wave
        orbital basis, free-electron init).
      * :class:`~.gnn.electron_gnn.ElectronGNN` with periodic edge
        features and type-only electron embedding.
      * :class:`~.omni.OmniNet` wiring (optional deep Jastrow head
        + backflow head).
      * Optional electronic cusp (periodic distances).
      * Optional extra pair Jastrow.

    Args:
        config: :class:`HEGPsiFormerConfig`.
        rngs: NNX RNG state.

    Returns:
        :class:`HEGPsiFormerWaveFunction`.
    """
    from .heg_wf import HEGPsiFormerConfig  # local import avoids cycle
    assert isinstance(config, HEGPsiFormerConfig)

    n_up = config.n_up
    n_down = config.n_down
    n_det = config.n_det
    L = float(config.L)
    n_elec = n_up + n_down
    emb_dim = config.embedding_dim
    tp_dim = config.two_particle_stream_dim
    dim = int(getattr(config, 'dim', 3))

    if dim == 3:
        lattice = make_cubic_lattice(L)
    elif dim == 2:
        from .periodic import make_square_lattice
        lattice = make_square_lattice(L)
    else:
        raise ValueError(f"config.dim must be 2 or 3, got {dim}")

    # --- Envelope: dispatch on config.envelope_type ---
    envelope_type = getattr(config, 'envelope_type', 'plane_wave')
    if envelope_type == 'plane_wave':
        # The PW basis is extended beyond the Fermi shell by
        # ``config.n_virt_pw`` functions so the multi-det expansion
        # has virtual orbitals to specialise into.  Without virtuals
        # (basis == Fermi shell exactly), every Slater determinant
        # over the occupied subspace is equal up to a constant, and
        # ``n_det > 1`` is structurally degenerate.  ``det_jitter``
        # seeds dets d>=1 with a small random perturbation so they
        # start as distinct particle-hole admixtures of the Fermi sea.
        pw_basis_size = max(n_up, n_down, 1) + int(config.n_virt_pw)
        envelope = PlaneWaveEnvelope(
            n_up=n_up, n_down=n_down, n_det=n_det, L=L,
            init_pw_count=pw_basis_size,
            det_jitter=config.det_jitter,
            dim=dim,
        )
    elif envelope_type == 'crystal_gaussian':
        if dim != 2:
            raise ValueError(
                "crystal_gaussian envelope requires dim=2 "
                f"(got dim={dim}).  Wigner-crystal ansatz is only "
                "implemented in 2D.",
            )
        from .env_localized_2d import GaussianLocalizedEnvelope2D
        # Recover r_s from the cell geometry: pi * rs^2 = A/N -> rs.
        rs = L / np.sqrt(np.pi * (n_up + n_down))
        envelope = GaussianLocalizedEnvelope2D(
            n_up=n_up, n_down=n_down, n_det=n_det,
            rs=float(rs), L=L,
            sigma_init=float(getattr(
                config, 'crystal_sigma_init', 0.25,
            )),
            spin_pattern=str(getattr(
                config, 'crystal_spin_pattern', 'neel',
            )),
            det_jitter=float(getattr(
                config, 'crystal_det_jitter', 0.0,
            )),
        )
    else:
        raise ValueError(
            f"Unknown envelope_type {envelope_type!r}; "
            f"expected 'plane_wave' or 'crystal_gaussian'.",
        )

    # --- Backbone dispatch: 'psiformer' (legacy attention GNN) vs.
    # 'mpnqs' (Pescia 2024 / Smith 2024 dual-stream message passing).
    # Embedding / per-layer / edge-feature modules below are
    # PsiFormer-specific and skipped entirely on the MP-NQS path —
    # the YAML fields ``layers``, ``heads``, ``two_particle_dim`` are
    # ignored when ``backbone='mpnqs'``.
    backbone = str(getattr(config, 'backbone', 'psiformer')).lower()
    if backbone == 'mpnqs':
        emb_dim = int(getattr(config, 'mpnqs_d1', 32))

    if backbone == 'psiformer':
        # --- Electron embedding: spin-typed, optionally ghost-atom enriched.
        electron_embedding = HEGElectronEmbedding(
            n_up=n_up, n_down=n_down,
            embedding_dim=emb_dim,
            lattice=lattice,
            use_ghost_atom=config.use_ghost_atom,
            rngs=rngs,
        )
    else:
        electron_embedding = None

    # --- Periodic edge features for 'same' and 'anti' edges ---
    #
    # log_rescale=True on the distance feature is critical: without
    # it the raw periodic_norm at typical HEG distances (|r| ~ 2 Bohr
    # → periodic_norm ~ 12; |r| ~ 4 Bohr → ~20) is ~10× the sin/cos
    # feature magnitude, unbalancing the MLP input and slowing
    # convergence.  log_rescale transforms the distance to
    # ``log(1 + r_p)`` which is smooth, bounded, and O(1) at typical
    # pair separations.
    def _periodic_ee_feature():
        return PeriodicCombinedEdgeFeature(features=[
            PeriodicSinCosFeature(lattice=lattice),
            PeriodicDistancePowerEdgeFeature(
                lattice=lattice, powers=[1], log_rescale=True,
            ),
        ])

    # --- PsiFormer-specific GNN scaffolding (skipped for MP-NQS) ---
    if backbone == 'psiformer':
        edge_types = ['same', 'anti']
        edge_features = {et: _periodic_ee_feature() for et in edge_types}
        # Dim of the 'same'/'anti' edge feature: 2*dim (sin+cos) + 1 (|r|).
        # 3D: 6 + 1 = 7; 2D: 4 + 1 = 5.
        ee_feat_dim = 2 * dim + 1

        layers = []
        for idx in range(config.n_interactions):
            layer = _build_heg_gnn_layer(
                config, idx=idx,
                emb_dim=emb_dim, tp_dim=tp_dim,
                edge_types=edge_types,
                ee_feat_dim=ee_feat_dim,
                n_up=n_up, n_down=n_down,
                rngs=rngs,
            )
            layers.append(layer)
    else:
        edge_types = None
        edge_features = None
        layers = None

    # --- Assemble GNN backbone.  The PsiFormer (single-stream
    # attention) path uses ElectronGNN; the MP-NQS path uses MPNqsGnn
    # which is a drop-in replacement (same GraphNodes interface).
    if backbone == 'mpnqs':
        from .gnn.mpnqs import MPNqsGnn
        gnn = MPNqsGnn(
            n_up=n_up, n_down=n_down,
            lattice=lattice,
            n_layers=int(getattr(config, 'mpnqs_n_layers', 4)),
            d1=int(getattr(config, 'mpnqs_d1', 32)),
            d2=int(getattr(config, 'mpnqs_d2', 26)),
            hidden=int(getattr(config, 'mpnqs_hidden', 32)),
            include_spin_in_v_ij=True,
            rngs=rngs,
        )
    else:
        gnn = ElectronGNN(
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

    # --- Backflow head (from OmniNet) ---
    backflow = None
    if config.use_backflow:
        n_bf = 1  # multiplicative-only
        bf_dict = {}
        if config.full_determinant:
            n_orb_up = n_orb_down = n_elec
        else:
            n_orb_up, n_orb_down = n_up, n_down
        for spin, n_orb in [('up', n_orb_up), ('down', n_orb_down)]:
            out_dim = n_orb * n_det
            nets = [
                MLP(
                    emb_dim, out_dim,
                    hidden_layers=config.bf_mlp_hidden_layers,
                    bias=False,
                    last_linear=True,
                    activation=None,
                    init='ferminet',
                    rngs=rngs,
                ),
            ]
            # Default variance-scaled init — the last layer is NOT
            # zero-initialised.  With zero/small-init the Jacobian
            # through earlier GNN layers is cut to essentially
            # zero, preventing training.  We instead use the
            # supervised pre-training stage to drive the
            # randomly-initialised backflow toward "outputs zero"
            # (i.e. reproduces HF on walker samples), which leaves
            # the GNN/backflow in a meaningfully-trained state
            # before energy VMC begins.  See
            # :mod:`OmegaQMC.pretrain_heg` for the pre-training
            # pipeline; without it the default init gives initial
            # Metropolis acceptance ≈ 0 and should not be used
            # for direct energy VMC.
            bf_dict[spin] = Backflow(
                n_orb, n_det, n_bf, spin,
                nets=nets,
            )
        backflow = bf_dict

    # --- Deep jastrow head (from OmniNet) ---
    # When ``use_smith_deep_jastrow=True``, the OmniNet-side jastrow is
    # disabled to avoid double-counting; the Smith variant takes its
    # place (built below, attached directly to the wavefunction
    # module rather than via OmniNet).
    deep_jastrow = None
    use_smith_dj = bool(getattr(config, 'use_smith_deep_jastrow', False))
    if config.use_deep_jastrow and not use_smith_dj:
        from .omni import Jastrow as _DeepJastrow
        jas_mlp = MLP(
            emb_dim, 1,
            hidden_layers=config.jas_mlp_hidden_layers,
            bias=config.jas_mlp_bias,
            last_linear=True,
            activation=config.jas_mlp_activation,
            init='ferminet',
            rngs=rngs,
        )
        if config.jas_mlp_zero_init_last:
            last = jas_mlp.layers[-1]
            last.kernel = nnx.Param(
                jnp.zeros_like(param_value(last.kernel)),
            )
            if last.use_bias:
                last.bias = nnx.Param(
                    jnp.zeros_like(param_value(last.bias)),
                )
        deep_jastrow = _DeepJastrow(
            net=jas_mlp, sum_first=False,
        )

    omni = OmniNet(
        n_up=n_up, gnn=gnn,
        jastrow=deep_jastrow,
        backflow=backflow,
        nuclear_gnn_head=None,
    )

    # --- Cusp ---
    # Uses :class:`PeriodicElectronicCusp` (defined in this module),
    # which wraps α in softplus so training cannot push it
    # non-positive, and multiplies the cusp by a smooth cosine
    # cutoff that vanishes at ``r_cut = r_cut_frac · L`` so the
    # cell-face kink of the minimum-image distance does not
    # propagate into the local energy.  See the class docstring
    # for the Kato-slope derivation.
    cusp_electrons = None
    if config.use_cusp:
        # Kato slope in d dimensions for 1/r Coulomb scales as
        # 1/(d-1): 3D anti-spin = 1/2, 3D same-spin = 1/4 (Pauli);
        # 2D anti-spin = 1, 2D same-spin = 1/2 (factor-of-two larger
        # because the radial volume element is 2*pi*r dr in 2D vs.
        # 4*pi*r^2 dr in 3D, so the kinetic-energy 1/r divergence
        # at coincidence is twice as severe).
        if dim == 3:
            same_scale, anti_scale = 0.25, 0.5
        else:  # dim == 2
            same_scale, anti_scale = 0.5, 1.0
        cusp_electrons = PeriodicElectronicCusp(
            L=L,
            same_scale=same_scale,
            anti_scale=anti_scale,
            alpha_init=1.0,
            r_cut_frac=0.45,
            trainable_alpha=bool(getattr(
                config, 'cusp_trainable_alpha', False,
            )),
            softplus_alpha=True,
        )

    # --- Optional extra pair Jastrow ---
    pair_jastrow = None
    if config.use_pair_jastrow:
        pair_jastrow = PeriodicPairJastrow(
            n_up, n_down,
            hidden=config.pair_jastrow_hidden,
            rngs=rngs,
        )

    # --- Optional Smith 2024 deep Jastrow (eqs. 20-21) ---
    # Takes (h_i^(T), Linear_pre(x_i)) — strictly more expressive than
    # the OmniNet ``deep_jastrow`` above, which uses h_i only.
    # Mutually exclusive with that head (handled at builder level).
    smith_deep_jastrow = None
    if use_smith_dj:
        smith_deep_jastrow = SmithDeepJastrow(
            d1=emb_dim, dim=dim,
            hidden=int(getattr(config, 'smith_jastrow_hidden', 32)),
            n_layers=int(getattr(config, 'smith_jastrow_n_layers', 4)),
            zero_init_last=True,
            rngs=rngs,
        )

    # --- Optional coord-transform backflow (Smith 2024) ---
    coord_backflow = None
    if getattr(config, 'use_coord_backflow', False):
        coord_backflow = nnx.Linear(
            in_features=emb_dim,
            out_features=dim,
            use_bias=False,
            rngs=rngs,
        )
        if getattr(config, 'coord_bf_zero_init', True):
            coord_backflow.kernel = nnx.Param(
                jnp.zeros_like(param_value(coord_backflow.kernel)),
            )

    return HEGPsiFormerWaveFunction(
        n_up=n_up, n_down=n_down, n_det=n_det, L=L,
        omni=omni, envelope=envelope, lattice=lattice,
        cusp_electrons=cusp_electrons,
        pair_jastrow=pair_jastrow,
        coord_backflow=coord_backflow,
        smith_deep_jastrow=smith_deep_jastrow,
    )


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


def _scale_last_layer_kernel(mlp, scale: float):
    """Rescale the final ``nnx.Linear`` kernel by *scale*.

    At ``scale = 0`` this zeros the kernel — but that kills
    gradient flow through the rest of the backflow network
    (``∂output/∂inputs = W_last = 0``), so only the last-layer
    weights can learn in the first few iterations.  A small
    non-zero ``scale`` (1e-2 to 1e-3) keeps the initial backflow
    multiplicative factor close to 1 while preserving
    differentiable dependence on the GNN/attention weights.

    This matches FermiNet / PsiFormer's standard practice of
    initialising the backflow "small but non-zero".
    """
    last = mlp.layers[-1]
    k = last.kernel.value
    last.kernel = nnx.Param(scale * k)


class _HEGMolInfoShim:
    """Lightweight stand-in for ``Mole_custom`` exposing just the
    fields that :class:`ElectronGNN` reads.

    HEG has no nuclei, so ``charges`` is an empty array and
    ``n_nuc = 0``.  The GNN does not actually need ``charges`` when
    ``edge_types`` excludes nucleus-involving edges and no nuclear
    embedding is used.
    """

    def __init__(self, *, n_up, n_down):
        self.n_up = n_up
        self.n_down = n_down
        self.charges = jnp.zeros((0,), dtype=jnp.float64)


# =====================================================================
# Complex (twist-aware) PsiFormer variant
# =====================================================================

class HEGPsiFormerWaveFunctionComplex(nnx.Module):
    """Complex-valued PsiFormer HEG wavefunction for TABC.

    Identical architecture to :class:`HEGPsiFormerWaveFunction` but
    uses :class:`~.env_periodic.ComplexPlaneWaveEnvelope` so the
    orbital matrix — and therefore ``log ψ`` — is complex-valued.
    The GNN and backflow remain real-valued (they operate on real
    electron coordinates and produce a real multiplicative backflow
    factor); the complex nature enters only through the envelope.

    Returns a complex scalar ``log ψ = log|ψ| + i·arg(ψ)``, matching
    the convention used by :class:`~.heg_wf.HEGSlaterJastrowComplex`
    so the twist-aware VMC driver consumes either transparently.
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
    ):
        self.n_up = n_up
        self.n_down = n_down
        self.n_det = n_det
        self.L = float(L)
        self.lattice = nnx.data(lattice)
        self.omni = omni
        self.envelope = envelope
        self.cusp_electrons = cusp_electrons
        self.pair_jastrow = pair_jastrow
        self.coord_backflow = coord_backflow
        self.smith_deep_jastrow = smith_deep_jastrow

    @property
    def _spin_slices(self):
        return (
            slice(None, self.n_up),
            slice(self.n_up, None),
        )

    def __call__(self, r: jax.Array) -> jax.Array:
        pc = _make_heg_phys_conf(r)

        # GNN + backflow (and optional deep jastrow) — real-valued.
        jastrow, fs, _nuc_params, emb = self.omni(pc)

        # Coord-transform backflow (Smith 2024).  Same as the real
        # Γ-point variant.
        if (getattr(self, 'coord_backflow', None) is not None
                and emb is not None):
            r_bf = r + self.coord_backflow(emb)
            pc_bf = _make_heg_phys_conf(r_bf)
        else:
            pc_bf = pc

        # Envelope orbitals — complex.
        orb = self.envelope(pc_bf)  # complex (n_det, n_elec, n_up+n_down)

        orb_up, orb_down = jnp.split(orb, [self.n_up], axis=-1)
        orb_up = orb_up[:, :self.n_up]
        orb_down = orb_down[:, self.n_up:]

        # Apply multiplicative backflow (real factor × complex orb).
        if fs is not None:
            orb_up = apply_heg_backflow_mult(orb_up, fs[0])
            if orb_down.shape[-1] > 0:
                orb_down = apply_heg_backflow_mult(orb_down, fs[1])

        # Complex Slater determinant — slogdet handles complex inputs.
        sign_u, log_u = eval_log_slater(orb_up)
        sign_d, log_d = eval_log_slater(orb_down)
        sign = sign_u * sign_d      # (n_det,) unit-magnitude complex
        xs = log_u + log_d          # (n_det,) real

        xs_shift = jnp.max(xs)
        xs_shift = jnp.where(
            jnp.isfinite(xs_shift),
            xs_shift, jnp.zeros_like(xs_shift),
        )
        xs_exp = sign * jnp.exp(xs - xs_shift)  # complex (n_det,)
        psi_sum = jnp.sum(xs_exp)               # complex scalar

        # Complex log ψ = log|ψ| + i·arg(ψ) + xs_shift (real offset).
        log_psi_complex = jnp.log(psi_sum) + xs_shift

        # Cusp + jastrow are real — they add to Re(log ψ) only.
        if self.cusp_electrons is not None:
            dists_elec = _periodic_pair_distances_full(
                r, self.lattice,
            )
            same_dists = jnp.concatenate([
                triu_flat(dists_elec[idx, idx])
                for idx in self._spin_slices
            ], axis=-1)
            anti_dists = flatten(
                dists_elec[:self.n_up, self.n_up:],
            )
            log_psi_complex = log_psi_complex + self.cusp_electrons(
                same_dists, anti_dists,
            )
        if jastrow is not None:
            log_psi_complex = log_psi_complex + jastrow
        if self.pair_jastrow is not None:
            log_psi_complex = log_psi_complex + self.pair_jastrow(
                r, self.lattice,
            )

        return log_psi_complex


def build_heg_psiformer_wf_complex(
    config,
    rngs: nnx.Rngs,
    *,
    kappa=None,
):
    """Assemble a complex (twist-aware) PsiFormer HEG wavefunction.

    Shares the GNN / backflow / cusp / jastrow construction logic
    with :func:`build_heg_psiformer_wf` — only the envelope and the
    wrapping wavefunction class differ.  The built model returns a
    *complex* scalar ``log psi`` suitable for the TABC driver.

    Args:
        config: :class:`~.heg_wf.HEGPsiFormerConfig`.
        rngs: NNX RNG state.
        kappa: Twist angle in fractional coordinates ``[-0.5, 0.5)``.
            Shape ``(dim,)`` matching ``config.dim``.  Default
            ``None`` -> Gamma point.
    """
    dim = int(getattr(config, 'dim', 3))
    if kappa is None:
        kappa = (0.0,) * dim
    if dim == 3:
        lattice = make_cubic_lattice(float(config.L))
    elif dim == 2:
        from .periodic import make_square_lattice
        lattice = make_square_lattice(float(config.L))
    else:
        raise ValueError(f"config.dim must be 2 or 3, got {dim}")

    # Build the real-valued PsiFormer to get all the GNN / backflow /
    # cusp / jastrow modules, then swap in a complex envelope and
    # rewrap.  This guarantees the non-envelope parameter pytree is
    # structurally identical between the real Gamma-point training
    # ansatz and the per-twist complex evaluation ansatz, so
    # ``transfer_trained_params`` can copy everything except the
    # envelope block.
    base = build_heg_psiformer_wf(config, rngs)
    complex_envelope = ComplexPlaneWaveEnvelope(
        n_up=config.n_up, n_down=config.n_down,
        n_det=config.n_det, L=float(config.L),
        kappa=kappa, dim=dim,
    )
    return HEGPsiFormerWaveFunctionComplex(
        n_up=config.n_up, n_down=config.n_down,
        n_det=config.n_det, L=float(config.L),
        omni=base.omni,
        envelope=complex_envelope,
        lattice=lattice,
        cusp_electrons=base.cusp_electrons,
        pair_jastrow=base.pair_jastrow,
    )
