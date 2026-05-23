"""Homogeneous-electron-gas neural-network wavefunction.

Minimal Slater–Jastrow ansatz on a cubic simulation cell with
periodic boundary conditions:

  * Orbital basis: real plane-wave envelope
    (:class:`~.env_periodic.PlaneWaveEnvelope`) initialised to the
    non-interacting Fermi sea.
  * Slater determinant per spin block (``full_determinant=False``
    convention so the up and down dets are independent).
  * Optional MLP-based Jastrow factor fed by the smooth periodic
    distances :func:`~.periodic.periodic_norm` between electron
    pairs.

Designed as the drop-in "trial" for :mod:`OmegaQMC.vmc_nn_heg`.
Layers from the existing molecular PsiFormer stack (attention,
backflow, deep Jastrow) are intentionally omitted from this first
version — they can be grafted on later once the plumbing is
validated against the non-interacting Fermi sea.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from .env_periodic import (
    ComplexPlaneWaveEnvelope,
    PlaneWaveEnvelope,
)
from .periodic import (
    PeriodicLattice,
    make_cubic_lattice,
    periodic_norm,
    periodic_norm_sq,
    minimum_image_diff,
)
from .types import PhysicalConfiguration, Psi
from .wf import eval_log_slater
from .layers import MLP


# ---------------------------------------------------------------------
# Lightweight phys-conf for HEG (no nuclei)
# ---------------------------------------------------------------------

def _make_heg_phys_conf(r: jax.Array) -> PhysicalConfiguration:
    """Wrap electron coords in a :class:`PhysicalConfiguration`.

    The HEG has no nuclei, so ``R`` is an empty ``(0, dim)`` array;
    ``dim`` is inferred from ``r.shape[-1]`` so the same helper works
    in 2D and 3D without modification.
    """
    dim = int(r.shape[-1])
    return PhysicalConfiguration(
        R=jnp.zeros((0, dim), dtype=r.dtype),
        r=r,
        mol_idx=jnp.asarray(0),
    )


# ---------------------------------------------------------------------
# Jastrow on periodic e-e distances
# ---------------------------------------------------------------------

class PeriodicPairJastrow(nnx.Module):
    """Sum-of-pairs Jastrow factor on the smooth periodic distance.

    ``J(r) = Σ_{i<j} u(|r_ij|_periodic)``

    with ``u`` a small MLP and ``|·|_periodic`` the smooth torus
    distance from :mod:`~.periodic`.  Separate MLPs are used for
    parallel and antiparallel spin pairs, mirroring the two-body
    Jastrow conventions in molecular QMC.

    Args:
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        hidden: Hidden-layer widths for the pair MLPs.
        rngs: NNX RNG state.
    """

    def __init__(
        self, n_up: int, n_down: int,
        *, hidden=(16, 16), rngs,
    ):
        self.n_up = n_up
        self.n_down = n_down
        self.u_same = MLP(
            1, 1,
            hidden_layers=list(hidden),
            bias=True,
            last_linear=True,
            activation='tanh',
            init='ferminet',
            rngs=rngs,
        )
        self.u_anti = MLP(
            1, 1,
            hidden_layers=list(hidden),
            bias=True,
            last_linear=True,
            activation='tanh',
            init='ferminet',
            rngs=rngs,
        )

    def __call__(
        self, r: jax.Array, lattice: PeriodicLattice,
    ) -> jax.Array:
        """Return the scalar Jastrow exponent at configuration *r*.

        Args:
            r: Electron positions ``(n_elec, 3)``.
            lattice: :class:`PeriodicLattice`.
        """
        n_up = self.n_up
        n_down = self.n_down

        def _pair_contrib(mlp, dists_sq):
            # dists_sq: (n_pairs,) — squared periodic distance,
            # even in r (→ 0 as r² → 0).  Feeding r² (not r)
            # makes the MLP's slope w.r.t. r structurally zero at
            # r = 0, so the Jastrow cannot compete with the
            # explicit cusp's Kato slope.  See the cusp/Jastrow
            # audit notes for the derivation.
            inp = dists_sq[:, None]
            return jnp.sum(mlp(inp))

        # Same-spin up-up
        total = jnp.asarray(0.0, dtype=r.dtype)
        if n_up > 1:
            i, j = jnp.triu_indices(n_up, k=1)
            d = r[i] - r[j]
            dist_sq = periodic_norm_sq(d, lattice)
            total = total + _pair_contrib(self.u_same, dist_sq)
        if n_down > 1:
            i, j = jnp.triu_indices(n_down, k=1)
            d = r[n_up + i] - r[n_up + j]
            dist_sq = periodic_norm_sq(d, lattice)
            total = total + _pair_contrib(self.u_same, dist_sq)
        if n_up > 0 and n_down > 0:
            # Antiparallel all-pairs
            ru = r[:n_up]
            rd = r[n_up:]
            d = ru[:, None, :] - rd[None, :, :]
            dist_sq = periodic_norm_sq(d, lattice).reshape(-1)
            total = total + _pair_contrib(self.u_anti, dist_sq)

        return total


# ---------------------------------------------------------------------
# HEG wavefunction
# ---------------------------------------------------------------------

class HEGSlaterJastrow(nnx.Module):
    """Minimal Slater–Jastrow HEG wavefunction.

    Args:
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        n_det: Number of Slater determinants (multi-det expansion
            is summed with uniform coefficients).
        L: Cubic simulation-cell side length.
        use_jastrow: Include the :class:`PeriodicPairJastrow`.
        jastrow_hidden: Hidden-layer widths for the Jastrow MLPs.
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        n_up: int,
        n_down: int,
        n_det: int,
        L: float,
        *,
        use_jastrow: bool = True,
        jastrow_hidden=(16, 16),
        dim: int = 3,
        rngs,
    ):
        self.n_up = n_up
        self.n_down = n_down
        self.n_det = n_det
        self.L = float(L)
        self.dim = int(dim)
        if dim == 3:
            self.lattice = nnx.data(make_cubic_lattice(L))
        elif dim == 2:
            from .periodic import make_square_lattice
            self.lattice = nnx.data(make_square_lattice(L))
        else:
            raise ValueError(f"dim must be 2 or 3, got {dim}")

        self.envelope = PlaneWaveEnvelope(
            n_up=n_up, n_down=n_down, n_det=n_det, L=L, dim=dim,
        )
        if use_jastrow:
            self.jastrow = PeriodicPairJastrow(
                n_up, n_down, hidden=jastrow_hidden, rngs=rngs,
            )
        else:
            self.jastrow = None

    def __call__(self, r: jax.Array) -> Psi:
        """Evaluate the wavefunction.

        Args:
            r: Electron positions ``(n_elec, 3)`` in the grouped
                (up-then-down) ordering used by the molecular
                PsiFormer stack.

        Returns:
            :class:`Psi` ``(sign, log)``.
        """
        pc = _make_heg_phys_conf(r)
        orb = self.envelope(pc)  # (n_det, n_elec, n_up+n_down)

        # full_determinant=False: split columns at n_up, slice rows.
        orb_up, orb_down = jnp.split(orb, [self.n_up], axis=-1)
        orb_up = orb_up[:, :self.n_up]          # (n_det, n_up, n_up)
        orb_down = orb_down[:, self.n_up:]      # (n_det, n_dn, n_dn)

        sign_u, log_u = eval_log_slater(orb_up)
        sign_d, log_d = eval_log_slater(orb_down)
        sign = sign_u * sign_d                   # (n_det,)
        xs = log_u + log_d                       # (n_det,)

        xs_shift = jnp.max(xs)
        xs_shift = jnp.where(
            jnp.isfinite(xs_shift), xs_shift, jnp.zeros_like(xs_shift),
        )
        xs_exp = sign * jnp.exp(xs - xs_shift)
        psi_sum = jnp.sum(xs_exp)
        log_psi = jnp.log(jnp.abs(psi_sum)) + xs_shift
        sign_psi = jax.lax.stop_gradient(jnp.sign(psi_sum))

        if self.jastrow is not None:
            j = self.jastrow(r, self.lattice)
            log_psi = log_psi + j

        return Psi(sign_psi, log_psi)


# ---------------------------------------------------------------------
# log_psi adapter
# ---------------------------------------------------------------------

class HEGConfig(NamedTuple):
    """HEG ansatz hyperparameters (minimal).

    Attributes:
        n_up: Spin-up electrons.
        n_down: Spin-down electrons.
        L: Cubic (3D) or square (2D) simulation-cell side length.
        n_det: Number of Slater determinants.
        use_jastrow: Enable the periodic pair Jastrow.
        jastrow_hidden: Hidden-layer widths for Jastrow MLPs.
        dim: Spatial dimension (3 for 3D HEG, 2 for 2D HEG).
            Defaults to 3 for backward compatibility.
    """

    n_up: int
    n_down: int
    L: float
    n_det: int = 1
    use_jastrow: bool = True
    jastrow_hidden: tuple = (16, 16)
    dim: int = 3


class HEGPsiFormerConfig(NamedTuple):
    """Hyperparameters for the full PsiFormer-style HEG ansatz.

    Strict superset of :class:`HEGConfig` — adds the GNN / attention
    / backflow knobs that turn the minimal Slater-Jastrow into a
    PsiFormer-class wavefunction.

    Attributes:
        n_up, n_down, L, n_det: System and Slater-det size.
        full_determinant: If True, each determinant uses
            ``n_elec`` orbitals (rather than per-spin).  False
            matches the existing Slater-Jastrow plumbing.
        embedding_dim: Per-electron embedding dimension in the GNN.
        n_interactions: Number of GNN / attention layers.
        two_particle_stream_dim: Edge-feature dimension after the
            first layer.
        n_attention_heads: Heads in each self-attention update
            feature.
        mlp_hidden_layers: Main per-layer MLP widths (e.g.
            ``['log', 2]``).
        g_mlp_hidden_layers: Two-particle-stream MLP widths.
        bf_mlp_hidden_layers: Backflow readout MLP widths.
        jas_mlp_hidden_layers: Deep-Jastrow MLP widths.
        use_backflow: Enable the per-orbital multiplicative backflow.
        use_cusp: Enable the electronic-cusp correction (smooth
            periodic distances).
        use_deep_jastrow: Enable the GNN-side scalar Jastrow head.
        use_pair_jastrow: Enable the extra explicit pair Jastrow
            (legacy two-body term on top of the deep stack).
        pair_jastrow_hidden: Widths for the pair Jastrow MLPs.
    """

    n_up: int
    n_down: int
    L: float
    L_y: Optional[float] = None     # if set, cell is rectangular with (L_x=L, L_y); else square
    n_det: int = 4
    full_determinant: bool = False
    embedding_dim: int = 64
    n_interactions: int = 2
    two_particle_stream_dim: int = 16
    n_attention_heads: int = 2
    mlp_hidden_layers: tuple = ('log', 2)
    g_mlp_hidden_layers: tuple = ('log', 1)
    bf_mlp_hidden_layers: tuple = ('log', 1)
    # Note: ``('log', 1)`` collapses to a single linear layer (no
    # activation slot), making the deep Jastrow degenerate even when
    # the builder supplies an activation.  Use ``('log', 2)`` or
    # larger to actually get a nonlinear Jastrow head.
    jas_mlp_hidden_layers: tuple = ('log', 2)
    # Deep Jastrow MLP details (used when ``use_deep_jastrow=True``).
    # With ``activation=None`` the whole MLP collapses to a linear
    # readout regardless of depth; default to ``'tanh'`` for a
    # genuinely nonlinear head.  ``zero_init_last=True`` initialises
    # the final layer weights/bias to zero so the deep Jastrow
    # contributes nothing at iter 0 — standard FermiNet/PsiFormer
    # practice, keeps the walker distribution close to the envelope
    # prior at start-up.
    jas_mlp_activation: Optional[str] = 'tanh'
    jas_mlp_bias: bool = True
    jas_mlp_zero_init_last: bool = True
    use_backflow: bool = True
    # Backbone choice.  'psiformer' (default) → single-stream attention
    # GNN (FermiNet/PsiFormer family).  'mpnqs' → dual-stream message-
    # passing (Pescia 2024 / Smith 2024) with persistent two-body
    # stream — better inductive bias for HEG-type pair-additive
    # interactions.  Both feed the same backflow / Jastrow / envelope.
    backbone: str = 'psiformer'
    # MP-NQS-specific dims (used when backbone='mpnqs'; Smith uses
    # d1=32, d2=26, hidden=32, T=4).
    mpnqs_d1: int = 32
    mpnqs_d2: int = 26
    mpnqs_hidden: int = 32
    mpnqs_n_layers: int = 4
    # Smith 2024 supplement (page 7) applies layer normalization in
    # the MPNN.  Bounds activation magnitude after each residual sum.
    # Critical when coord-backflow is enabled — without LayerNorm the
    # h_i^(T) features grow during training, making W_bf · h_i^(T)
    # large enough to collapse electron coordinates and produce
    # near-singular Slater determinants (a sub-DMC bias).
    mpnqs_use_layer_norm: bool = False
    # LayerNorm placement (only used when mpnqs_use_layer_norm=True):
    #   'post_each':       LN after residual sum every layer (most restrictive,
    #                      Smith's most-likely default, but over-bounds h_i^(T))
    #   'pre_each':        LN inputs to f1/f2 every layer (residuals can grow)
    #   'post_final_only': LN only on final h_i/h_ij (surgical: bounds the
    #                      input to BF readout without restricting per-layer
    #                      dynamics)
    mpnqs_layer_norm_mode: str = 'post_each'
    # Coord-transform backflow (Smith 2024 PRL 133 266504, eq. 19):
    # x_i = r_i + W_bf · h_i^(T).  When True, a small Linear readout
    # from the post-GNN one-body stream produces a per-electron
    # displacement that shifts positions before the orbital basis is
    # evaluated.  Multiplicative BF (use_backflow=True) and
    # coord-transform BF can be enabled simultaneously — they
    # represent different (non-redundant) families of correlations.
    use_coord_backflow: bool = False
    coord_bf_zero_init: bool = True   # zero-init last layer so initial Δr = 0
    # Smith 2024 deep Jastrow (eqs. 20-21): U(R) = Σ_i J(h_i, Linear_pre(x_i)).
    # When True, this REPLACES the standard ``use_deep_jastrow`` head —
    # builder logs an info message and disables the latter to avoid
    # double counting.
    use_smith_deep_jastrow: bool = False
    smith_jastrow_hidden: int = 32        # Smith's hidden width
    smith_jastrow_n_layers: int = 4       # Smith's L=4 (3 hidden)
    # ``use_cusp`` is ON by default.  Previously it was False
    # because the naïve PsiformerCusp implementation had three
    # separable bugs: (i) unconstrained trainable ``α`` that
    # could collapse toward 0 or go negative under Adam/SR,
    # (ii) gradient discontinuity at the simulation-cell face
    # from the minimum-image Euclidean distance, (iii) the Ewald
    # origin-image term was masked to 0 at coincidence while the
    # cusp still saw a regularized ``sqrt(eps)`` distance —
    # producing a spurious ``−1/√eps`` kinetic term and the
    # ``⟨E⟩ < E_DMC`` variational-violation at init.  All three
    # are now fixed (see
    # :class:`~OmegaQMC.psi.nn.heg_wf_module.PeriodicElectronicCusp`
    # for (i) softplus-parameterised α and (ii) smooth
    # cell-face cutoff, and
    # :func:`OmegaQMC.observables.ewald.ewald_pair_potential`
    # for (iii) consistent eps-floored origin distance), so
    # enabling the cusp gives a correctly Kato-cusped trial
    # wavefunction from iter 0.
    use_cusp: bool = True
    # Cusp α trainability.  Default False (locked to the analytic Kato
    # slope) — runaway α drift caused a variational catastrophe in
    # 100k-iter MP-NQS runs (E went unphysically below DMC, then var
    # exploded).  Smith uses an analytical u_2 with hard cusp
    # constraint; we mirror that here by default.  Set True only for
    # diagnostic exploration of cusp sensitivity.
    cusp_trainable_alpha: bool = False
    use_deep_jastrow: bool = False
    use_pair_jastrow: bool = False
    pair_jastrow_hidden: tuple = (16, 16)
    # Multi-determinant diversification.
    #
    # ``n_virt_pw``: extra plane-wave basis functions beyond the
    # Fermi sea.  With the default ``init_pw_count = max(n_up, n_down)``
    # the basis has zero virtual orbitals — the occupied-orbital
    # subspace spans the whole basis, and every Slater determinant
    # over that subspace is equal to every other up to a constant.
    # The ``n_det`` expansion is then structurally degenerate
    # regardless of random initialisation.  Adding at least one
    # full next shell of PWs (``+12`` for the cubic HEG at the N=14
    # closed shell) gives the expansion virtual room to specialise
    # into.
    #
    # ``det_jitter``: magnitude of a random Gaussian perturbation
    # applied to the envelope coefficients of determinants
    # ``d = 1, … n_det − 1`` at init.  Det 0 remains at the pure
    # Fermi-sea reference.  Perturbation is structural — it seeds
    # each higher det as a small particle-hole admixture — and is
    # what FermiNet-class wavefunctions rely on to get meaningful
    # multi-det diversity during training.
    n_virt_pw: int = 0
    det_jitter: float = 0.02
    # Ghost-atom (Z=0 phantom nucleus at cell origin) positional
    # fingerprint in the electron embedding.  Without this, same-spin
    # electrons get near-identical embeddings at init (the GNN
    # aggregation over neighbours averages out per-electron
    # differences in the bulk-like HEG walker), leaving the backflow
    # head with ~no signal to specialise per electron.  FermiNet's
    # HEG config (Cassella et al. 2022, ``ferminet/configs/heg.py``)
    # uses this trick — it structurally breaks the embedding
    # degeneracy at the architectural level and is the single
    # highest-impact departure from a molecular-style embedding.
    # Strongly recommended (and the default); the ``--pf-no-ghost-
    # atom`` CLI flag disables it for diagnostic comparisons.
    use_ghost_atom: bool = True
    # Spatial dimension: 3 (default) for 3D HEG; 2 for 2D HEG.
    # Selects the cubic-vs-square lattice and 3D-vs-2D plane-wave
    # basis enumeration in :func:`build_heg_psiformer_wf`.
    dim: int = 3
    # Envelope choice: 'plane_wave' (default — Slater determinant of
    # plane waves at the closed-shell Fermi sea, suitable for the
    # *fluid* phase of the HEG) or 'crystal_gaussian' (Slater
    # determinant of localised Gaussians on a triangular Bravais
    # lattice, suitable for the *Wigner crystal* phase of the 2D HEG
    # at r_s > r_s^c ~ 31).  Only 'crystal_gaussian' with dim=2 is
    # supported as the crystal envelope.
    envelope_type: str = 'plane_wave'
    # Crystal-envelope hyperparameters (used only when
    # envelope_type == 'crystal_gaussian').
    crystal_sigma_init: float = 0.25       # fraction of NN spacing
    crystal_spin_pattern: str = 'neel'     # 'neel' (AFM) or 'all_up' (FM)
    crystal_det_jitter: float = 0.0        # site-position jitter for det>=1
    crystal_lattice_type: str = 'triangular'  # 'triangular' | 'square' | 'stripe_x' | 'stripe_y'
    crystal_anisotropic_sigma: bool = False   # if True, per-site (σ_x, σ_y) variational — tests nematic distortion
    crystal_site_offset: float = 0.5          # fractional shift within unit cell (square lattice only); 0.5 = cell centers (bare WC), 0.0 = corners (matches cosine v_ext minima)
    # Walker initialisation strategy (consumed by the SR/eval drivers).
    # 'auto' (default) → crystal_perturbed when envelope is
    # crystal_gaussian, uniform otherwise.  'crystal_perturbed' forces
    # WC-position init even for plane-wave (fluid) envelopes — Smith
    # 2024 uses this for both phases (PRL 133, 266504, supplement page
    # 7).  'uniform' forces uniform init regardless of envelope.
    walker_init: str = 'auto'


def make_heg_log_psi(
    config: HEGConfig, rng_key: jax.Array,
):
    """Build a HEG log-ψ callable with the molecular-driver signature.

    Args:
        config: :class:`HEGConfig`.
        rng_key: JAX PRNG key.

    Returns:
        ``(log_psi, init_params, graphdef)`` with

        * ``log_psi(r, params) -> scalar`` — ``r`` is ``(n_elec, 3)``.
        * ``init_params`` — NNX ``State`` pytree (trainable).
        * ``graphdef`` — NNX ``GraphDef`` needed by ``nnx.merge``.
    """
    rngs = nnx.Rngs(rng_key)
    model = HEGSlaterJastrow(
        n_up=config.n_up,
        n_down=config.n_down,
        n_det=config.n_det,
        L=config.L,
        use_jastrow=config.use_jastrow,
        jastrow_hidden=config.jastrow_hidden,
        dim=getattr(config, 'dim', 3),
        rngs=rngs,
    )
    graphdef, params, other = nnx.split(model, nnx.Param, ...)

    def log_psi(r, params):
        mdl = nnx.merge(graphdef, params, other)
        return mdl(r).log

    return log_psi, params, graphdef


# ---------------------------------------------------------------------
# Complex (twist-averaged) variant
# ---------------------------------------------------------------------

class HEGSlaterJastrowComplex(nnx.Module):
    """Complex Slater–Jastrow HEG wavefunction for TABC.

    Identical structure to :class:`HEGSlaterJastrow` but uses
    :class:`ComplexPlaneWaveEnvelope` so the orbital matrix — and
    hence ``log ψ`` — is complex-valued.  The Jastrow factor is
    real (same MLP as the Γ-point path) and adds to ``Re log ψ``
    only.  Used by the twist-averaged VMC driver; at ``κ = 0`` the
    physical density matches the real path to floating-point
    precision (up to the basis-change normalization constant that
    drops out of |ψ|² ratios).

    Args:
        n_up, n_down, n_det, L: as :class:`HEGSlaterJastrow`.
        kappa: Twist ``(3,)`` in fractional coordinates
            ``[-0.5, 0.5)``.  Default ``(0, 0, 0)``.
        use_jastrow, jastrow_hidden, rngs: as
            :class:`HEGSlaterJastrow`.
    """

    def __init__(
        self,
        n_up: int,
        n_down: int,
        n_det: int,
        L: float,
        *,
        kappa=None,
        use_jastrow: bool = True,
        jastrow_hidden=(16, 16),
        dim: int = 3,
        rngs,
    ):
        if kappa is None:
            kappa = (0.0,) * dim
        self.n_up = n_up
        self.n_down = n_down
        self.n_det = n_det
        self.L = float(L)
        self.dim = int(dim)
        if dim == 3:
            self.lattice = nnx.data(make_cubic_lattice(L))
        elif dim == 2:
            from .periodic import make_square_lattice
            self.lattice = nnx.data(make_square_lattice(L))
        else:
            raise ValueError(f"dim must be 2 or 3, got {dim}")

        self.envelope = ComplexPlaneWaveEnvelope(
            n_up=n_up, n_down=n_down, n_det=n_det, L=L,
            kappa=kappa, dim=dim,
        )
        if use_jastrow:
            self.jastrow = PeriodicPairJastrow(
                n_up, n_down, hidden=jastrow_hidden, rngs=rngs,
            )
        else:
            self.jastrow = None

    def __call__(self, r: jax.Array) -> jax.Array:
        """Evaluate the wavefunction, returning complex ``log ψ``.

        Args:
            r: Electron positions ``(n_elec, 3)`` in grouped
                (up-then-down) order.

        Returns:
            Complex scalar ``log ψ = log|ψ| + i·arg(ψ)``.
        """
        pc = _make_heg_phys_conf(r)
        orb = self.envelope(pc)  # complex (n_det, n_elec, n_up+n_down)

        orb_up, orb_down = jnp.split(orb, [self.n_up], axis=-1)
        orb_up = orb_up[:, :self.n_up]
        orb_down = orb_down[:, self.n_up:]

        # slogdet on complex: sign is unit-modulus complex, log is real
        # (= log|det|).
        sign_u, log_u = eval_log_slater(orb_up)
        sign_d, log_d = eval_log_slater(orb_down)
        sign = sign_u * sign_d        # (n_det,) complex, |·| = 1
        xs = log_u + log_d            # (n_det,) real

        xs_shift = jnp.max(xs)
        xs_shift = jnp.where(
            jnp.isfinite(xs_shift), xs_shift, jnp.zeros_like(xs_shift),
        )
        xs_exp = sign * jnp.exp(xs - xs_shift)   # complex (n_det,)
        psi_sum = jnp.sum(xs_exp)                # complex scalar

        # log(complex psi) = log|psi| + i·arg(psi); add xs_shift to the
        # real part only.  ``jnp.log`` on complex input does exactly
        # that and is differentiable (with a branch cut at the
        # negative real axis which our sampler visits with measure
        # zero — wavefunction nodes).
        log_psi_complex = jnp.log(psi_sum) + xs_shift

        if self.jastrow is not None:
            # Jastrow is real-valued, adds to Re(log ψ) only.
            j = self.jastrow(r, self.lattice)
            log_psi_complex = log_psi_complex + j

        return log_psi_complex


def make_heg_log_psi_complex(
    config,
    rng_key: jax.Array,
    *,
    kappa=(0.0, 0.0, 0.0),
):
    """Build a complex HEG ``log ψ`` callable at a given twist.

    Dispatches on config type:
      * :class:`HEGConfig` → complex Slater-Jastrow
        (:class:`HEGSlaterJastrowComplex`).
      * :class:`HEGPsiFormerConfig` → complex PsiFormer
        (:class:`~.heg_wf_module.HEGPsiFormerWaveFunctionComplex`).

    In both cases the returned ``log_psi`` is complex-scalar-valued.
    Downstream code (kinetic energy, Metropolis) splits it into
    ``Re`` (= ``log|ψ|``, for MCMC acceptance) and ``Im`` (= phase,
    for the complex-kinetic formula).

    Args:
        config: :class:`HEGConfig` or :class:`HEGPsiFormerConfig`.
        rng_key: JAX PRNG key.
        kappa: Twist ``(3,)`` in fractional coordinates.

    Returns:
        ``(log_psi_complex, init_params, graphdef)``.
    """
    rngs = nnx.Rngs(rng_key)
    if isinstance(config, HEGPsiFormerConfig):
        from .heg_wf_module import build_heg_psiformer_wf_complex
        model = build_heg_psiformer_wf_complex(
            config, rngs, kappa=kappa,
        )
    elif isinstance(config, HEGConfig):
        model = HEGSlaterJastrowComplex(
            n_up=config.n_up,
            n_down=config.n_down,
            n_det=config.n_det,
            L=config.L,
            kappa=kappa,
            use_jastrow=config.use_jastrow,
            jastrow_hidden=config.jastrow_hidden,
            rngs=rngs,
        )
    else:
        raise TypeError(
            f"Unsupported config type: {type(config).__name__}. "
            "Expected HEGConfig or HEGPsiFormerConfig."
        )
    graphdef, params, other = nnx.split(model, nnx.Param, ...)

    def log_psi_complex(r, params):
        mdl = nnx.merge(graphdef, params, other)
        return mdl(r)  # complex scalar

    return log_psi_complex, params, graphdef


def transfer_jastrow_params(src_params, dst_params):
    """Copy Jastrow sub-parameters from *src_params* into *dst_params*.

    Used by the TABC driver: the Jastrow is trained once at Γ with
    the real ansatz; each per-twist evaluation builds a fresh
    :class:`HEGSlaterJastrowComplex` whose envelope coefficients are
    the Hartree–Fock identity init, but whose Jastrow block must
    match the trained values.

    Both pytrees must have been produced by
    :func:`make_heg_log_psi` / :func:`make_heg_log_psi_complex` with
    identical ``use_jastrow`` and ``jastrow_hidden`` settings; the
    envelope subtrees may differ.

    Implementation: walks both NNX ``State`` pytrees as plain pytrees
    (since they are valid JAX pytrees) and replaces every leaf whose
    path includes the string ``"jastrow"`` with the corresponding
    leaf from the source.

    Args:
        src_params: NNX ``State`` (or dict) containing trained
            Jastrow parameters.
        dst_params: Fresh NNX ``State`` into which the Jastrow
            subtree should be grafted.

    Returns:
        A new pytree identical to *dst_params* except the Jastrow
        leaves are replaced by those of *src_params*.
    """
    src_leaves = dict(jax.tree_util.tree_flatten_with_path(src_params)[0])
    dst_leaves, dst_def = jax.tree_util.tree_flatten_with_path(dst_params)
    new_leaves = []
    for path, leaf in dst_leaves:
        path_repr = _path_str(path)
        if 'jastrow' in path_repr and path in src_leaves:
            new_leaves.append(src_leaves[path])
        else:
            new_leaves.append(leaf)
    return jax.tree_util.tree_unflatten(dst_def, new_leaves)


def _path_str(path) -> str:
    """Stringify a JAX key path for substring matching."""
    parts = []
    for k in path:
        # DictKey/GetAttrKey/SequenceKey all have .key or .idx; fall
        # back to repr which includes the relevant token.
        parts.append(getattr(k, 'key', getattr(k, 'name', repr(k))))
    return '/'.join(str(p) for p in parts)


def transfer_trained_params(src_params, dst_params):
    """Copy all trained parameters *except* the envelope subtree.

    Used by the TABC driver for both the Slater-Jastrow and
    PsiFormer ansätze: the Γ-point trained parameters include the
    envelope coefficients, but those must be replaced by a fresh
    identity-init on the twisted k-grid at each twist.  All other
    parameters (Jastrow, GNN, attention, backflow, cusp, …) carry
    over unchanged.

    Identical interface to :func:`transfer_jastrow_params`.

    Args:
        src_params: Trained parameter pytree (from the optimizer).
        dst_params: Fresh parameter pytree from
            :func:`make_heg_log_psi_complex` at the target twist.

    Returns:
        A new pytree identical to *dst_params* except every non-
        envelope leaf is replaced by the corresponding source leaf.
    """
    src_leaves = dict(jax.tree_util.tree_flatten_with_path(src_params)[0])
    dst_leaves, dst_def = jax.tree_util.tree_flatten_with_path(dst_params)
    new_leaves = []
    for path, leaf in dst_leaves:
        path_repr = _path_str(path)
        if 'envelope' in path_repr:
            # Keep the fresh twisted envelope as-is.
            new_leaves.append(leaf)
        elif path in src_leaves:
            # Transfer every other matching leaf.
            new_leaves.append(src_leaves[path])
        else:
            new_leaves.append(leaf)
    return jax.tree_util.tree_unflatten(dst_def, new_leaves)


# ---------------------------------------------------------------------
# PsiFormer adapter
# ---------------------------------------------------------------------

def make_heg_psiformer_log_psi(
    config: 'HEGPsiFormerConfig', rng_key: jax.Array,
):
    """Build a PsiFormer-style HEG ``log|ψ|`` callable.

    Same signature as :func:`make_heg_log_psi`, so downstream VMC
    and optimizer drivers consume it unchanged — a HEGPsiFormer is
    drop-in for the minimal Slater-Jastrow.

    When ``config.use_coord_backflow`` is True, the returned ``log_psi``
    function gets an attribute ``log_psi.bf_diagnostics`` — a callable
    ``(walkers, params) -> dict`` that returns BF displacement and
    quasiparticle-separation statistics.  Used by the SR training loop
    to monitor for coord-collapse pathology (``min_pair_sep_min → 0``).

    Args:
        config: :class:`HEGPsiFormerConfig`.
        rng_key: JAX PRNG key.

    Returns:
        ``(log_psi, init_params, graphdef)``.
    """
    from .heg_wf_module import (
        build_heg_psiformer_wf, _make_heg_phys_conf,
        _pair_distances_mi_full,
    )

    rngs = nnx.Rngs(rng_key)
    model = build_heg_psiformer_wf(config, rngs)
    graphdef, params, other = nnx.split(model, nnx.Param, ...)

    def log_psi(r, params):
        mdl = nnx.merge(graphdef, params, other)
        return mdl(r).log

    # Attach BF diagnostics callable when coord backflow is active.
    # Walks through the GNN + coord_backflow on every walker, computes
    # per-walker statistics, then aggregates across walkers.
    has_bf = bool(getattr(config, 'use_coord_backflow', False))
    if has_bf:
        @jax.jit
        def bf_diagnostics(walkers, params):
            mdl = nnx.merge(graphdef, params, other)

            def per_walker(r):
                pc = _make_heg_phys_conf(r)
                _, _, _, emb = mdl.omni(pc)
                disp = mdl.coord_backflow(emb)        # (n_elec, dim)
                x_bf = r + disp
                disp_norms = jnp.linalg.norm(disp, axis=-1)  # (n_elec,)
                # Min-image pair distances of QUASIPARTICLES (post-BF).
                # Diagonal is 0 — we mask it before taking min.
                d_full = _pair_distances_mi_full(x_bf, mdl.lattice)
                n = x_bf.shape[0]
                # Set diagonal to +inf so min skips it.
                d_offdiag = d_full + jnp.eye(n, dtype=d_full.dtype) * 1e30
                return jnp.array([
                    jnp.mean(disp_norms),     # mean displacement / walker
                    jnp.max(disp_norms),      # max displacement / walker
                    jnp.min(d_offdiag),       # min quasiparticle pair sep
                ])

            # Flatten any leading axes (handles pmap shape
            # (n_dev, n_w/dev, n_elec, dim) as well as plain
            # (n_w, n_elec, dim)) so vmap iterates over walkers.
            n_elec = walkers.shape[-2]
            dim = walkers.shape[-1]
            walkers_flat = walkers.reshape(-1, n_elec, dim)
            stats = jax.vmap(per_walker)(walkers_flat)  # (n_w_total, 3)
            return {
                'mean_disp_avg':       jnp.mean(stats[:, 0]),
                'max_disp_avg':        jnp.mean(stats[:, 1]),
                'max_disp_max':        jnp.max(stats[:, 1]),
                'min_pair_sep_avg':    jnp.mean(stats[:, 2]),
                'min_pair_sep_min':    jnp.min(stats[:, 2]),
            }

        log_psi.bf_diagnostics = bf_diagnostics
    else:
        log_psi.bf_diagnostics = None

    return log_psi, params, graphdef


def make_heg_log_psi_any(config, rng_key):
    """Dispatch on config type to the right ``make_heg_log_psi_*``.

    Accepts a :class:`HEGConfig` (minimal Slater-Jastrow ansatz)
    or a :class:`HEGPsiFormerConfig` (PsiFormer ansatz).  Lets the
    VMC and optimiser drivers consume either config transparently.
    """
    if isinstance(config, HEGPsiFormerConfig):
        return make_heg_psiformer_log_psi(config, rng_key)
    if isinstance(config, HEGConfig):
        return make_heg_log_psi(config, rng_key)
    raise TypeError(
        f"Unsupported config type: {type(config).__name__}. "
        "Expected HEGConfig or HEGPsiFormerConfig."
    )
