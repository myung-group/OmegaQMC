"""Plane-wave envelopes for HEG neural-network wavefunctions.

Provides :class:`PlaneWaveEnvelope`, a drop-in replacement for
:class:`~.env.ExponentialEnvelopes` on simulation cells with
periodic boundary conditions.  The envelope produces orbital
matrices of shape ``(n_det, n_elec, n_orb)`` that are linear
combinations of real plane waves (``cos(k·r)`` and ``sin(k·r)``).

At the Γ point the envelope is real-valued, and the default
initialization sets each orbital to a single free-electron
plane wave ordered by increasing kinetic energy.  The resulting
Slater determinant of the initial envelope is the Hartree–Fock
Fermi sea of the non-interacting HEG — a well-defined and easily
reproducible reference for validation.

Complex envelopes (needed for twist-averaged boundary conditions)
are a follow-up: the module is structured so ``coeff_sin`` can be
swapped for a phase-enabled complex version without touching the
rest of the ansatz stack.
"""

from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from .compat import param_value


# ---------------------------------------------------------------------
# Real plane-wave basis enumeration
# ---------------------------------------------------------------------

class RealPWBasis(NamedTuple):
    """Real plane-wave basis on a cubic reciprocal grid.

    Attributes:
        kvecs: ``(n_pw, 3)`` array of Cartesian k-vectors.
        basis_idx: ``(n_orb,)`` index into ``kvecs`` for each basis
            function.
        basis_is_sin: ``(n_orb,)`` 0 for cos, 1 for sin.
        k_sq: ``(n_pw,)`` squared k-magnitudes (non-interacting
            single-particle energies up to a factor of 1/2).
    """

    kvecs: jax.Array
    basis_idx: jax.Array
    basis_is_sin: jax.Array
    k_sq: jax.Array


def enumerate_real_pw_basis(
    n_orb: int, L: float,
) -> RealPWBasis:
    """Build a real plane-wave basis on a cubic lattice of side L.

    Returns the first *n_orb* real basis functions in order of
    ascending single-particle kinetic energy:

        1 (= cos(0·r)),   cos(k₁·r), sin(k₁·r),
        cos(k₂·r), sin(k₂·r),   …

    where the k's are integer multiples of ``2π/L`` sorted by |k|².
    One representative is chosen for each (+k, -k) pair: the
    lexicographically-positive member.  At closed-shell fills
    (``n_orb`` ∈ {1, 7, 19, 27, 33, 57, …}) the basis covers exactly
    the filled shells of the free-electron Fermi sea.

    Args:
        n_orb: Number of basis functions required (must be positive).
        L: Side length of the cubic simulation cell.

    Returns:
        :class:`RealPWBasis`.
    """
    if n_orb < 1:
        raise ValueError(f"n_orb must be ≥ 1, got {n_orb}")

    dk = 2.0 * np.pi / L
    # Shell enumeration buffer: safely larger than needed.
    nmax = int(np.ceil((n_orb / 2.0) ** (1.0 / 3.0))) + 4

    # Collect all integer k-points, sorted by |n|² then lexicographic.
    pts = []
    for nx in range(-nmax, nmax + 1):
        for ny in range(-nmax, nmax + 1):
            for nz in range(-nmax, nmax + 1):
                pts.append((nx * nx + ny * ny + nz * nz,
                            (nx, ny, nz)))
    pts.sort(key=lambda t: (t[0], t[1]))

    # Build pairs of ±k representatives in order.  Each +k pair
    # contributes (cos, sin) basis functions; k=0 contributes only
    # cos (the constant).
    kvecs = []
    basis_idx = []
    basis_is_sin = []
    k_sq = []
    seen = set()

    def _is_positive_rep(n):
        # Lexicographic "positive" — strictly first nonzero > 0.
        for x in n:
            if x != 0:
                return x > 0
        return True  # zero vector counts as positive rep

    for n2, n in pts:
        if n in seen or tuple(-x for x in n) in seen:
            continue
        if not _is_positive_rep(n):
            continue
        idx = len(kvecs)
        kvecs.append((n[0] * dk, n[1] * dk, n[2] * dk))
        k_sq.append(n2 * dk * dk)
        seen.add(n)
        if n == (0, 0, 0):
            # k=0: only the constant (cos).
            basis_idx.append(idx)
            basis_is_sin.append(0)
        else:
            # Paired (+k): cos then sin.
            basis_idx.append(idx)
            basis_is_sin.append(0)
            basis_idx.append(idx)
            basis_is_sin.append(1)
        if len(basis_idx) >= n_orb:
            break

    if len(basis_idx) < n_orb:
        raise RuntimeError(
            f"Could not enumerate {n_orb} basis functions "
            f"within nmax={nmax}. Increase nmax buffer."
        )

    # Truncate to exactly n_orb (last shell may overshoot by 1).
    basis_idx = basis_idx[:n_orb]
    basis_is_sin = basis_is_sin[:n_orb]

    return RealPWBasis(
        kvecs=jnp.asarray(kvecs, dtype=jnp.float64),
        basis_idx=jnp.asarray(basis_idx, dtype=jnp.int32),
        basis_is_sin=jnp.asarray(basis_is_sin, dtype=jnp.int32),
        k_sq=jnp.asarray(k_sq, dtype=jnp.float64),
    )


# ---------------------------------------------------------------------
# Plane-wave envelope (real, Γ-point)
# ---------------------------------------------------------------------

class PlaneWaveEnvelope(nnx.Module):
    """Real plane-wave envelope at the Γ point.

    For each orbital ``i`` and determinant ``d`` the envelope is

        φ_{d,i}(r) = Σ_m [ ν_{d,i,m} cos(k_m·r) + μ_{d,i,m} sin(k_m·r) ]

    where ``{k_m}`` are the Cartesian k-vectors of a real plane-wave
    basis (see :func:`enumerate_real_pw_basis`).  The coefficients
    ``ν``, ``μ`` are trainable parameters initialised so that orbital
    ``i`` equals the ``i``-th single-particle free-electron wave
    function (constant, cos, sin, …).

    The output shape is ``(n_det, n_elec, n_orb_total)`` with
    ``n_orb_total = n_up + n_down`` — matching the molecular envelope
    so that :class:`~.wf.NeuralNetworkWaveFunction` (with
    ``full_determinant=False``) can consume it unchanged.

    Args:
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        n_det: Number of determinants.
        L: Cubic simulation cell side length.
        spin_restricted: If True, share the basis between up and
            down spins at initialization (the coefficient tensors
            are still separate parameters so SR can break the
            symmetry during training).
        init_pw_count: Number of plane waves to include in the
            basis; must be at least ``max(n_up, n_down)``.  Default
            is ``max(n_up, n_down)`` (closed-shell Fermi sea only).
    """

    def __init__(
        self,
        n_up: int,
        n_down: int,
        n_det: int,
        L: float,
        *,
        spin_restricted: bool = True,
        init_pw_count: Optional[int] = None,
        det_jitter: float = 0.0,
    ):
        if init_pw_count is None:
            init_pw_count = max(n_up, n_down, 1)

        basis = enumerate_real_pw_basis(init_pw_count, L)
        self.n_up = n_up
        self.n_down = n_down
        self.n_det = n_det
        self.spin_restricted = spin_restricted
        self.n_pw = int(basis.kvecs.shape[0])
        self._det_jitter = float(det_jitter)

        # Static: k-vectors (wrapped for NNX pytree compliance).
        self.kvecs = nnx.data(basis.kvecs)

        # Coefficients: (n_det, n_orb_spin, n_pw).  ``det_jitter``
        # adds a Gaussian perturbation to dets d≥1 so the multi-det
        # expansion is not structurally degenerate — see the class
        # docstring and :class:`HEGPsiFormerConfig` notes for why
        # this matters even when ``init_pw_count`` > the Fermi
        # shell.  Det 0 stays at the pure Fermi-sea reference.
        coeff_cos_up = self._init_coeffs(
            n_det, n_up, basis, det_jitter=det_jitter,
        )
        coeff_sin_up = self._init_coeffs(
            n_det, n_up, basis, take_sin=True, det_jitter=det_jitter,
        )

        self.coeff_cos_up = nnx.Param(coeff_cos_up)
        self.coeff_sin_up = nnx.Param(coeff_sin_up)

        if spin_restricted:
            # Down spins will read the up coefficients (for n_up=n_down)
            # via _call_one_spin.  We still store independent params
            # for the down channel so SR can break symmetry.
            if n_down > 0:
                coeff_cos_dn = self._init_coeffs(
                    n_det, n_down, basis, det_jitter=det_jitter,
                )
                coeff_sin_dn = self._init_coeffs(
                    n_det, n_down, basis, take_sin=True,
                    det_jitter=det_jitter,
                )
                self.coeff_cos_dn = nnx.Param(coeff_cos_dn)
                self.coeff_sin_dn = nnx.Param(coeff_sin_dn)
        else:
            if n_down > 0:
                coeff_cos_dn = self._init_coeffs(
                    n_det, n_down, basis, det_jitter=det_jitter,
                )
                coeff_sin_dn = self._init_coeffs(
                    n_det, n_down, basis, take_sin=True,
                    det_jitter=det_jitter,
                )
                self.coeff_cos_dn = nnx.Param(coeff_cos_dn)
                self.coeff_sin_dn = nnx.Param(coeff_sin_dn)

        self._has_down = n_down > 0

    @staticmethod
    def _init_coeffs(
        n_det: int, n_orb: int, basis: RealPWBasis,
        *, take_sin: bool = False,
        jitter: float = 0.0,
        det_jitter: float = 0.0,
    ) -> jax.Array:
        """Initialise coefficients to free-electron basis functions.

        Places a 1 at the entry corresponding to each orbital's
        natural plane wave (cos or sin) and zeros elsewhere.  Det
        0 is the exact free-electron (Fermi sea) Slater determinant
        — so the non-interacting energy is reproducible.

        For orbital ``i`` with ``basis_is_sin[i] == 0``: the cos
        tensor gets a 1 at column ``basis_idx[i]`` and the sin
        tensor stays zero.  Inverted for ``basis_is_sin[i] == 1``.

        Args:
            n_det: Number of determinants.
            n_orb: Number of occupied orbitals per determinant.
            basis: Precomputed real-PW basis.
            take_sin: Selects the sin channel of coefficient
                storage (otherwise cos).
            jitter: Global Gaussian noise applied to *all*
                determinants, including det 0.  Default 0; mostly
                useful for numerical-robustness tests.
            det_jitter: Gaussian noise applied *only* to
                determinants ``d ≥ 1``.  Det 0 stays at exact HF.
                Breaks the multi-det symmetry — without this (and
                without the jitter path below), all dets are
                identical at init, move in lockstep, and are
                equivalent to ``n_det = 1``.  A value of ``~0.02``
                is enough to seed distinct trajectories;
                larger values drive the higher dets further from
                the Fermi sea.  Random state is seeded by
                ``take_sin`` so cos and sin channels get
                independent noise.
        """
        n_pw = basis.kvecs.shape[0]
        coeff = np.zeros((n_det, n_orb, n_pw), dtype=np.float64)
        is_sin = np.asarray(basis.basis_is_sin[:n_orb])
        idx = np.asarray(basis.basis_idx[:n_orb])
        for i in range(n_orb):
            if take_sin and is_sin[i] == 1:
                coeff[:, i, idx[i]] = 1.0
            elif (not take_sin) and is_sin[i] == 0:
                coeff[:, i, idx[i]] = 1.0
        # Per-determinant symmetry breaking.  Det 0 is preserved as
        # the exact Fermi-sea reference; dets d≥1 get a Gaussian
        # perturbation of size ``det_jitter``.  When ``n_pw > n_orb``
        # (i.e. the basis includes virtual PWs above the Fermi
        # shell) this perturbation mixes occupied with virtual
        # orbitals — a particle-hole-like excitation.  When
        # ``n_pw == n_orb`` it only rotates within the occupied
        # subspace, which is equivalent to HF up to a constant
        # and relies on downstream backflow/Jastrow to carry the
        # diversity; in that case set a larger ``init_pw_count``
        # in :class:`PlaneWaveEnvelope` so this jitter has virtual
        # room to excite into.
        if det_jitter > 0.0 and n_det > 1:
            rng = np.random.default_rng(17 + int(take_sin))
            coeff[1:] += det_jitter * rng.normal(
                size=(n_det - 1, n_orb, n_pw),
            )
        if jitter > 0.0:
            coeff += jitter * np.random.default_rng(
                23 + int(take_sin),
            ).normal(size=coeff.shape)
        return jnp.asarray(coeff, dtype=jnp.float64)

    def _call_one_spin(
        self, r: jax.Array, cos_kr: jax.Array, sin_kr: jax.Array,
        coeff_cos: jax.Array, coeff_sin: jax.Array,
    ) -> jax.Array:
        """Evaluate orbitals for a single spin channel.

        Args:
            r: Electron positions for this spin, ``(n_spin, 3)``.
            cos_kr: ``cos(k_m · r_i)``, shape ``(n_spin, n_pw)``.
            sin_kr: ``sin(k_m · r_i)``, shape ``(n_spin, n_pw)``.
            coeff_cos: ``(n_det, n_orb, n_pw)`` cosine coefficients.
            coeff_sin: ``(n_det, n_orb, n_pw)`` sine coefficients.

        Returns:
            Orbital matrix ``(n_det, n_spin, n_orb)``.
        """
        # (n_det, n_orb, n_pw) · (n_spin, n_pw) -> (n_det, n_spin, n_orb)
        orb_cos = jnp.einsum('dim,em->dei', coeff_cos, cos_kr)
        orb_sin = jnp.einsum('dim,em->dei', coeff_sin, sin_kr)
        return orb_cos + orb_sin

    def __call__(self, phys_conf, nuc_params=None):
        """Evaluate the envelope at *phys_conf.r*.

        Args:
            phys_conf: :class:`~.types.PhysicalConfiguration`-like
                NamedTuple exposing ``.r`` of shape ``(n_elec, 3)``.
                For HEG the ``.R`` field is ignored (may be empty).
            nuc_params: Unused — accepted for interface parity with
                :class:`ExponentialEnvelopes`.

        Returns:
            Orbital matrix ``(n_det, n_elec, n_up + n_down)``.
            The first ``n_up`` columns are the up-spin orbital basis
            and the remaining columns are the down-spin basis, in the
            format expected by
            :class:`~.wf.NeuralNetworkWaveFunction` with
            ``full_determinant=False``.
        """
        del nuc_params
        r = phys_conf.r

        # Dot products with all k-vectors.  (n_elec, n_pw)
        kr_all = r @ self.kvecs.T
        cos_kr = jnp.cos(kr_all)
        sin_kr = jnp.sin(kr_all)

        r_up = r[:self.n_up]
        r_dn = r[self.n_up:]
        cos_up, sin_up = cos_kr[:self.n_up], sin_kr[:self.n_up]
        cos_dn, sin_dn = cos_kr[self.n_up:], sin_kr[self.n_up:]

        orb_up = self._call_one_spin(
            r_up, cos_up, sin_up,
            param_value(self.coeff_cos_up),
            param_value(self.coeff_sin_up),
        )

        n_elec = self.n_up + self.n_down

        if self._has_down:
            orb_dn = self._call_one_spin(
                r_dn, cos_dn, sin_dn,
                param_value(self.coeff_cos_dn),
                param_value(self.coeff_sin_dn),
            )
        else:
            orb_dn = jnp.zeros(
                (self.n_det, 0, self.n_up), dtype=orb_up.dtype,
            )

        # Arrange into ``(n_det, n_elec, n_orb_total)`` layout
        # expected by NeuralNetworkWaveFunction (full_determinant=False):
        # columns [0:n_up] are up-spin orbitals, columns [n_up:] are
        # down-spin orbitals.  Each spin's electrons occupy their
        # own row block.
        n_det = self.n_det
        n_up = self.n_up
        n_dn = self.n_down

        # Padded matrices that align rows and columns.
        # orb_up_padded: (n_det, n_elec, n_up) — up electrons active,
        #                down rows are zero (unused after col-split).
        orb_up_padded = jnp.zeros(
            (n_det, n_elec, n_up), dtype=orb_up.dtype,
        )
        orb_up_padded = orb_up_padded.at[:, :n_up, :].set(orb_up)

        orb_dn_padded = jnp.zeros(
            (n_det, n_elec, n_dn), dtype=orb_up.dtype,
        )
        if n_dn > 0:
            orb_dn_padded = orb_dn_padded.at[:, n_up:, :].set(orb_dn)

        return jnp.concatenate(
            [orb_up_padded, orb_dn_padded], axis=-1,
        )


# ---------------------------------------------------------------------
# Complex plane-wave basis and envelope (for twist-averaged BC)
# ---------------------------------------------------------------------

class ComplexPWBasis(NamedTuple):
    """Complex plane-wave basis at an arbitrary twist ``κ``.

    Attributes:
        kvecs: ``(n_orb, 3)`` Cartesian k-vectors
            ``k_m = (n_m + κ) · 2π/L``.
        k_sq: ``(n_orb,)`` squared magnitudes of ``kvecs``.
        n_ints: ``(n_orb, 3)`` integer triples of the selected orbitals.
        kappa: ``(3,)`` twist angle in fractional coordinates.
    """

    kvecs: jax.Array
    k_sq: jax.Array
    n_ints: jax.Array
    kappa: jax.Array


def enumerate_complex_pw_basis(
    n_orb: int, L: float,
    kappa=(0.0, 0.0, 0.0),
) -> ComplexPWBasis:
    """Enumerate the lowest-|k+κ|² complex plane waves on a cubic cell.

    At κ = 0 the function reproduces the non-interacting Γ-point
    Fermi-sea ordering: orbital 0 is k=0, then the six first-shell
    vectors ``±e_x, ±e_y, ±e_z`` (any closed-shell fill of 7, 19,
    27, … electrons per spin is returned correctly).  At κ ≠ 0 the
    shell degeneracy is broken and each orbital has a unique
    ``|k|²``.

    Args:
        n_orb: Number of k-vectors to return.
        L: Cubic simulation-cell side length.
        kappa: Twist angle (3,) in fractional coordinates
            ``[-0.5, 0.5)``.

    Returns:
        :class:`ComplexPWBasis`.
    """
    if n_orb < 1:
        raise ValueError(f"n_orb must be ≥ 1, got {n_orb}")
    kappa = np.asarray(kappa, dtype=np.float64)
    if kappa.shape != (3,):
        raise ValueError(f"kappa must have shape (3,), got {kappa.shape}")

    dk = 2.0 * np.pi / L
    # Enumerate integer triples inside a generous shell.
    # |n + κ|² grows ∝ (max |n|)², so take nmax ≈ (n_orb)^{1/3} + buffer.
    nmax = int(np.ceil(n_orb ** (1.0 / 3.0))) + 4
    rng = np.arange(-nmax, nmax + 1)
    nx, ny, nz = np.meshgrid(rng, rng, rng, indexing='ij')
    n_ints = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    # Shifted magnitudes in units of (2π/L).
    shifted = n_ints + kappa[None, :]
    shifted_sq = np.sum(shifted ** 2, axis=-1)

    # Stable sort: primary by |n+κ|², secondary by lexicographic n for
    # reproducibility when the twist is exactly zero and many triples
    # share |n|².
    lex_key = (
        n_ints[:, 0] * (2 * nmax + 1) ** 2
        + n_ints[:, 1] * (2 * nmax + 1)
        + n_ints[:, 2]
    )
    order = np.lexsort((lex_key, shifted_sq))
    order = order[:n_orb]

    selected_n = n_ints[order]
    kvecs = (selected_n.astype(np.float64) + kappa[None, :]) * dk
    k_sq = np.sum(kvecs ** 2, axis=-1)

    return ComplexPWBasis(
        kvecs=jnp.asarray(kvecs, dtype=jnp.float64),
        k_sq=jnp.asarray(k_sq, dtype=jnp.float64),
        n_ints=jnp.asarray(selected_n, dtype=jnp.int32),
        kappa=jnp.asarray(kappa, dtype=jnp.float64),
    )


class ComplexPlaneWaveEnvelope(nnx.Module):
    """Complex plane-wave envelope at an arbitrary twist ``κ``.

    Each orbital is

        φ_{d,i}(r) = Σ_m c_{d,i,m} · exp(i k_m · r)

    with ``k_m = (n_m + κ) · 2π/L`` and complex coefficients
    ``c_{d,i,m}`` initialised to the identity — orbital ``i`` of every
    determinant is exactly the ``i``-th single-particle plane wave of
    the twisted Fermi sea.  At κ = 0 the corresponding Slater
    determinant is equivalent (up to an overall global phase) to that
    of the real :class:`PlaneWaveEnvelope`, so the physical |ψ|² and
    all observables match to floating-point precision.

    Output shape: ``(n_det, n_elec, n_up + n_down)`` complex, matching
    the layout consumed by :class:`~.wf.NeuralNetworkWaveFunction`
    with ``full_determinant=False``.

    Args:
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        n_det: Number of determinants.
        L: Cubic simulation-cell side length.
        kappa: Twist ``(3,)`` in fractional coords ``[-0.5, 0.5)``.
            Default ``(0, 0, 0)`` (Γ point).
    """

    def __init__(
        self,
        n_up: int,
        n_down: int,
        n_det: int,
        L: float,
        *,
        kappa=(0.0, 0.0, 0.0),
    ):
        self.n_up = n_up
        self.n_down = n_down
        self.n_det = n_det
        self.L = float(L)

        n_pw = max(n_up, n_down, 1)
        basis = enumerate_complex_pw_basis(n_pw, L, kappa=kappa)
        self.n_pw = int(basis.kvecs.shape[0])
        self.kvecs = nnx.data(basis.kvecs)
        self.k_sq = nnx.data(basis.k_sq)
        self.kappa_vec = nnx.data(basis.kappa)
        self._has_down = n_down > 0

        # Identity-init complex coefficients.
        coeff_up_init = np.zeros(
            (n_det, n_up, self.n_pw), dtype=np.complex128,
        )
        for i in range(n_up):
            coeff_up_init[:, i, i] = 1.0 + 0.0j
        self.coeff_up = nnx.Param(jnp.asarray(coeff_up_init))

        if n_down > 0:
            coeff_dn_init = np.zeros(
                (n_det, n_down, self.n_pw), dtype=np.complex128,
            )
            for i in range(n_down):
                coeff_dn_init[:, i, i] = 1.0 + 0.0j
            self.coeff_dn = nnx.Param(jnp.asarray(coeff_dn_init))

    def __call__(self, phys_conf, nuc_params=None):
        """Evaluate the complex envelope at ``phys_conf.r``.

        Returns a complex ``(n_det, n_elec, n_up + n_down)`` tensor;
        the first ``n_up`` columns are the up-spin orbital basis and
        the remaining columns are the down-spin basis.
        """
        del nuc_params
        r = phys_conf.r  # (n_elec, 3), real

        # k·r for every walker electron and every k-vector.
        kr = r @ self.kvecs.T  # (n_elec, n_pw) real
        exp_ikr = jnp.exp(1j * kr)  # (n_elec, n_pw) complex

        # Up-spin orbitals: only evaluate on the up electrons.
        exp_up = exp_ikr[:self.n_up]  # (n_up, n_pw)
        orb_up = jnp.einsum(
            'dim,em->dei',
            param_value(self.coeff_up), exp_up,
        )  # (n_det, n_up, n_up)

        if self._has_down:
            exp_dn = exp_ikr[self.n_up:]  # (n_dn, n_pw)
            orb_dn = jnp.einsum(
                'dim,em->dei',
                param_value(self.coeff_dn), exp_dn,
            )  # (n_det, n_dn, n_dn)
        else:
            orb_dn = jnp.zeros(
                (self.n_det, 0, self.n_up), dtype=orb_up.dtype,
            )

        # Pad into the ``(n_det, n_elec, n_up + n_down)`` layout.
        n_det, n_up, n_dn = self.n_det, self.n_up, self.n_down
        n_elec = n_up + n_dn

        orb_up_padded = jnp.zeros(
            (n_det, n_elec, n_up), dtype=orb_up.dtype,
        )
        orb_up_padded = orb_up_padded.at[:, :n_up, :].set(orb_up)

        orb_dn_padded = jnp.zeros(
            (n_det, n_elec, n_dn), dtype=orb_up.dtype,
        )
        if n_dn > 0:
            orb_dn_padded = orb_dn_padded.at[:, n_up:, :].set(orb_dn)

        return jnp.concatenate(
            [orb_up_padded, orb_dn_padded], axis=-1,
        )
