"""Localised-Gaussian envelope for the 2D Wigner crystal phase.

For ``r_s > r_s^c ~ 31`` (Drummond-Needs 2009) the 2D HEG ground
state is a Wigner crystal: electrons localise on a triangular Bravais
lattice and develop long-range positional order.  The plane-wave
:class:`~.env_periodic.PlaneWaveEnvelope` (a delocalised Slater
determinant) cannot represent this broken-translational-symmetry
state — at low density it is a poor variational ansatz and the SR/Adam
optimiser cannot bridge it to the crystal.

This module implements :class:`GaussianLocalizedEnvelope2D`:

    phi_{d, i}(r) = exp(-(1/(2 sigma^2)) |r - R_{pi(i), d}|^2_periodic)

where ``R_{i, d}`` are 2D Bravais lattice sites assigned to orbital
``i`` of determinant ``d``, ``sigma`` is a per-spin variational width,
and the periodic norm uses the smooth torus distance from
:mod:`OmegaQMC.psi.nn.periodic`.  The lattice primitive vectors
``(a1, a2)`` themselves are *variational* parameters initialised from
the triangular Wigner-crystal geometry — letting the optimiser pick the
true Bravais sector (triangular, centred-rectangular, rectangular,
oblique) without bias.  This is essential for the cavity-coupled phase
where uniaxial anisotropy is expected to distort the lattice
(centred-rectangular -> rectangular -> stripe phase as lambda grows).

The Slater determinant of localised Gaussians on the triangular lattice
gives a manifestly broken-translational-symmetry ansatz; the SR
optimiser then refines it (with backflow + Jastrow) into the
interacting WC ground state.

References:
    Cassella et al., Phys. Rev. Lett. 130, 036401 (2023) — used the
    same approach for the 3D Wigner crystal.
    Drummond & Needs, Phys. Rev. Lett. 102, 126402 (2009) — phase
    boundary at r_s^c ~ 31 in 2D.
"""

from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from .compat import param_value


# ---------------------------------------------------------------------
# Lattice site placement
# ---------------------------------------------------------------------

def triangular_lattice_sites(
    rs: float, n_sites: int,
) -> np.ndarray:
    """Place ``n_sites`` electrons on a triangular Bravais lattice.

    The lattice constant ``a = sqrt(2 / (sqrt(3) n))`` with
    ``n = 1 / (pi rs^2)`` gives one electron per primitive cell of
    area ``sqrt(3) a^2 / 2 = pi rs^2``.

    A square ``M x M`` super-cell of ``M^2`` sites with
    ``M = ceil(sqrt(n_sites / 2))`` is constructed (a triangular
    lattice has 2 sites per orthorhombic super-cell), and the first
    ``n_sites`` are returned in row-major order.

    For non-commensurate ``n_sites`` the assignment may not match the
    simulation cell exactly — in those cases the user should pick
    ``n_sites`` from the commensurate sequence
    ``2 * M^2`` for ``M = 1, 2, 3, ...`` (i.e. 2, 8, 18, 32, 50, 72,
    98, 128, 162, ...).

    Args:
        rs: Wigner-Seitz radius (Bohr).
        n_sites: Number of sites to return.

    Returns:
        ``(n_sites, 2)`` Cartesian site positions in Bohr.
    """
    n_density = 1.0 / (np.pi * rs ** 2)
    a = np.sqrt(2.0 / (np.sqrt(3.0) * n_density))
    a1 = a * np.array([1.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3.0) / 2.0])
    M = int(np.ceil(np.sqrt(n_sites)))
    sites = []
    for i in range(M):
        for j in range(M):
            sites.append(i * a1 + j * a2)
            if len(sites) >= n_sites:
                return np.array(sites)
    return np.array(sites)


def triangular_lattice_sites_rect_cell(rs: float, n_sites: int):
    """Generate triangular WC sites in their NATURAL centered-rectangular cell.

    Unlike :func:`triangular_lattice_sites` (which packs sites into an
    oblique M×M grid that does not perfectly tile a square cell), this
    function returns sites that EXACTLY tile a rectangular cell of
    dimensions ``L_x × L_y = M·a × M·a·√3`` with ``2·M²`` sites — the
    natural commensurate geometry for the 2D triangular Wigner crystal.

    Args:
        rs: Wigner-Seitz radius (Bohr).
        n_sites: Must be of the form ``2·M²`` (i.e. 2, 8, 18, 32, 50,
            72, 98, 128, 162, 200, ...).

    Returns:
        Tuple ``(sites, L_x, L_y)``:
          * ``sites`` shape ``(n_sites, 2)`` — Cartesian positions.
          * ``L_x``, ``L_y`` — rectangular cell dimensions.
    """
    M = int(round(np.sqrt(n_sites / 2)))
    if 2 * M * M != n_sites:
        raise ValueError(
            f"triangular_lattice_sites_rect_cell requires n_sites in "
            f"the sequence 2·M² = 2, 8, 18, 32, 50, 72, 98, ...; "
            f"got n_sites={n_sites}",
        )
    n_density = 1.0 / (np.pi * rs ** 2)
    a = np.sqrt(2.0 / (np.sqrt(3.0) * n_density))
    L_x = M * a
    L_y = M * a * np.sqrt(3.0)
    sites = []
    for j in range(M):
        for i in range(M):
            sites.append([i * a, j * a * np.sqrt(3.0)])
            sites.append([(i + 0.5) * a, (j + 0.5) * a * np.sqrt(3.0)])
    return np.array(sites), L_x, L_y


def stripe_lattice_sites(
    rs: float, n_sites: int,
    orientation: str = 'x', n_rows: int = 0,
) -> np.ndarray:
    """``n_sites`` electrons in horizontal (``orientation='x'``) or
    vertical (``orientation='y'``) stripes inside a square cell.

    The square cell has side ``L = rs * sqrt(pi * n_sites)``.  Sites are
    laid out on a rectangular grid: ``n_rows`` rows × ``n_per_row``
    columns (where ``n_per_row = n_sites // n_rows``).  Setting
    ``n_rows`` small (e.g. 2 or 5) gives elongated stripes; setting
    ``n_rows = ceil(sqrt(n_sites))`` gives a near-square rectangular
    arrangement.

    If ``n_rows = 0`` (default), picks ``ceil(sqrt(n_sites/2))`` for a
    moderate stripe shape (rows roughly half as many as columns).

    Args:
        rs: Wigner-Seitz radius (Bohr).
        n_sites: Number of electrons.
        orientation: ``'x'`` (rows along x) or ``'y'`` (rows along y).
        n_rows: Number of rows.  0 → auto.

    Returns:
        ``(n_sites, 2)`` site positions in Bohr.
    """
    if orientation not in ('x', 'y'):
        raise ValueError(
            f"orientation must be 'x' or 'y', got {orientation!r}",
        )
    n_density = 1.0 / (np.pi * rs ** 2)
    L = np.sqrt(n_sites / n_density)
    if n_rows <= 0:
        n_rows = max(1, int(np.ceil(np.sqrt(n_sites / 2.0))))
    n_per_row = int(np.ceil(n_sites / n_rows))
    if orientation == 'x':
        a_par = L / n_per_row   # spacing along x within a row
        a_perp = L / n_rows     # row separation along y
        sites = []
        for j in range(n_rows):
            for i in range(n_per_row):
                sites.append([(i + 0.5) * a_par, (j + 0.5) * a_perp])
                if len(sites) >= n_sites:
                    return np.array(sites)
    else:  # 'y' — same as 'x' but transposed
        a_par = L / n_per_row
        a_perp = L / n_rows
        sites = []
        for j in range(n_rows):
            for i in range(n_per_row):
                sites.append([(j + 0.5) * a_perp, (i + 0.5) * a_par])
                if len(sites) >= n_sites:
                    return np.array(sites)
    return np.array(sites)


def square_lattice_sites(
    rs: float, n_sites: int,
    site_offset: float = 0.5,
) -> np.ndarray:
    """``n_sites`` electrons on a near-square Bravais lattice.

    Picks ``M = ceil(sqrt(n_sites))``, builds an ``M x M`` grid with
    spacing ``L/M``, returns the first ``n_sites`` sites in row-major
    order.  Exactly square for ``n_sites = M²`` (e.g. 4, 9, 16, 25, 49);
    slightly imperfect otherwise.

    Args:
      site_offset: Fractional shift within a unit cell.  Default 0.5
        places sites at cell CENTERS (good for bare WC).  Use 0.0 to
        place sites at cell CORNERS (good for cosine-modulated systems
        where v_ext = -v·Σ cos(2πr/a) has minima at r = n·a).
    """
    n_density = 1.0 / (np.pi * rs ** 2)
    L = np.sqrt(n_sites / n_density)
    M = int(np.ceil(np.sqrt(n_sites)))
    a = L / M
    off = float(site_offset)
    sites = []
    for i in range(M):
        for j in range(M):
            sites.append([(i + off) * a, (j + off) * a])
            if len(sites) >= n_sites:
                return np.array(sites)
    return np.array(sites)


def make_lattice_sites(
    rs: float, n_sites: int,
    lattice_type: str = 'triangular',
    **kwargs,
) -> np.ndarray:
    """Dispatch to the correct lattice-sites generator.

    Args:
        rs: Wigner-Seitz radius (Bohr).
        n_sites: Number of electrons.
        lattice_type: One of ``'triangular'`` (default), ``'square'``,
            ``'stripe_x'``, ``'stripe_y'``.
        **kwargs: Passed through to the specific generator (e.g.
            ``n_rows`` for stripe).

    Returns:
        ``(n_sites, 2)`` site positions in Bohr.
    """
    lt = lattice_type.lower()
    if lt == 'triangular':
        return triangular_lattice_sites(rs, n_sites)
    elif lt == 'square':
        return square_lattice_sites(rs, n_sites, **kwargs)
    elif lt in ('stripe_x', 'stripe'):
        return stripe_lattice_sites(
            rs, n_sites, orientation='x', **kwargs,
        )
    elif lt == 'stripe_y':
        return stripe_lattice_sites(
            rs, n_sites, orientation='y', **kwargs,
        )
    else:
        raise ValueError(
            f"lattice_type must be one of "
            f"('triangular', 'square', 'stripe_x', 'stripe_y'); "
            f"got {lattice_type!r}",
        )


def commensurate_triangular_supercell(
    rs: float, n_sites: int,
):
    """Compute simulation-cell vectors for a triangular WC of N sites.

    Returns ``(L1, L2)`` for an oblique super-cell that exactly
    contains ``n_sites`` triangular Bravais points; the corresponding
    cell area is ``n_sites * pi rs^2``.

    Picks an ``M`` such that ``n_sites = M^2`` (square sub-block of the
    triangular lattice).  Falls back to nearest if exact ``M^2``
    impossible.
    """
    n_density = 1.0 / (np.pi * rs ** 2)
    a = np.sqrt(2.0 / (np.sqrt(3.0) * n_density))
    M = int(round(np.sqrt(n_sites)))
    L1 = M * a * np.array([1.0, 0.0])
    L2 = M * a * np.array([0.5, np.sqrt(3.0) / 2.0])
    return L1, L2


# ---------------------------------------------------------------------
# Crystal-aware walker initialisation
# ---------------------------------------------------------------------

def crystal_init_walkers_2d(
    rng_key,
    num_walkers: int,
    n_up: int,
    n_down: int,
    L,
    *,
    sigma_init: float = 0.25,
    spin_pattern: str = 'neel',
    noise_scale_factor: float = 0.5,
    lattice_type: str = 'triangular',
    site_offset: float = 0.5,
):
    """Initialise walkers near triangular Bravais lattice sites.

    For Wigner-crystal trial wavefunctions ``|psi|^2`` is sharply peaked
    at the lattice sites with width ``sigma ~ sigma_init * a_NN``; the
    inter-site spacing ``a_NN`` is far larger than any Metropolis step
    a uniform-init walker can take in a few decorrelation moves.  As a
    result, walkers initialised uniformly in the cell sit overwhelmingly
    in low-``|psi|^2`` regions and the SR optimiser then "trains away"
    the localised character of the envelope.  This helper places each
    walker's electrons directly at the triangular sites with a small
    Gaussian noise of width ``noise_scale_factor * sigma_init * a_NN``
    (default ``0.5 sigma_init a_NN``, half the envelope width — keeps
    walkers comfortably inside the |psi|^2 peak), which lets MCMC
    immediately sample the dominant region of |psi|^2.

    Args:
        rng_key: JAX PRNG key.
        num_walkers: Number of MCMC walkers.
        n_up, n_down: Per-spin electron counts.
        L: Square-cell side length (Bohr).
        sigma_init: Same as the corresponding
            :class:`GaussianLocalizedEnvelope2D` parameter.
        spin_pattern: ``'neel'`` (alternating up/down on adjacent
            triangular sites) or ``'all_up'``.
        noise_scale_factor: Multiplier on ``sigma_init * a_NN`` for the
            walker-position Gaussian noise.

    Returns:
        ``(num_walkers, n_up + n_down, 2)`` walker positions, in Bohr,
        wrapped into ``[0, L)^2`` for safety.
    """
    if spin_pattern not in ('neel', 'all_up'):
        raise ValueError(
            f"spin_pattern must be 'neel' or 'all_up', "
            f"got {spin_pattern!r}",
        )
    n_elec = n_up + n_down
    # Accept either scalar L (square cell) or 2-tuple (rectangular)
    if isinstance(L, (tuple, list, np.ndarray, jnp.ndarray)):
        L_x, L_y = float(L[0]), float(L[1])
    else:
        L_x = L_y = float(L)
    area = L_x * L_y
    rs = np.sqrt(area / (np.pi * n_elec))
    a_nn = np.sqrt(2.0 * np.pi / np.sqrt(3.0)) * rs   # ~ 1.905 * rs
    L_arr = np.asarray([L_x, L_y])

    # For rectangular cells with native triangular aspect, use the
    # 2·M²-commensurate generator (sites exactly tile the cell).
    is_rect = abs(L_x - L_y) > 1e-6
    if is_rect and lattice_type == 'triangular':
        all_sites, _, _ = triangular_lattice_sites_rect_cell(rs, n_elec)
    else:
        extra = {}
        if lattice_type == 'square':
            extra['site_offset'] = float(site_offset)
        all_sites = make_lattice_sites(
            rs, n_elec, lattice_type=lattice_type, **extra,
        )
    if spin_pattern == 'neel':
        up_sites = all_sites[0::2][:n_up]
        dn_sites = (
            all_sites[1::2][:n_down] if n_down > 0 else None
        )
    else:
        up_sites = all_sites[:n_up]
        dn_sites = (
            all_sites[n_up:n_up + n_down] if n_down > 0 else None
        )

    # Wrap to cell to match envelope's site convention.
    up_sites = np.mod(up_sites, L_arr)
    if dn_sites is not None:
        dn_sites = np.mod(dn_sites, L_arr)
        sites = np.concatenate([up_sites, dn_sites], axis=0)
    else:
        sites = up_sites
    sites = jnp.asarray(sites, dtype=jnp.float64)

    noise_scale = float(noise_scale_factor) * float(sigma_init) * a_nn
    noise = noise_scale * jax.random.normal(
        rng_key, (num_walkers, n_elec, 2),
    )
    walkers = sites[None, :, :] + noise

    # Wrap into [0, L_x)x[0, L_y) (scalar L_arr broadcasts to last axis).
    walkers = jnp.mod(walkers, jnp.asarray(L_arr, dtype=jnp.float64))
    return walkers


# ---------------------------------------------------------------------
# Localised Gaussian envelope
# ---------------------------------------------------------------------

class GaussianLocalizedEnvelope2D(nnx.Module):
    """Slater determinant of localised Gaussians on a 2D Bravais lattice.

    Each orbital ``i`` of each determinant ``d`` is

        phi_{d, i}(r) = exp(-(1/(2 sigma_d^2)) |r - R_i_d|^2_periodic)

    where the periodic ``|.|`` is the smooth torus distance.

    Variational parameters
    ----------------------

    * ``sigma`` per spin per determinant — Gaussian width.  Initial
      value is ``sigma_init * a_NN`` where ``a_NN`` is the
      nearest-neighbour spacing.
    * ``sites`` per spin per determinant -- (n_orb, 2) Cartesian
      positions of the Gaussian centres.  Initialised on the
      triangular lattice but free to relax (within the simulation
      cell).  Optimising these lets the lattice deform into
      non-triangular Bravais sectors (centred-rectangular,
      rectangular, oblique) when the underlying physics prefers it
      (e.g. cavity-induced anisotropy).

    Spin assignment
    ---------------

    For the antiferromagnetic Wigner crystal the natural choice is a
    Neel pattern: alternate sites get up vs down.  This module assigns
    sites in interleaved order (sites[::2] -> up, sites[1::2] -> down)
    when ``spin_pattern == 'neel'``; for ferromagnetic crystals
    ``'all_up'`` puts all sites in the up-spin sector.

    Args:
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        n_det: Number of determinants.
        rs: Wigner-Seitz radius (sets initial site spacing).
        L: Simulation-cell side length (square cell convention).
        sigma_init: Initial Gaussian width as fraction of the
            nearest-neighbour distance.  Smaller -> more localised;
            <= 0.3 recommended to keep the optimiser inside the
            crystal basin.
        spin_pattern: 'neel' (alternating, AFM) or 'all_up' (FM).
        det_jitter: Random Gaussian perturbation of site positions
            for det >= 1 (det 0 stays at the bare triangular lattice).
            Magnitude in Bohr.
    """

    def __init__(
        self,
        n_up: int,
        n_down: int,
        n_det: int,
        rs: float,
        L,
        *,
        sigma_init: float = 0.25,
        spin_pattern: str = 'neel',
        det_jitter: float = 0.0,
        lattice_type: str = 'triangular',
        anisotropic_sigma: bool = False,
        site_offset: float = 0.5,
    ):
        if spin_pattern not in ('neel', 'all_up'):
            raise ValueError(
                f"spin_pattern must be 'neel' or 'all_up', "
                f"got {spin_pattern!r}",
            )
        # Accept scalar L (square cell) or 2-tuple (rectangular).
        if isinstance(L, (tuple, list, np.ndarray, jnp.ndarray)):
            L_x, L_y = float(L[0]), float(L[1])
        else:
            L_x = L_y = float(L)
        n_total = n_up + n_down
        is_rect = abs(L_x - L_y) > 1e-6
        if is_rect and lattice_type == 'triangular':
            all_sites, _, _ = triangular_lattice_sites_rect_cell(
                rs, n_total,
            )
        else:
            # site_offset is forwarded to lattice generators that support
            # it (currently 'square').  Default 0.5 = cell-centered (bare
            # WC); 0.0 = corner-aligned (matches cosine v_ext minima).
            extra = {}
            if lattice_type == 'square':
                extra['site_offset'] = float(site_offset)
            all_sites = make_lattice_sites(
                rs, n_total, lattice_type=lattice_type, **extra,
            )
        # Wrap sites to be inside [0, L_x) x [0, L_y) (paranoia: with the
        # commensurate cell they should all already lie in the cell).
        L_arr_np = np.asarray([L_x, L_y])
        all_sites = np.mod(all_sites, L_arr_np)

        if spin_pattern == 'neel':
            # Alternating assignment: site[::2] is up, site[1::2] down.
            up_sites = all_sites[0::2][:n_up]
            dn_sites = all_sites[1::2][:n_down] if n_down > 0 else None
        else:  # 'all_up'
            up_sites = all_sites[:n_up]
            dn_sites = all_sites[n_up:n_up + n_down] if n_down > 0 else None

        # nearest-neighbour spacing
        a_nn = np.sqrt(2.0 / (np.sqrt(3.0) / (np.pi * rs ** 2)))
        sigma0 = float(sigma_init * a_nn)

        # ---- Per-det site arrays + jitter ----
        rng = np.random.default_rng(11)
        sites_up = np.broadcast_to(
            up_sites, (n_det, n_up, 2),
        ).copy()
        if det_jitter > 0.0 and n_det > 1:
            sites_up[1:] += det_jitter * rng.normal(
                size=(n_det - 1, n_up, 2),
            )
        self.sites_up = nnx.Param(jnp.asarray(sites_up))

        if n_down > 0:
            sites_dn = np.broadcast_to(
                dn_sites, (n_det, n_down, 2),
            ).copy()
            if det_jitter > 0.0 and n_det > 1:
                sites_dn[1:] += det_jitter * rng.normal(
                    size=(n_det - 1, n_down, 2),
                )
            self.sites_dn = nnx.Param(jnp.asarray(sites_dn))
        else:
            self.sites_dn = None

        # Sigma: scalar (isotropic) or (sigma_x, sigma_y) when
        # anisotropic_sigma=True.  Init both components to the same
        # sigma0 → at λ=0 they should stay equal; at finite-λ they
        # can diverge (nematic order parameter).
        self.anisotropic_sigma = bool(anisotropic_sigma)
        if self.anisotropic_sigma:
            self.sigma_up = nnx.Param(jnp.asarray([sigma0, sigma0]))
            self.sigma_dn = (
                nnx.Param(jnp.asarray([sigma0, sigma0]))
                if n_down > 0 else None
            )
        else:
            self.sigma_up = nnx.Param(jnp.asarray(sigma0))
            self.sigma_dn = (
                nnx.Param(jnp.asarray(sigma0)) if n_down > 0 else None
            )

        self.n_up = n_up
        self.n_down = n_down
        self.n_det = n_det
        # Store as 2-array so _orb_one_spin's min-image broadcasts
        # naturally for both square and rectangular cells.
        self.L = jnp.asarray([L_x, L_y], dtype=jnp.float64)
        self.dim = 2

    def _orb_one_spin(
        self, r_spin: jax.Array, sites: jax.Array, sigma: jax.Array,
    ) -> jax.Array:
        """Evaluate (n_det, n_spin, n_orb_spin) Gaussian orbitals.

        ``sigma`` may be a scalar (isotropic Gaussian) or a length-2
        array ``[sigma_x, sigma_y]`` (anisotropic — nematic distortion).
        """
        # r_spin: (n_spin, 2), sites: (n_det, n_orb, 2)
        diff = r_spin[None, :, None, :] - sites[:, None, :, :]
        s = diff / self.L
        s = s - jnp.round(s)
        diff_mi = s * self.L                  # (n_det, n_spin, n_orb, 2)
        if sigma.ndim == 0:
            d2 = jnp.sum(diff_mi ** 2, axis=-1)
            return jnp.exp(-d2 / (2.0 * sigma ** 2))
        else:
            # sigma shape (2,) — anisotropic
            inv_2s2 = 1.0 / (2.0 * sigma ** 2)         # (2,)
            exponent = jnp.sum(
                (diff_mi ** 2) * inv_2s2, axis=-1,
            )
            return jnp.exp(-exponent)

    def __call__(self, phys_conf, nuc_params=None):
        """Return ``(n_det, n_elec, n_up + n_down)`` orbital matrix.

        Format matches :class:`PlaneWaveEnvelope`: first ``n_up``
        columns hold the up-spin orbitals (filled in rows 0..n_up),
        remaining columns hold the down-spin orbitals (filled in rows
        n_up..n_elec).  Off-diagonal blocks are zero.
        """
        del nuc_params
        r = phys_conf.r           # (n_elec, 2)
        n_up = self.n_up
        n_down = self.n_down
        n_elec = n_up + n_down
        n_det = self.n_det

        sigma_up = jax.nn.softplus(param_value(self.sigma_up)) + 1e-3
        orb_up = self._orb_one_spin(
            r[:n_up], param_value(self.sites_up), sigma_up,
        )  # (n_det, n_up, n_up)

        if n_down > 0:
            sigma_dn = (
                jax.nn.softplus(param_value(self.sigma_dn)) + 1e-3
            )
            orb_dn = self._orb_one_spin(
                r[n_up:], param_value(self.sites_dn), sigma_dn,
            )
        else:
            orb_dn = jnp.zeros((n_det, 0, 0), dtype=orb_up.dtype)

        # Pad into the (n_det, n_elec, n_up + n_down) layout used by
        # the molecular-style downstream wavefunction.
        orb_up_padded = jnp.zeros(
            (n_det, n_elec, n_up), dtype=orb_up.dtype,
        )
        orb_up_padded = orb_up_padded.at[:, :n_up, :].set(orb_up)

        orb_dn_padded = jnp.zeros(
            (n_det, n_elec, n_down), dtype=orb_up.dtype,
        )
        if n_down > 0:
            orb_dn_padded = orb_dn_padded.at[:, n_up:, :].set(orb_dn)

        return jnp.concatenate(
            [orb_up_padded, orb_dn_padded], axis=-1,
        )
