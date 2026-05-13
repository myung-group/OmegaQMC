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
    L: float,
    *,
    sigma_init: float = 0.25,
    spin_pattern: str = 'neel',
    noise_scale_factor: float = 0.5,
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
    rs = L / np.sqrt(np.pi * n_elec)
    a_nn = np.sqrt(2.0 * np.pi / np.sqrt(3.0)) * rs   # ~ 1.905 * rs

    all_sites = triangular_lattice_sites(rs, n_elec)
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
    up_sites = np.mod(up_sites, L)
    if dn_sites is not None:
        dn_sites = np.mod(dn_sites, L)
        sites = np.concatenate([up_sites, dn_sites], axis=0)
    else:
        sites = up_sites
    sites = jnp.asarray(sites, dtype=jnp.float64)

    noise_scale = float(noise_scale_factor) * float(sigma_init) * a_nn
    noise = noise_scale * jax.random.normal(
        rng_key, (num_walkers, n_elec, 2),
    )
    walkers = sites[None, :, :] + noise

    # Wrap into [0, L)^2 in case noise pushed walkers outside
    # (Metropolis sampler also wraps, but this keeps the very first
    # log_psi evaluation inside the cell for sanity).
    walkers = jnp.mod(walkers, L)
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
        L: float,
        *,
        sigma_init: float = 0.25,
        spin_pattern: str = 'neel',
        det_jitter: float = 0.0,
    ):
        if spin_pattern not in ('neel', 'all_up'):
            raise ValueError(
                f"spin_pattern must be 'neel' or 'all_up', "
                f"got {spin_pattern!r}",
            )
        n_total = n_up + n_down
        all_sites = triangular_lattice_sites(rs, n_total)
        # Wrap sites to be inside [0, L)^2 (paranoia: with the
        # commensurate cell they should all already lie in the cell).
        all_sites = np.mod(all_sites, L)

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

        self.sigma_up = nnx.Param(jnp.asarray(sigma0))
        self.sigma_dn = (
            nnx.Param(jnp.asarray(sigma0)) if n_down > 0 else None
        )

        self.n_up = n_up
        self.n_down = n_down
        self.n_det = n_det
        self.L = float(L)
        self.dim = 2

    def _orb_one_spin(
        self, r_spin: jax.Array, sites: jax.Array, sigma: jax.Array,
    ) -> jax.Array:
        """Evaluate (n_det, n_spin, n_orb_spin) Gaussian orbitals."""
        # r_spin: (n_spin, 2), sites: (n_det, n_orb, 2)
        # diff: (n_det, n_spin, n_orb, 2) via broadcasting
        diff = r_spin[None, :, None, :] - sites[:, None, :, :]
        # Minimum-image via wrapping fractional coords to [-0.5, 0.5).
        s = diff / self.L
        s = s - jnp.round(s)
        diff_mi = s * self.L
        d2 = jnp.sum(diff_mi ** 2, axis=-1)
        return jnp.exp(-d2 / (2.0 * sigma ** 2))

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
