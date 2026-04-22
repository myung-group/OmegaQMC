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

from .env_periodic import PlaneWaveEnvelope
from .periodic import (
    PeriodicLattice,
    make_cubic_lattice,
    periodic_norm,
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

    The HEG has no nuclei, so ``R`` is an empty ``(0, 3)`` array.
    """
    return PhysicalConfiguration(
        R=jnp.zeros((0, 3), dtype=r.dtype),
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

        def _pair_contrib(mlp, dists):
            # dists: (n_pairs,)
            inp = dists[:, None]
            return jnp.sum(mlp(inp))

        # Same-spin up-up
        total = jnp.asarray(0.0, dtype=r.dtype)
        if n_up > 1:
            i, j = jnp.triu_indices(n_up, k=1)
            d = r[i] - r[j]
            dist = periodic_norm(d, lattice)
            total = total + _pair_contrib(self.u_same, dist)
        if n_down > 1:
            i, j = jnp.triu_indices(n_down, k=1)
            d = r[n_up + i] - r[n_up + j]
            dist = periodic_norm(d, lattice)
            total = total + _pair_contrib(self.u_same, dist)
        if n_up > 0 and n_down > 0:
            # Antiparallel all-pairs
            ru = r[:n_up]
            rd = r[n_up:]
            d = ru[:, None, :] - rd[None, :, :]
            dist = periodic_norm(d, lattice).reshape(-1)
            total = total + _pair_contrib(self.u_anti, dist)

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
        rngs,
    ):
        self.n_up = n_up
        self.n_down = n_down
        self.n_det = n_det
        self.L = float(L)
        self.lattice = nnx.data(make_cubic_lattice(L))

        self.envelope = PlaneWaveEnvelope(
            n_up=n_up, n_down=n_down, n_det=n_det, L=L,
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
        L: Cubic simulation-cell side length.
        n_det: Number of Slater determinants.
        use_jastrow: Enable the periodic pair Jastrow.
        jastrow_hidden: Hidden-layer widths for Jastrow MLPs.
    """

    n_up: int
    n_down: int
    L: float
    n_det: int = 1
    use_jastrow: bool = True
    jastrow_hidden: tuple = (16, 16)


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
        rngs=rngs,
    )
    graphdef, params, other = nnx.split(model, nnx.Param, ...)

    def log_psi(r, params):
        mdl = nnx.merge(graphdef, params, other)
        return mdl(r).log

    return log_psi, params, graphdef
