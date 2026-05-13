"""Core types for NN trial wavefunctions.

Replaces ``jax_dataclasses``-based types from DeepQMC with
plain NamedTuples that are valid JAX pytrees.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class PhysicalConfiguration(NamedTuple):
    """Nuclear and electronic coordinates.

    Attributes:
        R: Nuclear coordinates, shape ``(natom, 3)``.
        r: Electron coordinates, shape ``(nelec, 3)``.
        mol_idx: Molecule index (scalar, always 0
            for single-molecule runs).
        n: Photon Fock index (scalar, default 0).
            Only used by QED ansatze with
            per-electron one-hot photon injection
            (Tang 2025, 2503.15644). Non-QED code
            ignores this field.
    """

    R: jax.Array
    r: jax.Array
    mol_idx: jax.Array
    n: jax.Array = jnp.array(0, dtype=jnp.int32)


class Psi(NamedTuple):
    """Wavefunction value as sign and log magnitude.

    Attributes:
        sign: Sign of the wavefunction.
        log: Log of the absolute value.
    """

    sign: jax.Array
    log: jax.Array
