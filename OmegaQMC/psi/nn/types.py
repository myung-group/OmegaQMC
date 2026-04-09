"""Core types for NN trial wavefunctions.

Replaces ``jax_dataclasses``-based types from DeepQMC with
plain NamedTuples that are valid JAX pytrees.
"""

from typing import NamedTuple

import jax


class PhysicalConfiguration(NamedTuple):
    """Nuclear and electronic coordinates.

    Attributes:
        R: Nuclear coordinates, shape ``(natom, 3)``.
        r: Electron coordinates, shape ``(nelec, 3)``.
        mol_idx: Molecule index (scalar, always 0
            for single-molecule runs).
    """

    R: jax.Array
    r: jax.Array
    mol_idx: jax.Array


class Psi(NamedTuple):
    """Wavefunction value as sign and log magnitude.

    Attributes:
        sign: Sign of the wavefunction.
        log: Log of the absolute value.
    """

    sign: jax.Array
    log: jax.Array
