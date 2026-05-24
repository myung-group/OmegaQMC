"""
Walker dump utilities for the compressed-sensing pipeline.

Two pieces:

- :class:`WalkerDumper` is a context-managed HDF5 writer used by the
  VMC driver. After each production block the driver calls
  :meth:`write_block` to stream the current walker positions and
  ``log|Psi|`` values into pre-allocated datasets.

- :func:`load_walker_bank` reads back the dumped file into NumPy arrays
  for the CS scaling pipeline.

The dump deliberately stores only ``log|Psi|``, not signed ``Psi``.
Reconstructing the sign requires reloading the trained ansatz; the
sign-aware helper is a separate utility owned by the experiment driver
(:mod:`OmegaQMC.cs.scaling` consumes ``(walker_bank, psi_vals_bank)`` from
that helper, not directly from the dump file).
"""

from typing import Mapping, Optional, Tuple

import numpy as np


class WalkerDumper:
    """HDF5 streaming writer for VMC walker configurations.

    Dataset layout
    --------------
    - ``/walkers`` shape ``(num_blocks * num_walkers, nelec, 3)``, ``f4``
    - ``/log_psi`` shape ``(num_blocks * num_walkers,)``, ``f8``
    - root attrs: ``num_blocks``, ``num_walkers``, ``nelec``,
      ``mc_timestep``, ``num_steps_decorr`` (when supplied),
      ``schema_version``.

    Use as a context manager. ``write_block`` may be called at most
    ``num_blocks`` times; further calls raise.
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        path: str,
        num_blocks: int,
        num_walkers: int,
        nelec: int,
        mc_timestep: float = 0.0,
        num_steps_decorr: int = 1,
        walker_dtype: str = "f4",
    ):
        import h5py
        self._h5py = h5py
        self._path = str(path)
        self.num_blocks = int(num_blocks)
        self.num_walkers = int(num_walkers)
        self.nelec = int(nelec)
        self._next_block = 0
        self._f = h5py.File(self._path, "w")
        self._ds_walkers = self._f.create_dataset(
            "walkers",
            shape=(self.num_blocks * self.num_walkers, self.nelec, 3),
            dtype=walker_dtype,
        )
        self._ds_log_psi = self._f.create_dataset(
            "log_psi",
            shape=(self.num_blocks * self.num_walkers,),
            dtype="f8",
        )
        self._f.attrs["num_blocks"] = self.num_blocks
        self._f.attrs["num_walkers"] = self.num_walkers
        self._f.attrs["nelec"] = self.nelec
        self._f.attrs["mc_timestep"] = float(mc_timestep)
        self._f.attrs["num_steps_decorr"] = int(num_steps_decorr)
        self._f.attrs["schema_version"] = self.SCHEMA_VERSION

    def write_block(self, walkers, log_psi) -> None:
        """Append one block. ``walkers`` shape ``(num_walkers, nelec, 3)``."""
        if self._next_block >= self.num_blocks:
            raise RuntimeError(
                f"WalkerDumper already wrote {self.num_blocks} blocks"
            )
        w = np.asarray(walkers)
        lp = np.asarray(log_psi)
        if w.shape != (self.num_walkers, self.nelec, 3):
            raise ValueError(
                f"walkers shape {w.shape} expected "
                f"({self.num_walkers}, {self.nelec}, 3)"
            )
        if lp.shape != (self.num_walkers,):
            raise ValueError(
                f"log_psi shape {lp.shape} expected ({self.num_walkers},)"
            )
        start = self._next_block * self.num_walkers
        end = start + self.num_walkers
        self._ds_walkers[start:end] = w
        self._ds_log_psi[start:end] = lp
        self._next_block += 1

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    @property
    def blocks_written(self) -> int:
        return self._next_block

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def load_walker_bank(
    path: str,
    max_K_s: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Read back a dump file.

    Returns ``(walkers, log_psi, metadata)``. If ``max_K_s`` is given the
    first ``max_K_s`` rows are returned.
    """
    import h5py
    with h5py.File(str(path), "r") as f:
        n = f["walkers"].shape[0]
        cap = n if max_K_s is None else min(int(max_K_s), n)
        walkers = f["walkers"][:cap]
        log_psi = f["log_psi"][:cap]
        meta: dict = {k: v for k, v in f.attrs.items()}
    return np.asarray(walkers), np.asarray(log_psi), meta
