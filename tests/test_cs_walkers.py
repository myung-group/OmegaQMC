"""
Tests for OmegaQMC.cs.walkers — WalkerDumper and load_walker_bank.

Dumper logic is tested in isolation against synthetic per-block writes.
The driver wiring is exercised indirectly by an import-smoke test on
:mod:`OmegaQMC.vmc_nn` so any syntax error in the dump_walkers patch
fails here, not in a downstream pilot run.
"""

import h5py
import numpy as np
import pytest

from OmegaQMC.cs.walkers import WalkerDumper, load_walker_bank


def test_dumper_roundtrip_shapes_and_attrs(tmp_path):
    path = tmp_path / "walkers.h5"
    nb, nw, ne = 4, 13, 6
    rng = np.random.default_rng(0)
    walker_blocks = [rng.normal(size=(nw, ne, 3)).astype("f4")
                     for _ in range(nb)]
    log_psi_blocks = [rng.normal(size=(nw,)) for _ in range(nb)]

    with WalkerDumper(str(path), num_blocks=nb, num_walkers=nw, nelec=ne,
                      mc_timestep=0.07, num_steps_decorr=3) as d:
        for w, lp in zip(walker_blocks, log_psi_blocks):
            d.write_block(w, lp)
        assert d.blocks_written == nb

    walkers, log_psi, meta = load_walker_bank(str(path))
    assert walkers.shape == (nb * nw, ne, 3)
    assert log_psi.shape == (nb * nw,)
    np.testing.assert_allclose(walkers, np.concatenate(walker_blocks),
                                rtol=0, atol=1e-6)
    np.testing.assert_allclose(log_psi, np.concatenate(log_psi_blocks))
    assert int(meta["num_blocks"]) == nb
    assert int(meta["num_walkers"]) == nw
    assert int(meta["nelec"]) == ne
    assert int(meta["num_steps_decorr"]) == 3
    assert float(meta["mc_timestep"]) == pytest.approx(0.07)
    assert str(meta["schema_version"]) == "1.1.0"
    assert "jastrow" not in meta  # legacy path writes no jastrow dataset


def test_dumper_jastrow_roundtrip(tmp_path):
    path = tmp_path / "walkers_j.h5"
    nb, nw, ne = 3, 5, 4
    rng = np.random.default_rng(1)
    wb = [rng.normal(size=(nw, ne, 3)).astype("f4") for _ in range(nb)]
    lpb = [rng.normal(size=(nw,)) for _ in range(nb)]
    jb = [rng.normal(size=(nw,)) for _ in range(nb)]
    with WalkerDumper(str(path), num_blocks=nb, num_walkers=nw, nelec=ne) as d:
        for w, lp, j in zip(wb, lpb, jb):
            d.write_block(w, lp, jastrow=j)
    _, _, meta = load_walker_bank(str(path))
    assert "jastrow" in meta
    np.testing.assert_allclose(meta["jastrow"], np.concatenate(jb))


def test_dumper_rejects_jastrow_midstream(tmp_path):
    path = tmp_path / "w.h5"
    with WalkerDumper(str(path), num_blocks=2, num_walkers=2, nelec=2) as d:
        d.write_block(np.zeros((2, 2, 3)), np.zeros(2))  # no jastrow
        with pytest.raises(ValueError, match="jastrow supplied mid-stream"):
            d.write_block(np.zeros((2, 2, 3)), np.zeros(2), jastrow=np.zeros(2))


def test_dumper_rejects_overshoot(tmp_path):
    path = tmp_path / "w.h5"
    with WalkerDumper(str(path), num_blocks=1, num_walkers=2, nelec=2) as d:
        d.write_block(np.zeros((2, 2, 3)), np.zeros(2))
        with pytest.raises(RuntimeError, match="already wrote"):
            d.write_block(np.zeros((2, 2, 3)), np.zeros(2))


def test_dumper_rejects_wrong_walker_shape(tmp_path):
    path = tmp_path / "w.h5"
    with WalkerDumper(str(path), num_blocks=2, num_walkers=4, nelec=2) as d:
        with pytest.raises(ValueError, match="walkers shape"):
            d.write_block(np.zeros((3, 2, 3)), np.zeros(3))


def test_dumper_rejects_wrong_log_psi_shape(tmp_path):
    path = tmp_path / "w.h5"
    with WalkerDumper(str(path), num_blocks=2, num_walkers=4, nelec=2) as d:
        with pytest.raises(ValueError, match="log_psi shape"):
            d.write_block(np.zeros((4, 2, 3)), np.zeros(5))


def test_load_with_max_K_s_truncates(tmp_path):
    path = tmp_path / "w.h5"
    nb, nw, ne = 5, 4, 2
    with WalkerDumper(str(path), num_blocks=nb, num_walkers=nw, nelec=ne) as d:
        for _ in range(nb):
            d.write_block(np.zeros((nw, ne, 3)), np.zeros(nw))
    walkers, log_psi, _ = load_walker_bank(str(path), max_K_s=7)
    assert walkers.shape == (7, ne, 3)
    assert log_psi.shape == (7,)


def test_dumper_partial_write_then_close(tmp_path):
    """An aborted run still produces a readable file; unfilled rows are zeros
    (HDF5 default), which the loader exposes verbatim."""
    path = tmp_path / "w.h5"
    nb, nw, ne = 4, 3, 2
    with WalkerDumper(str(path), num_blocks=nb, num_walkers=nw, nelec=ne) as d:
        ones = np.ones((nw, ne, 3))
        d.write_block(ones, np.ones(nw))
        d.write_block(ones * 2, np.full(nw, 2.0))
    walkers, log_psi, meta = load_walker_bank(str(path))
    assert walkers.shape == (nb * nw, ne, 3)
    np.testing.assert_allclose(walkers[:nw], 1.0)
    np.testing.assert_allclose(walkers[nw:2 * nw], 2.0)
    np.testing.assert_allclose(walkers[2 * nw:], 0.0)
    assert int(meta["num_blocks"]) == nb


def test_vmc_nn_module_imports_after_hook_patch():
    """Catch any syntax error introduced by the dump_walkers patch."""
    import importlib
    import OmegaQMC.vmc_nn as mod
    importlib.reload(mod)
    assert hasattr(mod, "get_vmc_nn_func")
    sig = mod._VMCDriverNN.__call__.__doc__
    assert "dump_walkers_path" in sig
