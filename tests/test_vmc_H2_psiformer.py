"""Integration test: NN VMC driver on H2.

Runs a short VMC simulation with the PsiFormer ansatz
on H2 and checks that the energy is finite and in a
reasonable range.
"""

import h5py
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmc_nn import get_vmc_nn_func


@pytest.fixture
def h2_driver():
    """Build a PsiFormer VMC driver for H2."""
    L = 1.4010
    mol = Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[
            [0.0, 0.0, -L / 2],
            [0.0, 0.0, L / 2],
        ],
        n_up=1,
        n_down=1,
    )
    init_key = jax.random.key(99)
    return get_vmc_nn_func(mol, 'psiformer', init_key)


def test_vmc_nn_h2_energy(h2_driver):
    """Energy is finite and in [-2, 0] Ha for H2."""
    run_key = jax.random.key(77)
    result = h2_driver(
        run_key,
        num_walkers=50,
        num_steps_per_block=20,
        num_blocks=5,
        num_blocks_equil=2,
        mc_timestep=0.1,
        verbose=0,
    )
    E = result['E_mean']
    assert jnp.isfinite(E), (
        f"Energy not finite: {E}"
    )
    assert -2.0 < E < 0.0, (
        f"Energy {E} outside [-2, 0] Ha for H2"
    )


def test_vmc_nn_h2_serr(h2_driver):
    """Statistical error is finite and positive."""
    run_key = jax.random.key(77)
    result = h2_driver(
        run_key,
        num_walkers=50,
        num_steps_per_block=20,
        num_blocks=5,
        num_blocks_equil=2,
        mc_timestep=0.1,
        verbose=0,
    )
    serr = result['E_serr']
    assert jnp.isfinite(serr), (
        f"Standard error not finite: {serr}"
    )
    assert serr > 0, (
        f"Standard error should be positive: {serr}"
    )


def test_vmc_nn_h2_blocks(h2_driver):
    """Correct number of block energies returned."""
    run_key = jax.random.key(77)
    n_blocks = 5
    result = h2_driver(
        run_key,
        num_walkers=50,
        num_steps_per_block=20,
        num_blocks=n_blocks,
        num_blocks_equil=2,
        mc_timestep=0.1,
        verbose=0,
    )
    assert len(result['E_blocks']) == n_blocks


def test_vmc_nn_h2_gradients(tmp_path):
    """ZVZB gradient components are recorded correctly
    in ``ofname_grd``.

    Runs a short H2 VMC with ``compute_gradients=True``,
    then checks the written ``<prefix>.grd.h5`` for:
    nuclear-nuclear gradient shape/value, cached
    system metadata (atom_symbols, atom_coords,
    charges, units), and per-block ZVZB decomposed
    gradient components (``grd_ee_en``, ``grd_ke``,
    ``grd_logpsi``) together with local-energy
    datasets, in the schema shared with
    ``save_gto_gradients``.

    Uses a tilted (off-axis) H2 geometry: the ZVZB
    estimator evaluates ``log|h_{ia,k}|`` where
    ``h = grad_R log|psi|``, so perfect C_inf_v symmetry
    of ψ would force ``h_x = h_y = 0`` and produce NaN
    components.
    """
    r0 = np.array([0.1, 0.0, -0.7])
    r1 = np.array([0.0, 0.1, 0.7])
    mol = Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[r0.tolist(), r1.tolist()],
        n_up=1,
        n_down=1,
    )
    init_key = jax.random.key(99)
    prefix = str(tmp_path / 'h2_grd')
    driver = get_vmc_nn_func(
        mol, 'psiformer', init_key, prefix=prefix,
    )

    n_blocks = 3
    num_walkers = 50
    num_steps_per_block = 10
    run_key = jax.random.key(77)
    driver(
        run_key,
        num_walkers=num_walkers,
        num_steps_per_block=num_steps_per_block,
        num_blocks=n_blocks,
        num_blocks_equil=2,
        mc_timestep=0.1,
        compute_gradients=True,
        verbose=0,
    )

    grd_path = prefix + '.grd.h5'
    with h5py.File(grd_path, 'r') as f:
        grd_nn = f['grd_nn'][:]
        atom_symbols = f['system/atom_symbols'][()]
        atom_coords = f['system/atom_coords'][:]
        charges = f['system/charges'][:]
        units = f['system/units'][()]
        grp_names = [
            'grd_ee_en', 'grd_ke', 'grd_logpsi',
        ]
        block_keys = sorted(
            f['grd_ee_en'].keys(), key=int,
        )
        grd_blocks = {
            name: [f[name][k][:] for k in block_keys]
            for name in grp_names
        }
        le_blocks = [
            f['local_energies'][k][:]
            for k in block_keys
        ]

    # Nuclear-nuclear gradient for Z=1 atoms:
    # grad_{R_0} (1/|R_0-R_1|) = (R_1 - R_0)/|R|^3;
    # atom 1 receives the opposite (Newton's 3rd law).
    dR = r1 - r0
    L3 = float(np.linalg.norm(dR)) ** 3
    expected_grd = np.stack([dR / L3, -dR / L3], axis=0)
    assert grd_nn.shape == (2, 3), (
        f"grd_nn shape {grd_nn.shape} != (2, 3)"
    )
    np.testing.assert_allclose(
        grd_nn, expected_grd, atol=1e-10,
    )

    # System metadata cached with expected shapes and
    # values.
    assert atom_symbols.decode().split() == ['H', 'H']
    assert atom_coords.shape == (2, 3)
    np.testing.assert_allclose(
        atom_coords, np.stack([r0, r1], axis=0),
    )
    np.testing.assert_allclose(charges, [1.0, 1.0])
    assert units.decode().upper() == 'BOHR'

    # Per-block decomposed ZVZB gradient datasets:
    # one per production block, with matching sample
    # counts and correct per-sample shape.  Gradient-
    # value finiteness is not asserted here — the ZVZB
    # estimator is numerically unstable for an
    # untrained PsiFormer (fourth-order derivatives of
    # log|psi|) and can return NaN; that is an
    # estimator-stability concern, not a recording
    # concern.  Local energies come from independent
    # ee/en/ke estimators and should always be finite,
    # and are stored in (steps, walkers) shape so that
    # downstream postprocessing can recover the
    # original sampling layout.
    for name in grp_names:
        assert len(grd_blocks[name]) == n_blocks, (
            f"Expected {n_blocks} {name} blocks, "
            f"got {len(grd_blocks[name])}"
        )
    assert len(le_blocks) == n_blocks
    expected_samples = num_steps_per_block * num_walkers
    for blk_idx in range(n_blocks):
        for name in grp_names:
            g = grd_blocks[name][blk_idx]
            assert g.shape == (
                expected_samples, 2, 3,
            ), (
                f"block {blk_idx + 1}: {name} shape "
                f"{g.shape} != "
                f"({expected_samples}, 2, 3)"
            )
        le = le_blocks[blk_idx]
        assert le.shape == (
            num_steps_per_block, num_walkers,
        ), (
            f"block {blk_idx + 1}: local_energies "
            f"shape {le.shape} != "
            f"({num_steps_per_block}, {num_walkers})"
        )
        assert np.all(np.isfinite(le)), (
            f"block {blk_idx + 1}: local_energies has "
            f"non-finite values"
        )


def test_vmc_nn_h2_gradients_pgcs(tmp_path):
    """PGCS combo data is recorded correctly when
    ``symmop_list`` is non-trivial.

    Uses the same tilted H2 geometry as
    ``test_vmc_nn_h2_gradients`` but activates Point
    Group Correlated Sampling via
    ``symmop_list=['E', 'C2z', 'i']``.  Verifies:

    1. ``system/atom_fragment_map`` is written.
    2. Per-combo subgroups appear under ``grd_ee_en``,
       ``grd_ke``, ``grd_logpsi``,
       ``fragment_weights``, and ``local_energies``
       with one dataset per block.
    3. Gradient combo datasets have shape
       ``(num_samples, 2, 3)``; fragment weights have
       shape ``(num_samples,)`` and are finite and
       non-negative.
    """
    r0 = np.array([0.1, 0.0, -0.7])
    r1 = np.array([0.0, 0.1, 0.7])
    mol = Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[r0.tolist(), r1.tolist()],
        n_up=1,
        n_down=1,
    )
    init_key = jax.random.key(99)
    prefix = str(tmp_path / 'h2_pgcs')
    vmc_run = get_vmc_nn_func(
        mol, 'psiformer', init_key, prefix=prefix,
        symmop_list=['E', 'C2z', 'i'],
    )
    expected_combos = [c[2] for c in vmc_run.single_frag_combos]
    assert len(expected_combos) == 2, (
        f"Expected 2 non-E combos for "
        f"['E', 'C2z', 'i'], got {expected_combos}"
    )

    n_blocks = 2
    num_walkers = 50
    num_steps_per_block = 10
    run_key = jax.random.key(77)
    vmc_run(
        run_key,
        num_walkers=num_walkers,
        num_steps_per_block=num_steps_per_block,
        num_blocks=n_blocks,
        num_blocks_equil=2,
        mc_timestep=0.1,
        compute_gradients=True,
        verbose=0,
    )

    grd_path = prefix + '.grd.h5'
    expected_samples = num_steps_per_block * num_walkers
    with h5py.File(grd_path, 'r') as f:
        assert 'atom_fragment_map' in f['system'], (
            "atom_fragment_map missing from system group"
        )
        afmap = f['system']['atom_fragment_map'][:]
        assert list(afmap) == [0, 0]

        for label in expected_combos:
            for grp in (
                'grd_ee_en', 'grd_ke',
                'grd_logpsi', 'fragment_weights',
                'local_energies',
            ):
                assert label in f[grp], (
                    f"combo '{label}' missing from "
                    f"'{grp}'"
                )
                block_keys = sorted(
                    f[grp][label].keys(), key=int,
                )
                assert len(block_keys) == n_blocks, (
                    f"{grp}/{label}: expected "
                    f"{n_blocks} blocks, got "
                    f"{len(block_keys)}"
                )

            for blk_key in sorted(
                f['grd_ee_en'][label].keys(), key=int,
            ):
                for grp in (
                    'grd_ee_en', 'grd_ke', 'grd_logpsi',
                ):
                    g = f[grp][label][blk_key][:]
                    assert g.shape == (
                        expected_samples, 2, 3,
                    ), (
                        f"{grp}/{label}/{blk_key}: "
                        f"shape {g.shape} != "
                        f"({expected_samples}, 2, 3)"
                    )
                w = f['fragment_weights'][label][
                    blk_key
                ][:]
                assert w.shape == (expected_samples,), (
                    f"fragment_weights/{label}/"
                    f"{blk_key}: shape {w.shape} != "
                    f"({expected_samples},)"
                )
                assert np.all(np.isfinite(w)), (
                    f"fragment_weights/{label}/"
                    f"{blk_key}: non-finite"
                )
                assert np.all(w >= 0.0), (
                    f"fragment_weights/{label}/"
                    f"{blk_key}: negative weight"
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
