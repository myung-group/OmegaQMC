"""Electron-gas VMC driver, at the Gamma point and with twist averaging.

The drivers compile and step; VMC at the HF initialization reproduces
the finite-cell HF energy; energies and error bars are finite at every
twist.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from OmegaQMC.psi.nn.heg_wf import (  # noqa: E402
    HEGConfig,
    make_heg_log_psi,
)
from OmegaQMC.vmc_nn_heg import (  # noqa: E402
    get_vmc_nn_heg_func,
    get_vmc_nn_heg_twist_func,
)
from OmegaQMC.afqmc_pw_heg import (  # noqa: E402
    build_3deg_system,
    get_afqmc_3deg_func,
)


# ---------------------------------------------------------------------------
# From test_vmc_heg.py
#
# Integration tests for the HEG VMC driver.
#
# Smoke tests:
#   * Driver constructs and local-energy evaluation compiles.
#   * One Metropolis move step runs end-to-end.
#
# A heavier test that runs enough blocks to compare against HF is
# gated behind ``pytest -m slow`` and lives in
# :func:`test_heg_vmc_init_is_close_to_hf`.
# ---------------------------------------------------------------------------


# -----------------------------------------------------------------
# Smoke tests (fast)
# -----------------------------------------------------------------
def test_heg_log_psi_shapes_and_finite():
    """log_psi of the HEG ansatz is a finite scalar."""
    rs = 2.0
    sys = build_3deg_system(rs, N_elec=14, N_pw=7,
                            polarization='unpolarized')
    L = sys['L']
    config = HEGConfig(
        n_up=7, n_down=7, L=L, n_det=1, use_jastrow=False,
    )
    log_psi, params, _ = make_heg_log_psi(config, jax.random.key(0))
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.uniform(0, L, size=(14, 3)))
    lp = float(log_psi(r, params))
    assert np.isfinite(lp)


def test_heg_driver_local_energy_compiles():
    rs = 2.0
    sys = build_3deg_system(rs, N_elec=14, N_pw=7,
                            polarization='unpolarized')
    L = sys['L']
    config = HEGConfig(
        n_up=7, n_down=7, L=L, n_det=1, use_jastrow=False,
    )
    driver = get_vmc_nn_heg_func(
        config, jax.random.key(0),
        ewald_n_real=2, ewald_n_recip=4,
    )
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.uniform(0, L, size=(14, 3)))
    e_loc = float(driver.local_energy(r, driver.params))
    assert np.isfinite(e_loc)


def test_heg_driver_metropolis_step_runs():
    rs = 2.0
    sys = build_3deg_system(rs, N_elec=14, N_pw=7,
                            polarization='unpolarized')
    L = sys['L']
    config = HEGConfig(
        n_up=7, n_down=7, L=L, n_det=1, use_jastrow=False,
    )
    driver = get_vmc_nn_heg_func(
        config, jax.random.key(0),
        ewald_n_real=2, ewald_n_recip=4,
    )
    walkers = driver.initialize_walkers(jax.random.key(1), 4)
    keys = jax.random.split(jax.random.key(2), 4)
    new_w, acc = driver._metropolis_move_allw(
        keys, walkers, 0.3, driver.params,
    )
    assert new_w.shape == walkers.shape
    assert acc.shape == (4,)
    # Walkers remain in [0, L) after wrapping.
    assert float(new_w.min()) >= -1e-9
    assert float(new_w.max()) < L + 1e-9


# -----------------------------------------------------------------
# HF-sanity end-to-end test (slow; gated)
# -----------------------------------------------------------------
@pytest.mark.slow
def test_heg_vmc_init_matches_finite_cell_hf():
    """The plane-wave envelope initialised to the non-interacting
    Fermi sea has |ψ|² equal to the finite-cell Hartree–Fock
    Slater determinant.  Its VMC total energy (kinetic + Ewald
    Coulomb + Madelung) must therefore agree with the AFQMC trial
    energy for the same system at fixed (N, rs)."""
    rs = 2.0
    N = 14
    sys = build_3deg_system(rs, N_elec=N, N_pw=7,
                            polarization='unpolarized')
    L = sys['L']

    # Finite-cell HF reference: AFQMC's HF Slater-determinant
    # trial energy includes the Madelung self-energy.
    afqmc = get_afqmc_3deg_func(
        sys, dt=0.005, include_coulomb=True, verbose=False,
    )
    e_hf_finite_ha = float(afqmc.e_trial) / N

    config = HEGConfig(
        n_up=7, n_down=7, L=L, n_det=1, use_jastrow=False,
    )
    driver = get_vmc_nn_heg_func(
        config, jax.random.key(0),
        ewald_n_real=3, ewald_n_recip=6,
    )
    result = driver(
        jax.random.key(1),
        num_walkers=128,
        num_steps_per_block=30,
        num_blocks=30,
        num_blocks_equil=15,
        mc_timestep=0.1,
        verbose=0,
    )
    e_vmc_ha = result['E_per_elec_ha']
    # Finite-MCMC statistical error + envelope Ewald-vs-Fourier
    # convergence mismatch.  10 mHa/elec is well inside variance.
    assert abs(e_vmc_ha - e_hf_finite_ha) < 0.010, (
        f"VMC E/N = {e_vmc_ha:.4f} Ha, "
        f"AFQMC HF E/N = {e_hf_finite_ha:.4f} Ha"
    )


# ---------------------------------------------------------------------------
# From test_vmc_heg_twist.py
#
# Tests for the twist-aware (complex-ansatz) HEG VMC driver.
#
# Ensures:
#   * Driver constructs, local-energy evaluation compiles.
#   * At κ = 0, the complex-ansatz *finite-cell HF* energy matches
#     the real-ansatz finite-cell HF energy within MCMC noise (this
#     is the non-trivial correctness check — same physics, two
#     representations).
#   * At κ ≠ 0, the energy at init is in a physically reasonable
#     range (bracketed by Γ-point values).
# ---------------------------------------------------------------------------


def test_twist_driver_constructs_and_local_energy_finite():
    rs = 2.0
    sys = build_3deg_system(rs, N_elec=14, N_pw=7,
                            polarization='unpolarized')
    L = sys['L']
    cfg = HEGConfig(n_up=7, n_down=7, L=L, n_det=1,
                    use_jastrow=False)
    drv = get_vmc_nn_heg_twist_func(
        cfg, jax.random.key(0),
        kappa=(0.0, 0.0, 0.0),
        ewald_n_real=2, ewald_n_recip=4,
    )
    rng = np.random.default_rng(0)
    import jax.numpy as jnp
    r = jnp.asarray(rng.uniform(0, L, size=(14, 3)))
    e_loc = float(drv.local_energy(r, drv.params))
    assert np.isfinite(e_loc)


def test_tabc_sweep_smoke():
    """TABC driver runs end-to-end for 2 twists with no trained
    Jastrow and produces finite per-twist energies with non-zero
    twist-to-twist scatter."""
    from OmegaQMC.vmc_nn_heg import run_twist_averaged_heg

    rs = 2.0
    sys = build_3deg_system(rs, N_elec=14, N_pw=7,
                            polarization='unpolarized')
    cfg = HEGConfig(n_up=7, n_down=7, L=sys['L'], n_det=1,
                    use_jastrow=False)

    tabc = run_twist_averaged_heg(
        cfg, jax.random.key(0),
        trained_params_real=None,
        n_twists=2,
        ewald_n_real=2, ewald_n_recip=4,
        num_walkers=32,
        num_steps_per_block=10,
        num_blocks=5,
        num_blocks_equil=3,
        mc_timestep=0.1,
        eval_seed=1,
        verbose=0,
    )
    assert tabc['twists'].shape == (2, 3)
    assert tabc['energies_per_twist'].shape == (2,)
    assert np.all(np.isfinite(tabc['energies_per_twist']))
    assert np.all(np.isfinite(tabc['errors_per_twist']))
    # At N=14, rs=2, two distinct twists cannot give exactly the
    # same energy — the Fermi-sea shell structure is different.
    assert (tabc['energies_per_twist'].max()
            - tabc['energies_per_twist'].min()) > 1e-4


@pytest.mark.slow
def test_twist_at_gamma_matches_real_hf():
    """The complex-ansatz energy at κ = (0,0,0) with no Jastrow must
    agree with the AFQMC finite-cell HF reference — same physics,
    different representation.  MCMC noise is the dominant error."""
    rs = 2.0
    N = 14
    sys = build_3deg_system(rs, N_elec=N, N_pw=7,
                            polarization='unpolarized')
    L = sys['L']

    e_hf_ha = float(get_afqmc_3deg_func(
        sys, dt=0.005, include_coulomb=True, verbose=False,
    ).e_trial) / N

    cfg = HEGConfig(n_up=7, n_down=7, L=L, n_det=1,
                    use_jastrow=False)
    drv = get_vmc_nn_heg_twist_func(
        cfg, jax.random.key(0),
        kappa=(0.0, 0.0, 0.0),
        ewald_n_real=3, ewald_n_recip=6,
    )
    result = drv(
        jax.random.key(1),
        num_walkers=128,
        num_steps_per_block=30,
        num_blocks=30,
        num_blocks_equil=15,
        mc_timestep=0.1,
        verbose=0,
    )
    assert abs(result['E_per_elec_ha'] - e_hf_ha) < 0.010, (
        f"κ=0 complex E/N = {result['E_per_elec_ha']:.4f} Ha, "
        f"AFQMC HF = {e_hf_ha:.4f} Ha"
    )
