"""Tests for OmegaQMC.qed_physics_heg (velocity-gauge Pauli-Fierz for HEG).

Three physics-limit checks on the velocity-gauge local-energy estimator:

  1. λ=0, n=0 → exactly the bare HEG E_loc (KE + Ewald V_ee).
  2. λ=0, n=k → bare HEG E_loc + Ω·k (only the photon energy contributes).
  3. λ→0+, vacuum-like trial → diamagnetic shift = N·λ²/(4Ω).

Tests use a hand-built positive-real Gaussian Slater × δ-like Fock
trial, so any deviation from the analytical prediction has to come
from the cavity-coupling code itself (no NN artefacts, no MCMC noise).

References
----------
- Module docstring of :mod:`OmegaQMC.qed_physics_heg` for the
  velocity-gauge PF Hamiltonian and Fock-ladder ratio formulas.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.qed_physics_heg import (
    pauli_fierz_local_energy_velocity_heg,
    _ke_real,
)
from OmegaQMC.observables.ewald_dispatch import (
    build_ewald_tables_dim, ewald_pair_energy_dim,
)


@pytest.fixture
def heg_2d_setup():
    """Small 2D HEG-like system + Gaussian-product trial wavefunction."""
    rs = 10.0
    n_up = n_down = 6
    n_elec = n_up + n_down
    dim = 2
    L = float(np.sqrt(np.pi * n_elec) * rs)
    omega = 0.1
    nph_max = 4
    sigma = 0.5 * L

    tables = build_ewald_tables_dim(L, dim=dim, n_real=3, n_recip=6)
    ewald_fn = lambda r: ewald_pair_energy_dim(r, tables, dim=dim)

    def log_psi_signed(elec, n, params):
        # Real positive Gaussian × rapidly decaying photon weight (≈ |0⟩).
        log_mag_e = -0.5 * jnp.sum((elec / sigma) ** 2)
        log_mag_n = -10.0 * jnp.asarray(n, dtype=elec.dtype)
        return log_mag_e + log_mag_n, jnp.asarray(1.0 + 0.0j)

    def phase_smooth(elec, n, params):
        # Trivial smooth phase = 0 for all (r, n).
        return jnp.asarray(0.0, dtype=elec.dtype)

    def log_mag_only(elec):
        return -0.5 * jnp.sum((elec / sigma) ** 2)

    rng = jax.random.PRNGKey(0)
    elec = jax.random.uniform(rng, (n_elec, dim), minval=0.0, maxval=L)

    return {
        "n_elec": n_elec, "dim": dim, "L": L, "omega": omega,
        "nph_max": nph_max, "ewald_fn": ewald_fn,
        "log_psi_signed": log_psi_signed, "phase_smooth": phase_smooth,
        "log_mag_only": log_mag_only,
        "elec": elec,
    }


def _bare_heg_e_loc(setup):
    return float(_ke_real(setup["log_mag_only"], setup["elec"])
                 + setup["ewald_fn"](setup["elec"]))


def test_zero_coupling_n0_reduces_to_bare_heg(heg_2d_setup):
    """λ=0, n=0 → E_loc = (KE + Ewald) exactly."""
    s = heg_2d_setup
    e0 = pauli_fierz_local_energy_velocity_heg(
        s["log_psi_signed"], None, s["elec"], jnp.asarray(0),
        omega=s["omega"], coupling_vec=jnp.zeros(s["dim"]),
        nph_max=s["nph_max"], ewald_pair_fn=s["ewald_fn"],
        complex_psi=True, phase_smooth_fn=s["phase_smooth"],
    )
    assert abs(float(jnp.real(e0)) - _bare_heg_e_loc(s)) < 1e-10


def test_zero_coupling_finite_n_picks_up_photon_energy(heg_2d_setup):
    """λ=0, n=k → E_loc = bare HEG + Ω·k (no other coupling)."""
    s = heg_2d_setup
    bare = _bare_heg_e_loc(s)
    for k in (1, 2, 3):
        e_k = pauli_fierz_local_energy_velocity_heg(
            s["log_psi_signed"], None, s["elec"], jnp.asarray(k),
            omega=s["omega"], coupling_vec=jnp.zeros(s["dim"]),
            nph_max=s["nph_max"], ewald_pair_fn=s["ewald_fn"],
            complex_psi=True, phase_smooth_fn=s["phase_smooth"],
        )
        assert abs(float(jnp.real(e_k)) - (bare + s["omega"] * k)) < 1e-8


def test_diamagnetic_shift_at_small_lambda(heg_2d_setup):
    """Vacuum-trial diamagnetic shift = N·λ²/(4Ω) in lowest order."""
    s = heg_2d_setup
    eps = jnp.array([1.0, 0.0])
    e0 = pauli_fierz_local_energy_velocity_heg(
        s["log_psi_signed"], None, s["elec"], jnp.asarray(0),
        omega=s["omega"], coupling_vec=jnp.zeros(s["dim"]),
        nph_max=s["nph_max"], ewald_pair_fn=s["ewald_fn"],
        complex_psi=True, phase_smooth_fn=s["phase_smooth"],
    )
    lam = 0.01
    e_lam = pauli_fierz_local_energy_velocity_heg(
        s["log_psi_signed"], None, s["elec"], jnp.asarray(0),
        omega=s["omega"], coupling_vec=lam * eps,
        nph_max=s["nph_max"], ewald_pair_fn=s["ewald_fn"],
        complex_psi=True, phase_smooth_fn=s["phase_smooth"],
    )
    g = lam / np.sqrt(2.0 * s["omega"])
    expected = s["n_elec"] * g ** 2 / 2.0
    actual = float(jnp.real(e_lam)) - float(jnp.real(e0))
    rel = abs(actual - expected) / abs(expected)
    assert rel < 1e-6, f"diamag rel err {rel:.2e}"
