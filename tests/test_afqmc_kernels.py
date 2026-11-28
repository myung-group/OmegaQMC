"""AFQMC fast paths against their reference implementations.

The specialised Green's-function variants and the slab-streamed
(memory-capped) local energies must equal the full versions.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto, scf

jax.config.update('jax_enable_x64', True)

from OmegaQMC.observables.greens import (  # noqa: E402
    greens_function,
    greens_function_force_bias,
    greens_function_overlap,
    greens_function_multidet,
)
from OmegaQMC.observables.energy import (  # noqa: E402
    local_energy,
    local_energy_2body,
    local_energy_2body_streamed,
    local_energy_streamed,
    local_energy_multidet,
    local_energy_multidet_streamed,
)


# ---------------------------------------------------------------------------
# From test_greens_specialized.py
#
# Regression: specialized Green's-function variants vs
# the canonical ``greens_function``.
#
# ``OmegaQMC.observables.greens`` exposes two specialized
# output-pruned variants of :func:`greens_function`:
#
# - :func:`greens_function_force_bias` — drops the full
#   ``G = trial.conj() @ Ghalf`` einsum (the
#   ``O(nwalkers · nbasis² · nocc)`` step), returning only
#   ``(Ghalfa, Ghalfb, overlap)``.  Used by the per-step
#   force-bias path inside AFQMC's ``propagate_walkers``.
#
# - :func:`greens_function_overlap` — drops the inverse and
#   Ghalf assembly entirely, returning only ``overlap`` via
#   ``slogdet`` of ``phi^T @ trial.conj()``.  Used by the
#   post-step weight update.
#
# This test pins the specialized outputs to the
# corresponding fields of :func:`greens_function` at 0.0
# absolute error in float64 across a batch of random
# walker configurations on H₂ / sto-3g.  Same shapes and
# dtypes as the AFQMC driver's per-step calls.
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def h2_trial():
    mol = gto.M(
        atom='H 0 0 0; H 0 0 1.4',
        basis='sto-3g', unit='Bohr', verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    nbasis = mol.nao
    nup = mol.nelec[0]
    ndn = mol.nelec[1]
    mo = jnp.asarray(mf.mo_coeff, dtype=jnp.complex128)
    trial_up = mo[:, :nup]
    trial_dn = mo[:, :ndn]
    return nbasis, nup, ndn, trial_up, trial_dn


@pytest.fixture(scope='module')
def walker_batch(h2_trial):
    nbasis, nup, ndn, trial_up, trial_dn = h2_trial
    rng = np.random.default_rng(7)
    nw = 8
    phia = (
        jnp.asarray(trial_up)[None, :, :].repeat(nw, axis=0)
        + 0.05 * jnp.asarray(
            rng.normal(size=(nw, nbasis, nup))
            + 1j * rng.normal(size=(nw, nbasis, nup)),
        )
    )
    phib = (
        jnp.asarray(trial_dn)[None, :, :].repeat(nw, axis=0)
        + 0.05 * jnp.asarray(
            rng.normal(size=(nw, nbasis, ndn))
            + 1j * rng.normal(size=(nw, nbasis, ndn)),
        )
    )
    return phia, phib


def test_force_bias_matches(h2_trial, walker_batch):
    _, _, _, trial_up, trial_dn = h2_trial
    phia, phib = walker_batch

    Ga, Gb, Ghalfa, Ghalfb, ovlp = greens_function(
        phia, phib, trial_up, trial_dn,
    )
    (Ghalfa_fb, Ghalfb_fb,
     ovlp_fb) = greens_function_force_bias(
        phia, phib, trial_up, trial_dn,
    )

    err_Ga = float(jnp.max(jnp.abs(Ghalfa - Ghalfa_fb)))
    err_Gb = float(jnp.max(jnp.abs(Ghalfb - Ghalfb_fb)))
    err_ov = float(jnp.max(jnp.abs(ovlp - ovlp_fb)))
    print(f"  Ghalfa  |Δ|={err_Ga:.3e}")
    print(f"  Ghalfb  |Δ|={err_Gb:.3e}")
    print(f"  overlap |Δ|={err_ov:.3e}")
    # Same numerical kernel — bit-for-bit equality.
    assert err_Ga == 0.0, ('Ghalfa', err_Ga)
    assert err_Gb == 0.0, ('Ghalfb', err_Gb)
    assert err_ov == 0.0, ('overlap', err_ov)


def test_overlap_matches(h2_trial, walker_batch):
    _, _, _, trial_up, trial_dn = h2_trial
    phia, phib = walker_batch

    _, _, _, _, ovlp = greens_function(
        phia, phib, trial_up, trial_dn,
    )
    ovlp_only = greens_function_overlap(
        phia, phib, trial_up, trial_dn,
    )
    err = float(jnp.max(jnp.abs(ovlp - ovlp_only)))
    print(f"  overlap |Δ|={err:.3e}")
    # Same slogdet — bit-for-bit equality.
    assert err == 0.0, ('overlap', err)


# ---------------------------------------------------------------------------
# From test_local_energy_streamed.py
#
# Pin streamed local-energy helpers against the original.
#
# ``local_energy_2body_streamed`` accumulates the exchange-trace in
# slabs and reorders the auxiliary-axis sum, so bit-equality is not
# expected; equality to round-off is: 1e-12 absolute, plus 1e-13
# relative for energies of a few hundred Ha (the reordering error grows
# with their magnitude, ~2e-15 relative).
#
# ``local_energy_streamed`` additionally takes a precontracted
# ``h1e @ trial.conj()`` instead of building full G, but the
# algebraic identity is exact — only the streamed-sum reordering
# introduces drift.
# ---------------------------------------------------------------------------


def _make_harness(seed=0, nwalkers=8, nbasis=10, nup=3, ndown=3,
                  naux=24):
    """Build a small consistent set of random tensors."""
    rng = np.random.default_rng(seed)

    def _crand(shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape))

    trial_up = jnp.asarray(_crand((nbasis, nup)))
    trial_dn = jnp.asarray(_crand((nbasis, ndown)))
    phia = jnp.asarray(_crand((nwalkers, nbasis, nup)))
    phib = jnp.asarray(_crand((nwalkers, nbasis, ndown)))
    rchol_a = jnp.asarray(_crand((naux, nup, nbasis)))
    rchol_b = jnp.asarray(_crand((naux, ndown, nbasis)))
    h1e = jnp.asarray(
        rng.standard_normal((nbasis, nbasis))
        + 1j * rng.standard_normal((nbasis, nbasis))
    )
    h1e = 0.5 * (h1e + h1e.conj().T)
    enuc = float(rng.standard_normal())
    return (trial_up, trial_dn, phia, phib,
            rchol_a, rchol_b, h1e, enuc)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("e_chunk_g", [4, 7, 24])
def test_local_energy_2body_streamed_matches(seed, e_chunk_g):
    (trial_up, trial_dn, phia, phib,
     rchol_a, rchol_b, h1e, enuc) = _make_harness(seed=seed)

    _, _, Ghalfa, Ghalfb, _ = greens_function(
        phia, phib, trial_up, trial_dn,
    )

    e_coul_ref, e_exch_ref = local_energy_2body(
        Ghalfa, Ghalfb, rchol_a, rchol_b,
    )
    e_coul_new, e_exch_new = local_energy_2body_streamed(
        Ghalfa, Ghalfb, rchol_a, rchol_b, e_chunk_g,
    )

    np.testing.assert_allclose(
        np.asarray(e_coul_new), np.asarray(e_coul_ref),
        rtol=1e-13, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(e_exch_new), np.asarray(e_exch_ref),
        rtol=1e-13, atol=1e-12,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_local_energy_streamed_matches(seed):
    (trial_up, trial_dn, phia, phib,
     rchol_a, rchol_b, h1e, enuc) = _make_harness(seed=seed)

    Ga, Gb, Ghalfa, Ghalfb, _ = greens_function(
        phia, phib, trial_up, trial_dn,
    )

    e_tot_ref, e_1b_ref, e_2b_ref = local_energy(
        h1e, Ga, Gb, Ghalfa, Ghalfb, rchol_a, rchol_b, enuc,
    )

    h1e_trial_a = h1e @ trial_up.conj()
    h1e_trial_b = h1e @ trial_dn.conj()
    e_tot_new, e_1b_new, e_2b_new = local_energy_streamed(
        h1e_trial_a, h1e_trial_b, Ghalfa, Ghalfb,
        rchol_a, rchol_b, enuc, 4,
    )

    np.testing.assert_allclose(
        np.asarray(e_1b_new), np.asarray(e_1b_ref),
        rtol=1e-13, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(e_2b_new), np.asarray(e_2b_ref),
        rtol=1e-13, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(e_tot_new), np.asarray(e_tot_ref),
        rtol=1e-13, atol=1e-12,
    )


def _make_multidet_harness(seed=0, nwalkers=8, nbasis=10,
                           nup=3, ndown=3, naux=24, ndet=3):
    """Build random multi-det tensors with consistent shapes."""
    rng = np.random.default_rng(seed)

    def _crand(shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape))

    trials_up = jnp.asarray(_crand((ndet, nbasis, nup)))
    trials_dn = jnp.asarray(_crand((ndet, nbasis, ndown)))
    phia = jnp.asarray(_crand((nwalkers, nbasis, nup)))
    phib = jnp.asarray(_crand((nwalkers, nbasis, ndown)))
    rchols_a = jnp.asarray(_crand((ndet, naux, nup, nbasis)))
    rchols_b = jnp.asarray(_crand((ndet, naux, ndown, nbasis)))
    ci_coeffs = jnp.asarray(_crand((ndet,)))
    h1e = jnp.asarray(
        rng.standard_normal((nbasis, nbasis))
        + 1j * rng.standard_normal((nbasis, nbasis))
    )
    h1e = 0.5 * (h1e + h1e.conj().T)
    enuc = float(rng.standard_normal())
    return (trials_up, trials_dn, phia, phib,
            rchols_a, rchols_b, ci_coeffs, h1e, enuc)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("e_chunk_g", [4, 7, 24])
@pytest.mark.parametrize("ndet", [1, 3, 5])
def test_local_energy_multidet_streamed_matches(
    seed, e_chunk_g, ndet,
):
    (trials_up, trials_dn, phia, phib,
     rchols_a, rchols_b, ci_coeffs, h1e, enuc) = \
        _make_multidet_harness(seed=seed, ndet=ndet)

    Ga, Gb, Ghalfa_all, Ghalfb_all, _, ovlp_a_all, ovlp_b_all = \
        greens_function_multidet(
            phia, phib, trials_up, trials_dn, ci_coeffs,
        )

    e_tot_ref, e_1b_ref, e_2b_ref = local_energy_multidet(
        h1e, Ga, Gb, Ghalfa_all, Ghalfb_all,
        rchols_a, rchols_b, ci_coeffs,
        ovlp_a_all, ovlp_b_all, enuc,
    )

    h1e_trials_a = jnp.einsum('pq,dqi->dpi', h1e, trials_up.conj())
    h1e_trials_b = jnp.einsum('pq,dqi->dpi', h1e, trials_dn.conj())

    e_tot_new, e_1b_new, e_2b_new = local_energy_multidet_streamed(
        h1e_trials_a, h1e_trials_b,
        Ghalfa_all, Ghalfb_all,
        rchols_a, rchols_b, ci_coeffs,
        ovlp_a_all, ovlp_b_all, enuc, e_chunk_g,
    )

    np.testing.assert_allclose(
        np.asarray(e_1b_new), np.asarray(e_1b_ref),
        rtol=0, atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(e_2b_new), np.asarray(e_2b_ref),
        rtol=0, atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(e_tot_new), np.asarray(e_tot_ref),
        rtol=0, atol=1e-10,
    )
