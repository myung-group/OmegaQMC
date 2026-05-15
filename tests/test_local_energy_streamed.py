"""
Pin streamed local-energy helpers against the original.

``local_energy_2body_streamed`` accumulates the exchange-trace in
slabs and reorders the auxiliary-axis sum, so bit-equality is not
expected; equality to ~1e-12 is.

``local_energy_streamed`` additionally takes a precontracted
``h1e @ trial.conj()`` instead of building full G, but the
algebraic identity is exact — only the streamed-sum reordering
introduces drift.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from OmegaQMC.observables.energy import (
    local_energy,
    local_energy_2body,
    local_energy_2body_streamed,
    local_energy_streamed,
    local_energy_multidet,
    local_energy_multidet_streamed,
)
from OmegaQMC.observables.greens import (
    greens_function,
    greens_function_multidet,
)


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
        rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(e_exch_new), np.asarray(e_exch_ref),
        rtol=0, atol=1e-12,
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
        rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(e_2b_new), np.asarray(e_2b_ref),
        rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(e_tot_new), np.asarray(e_tot_ref),
        rtol=0, atol=1e-12,
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


if __name__ == '__main__':
    test_local_energy_2body_streamed_matches(0, 4)
    test_local_energy_streamed_matches(0)
    test_local_energy_multidet_streamed_matches(0, 4, 3)
    print("All streamed-energy parity tests passed.")
