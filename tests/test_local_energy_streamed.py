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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from OmegaQMC.observables.energy import (
    local_energy,
    local_energy_2body,
    local_energy_2body_streamed,
    local_energy_streamed,
)
from OmegaQMC.observables.greens import greens_function


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


if __name__ == '__main__':
    test_local_energy_2body_streamed_matches(0, 4)
    test_local_energy_streamed_matches(0)
    print("All streamed-energy parity tests passed.")
