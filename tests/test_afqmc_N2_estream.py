"""
Compare _AFQMCDriverGTO_EStream vs. _AFQMCDriverGTO on N2/cc-pVDZ.

The streamed exchange-trace reorders accumulation along the
auxiliary axis, so block energies are not bit-identical with
the legacy driver.  Stable-mean equality is expected to ~1e-8.
"""

import jax
from pyscf import gto, scf

from OmegaQMC.afqmc_gto import _AFQMCDriverGTO
from OmegaQMC.afqmc_gto_estream import _AFQMCDriverGTO_EStream


def _build_mf():
    mol = gto.M(
        atom='N 0 0 0; N 0 0 2.074',
        basis='cc-pvdz',
        unit='Bohr',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mf


def test_n2_ccpvdz_estream_matches_baseline():
    mf = _build_mf()

    kwargs_run = dict(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=20,
        num_steps_per_block=10,
        stabilize_freq=5,
        pop_control_freq=5,
        num_eqlb_blocks=5,
        fname_log=None,
    )

    drv_old = _AFQMCDriverGTO(mf, dt=0.005, chol_cut=1e-6,
                              verbose=False)
    res_old = drv_old(**kwargs_run)

    drv_new = _AFQMCDriverGTO_EStream(mf, dt=0.005, chol_cut=1e-6,
                                      verbose=False, e_chunk_g=8)
    res_new = drv_new(**kwargs_run)

    e_old = res_old['energy_mean']
    e_new = res_new['energy_mean']
    err = max(res_old['energy_err'], res_new['energy_err'], 1e-10)

    print(f"E_old     = {e_old:.10f} +/- {res_old['energy_err']:.2e}")
    print(f"E_estream = {e_new:.10f} +/- {res_new['energy_err']:.2e}")
    print(f"|ΔE|      = {abs(e_old - e_new):.2e}  (3σ = {3 * err:.2e})")

    # Block means should agree well within their statistical noise.
    assert abs(e_old - e_new) < 3 * err, (
        f"Energy mean drifted beyond 3σ: "
        f"{e_old:.10f} vs {e_new:.10f}, σ ~ {err:.2e}"
    )


def test_estream_rejects_multidet():
    mf = _build_mf()
    fake_trial = {
        'ndet': 2,
        'mo_coeff': mf.mo_coeff,
        'ci_coeffs': None,
        'occ_up': None,
        'occ_dn': None,
    }
    try:
        _AFQMCDriverGTO_EStream(mf, trial=fake_trial)
    except TypeError as e:
        # ``trial`` is not in the EStream driver's signature.
        assert 'trial' in str(e) or 'keyword' in str(e)
        return
    raise AssertionError(
        "_AFQMCDriverGTO_EStream must reject 'trial' kwarg"
    )


if __name__ == '__main__':
    test_n2_ccpvdz_estream_matches_baseline()
    test_estream_rejects_multidet()
    print("All N2 EStream driver-equivalence tests passed.")
