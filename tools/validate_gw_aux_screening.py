"""Validate the auxiliary-basis QED-dRPA dielectric against the dense path.

Checks, on H2O/6-31g inside a cavity (and at lambda = 0):

1. Screened-interaction identity (machine precision):
       Wt_pm(i w') = sum_s M_pms^2 * 2 Omega_s / (-w'^2 - Omega_s^2)
   where (Omega, M) come from the dense `_screening_at_eps` eigensolve and
   the left-hand side from the Woodbury dielectric `_aux_wtilde_grid`, for
   all four (include_dse, include_photon) flag combinations.

2. Sigma_c(omega): dense sum-over-poles (`_eval_sigma`) vs contour
   deformation (`_sigma_aux_cd`) at off-pole test frequencies.

3. G0W0 and evGW quasiparticle energies: dense vs aux-cd (expected to
   agree to quadrature accuracy) and vs aux-pade (valence states).

4. The same on a density-fitted reference (auxbasis='weigend').

Run:  /Users/willow/miniforge3/bin/python tools/qed_ccsd_df_derivation/validate_gw_aux_screening.py
"""

import numpy as np
from pyscf import gto

from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons.qed_gw import _eval_sigma
from OmegaQMC.addons.qed_polariton_singlet import (
    _build_spatial_quantities, _screening_at_eps, _aux_setup,
    _aux_wtilde_grid, _sigma_aux_cd, _imag_freq_grid, _wt_spline,
    _aux_w_static_blocks, run_qed_gw_singlet,
    run_qed_bse_polaritonic_singlet,
)

EV = 27.211386245988


def _h2o(basis='6-31g'):
    return gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
                 basis=basis, verbose=0)


def check_wtilde_identity(qedhf, freqs):
    """Dense SOP vs aux-dielectric Wt_pm(i w') for all flag combinations."""
    sq = _build_spatial_quantities(qedhf)
    eps = np.diag(sq['F'])
    worst = 0.0
    for dse in (True, False):
        for ph in (True, False):
            Omega, M = _screening_at_eps(sq, eps, include_dse=dse,
                                         include_photon=ph)
            setup = _aux_setup(sq, eps, include_dse=dse, include_photon=ph)
            Wt_aux = _aux_wtilde_grid(setup, freqs)
            pole = 2.0 * Omega / (-freqs[:, None] ** 2 - Omega[None, :] ** 2)
            Wt_dense = np.einsum('pqs,ks->kpq', M * M, pole)
            err = float(np.max(np.abs(Wt_aux - Wt_dense)))
            scale = float(np.max(np.abs(Wt_dense)))
            print(f"    dse={str(dse):5s} photon={str(ph):5s}  "
                  f"max|dWt| = {err:.3e}   (max|Wt| = {scale:.3e})")
            worst = max(worst, err)
    return worst


def check_sigma_cd(qedhf, eta=1e-3, n_freq=200, freq_scale=0.5):
    """Dense Sigma_c vs contour-deformation Sigma_c at test frequencies."""
    sq = _build_spatial_quantities(qedhf)
    eps = np.diag(sq['F'])
    no, nmo = sq['nocc'], sq['nmo']
    Omega, M = _screening_at_eps(sq, eps)

    quad_f, quad_w = _imag_freq_grid(n_freq, freq_scale)
    freqs_all = np.concatenate([[0.0], quad_f])
    setup = _aux_setup(sq, eps)
    Wt = _aux_wtilde_grid(setup, freqs_all)

    worst = 0.0
    for p in (0, no - 1, no, nmo - 1):
        spl = _wt_spline(freqs_all, Wt[:, p, :])
        for shift in (-0.15, -0.05, 0.0, 0.05, 0.15):
            w = eps[p] + shift
            s_dense = _eval_sigma(w, M[p], Omega, eps, no, eta)[0].real
            s_cd = _sigma_aux_cd(setup, Wt, freqs_all, quad_w, spl,
                                 p, w, eps, eta)
            err = abs(s_cd - s_dense)
            worst = max(worst, err)
            print(f"    p={p:2d}  w=eps{shift:+.2f}: "
                  f"Sigma dense = {s_dense:+.8f}  cd = {s_cd:+.8f}  "
                  f"|d| = {err:.2e}")
    return worst


def check_qp(qedhf, mode, no, **kw):
    """QP energies: dense vs aux-cd vs aux-pade."""
    res = {}
    for scr in ('dense', 'aux-cd', 'aux-pade'):
        res[scr] = run_qed_gw_singlet(qedhf, mode=mode, verbose=False,
                                      screening=scr, **kw)['eps_QP']
    d_cd = np.abs(res['aux-cd'] - res['dense'])
    d_pd = np.abs(res['aux-pade'] - res['dense'])
    # Pade is a near-gap method: judge it on states within 1 Ha of the
    # gap centre (its documented domain); CD covers the full spectrum.
    mu = 0.5 * (res['dense'][no - 1] + res['dense'][no])
    near = np.abs(res['dense'] - mu) < 1.0
    print(f"    {mode}:  max|cd - dense|              = "
          f"{d_cd.max():.3e} Ha")
    print(f"    {mode}:  max|pade - dense|(near-gap)  = "
          f"{d_pd[near].max():.3e} Ha  ({int(near.sum())}/{len(near)} states)")
    print(f"    {mode}:  |pade - dense| HOMO/LUMO     = "
          f"{d_pd[no-1]:.3e} / {d_pd[no]:.3e} Ha")
    print(f"    HOMO/LUMO (dense): {res['dense'][no-1]:+.6f} / "
          f"{res['dense'][no]:+.6f} Ha")
    return d_cd.max(), d_pd[near].max()


def check_w_static(qedhf):
    """Static BSE screened-interaction blocks: aux (one nu=0 Woodbury
    inversion) vs dense (sum over the nov-dim dRPA poles)."""
    sq = _build_spatial_quantities(qedhf)
    eps = np.diag(sq['F'])
    no = sq['nocc']
    Omega, M = _screening_at_eps(sq, eps, include_dse=True,
                                 include_photon=False)
    inv_Om = 1.0 / Omega
    ref_ijab = -2.0 * np.einsum('ijs,abs,s->ijab', M[:no, :no, :],
                                M[no:, no:, :], inv_Om)
    ref_ibaj = -2.0 * np.einsum('ibs,ajs,s->ibaj', M[:no, no:, :],
                                M[no:, :no, :], inv_Om)
    Wc_ijab, Wc_ibaj = _aux_w_static_blocks(sq, eps)
    e1 = float(np.max(np.abs(Wc_ijab - ref_ijab)))
    e2 = float(np.max(np.abs(Wc_ibaj - ref_ibaj)))
    print(f"    max|dW_ijab| = {e1:.3e}   max|dW_ibaj| = {e2:.3e}")
    return max(e1, e2)


def check_bse(qedhf):
    """Polaritonic BSE spectra: full dense pipeline vs full aux pipeline
    (aux-cd GW quasiparticles + aux static W)."""
    dense = run_qed_bse_polaritonic_singlet(qedhf, gw_mode='evGW',
                                            verbose=False)
    aux = run_qed_bse_polaritonic_singlet(qedhf, gw_mode='evGW',
                                          verbose=False,
                                          gw_screening='aux-cd',
                                          w_screening='aux')
    d_om = float(np.max(np.abs(dense['Omega'] - aux['Omega'])))
    d_f = float(np.max(np.abs(dense['f_osc'] - aux['f_osc'])))
    print(f"    max|dOmega| = {d_om:.3e} Ha   max|df_osc| = {d_f:.3e}")
    print(f"    first roots (eV): {(aux['Omega'][:4] * EV).round(3)}")
    return max(d_om, d_f)


def check_davidson(qedhf, nroots=10):
    """Matrix-free paired Davidson vs dense diagonalisation (same QP)."""
    eps_QP = run_qed_gw_singlet(qedhf, mode='evGW',
                                verbose=False)['eps_QP']
    worst = 0.0
    for tda in (False, True):
        dense = run_qed_bse_polaritonic_singlet(qedhf, eps_QP=eps_QP,
                                                tda=tda, verbose=False)
        dav = run_qed_bse_polaritonic_singlet(qedhf, eps_QP=eps_QP,
                                              tda=tda, verbose=False,
                                              solver='davidson',
                                              nroots=nroots,
                                              davidson_tol=1e-8)
        n = len(dav['Omega'])
        d_om = float(np.max(np.abs(dense['Omega'][:n] - dav['Omega'])))
        d_f = float(np.max(np.abs(dense['f_osc'][:n] - dav['f_osc'])))
        d_w = float(np.max(np.abs(dense['photon_weight'][:n]
                                  - dav['photon_weight'])))
        print(f"    tda={str(tda):5s}: max|dOmega| = {d_om:.3e} Ha   "
              f"max|df_osc| = {d_f:.3e}   max|dph_wt| = {d_w:.3e}")
        worst = max(worst, d_om, d_f, d_w)
    return worst


def main():
    mol = _h2o()
    omega = 0.4
    lam = (0.0, 0.0, 0.05)

    freqs = np.array([0.0, 0.05, 0.3, 1.0, 5.0])

    print("=" * 72)
    print("H2O/6-31g, w_cav = 0.4 Ha, lambda = (0, 0, 0.05), exact ERI factor")
    print("=" * 72)
    qedhf = run_qed_hf(mol, omega, lam, verbose=False)

    print("\n[1] Wt identity: aux dielectric vs dense sum-over-poles")
    e1 = check_wtilde_identity(qedhf, freqs)

    print("\n[2] Sigma_c(omega): contour deformation vs dense (eta = 1e-3)")
    e2 = check_sigma_cd(qedhf)

    no = qedhf['nocc_spatial']
    print("\n[3] Quasiparticle energies (n_freq = 200)")
    e3 = []
    for mode in ('G0W0', 'evGW'):
        e3.append(check_qp(qedhf, mode, no, n_freq=200))

    print("\n" + "=" * 72)
    print("lambda = 0 (purely electronic limit)")
    print("=" * 72)
    qedhf0 = run_qed_hf(mol, omega, (0.0, 0.0, 0.0), verbose=False)
    print("\n[4] Wt identity")
    e4 = check_wtilde_identity(qedhf0, freqs)
    print("\n[5] Quasiparticle energies")
    e5 = check_qp(qedhf0, 'evGW', no, n_freq=200)

    print("\n" + "=" * 72)
    print("Density-fitted reference (auxbasis='weigend'), same cavity")
    print("=" * 72)
    qedhf_df = run_qed_hf(mol, omega, lam, verbose=False, auxbasis='weigend')
    print("\n[6] Wt identity")
    e6 = check_wtilde_identity(qedhf_df, freqs)
    print("\n[7] Quasiparticle energies")
    e7 = check_qp(qedhf_df, 'evGW', no, n_freq=200)

    print("\n" + "=" * 72)
    print("Static-W BSE from the aux dielectric")
    print("=" * 72)
    print("\n[8] W^stat correlation blocks: aux (nu=0 Woodbury) vs dense")
    e8 = check_w_static(qedhf)
    print("\n[9] pol-BSE spectrum: full aux pipeline vs full dense pipeline")
    e9 = check_bse(qedhf)
    print("\n[10] pol-BSE lowest roots: paired Davidson (matrix-free) "
          "vs dense")
    e10 = check_davidson(qedhf)

    print("\n" + "=" * 72)
    ok = True
    for name, err, tol in (
            ('Wt identity (cavity)', e1, 1e-10),
            ('Sigma CD vs dense', e2, 5e-5),
            ('G0W0 cd vs dense', e3[0][0], 1e-5),
            ('evGW cd vs dense', e3[1][0], 1e-5),
            ('G0W0 pade vs dense (near-gap)', e3[0][1], 2e-4),
            ('evGW pade vs dense (near-gap)', e3[1][1], 2e-4),
            ('Wt identity (lambda=0)', e4, 1e-10),
            ('evGW cd vs dense (lambda=0)', e5[0], 1e-5),
            ('Wt identity (DF)', e6, 1e-10),
            ('evGW cd vs dense (DF)', e7[0], 1e-5),
            ('W static blocks aux vs dense', e8, 1e-10),
            ('pol-BSE full aux vs dense', e9, 1e-5),
            ('pol-BSE davidson vs dense', e10, 1e-8)):
        status = 'ok' if err < tol else 'FAIL'
        if err >= tol:
            ok = False
        print(f"  {name:35s} {err:.3e}  (tol {tol:.0e})  {status}")
    print("=" * 72)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
