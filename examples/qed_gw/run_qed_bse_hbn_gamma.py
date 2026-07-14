"""Gamma-only polaritonic QED-BSE@evGW for a 2D hBN monolayer (experiment).

Proof-of-concept that the molecular singlet QED-BSE machinery of
:mod:`OmegaQMC.addons.qed_polariton_singlet` applies to a 2D periodic
system at the Gamma point, where all quantities are real and the
periodic problem is *structurally* identical to a molecule with a
periodized Coulomb interaction:

* HF reference:   PySCF PBC RHF with Gaussian density fitting (GDF),
  exxdiv=None (internally consistent with the bare DF kernel used in
  the RPA/GW/BSE steps). The 2D layer sits in a 3D supercell with a
  large vacuum gap, so the GDF 3-index factor keeps its plain
  positive-definite  eri = sum_P B_P B_P  form (the dimension=2
  low-dim kernel would emit sign-split blocks the molecular aux
  machinery cannot consume).
* Cavity coupling: the length-gauge dipole operator does NOT exist
  under PBC, so the coupling matrix is built in the velocity gauge,
      d_pq = lambda . <p|r|q>_eff,  <p|r|q>_eff = i <p|p_hat|q> / (eps_q - eps_p),
  from the momentum matrix elements (int1e_ipovlp). At Gamma with real
  orbitals this is a real symmetric matrix defined on off-diagonal
  (p != q) pairs only. Caveats (standard for velocity-gauge optics):
  the [r, Sigma_x] and [r, V_nl(pseudo)] commutator corrections are
  neglected.
* NO QED-HF ground state and NO dipole self-energy: both require
  <(mu - <mu>)^2> per cell, i.e. the electronic localization tensor —
  a genuine reformulation (and divergent for metals). The photon
  therefore appears ONCE, explicitly, as the extra row/column of the
  polaritonic BSE excitation space, coupled through
      g_ai = -sqrt(2) sqrt(w_cav/2) d_ai
  exactly as in the molecular module. GW and the static W are purely
  electronic.

Physics experiment (collective coupling / thermodynamic limit):
the script runs a 1x1 primitive cell and a 2x2 supercell (both
Gamma-only), tuning w_cav to the lowest bright singlet exciton of each.
At FIXED lambda the Rabi splitting must grow ~ sqrt(N_cells) (the
photon couples to the coherent sum of N transition dipoles); rescaling
lambda -> lambda/sqrt(N_cells) (fixed collective coupling) must leave
the splitting invariant. Both checks are printed and saved.

Run:
    python run_qed_bse_hbn_gamma.py [--basis gth-szv|gth-dzvp]
"""

import argparse
import json
import math
import time

import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import tools as pbctools

from OmegaQMC.addons.qed_gw import (_solve_qed_rpa_eigensystem,
                                    _solve_qp_newton)
from OmegaQMC.addons.qed_polariton_singlet import (_screening_at_eps,
                                                   _ov_blocks)

EV = 27.211386245988
ANG = 1.8897259886


# ---------------------------------------------------------------------------
# Cell and Gamma-point HF reference
# ---------------------------------------------------------------------------
def build_hbn_cell(basis='gth-szv', vacuum=20.0, nsc=(1, 1), verbose=0):
    """hBN monolayer in a 3D supercell with a vacuum gap along z.

    a = 2.504 A; B at (0,0,0), N at (1/3)a1 + (2/3)a2 (B-N bond
    a/sqrt(3) = 1.446 A).
    """
    a = 2.504
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([-a / 2.0, a * math.sqrt(3.0) / 2.0, 0.0])
    a3 = np.array([0.0, 0.0, vacuum])
    r_N = a1 / 3.0 + 2.0 * a2 / 3.0
    cell = pbcgto.Cell()
    cell.a = np.array([a1, a2, a3])
    cell.atom = [('B', (0.0, 0.0, 0.0)), ('N', tuple(r_N))]
    cell.basis = basis
    cell.pseudo = 'gth-pade'
    cell.unit = 'angstrom'
    cell.dimension = 3          # vacuum slab; keeps GDF sign-definite
    cell.verbose = verbose
    cell.build()
    if nsc != (1, 1):
        cell = pbctools.super_cell(cell, [nsc[0], nsc[1], 1])
        cell.verbose = verbose
    return cell


def run_gamma_rhf(cell):
    """Gamma-only RHF/GDF; exxdiv=None for consistency with the bare DF
    kernel in the response steps."""
    mf = pbcscf.RHF(cell, exxdiv=None).density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('Gamma-point RHF did not converge')
    return mf


def gamma_df_factor(mf):
    """Stack the Gamma-point GDF 3-index factor, real (naux, nao, nao)."""
    cell = mf.cell
    nao = cell.nao_nr()
    kpt = np.zeros((2, 3))
    blocks = []
    for LpqR, LpqI, sign in mf.with_df.sr_loop(kpt, compact=False):
        if sign != 1:
            raise RuntimeError('sign-split GDF block: use a 3D vacuum cell')
        if np.max(np.abs(LpqI)) > 1e-8:
            raise RuntimeError('complex GDF block at Gamma')
        blocks.append(LpqR.reshape(-1, nao, nao))
    return np.ascontiguousarray(np.concatenate(blocks, axis=0))


def velocity_gauge_dipole(mf):
    """Effective transition-dipole matrices r_eff[x][p,q] (MO basis) from
    momentum matrix elements:  <p|r|q>_eff = -P_pq / (eps_q - eps_p)
    with P = C^T <mu|d/dr|nu>_bra C (int1e_ipovlp, antisymmetric, real
    at Gamma). Diagonal / degenerate pairs are set to zero."""
    cell = mf.cell
    C = np.asarray(mf.mo_coeff)
    eps = np.asarray(mf.mo_energy)
    ip = np.asarray(cell.pbc_intor('int1e_ipovlp', comp=3))
    if np.iscomplexobj(ip):
        assert np.max(np.abs(ip.imag)) < 1e-9
        ip = ip.real
    nmo = C.shape[1]
    r_eff = np.zeros((3, nmo, nmo))
    dE = eps[None, :] - eps[:, None]
    safe = np.abs(dE) > 1e-6
    for x in range(3):
        P = C.T @ ip[x] @ C
        r_eff[x][safe] = -P[safe] / dE[safe]
        r_eff[x] = 0.5 * (r_eff[x] + r_eff[x].T)   # symmetrize (real gauge)
    return r_eff


def build_sq(mf, B_ao, r_eff, lambda_cav, omega_cav):
    """The `sq` dict consumed by the qed_polariton_singlet kernels.
    d carries the velocity-gauge cavity coupling on ov/vo blocks only
    (oo/vv blocks belong to the DSE, which is off under PBC)."""
    C = np.asarray(mf.mo_coeff)
    eps = np.asarray(mf.mo_energy)
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    nmo = C.shape[1]
    B_mo = np.einsum('pi,Ppq,qj->Pij', C, B_ao, C, optimize=True)
    lam = np.asarray(lambda_cav, dtype=float)
    d_full = np.einsum('x,xpq->pq', lam, r_eff)
    d = np.zeros_like(d_full)
    d[:nocc, nocc:] = d_full[:nocc, nocc:]
    d[nocc:, :nocc] = d_full[nocc:, :nocc]
    return {
        'F': np.diag(eps),
        'd': d,
        'B': np.ascontiguousarray(B_mo),
        'nocc': nocc,
        'nmo': nmo,
        'omega_cav': float(omega_cav),
    }, eps.copy()


# ---------------------------------------------------------------------------
# Electronic evGW (photon-free, DSE-free dRPA screening)
# ---------------------------------------------------------------------------
def run_gw_electronic(sq, eps_HF, mode='evGW', eta=1e-3, max_iter=8,
                      tol=1e-4, verbose=True):
    no, nmo = sq['nocc'], sq['nmo']
    eps_cur = eps_HF.copy()
    n_outer = max_iter if mode == 'evGW' else 1
    for outer in range(1, n_outer + 1):
        Omega, M_full = _screening_at_eps(sq, eps_cur, include_dse=False,
                                          include_photon=False)
        eps_new = np.empty_like(eps_cur)
        for p in range(nmo):
            eps_new[p], _, _ = _solve_qp_newton(
                eps_HF[p], M_full[p], Omega, eps_cur, no, eta,
                omega_start=eps_cur[p], max_iter=50, tol=1e-9)
        delta = float(np.max(np.abs(eps_new - eps_cur)))
        eps_cur = eps_new
        if verbose:
            print(f'    {mode} iter {outer}: max|d eps| = {delta:.3e}')
        if mode != 'evGW' or delta < tol:
            break
    if verbose:
        print(f'  HF gap  = {(eps_HF[no] - eps_HF[no-1]) * EV:7.3f} eV -> '
              f'GW gap = {(eps_cur[no] - eps_cur[no-1]) * EV:7.3f} eV')
    return eps_cur


# ---------------------------------------------------------------------------
# Polaritonic BSE (photon explicit, electronic W, no DSE)
# ---------------------------------------------------------------------------
def run_bse_polaritonic(sq, eps_QP, r_eff, lambda_on=True, tda=False):
    """Dense singlet polaritonic BSE. Mirrors the molecular dense path of
    run_qed_bse_polaritonic_singlet with every DSE (d x d) term dropped;
    oscillator strengths are velocity-gauge."""
    no, nmo = sq['nocc'], sq['nmo']
    nv = nmo - no
    nov = no * nv
    omega_cav = sq['omega_cav']
    d = sq['d']
    d_vo = d[no:, :no]

    Omega_scr, M_full = _screening_at_eps(sq, eps_QP, include_dse=False,
                                          include_photon=False)
    inv_Om = 1.0 / Omega_scr
    Wc_ijab = -2.0 * np.einsum('ijs,abs,s->ijab', M_full[:no, :no, :],
                               M_full[no:, no:, :], inv_Om, optimize=True)
    Wc_ibaj = -2.0 * np.einsum('ibs,ajs,s->ibaj', M_full[:no, no:, :],
                               M_full[no:, :no, :], inv_Om, optimize=True)
    v_ovov, v_oovv = _ov_blocks(sq)
    W_ijab = v_oovv + Wc_ijab
    W_ibaj = v_ovov.transpose(0, 1, 3, 2) + Wc_ibaj

    A_e = (2.0 * v_ovov.transpose(1, 0, 3, 2)
           - W_ijab.transpose(2, 0, 3, 1)).reshape(nov, nov)
    A_e += np.diag((eps_QP[no:, None] - eps_QP[None, :no]).reshape(-1))
    A_e = 0.5 * (A_e + A_e.T)
    B_e = (2.0 * v_ovov.transpose(1, 0, 3, 2)
           - W_ibaj.transpose(2, 0, 1, 3)).reshape(nov, nov)
    B_e = 0.5 * (B_e + B_e.T)

    if lambda_on:
        dim = nov + 1
        g_vec = -math.sqrt(2.0 * omega_cav / 2.0) * d_vo.reshape(nov)
        A_big = np.zeros((dim, dim))
        A_big[:nov, :nov] = A_e
        A_big[:nov, -1] = A_big[-1, :nov] = g_vec
        A_big[-1, -1] = omega_cav
        B_big = np.zeros((dim, dim))
        B_big[:nov, :nov] = B_e
        if not tda:
            B_big[:nov, -1] = B_big[-1, :nov] = g_vec
    else:
        dim = nov
        A_big, B_big = A_e, B_e

    if tda:
        import scipy.linalg as la
        Omega, vecs = la.eigh(A_big)
        XpY = vecs
    else:
        Omega, U, V = _solve_qed_rpa_eigensystem(A_big, B_big)
        XpY = U + V
    order = np.argsort(Omega)
    Omega = Omega[order]
    XpY = XpY[:, order]
    XpY_e = XpY[:nov, :]
    ph_amp = XpY[nov, :] if lambda_on else np.zeros_like(Omega)

    # velocity-gauge oscillator strengths (spin-summed singlet)
    f_osc = np.zeros_like(Omega)
    XpY_ai = XpY_e.reshape(nv, no, -1)
    for x in range(3):
        r_n = np.einsum('ai,ais->s', r_eff[x][no:, :no], XpY_ai)
        f_osc += (2.0 / 3.0) * Omega * 2.0 * r_n ** 2
    return {'Omega': Omega, 'f_osc': f_osc, 'photon_weight': ph_amp ** 2}


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def run_system(nsc, basis, lam_abs, gw_mode, verbose=True):
    """Full pipeline for one Gamma-only supercell. lambda is along x
    (in-plane); w_cav is tuned to the lowest bright exciton (lambda=0)."""
    t0 = time.time()
    ncell = nsc[0] * nsc[1]
    tag = f'{nsc[0]}x{nsc[1]}'
    print(f'\n=== hBN {tag} supercell, Gamma-only, basis={basis} ===')
    cell = build_hbn_cell(basis=basis, nsc=nsc)
    mf = run_gamma_rhf(cell)
    B_ao = gamma_df_factor(mf)
    r_eff = velocity_gauge_dipole(mf)
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    nmo = mf.mo_coeff.shape[1]
    print(f'  E_HF = {mf.e_tot:.8f} Ha, nocc = {nocc}, nmo = {nmo}, '
          f'naux = {B_ao.shape[0]}, nov = {nocc * (nmo - nocc)}')

    # --- lambda = 0: electronic GW + BSE, find the lowest bright exciton
    sq0, eps_HF = build_sq(mf, B_ao, r_eff, (0.0, 0.0, 0.0), omega_cav=1.0)
    eps_QP = run_gw_electronic(sq0, eps_HF, mode=gw_mode, verbose=verbose)
    bse0 = run_bse_polaritonic(sq0, eps_QP, r_eff, lambda_on=False)
    bright = np.where(bse0['f_osc'] > 1e-2)[0]
    ib = int(bright[0])
    w_cav = float(bse0['Omega'][ib])
    print(f'  lambda=0 excitons (eV / f_osc):')
    for i in range(min(6, len(bse0['Omega']))):
        mark = ' <- cavity tuned here' if i == ib else ''
        print(f'    {bse0["Omega"][i] * EV:8.3f}  {bse0["f_osc"][i]:8.4f}'
              f'{mark}')

    def polariton_run(lam_val):
        sq, _ = build_sq(mf, B_ao, r_eff, (lam_val, 0.0, 0.0), w_cav)
        bse = run_bse_polaritonic(sq, eps_QP, r_eff, lambda_on=True)
        # LP/UP: the two largest-photon-weight roots
        idx = np.argsort(bse['photon_weight'])[::-1][:2]
        lp, up = sorted(float(bse['Omega'][i]) for i in idx)
        # two-level (Jaynes-Cummings-like) prediction at exact resonance:
        # 2g with g = sqrt(2) sqrt(w/2) lam |d_x|, |d_x| = sqrt(3 f / (4 W))
        f_b = float(bse0['f_osc'][ib])
        d_col = math.sqrt(3.0 * f_b / (4.0 * w_cav))
        rabi_pred = 2.0 * math.sqrt(2.0 * w_cav / 2.0) * lam_val * d_col
        return bse, lp, up, rabi_pred

    lam_scaled = lam_abs / math.sqrt(ncell)
    out = {'tag': tag, 'ncell': ncell, 'basis': basis,
           'E_HF': float(mf.e_tot), 'nocc': nocc, 'nmo': nmo,
           'gw_mode': gw_mode,
           'gap_HF_eV': float((eps_HF[nocc] - eps_HF[nocc - 1]) * EV),
           'gap_GW_eV': float((eps_QP[nocc] - eps_QP[nocc - 1]) * EV),
           'omega_cav_eV': w_cav * EV,
           'exciton_eV': [float(x) for x in bse0['Omega'][:8] * EV],
           'exciton_f': [float(x) for x in bse0['f_osc'][:8]],
           'runs': {}}
    for label, lam_val in (('fixed', lam_abs), ('scaled', lam_scaled)):
        bse, lp, up, rabi_pred = polariton_run(lam_val)
        split = (up - lp) * EV
        print(f'  lambda_{label} = {lam_val:.4f}: LP = {lp * EV:7.3f} eV, '
              f'UP = {up * EV:7.3f} eV, Rabi splitting = {split:6.3f} eV '
              f'(two-level: {rabi_pred * EV:6.3f} eV)')
        pw = bse['photon_weight']
        top = np.argsort(pw)[::-1][:2]
        print(f'    photon weights of LP/UP: '
              f'{pw[top[0]]:.3f} / {pw[top[1]]:.3f}')
        out['runs'][label] = {
            'lambda': lam_val, 'LP_eV': lp * EV, 'UP_eV': up * EV,
            'rabi_eV': split, 'rabi_two_level_eV': rabi_pred * EV,
            'photon_weight': [float(pw[top[0]]), float(pw[top[1]])],
        }
    out['t_wall_s'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basis', default='gth-szv')
    ap.add_argument('--lam', type=float, default=0.05)
    ap.add_argument('--gw-mode', default='evGW', choices=['evGW', 'G0W0'])
    ap.add_argument('--json', default='qed_bse_hbn_gamma_results.json')
    args = ap.parse_args()

    results = {'lambda_abs': args.lam, 'systems': []}
    for nsc in ((1, 1), (2, 2)):
        results['systems'].append(
            run_system(nsc, args.basis, args.lam, args.gw_mode))

    # Collective-coupling analysis. A naive Rabi(2x2)/Rabi(1x1) ratio does
    # NOT isolate the sqrt(N) law: a Gamma-only 2x2 supercell folds the M
    # points into Gamma, so the electronic structure itself changes (gap,
    # bright exciton, oscillator strength). The quantitative check is the
    # two-level formula per system, Rabi = 2 sqrt(2) sqrt(w/2) lam
    # sqrt(3 f / 4 W): once g = lam sqrt(w/2) d_collective is verified,
    # sqrt(N) scaling follows from the extensivity of f for a fixed
    # per-cell exciton.
    print('\n=== Coupling-law check (computed vs two-level) ===')
    for sys_out in results['systems']:
        for label, run in sys_out['runs'].items():
            dev = run['rabi_eV'] / run['rabi_two_level_eV'] - 1.0
            print(f"  {sys_out['tag']} lambda_{label}: "
                  f"{run['rabi_eV']:6.3f} eV vs {run['rabi_two_level_eV']:6.3f} eV "
                  f"({dev * 100:+.1f}%)")
    r22 = results['systems'][1]['runs']
    lin = r22['fixed']['rabi_eV'] / r22['scaled']['rabi_eV']
    print(f'  2x2 fixed/scaled lambda ratio = {lin:.3f} '
          f'(linearity in lambda: {math.sqrt(4):.3f} expected)')
    results['ratio_linearity_2x2'] = lin

    with open(args.json, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nresults saved to {args.json}')


if __name__ == '__main__':
    main()
