"""k-point singlet polaritonic QED-BSE@evGW with q = 0 photon coupling.

k-sampled generalisation of :mod:`OmegaQMC.addons.qed_polariton_singlet`
for closed-shell periodic (KRHF) references. The excitation space is the
set of vertical (i k -> a k) singlet transitions over the whole
Gamma-centred mesh, plus ONE explicit photon channel that couples only
to the q = 0 (k-diagonal) block:

    g_ai(k) = -sqrt(2) sqrt(w_cav/2) d_ai(k),

with the cavity-coupling matrix built in the VELOCITY gauge per k-point
(the length-gauge position operator does not exist under PBC),

    d_pq(k) = lambda . r_eff(k),   r_eff_pq = i <p k|p_hat|q k> / (eps_q - eps_p),

from the int1e_ipovlp momentum matrix elements. There is NO QED-HF
ground-state shift and NO dipole self-energy: both require the
per-cell dipole-fluctuation (localization) tensor, which is a genuine
reformulation under PBC. GW and the static W are purely electronic;
the photon appears once, explicitly, in the BSE excitation space.

Coupling-strength convention: ``lambda_cav`` is the coupling of the
single cavity mode to the WHOLE Born-von-Karman supercell (N_k cells).
A Gamma-only supercell calculation with coupling lambda is reproduced
by the equivalent k-mesh with the same lambda; to compare meshes at
fixed collective Rabi splitting, keep lambda fixed (the per-cell
coupling then scales as it must, ~ 1/sqrt(N_k)).

DF conventions (pyscf GDF, momentum-conserving):

    L_pq(ki, kj) = (1/sqrt(N_k)) C(ki)^dag j3c(ki, kj) C(kj)
    (p ki q kj | r kr s ks) = sum_X L_pq,X(ki,kj) L_rs,X(kr,ks)

with L_qp(kj, ki) = conj(L_pq(ki, kj)) — the same conventions as
pyscf.pbc.mp.kmp2 with the 1/N_k folded into the factors, so at
Gamma-only meshes every quantity reduces to the molecular/supercell
module exactly.

Screening: the auxiliary-basis dielectric of the molecular module,
ported to complex arithmetic per momentum transfer q. For each q the
(naux x naux) Hermitian polarizability

    Pi_XY(q, nu) = sum_{k, ia} conj(L_ia,X(k, k+q)) dE/(nu^2 - dE^2) L_ia,Y(k, k+q)

gives the screened-interaction kernel Lam_eff(q, nu) = 4 Pi (1 - 4 Pi)^{-1}
by one Woodbury-type inversion (no nov-dimensional diagonalisation at
any q). The BSE uses Lam_eff(q, 0) (static W); the KGW quasiparticle
energies use the contour-deformation self-energy with the m-sum running
over all bands at k - q for every q.

Requires a 3D cell (2D materials: vacuum slab) so the GDF factor is a
plain positive factorisation; cell.dimension = 2 emits sign-split
blocks that this module does not consume.

Validation: a Gamma-centred 2x2 mesh on the primitive cell reproduces
the Gamma-only 2x2 supercell (HF, evGW QP energies, TDA and full
polaritonic BSE spectra) — see
examples/qed_gw/validate_qed_bse_kpts_hbn.py.
"""

import math

import numpy as np
import scipy.linalg as la

from .qed_polariton_singlet import _imag_freq_grid, _wt_spline, _qp_secant

EV = 27.211386245988


# ---------------------------------------------------------------------------
# k-point reference quantities
# ---------------------------------------------------------------------------
def _k_maps(cell, kpts, tol=6):
    """qidx[ki, kj] = mesh index of q = k_j - k_i (mod G);
    kplusq[ki, iq] = mesh index of k_i + q_iq. The Gamma-centred mesh is
    closed under these operations."""
    scaled = cell.get_scaled_kpts(kpts)
    key = lambda v: tuple(np.round(np.asarray(v) % 1.0, tol) % 1.0)
    lookup = {key(s): i for i, s in enumerate(scaled)}
    nk = len(kpts)
    qidx = np.empty((nk, nk), dtype=int)
    kplusq = np.empty((nk, nk), dtype=int)
    for i in range(nk):
        for j in range(nk):
            qidx[i, j] = lookup[key(scaled[j] - scaled[i])]
            kplusq[i, j] = lookup[key(scaled[i] + scaled[j])]
    return qidx, kplusq


def build_kpts_quantities(kmf, verbose=True):
    """Reference quantities from a converged KRHF/GDF calculation:
    MO energies, the DF factors L[ki][kj] (1/sqrt(nk) folded in) and the
    velocity-gauge effective dipole matrices r_eff(k)."""
    cell = kmf.cell
    kpts = kmf.kpts
    nk = len(kpts)
    C = [np.asarray(c) for c in kmf.mo_coeff]
    eps = [np.asarray(e).copy() for e in kmf.mo_energy]
    occs = [int(np.count_nonzero(o > 0)) for o in kmf.mo_occ]
    if len(set(occs)) != 1:
        raise NotImplementedError('non-uniform occupations (metallic mesh)')
    nocc = occs[0]
    nmo = C[0].shape[1]
    nao = cell.nao_nr()

    L = {}
    for ki in range(nk):
        for kj in range(nk):
            blocks = []
            for LpqR, LpqI, sign in kmf.with_df.sr_loop(
                    (kpts[ki], kpts[kj]), compact=False):
                if sign != 1:
                    raise RuntimeError(
                        'sign-split GDF block: use a 3D (vacuum-slab) cell')
                blocks.append((LpqR + 1j * LpqI).reshape(-1, nao, nao))
            Lao = np.concatenate(blocks, axis=0)
            L[ki, kj] = np.einsum('Lpq,pi,qj->Lij', Lao, C[ki].conj(), C[kj],
                                  optimize=True) / math.sqrt(nk)
    naux = L[0, 0].shape[0]

    # velocity-gauge effective dipole: r_eff = i p / dE, p = i <del mu|nu>
    ip_all = cell.pbc_intor('int1e_ipovlp', comp=3, kpts=kpts)
    r_eff = []
    for k in range(nk):
        ip = np.asarray(ip_all[k])
        rk = np.zeros((3, nmo, nmo), dtype=complex)
        dE = eps[k][None, :] - eps[k][:, None]
        safe = np.abs(dE) > 1e-6
        for x in range(3):
            p_ao = 1j * ip[x]
            p_mo = C[k].conj().T @ p_ao @ C[k]
            p_mo = 0.5 * (p_mo + p_mo.conj().T)
            rk[x][safe] = 1j * p_mo[safe] / dE[safe]
            rk[x] = 0.5 * (rk[x] + rk[x].conj().T)
        r_eff.append(rk)

    qidx, kplusq = _k_maps(cell, kpts)
    if verbose:
        print(f'  k-mesh: nk = {nk}, nocc = {nocc}, nmo = {nmo} (per k), '
              f'naux = {naux}')
    return {
        'nk': nk, 'nocc': nocc, 'nmo': nmo, 'naux': naux,
        'eps': eps, 'L': L, 'r_eff': r_eff,
        'qidx': qidx, 'kplusq': kplusq,
    }


# ---------------------------------------------------------------------------
# Auxiliary-basis dielectric per momentum transfer q
# ---------------------------------------------------------------------------
def _q_channels(kq, eps, iq):
    """Stacked ov channel vectors and bare gaps for momentum transfer q:
    b[X, (k,ia)] = L_ia,X(k, k+q), dE[(k,ia)] = eps_a(k+q) - eps_i(k)."""
    nk, no = kq['nk'], kq['nocc']
    cols, gaps = [], []
    for ki in range(nk):
        kj = int(kq['kplusq'][ki, iq])
        Lov = kq['L'][ki, kj][:, :no, no:]
        cols.append(Lov.reshape(Lov.shape[0], -1))
        gaps.append((eps[kj][None, no:] - eps[ki][:no, None]).reshape(-1))
    return np.concatenate(cols, axis=1), np.concatenate(gaps)


def _lambda_eff_q(b, dE, nu2):
    """Screened-interaction kernel Lam_eff(q, nu) = 4 Pi (1 - 4 Pi)^{-1}
    with Pi_XY = sum conj(b_X) dE/(nu^2 - dE^2) b_Y (complex Woodbury;
    Hermitian for real nu2). Same algebra as the molecular
    _aux_wtilde_core with the cavity channel absent."""
    x = dE / (nu2 - dE * dE)
    pi = (b.conj() * x) @ b.T
    nch = pi.shape[0]
    Q = np.linalg.solve(np.eye(nch, dtype=pi.dtype) - 4.0 * pi, pi)
    return 4.0 * (pi + 4.0 * pi @ Q)


# ---------------------------------------------------------------------------
# Electronic KGW (evGW / G0W0) by contour deformation
# ---------------------------------------------------------------------------
def _sigma_cd_kpts(Wt_p, spline, freqs_all, quad_w, eps_used, occ_flat,
                   omega, eta, wt_pt):
    """Contour-deformation Sigma_c,pp(k; omega). Identical structure to
    the molecular _sigma_aux_cd, with the m-sum over all (k-q, band)
    pairs (flattened); wt_pt(mflat, nu2) supplies the single-point
    screened interaction for the pole residues."""
    Delta = omega - eps_used
    aD = np.abs(Delta)
    wmax = freqs_all[-1]
    nM = eps_used.size
    idx = np.arange(nM)
    WD = spline(np.minimum(aD, wmax))[idx, idx]
    tail = aD > wmax
    if tail.any():
        WD[tail] *= (wmax / aD[tail]) ** 2
    om2 = freqs_all[1:] ** 2
    kern = Delta[None, :] / (Delta[None, :] ** 2 + om2[:, None])
    sig = -(quad_w[:, None] * (Wt_p[1:, :] - WD[None, :]) * kern).sum() \
        / math.pi
    sig -= 0.5 * (WD * np.sign(Delta)).sum()
    for m in range(nM):
        occ = occ_flat[m]
        if occ and eps_used[m] > omega:
            fac, nu = -1.0, eps_used[m] - omega
        elif (not occ) and eps_used[m] < omega:
            fac, nu = 1.0, omega - eps_used[m]
        elif eps_used[m] == omega:
            fac, nu = (-0.5 if occ else 0.5), 0.0
        else:
            continue
        sig += fac * wt_pt(m, (nu + 1j * eta) ** 2).real
    return sig


def run_kgw_electronic(kq, mode='evGW', eta=1e-3, max_iter=8, tol=1e-4,
                       n_freq=100, freq_scale=0.5, inner_max=50,
                       inner_tol=1e-9, verbose=True):
    """Purely electronic evGW/G0W0 quasiparticle energies on the k-mesh
    (dRPA screening, aux-basis dielectric, contour deformation).
    Returns a list of QP energy arrays, one per k-point."""
    nk, no, nmo, naux = kq['nk'], kq['nocc'], kq['nmo'], kq['naux']
    eps_HF = [e.copy() for e in kq['eps']]
    quad_f, quad_w = _imag_freq_grid(n_freq, freq_scale)
    freqs_all = np.concatenate([[0.0], quad_f])
    nfreq = len(freqs_all)
    occ_flat = np.concatenate([np.arange(nmo) < no for _ in range(nk)])

    if verbose:
        print(f'\n{mode}@dRPA@KRHF (k-point, aux-cd, electronic)')
        print(f'  nk = {nk}, nocc = {no}, nmo = {nmo}, naux = {naux}, '
              f'n_freq = {n_freq}')

    eps_cur = [e.copy() for e in eps_HF]
    n_outer = max_iter if mode == 'evGW' else 1
    for outer in range(1, n_outer + 1):
        chan = {iq: _q_channels(kq, eps_cur, iq) for iq in range(nk)}
        # Wt[kp, freq, p, (km, m)] — real by Hermiticity of Lam_eff
        Wt = np.zeros((nk, nfreq, nmo, nk * nmo))
        for iq in range(nk):
            b, dE = chan[iq]
            for f, wpr in enumerate(freqs_all):
                lam = _lambda_eff_q(b, dE, -wpr * wpr)
                for kp in range(nk):
                    km = int(kq['kplusq'][kp, iq])
                    R = kq['L'][kp, km].reshape(naux, -1)
                    T = lam @ R
                    blk = (R.conj() * T).sum(axis=0).real.reshape(nmo, nmo)
                    Wt[kp, f, :, km * nmo:(km + 1) * nmo] = blk

        eps_used = np.concatenate(eps_cur)
        eps_new = [np.empty(nmo) for _ in range(nk)]
        delta = 0.0
        for kp in range(nk):
            def wt_pt(mflat, nu2, kp=kp):
                km, m = divmod(mflat, nmo)
                iq = int(kq['qidx'][kp, km])
                b, dE = chan[iq]
                lam = _lambda_eff_q(b, dE, nu2)
                r = kq['L'][kp, km][:, :, m]
                return np.einsum('Xp,XY,Yp->p', r.conj(), lam, r)

            for p in range(nmo):
                spl = _wt_spline(freqs_all, Wt[kp, :, p, :])
                wt_pt_p = (lambda mflat, nu2, p=p, kp=kp:
                           wt_pt(mflat, nu2)[p])
                eps_new[kp][p] = _qp_secant(
                    lambda w, p=p, kp=kp, spl=spl, f=wt_pt_p:
                        w - eps_HF[kp][p] - _sigma_cd_kpts(
                            Wt[kp, :, p, :], spl, freqs_all, quad_w,
                            eps_used, occ_flat, w, eta, f),
                    eps_cur[kp][p], max_iter=inner_max, tol=inner_tol)
            delta = max(delta,
                        float(np.max(np.abs(eps_new[kp] - eps_cur[kp]))))
        eps_cur = eps_new
        if verbose:
            print(f'    iter {outer:3d}:  max |d eps^GW| = {delta:.3e}')
        if mode != 'evGW' or delta < tol:
            break

    if verbose:
        homo = max(e[no - 1] for e in eps_cur)
        lumo = min(e[no] for e in eps_cur)
        homo0 = max(e[no - 1] for e in eps_HF)
        lumo0 = min(e[no] for e in eps_HF)
        print(f'  HF gap = {(lumo0 - homo0) * EV:7.3f} eV -> '
              f'GW gap = {(lumo - homo) * EV:7.3f} eV')
    return eps_cur


# ---------------------------------------------------------------------------
# Complex RPA (paired) eigensolver
# ---------------------------------------------------------------------------
def _solve_rpa_cplx(A, B, imag_tol=1e-6):
    """Diagonalise [[A, B], [-B*, -A*]] (A Hermitian, B symmetric) and
    return the positive-norm modes with U^dag U - V^dag V = 1."""
    dim = A.shape[0]
    M = np.zeros((2 * dim, 2 * dim), dtype=complex)
    M[:dim, :dim] = A
    M[:dim, dim:] = B
    M[dim:, :dim] = -B.conj()
    M[dim:, dim:] = -A.conj()
    evals, evecs = la.eig(M)
    U = evecs[:dim]
    V = evecs[dim:]
    norms = (np.abs(U) ** 2).sum(axis=0) - (np.abs(V) ** 2).sum(axis=0)
    pos = np.where(norms > 1e-10)[0]
    if len(pos) != dim:
        raise RuntimeError(
            f'RPA instability: {len(pos)} positive-norm modes, expected {dim}')
    if np.max(np.abs(evals[pos].imag)) > imag_tol:
        raise RuntimeError('complex RPA eigenvalue: reference unstable')
    Omega = evals[pos].real
    scale = 1.0 / np.sqrt(norms[pos])
    U = U[:, pos] * scale
    V = V[:, pos] * scale
    order = np.argsort(Omega)
    return Omega[order], U[:, order], V[:, order]


# ---------------------------------------------------------------------------
# Polaritonic BSE on the k-mesh (photon explicit at q = 0)
# ---------------------------------------------------------------------------
def run_qed_bse_kpts(kq, omega_cav, lambda_cav, eps_QP=None, tda=False,
                     herm_tol=1e-6, verbose=True):
    """Singlet polaritonic BSE over the k-mesh with one explicit q = 0
    photon channel. lambda_cav couples the mode to the whole BvK
    supercell (see module docstring); eps_QP defaults to the KRHF
    eigenvalues (BSE@HF).

    Returns dict with 'Omega', 'f_osc' (velocity gauge),
    'photon_weight', 'omega_cav'.
    """
    nk, no, nmo, naux = kq['nk'], kq['nocc'], kq['nmo'], kq['naux']
    nv = nmo - no
    novk = nk * no * nv
    dim = novk + 1
    if eps_QP is None:
        eps_QP = kq['eps']
    eps_QP = [np.asarray(e, dtype=float) for e in eps_QP]
    lam_vec = np.asarray(lambda_cav, dtype=float)

    # static screened-interaction kernels per momentum transfer
    lam_stat = {}
    for iq in range(nk):
        b, dE = _q_channels(kq, eps_QP, iq)
        lam_stat[iq] = _lambda_eff_q(b, dE, 0.0)
    eye = np.eye(naux)

    Lov0 = [kq['L'][k, k][:, :no, no:] for k in range(nk)]
    A = np.zeros((dim, dim), dtype=complex)
    B = None if tda else np.zeros((dim, dim), dtype=complex)
    for ki in range(nk):
        si = ki * no * nv
        for kj in range(nk):
            sj = kj * no * nv
            iq = int(kq['qidx'][ki, kj])
            K = eye + lam_stat[iq]              # bare + correlation W(q)
            Lfull = kq['L'][ki, kj]
            Lij = Lfull[:, :no, :no]
            Lab = Lfull[:, no:, no:]
            # A: 2 (a_ki i_ki | j_kj b_kj) - W_direct
            blk = 2.0 * np.einsum('Xia,Xjb->iajb', Lov0[ki].conj(),
                                  Lov0[kj], optimize=True)
            blk -= np.einsum('Xab,XY,Yij->iajb', Lab, K, Lij.conj(),
                             optimize=True)
            A[si:si + no * nv, sj:sj + no * nv] = blk.reshape(no * nv,
                                                              no * nv)
            if not tda:
                Lvo = Lfull[:, no:, :no]        # L_aj (a at ki, j at kj)
                Lov = Lfull[:, :no, no:]        # L_ib (i at ki, b at kj)
                blkB = 2.0 * np.einsum('Xia,Xjb->iajb', Lov0[ki].conj(),
                                       Lov0[kj].conj(), optimize=True)
                blkB -= np.einsum('Xaj,XY,Yib->iajb', Lvo, K, Lov.conj(),
                                  optimize=True)
                B[si:si + no * nv, sj:sj + no * nv] = blkB.reshape(
                    no * nv, no * nv)
        dE_k = (eps_QP[ki][None, no:] - eps_QP[ki][:no, None]).reshape(-1)
        A[si:si + no * nv, si:si + no * nv] += np.diag(dE_k)

    # photon row/column: q = 0 velocity-gauge coupling
    g = np.empty(novk, dtype=complex)
    for k in range(nk):
        d_k = np.einsum('x,xpq->pq', lam_vec, kq['r_eff'][k])
        g[k * no * nv:(k + 1) * no * nv] = \
            -math.sqrt(2.0 * omega_cav / 2.0) * d_k[no:, :no].T.reshape(-1)
    A[:novk, -1] = g
    A[-1, :novk] = g.conj()
    A[-1, -1] = omega_cav
    if not tda:
        B[:novk, -1] = g
        B[-1, :novk] = g

    devA = float(np.max(np.abs(A - A.conj().T)))
    A = 0.5 * (A + A.conj().T)
    devB = 0.0
    if not tda:
        devB = float(np.max(np.abs(B - B.T)))
        B = 0.5 * (B + B.T)
    if verbose and max(devA, devB) > herm_tol:
        print(f'  warning: block-symmetry deviation A: {devA:.2e}, '
              f'B: {devB:.2e}')

    if tda:
        Omega, U = la.eigh(A)
        XpY = U
        ph_amp = U[-1, :]
    else:
        Omega, U, V = _solve_rpa_cplx(A, B)
        XpY = U + V
        ph_amp = XpY[-1, :]

    # velocity-gauge singlet oscillator strengths
    f_osc = np.zeros_like(Omega)
    XpY_e = XpY[:novk, :].reshape(nk, no, nv, -1)
    for x in range(3):
        t = np.zeros(XpY.shape[1], dtype=complex)
        for k in range(nk):
            r_ov = kq['r_eff'][k][x][:no, no:]
            t += np.einsum('ia,ias->s', r_ov.conj(), XpY_e[k])
        f_osc += (2.0 / 3.0) * Omega * 2.0 * np.abs(t) ** 2

    if verbose:
        kind = 'pol-TDA-BSE' if tda else 'pol-BSE'
        print(f'\n{kind} (k-point, photon explicit at q = 0)')
        print(f'  nk = {nk}, dim = {dim}, w_cav = {omega_cav * EV:.3f} eV, '
              f'lambda = {lam_vec}')
        bright = np.where((f_osc > 1e-3) | (np.abs(ph_amp) ** 2 > 0.05))[0]
        print('   Omega(eV)   f_osc     photon_wt')
        for i in bright[:12]:
            print(f'   {Omega[i] * EV:8.3f}   {f_osc[i]:8.4f}  '
                  f'{np.abs(ph_amp[i]) ** 2:8.3f}')

    return {
        'method': 'pol-TDA-BSE (kpts)' if tda else 'pol-BSE (kpts)',
        'tda': bool(tda),
        'Omega': Omega,
        'f_osc': f_osc,
        'photon_weight': np.abs(ph_amp) ** 2,
        'omega_cav': float(omega_cav),
        'novk': novk,
    }
