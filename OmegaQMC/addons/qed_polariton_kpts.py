"""k-point singlet polaritonic QED-BSE@evGW with q = 0 photon coupling.

k-sampled generalisation of :mod:`OmegaQMC.addons.qed_polariton_singlet`
for closed-shell periodic (KRHF) references. The excitation space is the
set of vertical (i k -> a k) singlet transitions over the whole
Gamma-centred mesh, plus ONE explicit photon channel that couples only
to the q = 0 (k-diagonal) block:

    g_ai(k) = -sqrt(2) sqrt(w_cav/2) d_ai(k),

with the cavity-coupling matrix built in the VELOCITY gauge per k-point
(the length-gauge position operator does not exist under PBC),

    d_pq(k) = lambda . r_eff(k).

Two velocity routes are provided (``velocity=`` of
:func:`build_kpts_quantities`):

* ``'exact'`` (default): the exact-within-basis transition dipole from
  the k-derivative of the fixed-density Fock/overlap matrices plus the
  intra-cell dipole (basis Berry-connection) term,

      r_eff_pq = i C_p^dag [dF/dk - (eps_p+eps_q)/2 dS/dk] C_q
                 / (eps_q - eps_p)  +  [C^dag Pbar C]_pq,

  with Pbar the Hermitized lattice-summed dipole integrals
  P_uv(k) = sum_R e^{ikR} <u 0|r|v R> (``int1e_r``). This is the
  nonorthogonal-basis momentum formula of Lee, Kim & Son
  [PRB 98, 115115 (2018), Eq. (7)] divided by the transition energy;
  it captures the [r, V_nl(pseudo)] and [r, Sigma_x] commutators that
  the bare-momentum route neglects, and reduces exactly to the
  length-gauge dipole in the isolated-molecule limit. dF/dk is a
  central finite difference with the converged density held fixed;
  hcore, S and the Hartree J are differenced at shifted band k-points
  (smooth), while the HF exchange is differenced over a RIGID SHIFT of
  the whole k-mesh (fresh GDF per shifted mesh) — a fixed-mesh
  difference of K diverges as 4pi/delta^2 through the G = 0 Coulomb
  component at momentum transfer q = delta (the exxdiv=None spike),
  whereas the rigid shift keeps every transfer exactly on-mesh and
  equals the fixed-density dK/dk in the thermodynamic limit
  (integration by parts over the BZ).

* ``'bare'``: the historical route, r_eff_pq = i P_pq/(eps_q - eps_p)
  from the bare int1e_ipovlp momentum matrix elements (commutator
  corrections neglected).

There is NO QED-HF ground-state shift and NO dipole self-energy: both
require the per-cell dipole-fluctuation (localization) tensor, which
is a genuine reformulation under PBC. GW and the static W are purely
electronic; the photon appears once, explicitly, in the BSE
excitation space.

Coupling-strength convention: ``lambda_cav`` is the coupling of the
single cavity mode to the WHOLE Born-von-Karman supercell (N_k cells).
A Gamma-only supercell calculation with coupling lambda is reproduced
by the equivalent k-mesh with the same lambda; to compare meshes at
fixed collective Rabi splitting, keep lambda fixed (the per-cell
coupling then scales as it must, ~ 1/sqrt(N_k)).

DF conventions (pyscf GDF, momentum-conserving):

    L_pq(ki, kj) = (1/sqrt(N_k)) C(ki)^dag j3c(ki, kj) C(kj)
    (p ki q kj | r kr s ks) = sum_X s_X L_pq,X(ki,kj) L_rs,X(kr,ks)

with L_qp(kj, ki) = conj(L_pq(ki, kj)) — the same conventions as
pyscf.pbc.mp.kmp2 with the 1/N_k folded into the factors, so at
Gamma-only meshes every quantity reduces to the molecular/supercell
module exactly. s_X = +/-1 is the auxiliary-space METRIC: for a 3D
cell the GDF factorisation is positive (s = +1 identically, the rule
above is the plain B.B one); for cell.dimension = 2 the truncated
Coulomb operator is indefinite and pyscf's low-dimensional GDF kernel
emits sign-split factor blocks (sign = -1 in with_df.sr_loop), i.e.
(pq|rs) = B.B - Bbar.Bbar. The metric S = diag(s) depends only on the
momentum transfer q = kj - ki (the j2c decomposition is per q) and is
carried through every auxiliary-space contraction below.

Screening: the auxiliary-basis dielectric of the molecular module,
ported to complex arithmetic per momentum transfer q. For each q the
(naux x naux) Hermitian polarizability

    Pi_XY(q, nu) = sum_{k, ia} conj(L_ia,X(k, k+q)) dE/(nu^2 - dE^2) L_ia,Y(k, k+q)

gives the screened-interaction kernel by one Woodbury-type inversion
(no nov-dimensional diagonalisation at any q),

    Lam_eff(q, nu) = S 4 Pi (1 - S 4 Pi)^{-1} S,   S = diag(s(q)),

which for a positive (3D) factorisation, S = 1, is the familiar
4 Pi (1 - 4 Pi)^{-1}. Every bare-interaction contraction carries the
same metric: the static-W BSE kernel is L [S + Lam_eff(q,0)] L*
instead of L [1 + Lam_eff] L*, and the Hartree (transition) channel is
2 sum_X s_X(0) L_ai L_jb. Lam_eff stays Hermitian for real nu^2
(S 4Pi (1-S 4Pi)^{-1} S is Hermitian whenever Pi is), so the
contour-deformation KGW machinery is untouched. The BSE uses
Lam_eff(q, 0) (static W); the KGW quasiparticle energies use the
contour-deformation self-energy with the m-sum running over all bands
at k - q for every q.

Supports both 3D cells (vacuum-slab treatment of 2D materials, all
signs +1) and cell.dimension = 2 low-dimensional GDF references
(truncated Coulomb, sign-split factors). The dimension = 2 path
removes the spurious interlayer screening of the vacuum-slab approach,
which converges only ~ 1/L_z in the interlayer distance
[Hueser, Olsen, Thygesen, PRB 88, 245309 (2013)].

Validation: a Gamma-centred 2x2 mesh on the primitive cell reproduces
the Gamma-only 2x2 supercell (HF, evGW QP energies, TDA and full
polaritonic BSE spectra) at the 1e-7 level with velocity='bare' —
see examples/qed_gw/validate_qed_bse_kpts_hbn.py; the equivalence
holds identically on the sign-split dimension = 2 path, and the
dimension = 2 results are the L_z -> infinity limit of the
vacuum-slab series — see
examples/qed_gw/validate_qed_bse_kpts_2d.py (machinery validations
are pinned to velocity='bare'). The exact velocity
route is validated independently (isolated-molecule length-gauge
limit; r_nm = i <u_nk|u_m,k+q>/q vertex identity on periodic hBN;
Hellmann-Feynman band velocities; delta-convergence). Note that the
exact route is representation-covariant only up to
basis-incompleteness residuals: evaluated in a SUPERCELL it acquires
small spurious momentum-forbidden couplings that the primitive-mesh
evaluation excludes by construction (k-diagonal), so the primitive
mesh is canonical — the effect on hBN gth-szv 2x2 polaritons is
quantified in examples/qed_gw/run_qed_velocity_exact_check.py.
"""

import math

import numpy as np
import scipy.linalg as la

from .qed_polariton_singlet import _imag_freq_grid, _wt_spline, _qp_secant

EV = 27.211386245988


# ---------------------------------------------------------------------------
# Velocity-gauge transition dipoles
# ---------------------------------------------------------------------------
def _fd_fock_ovlp(kmf, delta=1e-4):
    """dF[x][k], dS[x][k] (AO basis) by central finite differences at
    k +/- delta e_x with the converged density held fixed. hcore, S and
    the Hartree J are evaluated at shifted band k-points (smooth in k);
    the HF exchange is evaluated on rigidly shifted meshes so every
    momentum transfer stays exactly on-mesh (see module docstring)."""
    from pyscf.pbc import df as pbcdf
    cell = kmf.cell
    kpts = np.asarray(kmf.kpts).reshape(-1, 3)
    nk = len(kpts)
    nao = cell.nao_nr()
    dm = kmf.make_rdm1()
    kb = []
    for x in range(3):
        e = np.zeros(3)
        e[x] = delta
        kb += [kpts + e, kpts - e]
    kb = np.concatenate(kb, axis=0)                       # (6 nk, 3)

    hcore = np.asarray(kmf.get_hcore(cell, kb))
    S = np.asarray(kmf.get_ovlp(cell, kb)).reshape(3, 2, nk, nao, nao)
    vj = np.asarray(kmf.with_df.get_jk(dm, kpts=kpts, kpts_band=kb,
                                       with_k=False, exxdiv=None)[0])
    HJ = (hcore + vj).reshape(3, 2, nk, nao, nao)

    auxbasis = getattr(kmf.with_df, 'auxbasis', None)
    K = np.empty((3, 2, nk, nao, nao), dtype=complex)
    for x in range(3):
        e = np.zeros(3)
        e[x] = delta
        for s, sgn in enumerate((1.0, -1.0)):
            mesh_s = kpts + sgn * e
            df_s = pbcdf.GDF(cell, kpts=mesh_s)
            if auxbasis is not None:
                df_s.auxbasis = auxbasis
            K[x, s] = np.asarray(df_s.get_jk(dm, kpts=mesh_s, with_j=False,
                                             exxdiv=None)[1])

    F = HJ - 0.5 * K
    dF = (F[:, 0] - F[:, 1]) / (2.0 * delta)              # (3, nk, nao, nao)
    dS = (S[:, 0] - S[:, 1]) / (2.0 * delta)
    return dF, dS


def velocity_dipole_exact(kmf, delta=1e-4, deg_tol=1e-6):
    """Exact-within-basis velocity-gauge transition dipoles, one
    (3, nmo, nmo) complex Hermitian array per k-point (module
    docstring). Returns (r_eff, herm_dev) with herm_dev the maximal
    pre-symmetrization Hermiticity deviation (finite-difference noise
    diagnostic)."""
    cell = kmf.cell
    kpts = np.asarray(kmf.kpts).reshape(-1, 3)
    nk = len(kpts)
    C = [np.asarray(kmf.mo_coeff[k]) for k in range(nk)]
    eps = [np.asarray(kmf.mo_energy[k]) for k in range(nk)]
    dF, dS = _fd_fock_ovlp(kmf, delta)
    rints = cell.pbc_intor('int1e_r', comp=3, kpts=kpts)
    nao = cell.nao_nr()
    nmo = C[0].shape[1]
    r_eff = []
    herm_dev = 0.0
    for k in range(nk):
        rk = np.zeros((3, nmo, nmo), dtype=complex)
        dE = eps[k][None, :] - eps[k][:, None]
        eavg = 0.5 * (eps[k][None, :] + eps[k][:, None])
        safe = np.abs(dE) > deg_tol
        P = np.asarray(rints[k]).reshape(3, nao, nao)
        for x in range(3):
            dF_mo = C[k].conj().T @ dF[x, k] @ C[k]
            dS_mo = C[k].conj().T @ dS[x, k] @ C[k]
            P_mo = C[k].conj().T @ (0.5 * (P[x] + P[x].conj().T)) @ C[k]
            num = 1j * (dF_mo - eavg * dS_mo)
            rk[x][safe] = num[safe] / dE[safe] + P_mo[safe]
            herm_dev = max(herm_dev,
                           float(np.max(np.abs(rk[x] - rk[x].conj().T))))
            rk[x] = 0.5 * (rk[x] + rk[x].conj().T)
        r_eff.append(rk)
    return r_eff, herm_dev


def velocity_dipole_bare(kmf, deg_tol=1e-6):
    """Bare-momentum velocity-gauge dipoles r_eff = i p/dE per k-point
    ([r, V_nl] and [r, Sigma_x] commutators neglected)."""
    cell = kmf.cell
    kpts = np.asarray(kmf.kpts).reshape(-1, 3)
    nk = len(kpts)
    C = [np.asarray(kmf.mo_coeff[k]) for k in range(nk)]
    eps = [np.asarray(kmf.mo_energy[k]) for k in range(nk)]
    ip_all = cell.pbc_intor('int1e_ipovlp', comp=3, kpts=kpts)
    nmo = C[0].shape[1]
    r_eff = []
    for k in range(nk):
        ip = np.asarray(ip_all[k])
        rk = np.zeros((3, nmo, nmo), dtype=complex)
        dE = eps[k][None, :] - eps[k][:, None]
        safe = np.abs(dE) > deg_tol
        for x in range(3):
            p_mo = C[k].conj().T @ (1j * ip[x]) @ C[k]
            p_mo = 0.5 * (p_mo + p_mo.conj().T)
            rk[x][safe] = 1j * p_mo[safe] / dE[safe]
            rk[x] = 0.5 * (rk[x] + rk[x].conj().T)
        r_eff.append(rk)
    return r_eff


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


def build_kpts_quantities(kmf, verbose=True, velocity='exact',
                          fd_delta=1e-4):
    """Reference quantities from a converged KRHF/GDF calculation:
    MO energies, the DF factors L[ki][kj] (1/sqrt(nk) folded in), the
    per-q auxiliary metric s(q) (sign-split low-dimensional GDF) and
    the velocity-gauge effective dipole matrices r_eff(k) (``velocity``
    selects the 'exact' or 'bare' route, module docstring)."""
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
    qidx, kplusq = _k_maps(cell, kpts)

    L = {}
    sgn_q = [None] * nk
    for ki in range(nk):
        for kj in range(nk):
            blocks, signs = [], []
            for LpqR, LpqI, sign in kmf.with_df.sr_loop(
                    (kpts[ki], kpts[kj]), compact=False):
                blk = (LpqR + 1j * LpqI).reshape(-1, nao, nao)
                blocks.append(blk)
                signs.append(np.full(blk.shape[0], sign, dtype=float))
            Lao = np.concatenate(blocks, axis=0)
            L[ki, kj] = np.einsum('Lpq,pi,qj->Lij', Lao, C[ki].conj(), C[kj],
                                  optimize=True) / math.sqrt(nk)
            # the aux metric is a property of q = kj - ki only (the j2c
            # decomposition is per momentum transfer)
            s = np.concatenate(signs)
            iq = int(qidx[ki, kj])
            if sgn_q[iq] is None:
                sgn_q[iq] = s
            elif not np.array_equal(sgn_q[iq], s):
                raise RuntimeError(
                    f'inconsistent GDF sign structure within q-channel {iq}')
    naux_q = np.array([len(s) for s in sgn_q])
    naux = int(naux_q.max())
    indefinite = any((s < 0).any() for s in sgn_q)

    # velocity-gauge effective transition dipoles
    if velocity == 'exact':
        r_eff, herm_dev = velocity_dipole_exact(kmf, delta=fd_delta)
    elif velocity == 'bare':
        r_eff = velocity_dipole_bare(kmf)
        herm_dev = 0.0
    else:
        raise ValueError(f"velocity must be 'exact' or 'bare': {velocity}")

    if verbose:
        print(f'  k-mesh: nk = {nk}, nocc = {nocc}, nmo = {nmo} (per k), '
              f'naux = {naux}')
        if indefinite:
            nneg = [int((s < 0).sum()) for s in sgn_q]
            print(f'  low-dim GDF (dimension = {cell.dimension}): '
                  f'indefinite metric, naux per q = {naux_q.tolist()}, '
                  f'negative rows per q = {nneg}')
        msg = f'  velocity gauge: {velocity}'
        if velocity == 'exact':
            msg += (f' (fd_delta = {fd_delta:.1e}, '
                    f'herm dev = {herm_dev:.1e})')
        print(msg)
    return {
        'nk': nk, 'nocc': nocc, 'nmo': nmo, 'naux': naux,
        'eps': eps, 'L': L, 'r_eff': r_eff, 'velocity': velocity,
        'sgn': sgn_q, 'qidx': qidx, 'kplusq': kplusq,
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


def _lambda_eff_q(b, dE, nu2, sgn=None):
    """Screened-interaction kernel with the auxiliary metric S = diag(sgn),

        Lam_eff(q, nu) = S 4 Pi (1 - S 4 Pi)^{-1} S,

    with Pi_XY = sum conj(b_X) dE/(nu^2 - dE^2) b_Y (complex Woodbury;
    Hermitian for real nu2 — the metric-dressed form S 4Pi (1-S 4Pi)^{-1} S
    is Hermitian whenever Pi is, by the push-through identity). For
    sgn = None (positive 3D factorisation) this is 4 Pi (1 - 4 Pi)^{-1},
    the same algebra as the molecular _aux_wtilde_core with the cavity
    channel absent. The RPA series with an indefinite decomposition
    v = B^T S B inserts S between successive polarizabilities:
    Lam = S [4Pi + 4Pi S 4Pi + ...] S."""
    x = dE / (nu2 - dE * dE)
    pi = (b.conj() * x) @ b.T
    nch = pi.shape[0]
    eye = np.eye(nch, dtype=pi.dtype)
    if sgn is None or not (np.asarray(sgn) < 0).any():
        Q = np.linalg.solve(eye - 4.0 * pi, pi)
        return 4.0 * (pi + 4.0 * pi @ Q)
    sgn = np.asarray(sgn, dtype=float)
    # 4 Pi (1 - S 4 Pi)^{-1} = (1 - 4 Pi S)^{-1} 4 Pi  (push-through)
    Y = np.linalg.solve(eye - 4.0 * pi * sgn[None, :], 4.0 * pi)
    return sgn[:, None] * Y * sgn[None, :]


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
                lam = _lambda_eff_q(b, dE, -wpr * wpr, kq['sgn'][iq])
                for kp in range(nk):
                    km = int(kq['kplusq'][kp, iq])
                    Lb = kq['L'][kp, km]
                    R = Lb.reshape(Lb.shape[0], -1)
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
                lam = _lambda_eff_q(b, dE, nu2, kq['sgn'][iq])
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
    nk, no, nmo = kq['nk'], kq['nocc'], kq['nmo']
    nv = nmo - no
    novk = nk * no * nv
    dim = novk + 1
    if eps_QP is None:
        eps_QP = kq['eps']
    eps_QP = [np.asarray(e, dtype=float) for e in eps_QP]
    lam_vec = np.asarray(lambda_cav, dtype=float)

    # static screened-interaction kernels per momentum transfer,
    # bare metric included: K(q) = S(q) + Lam_eff(q, 0)
    Kq = {}
    for iq in range(nk):
        b, dE = _q_channels(kq, eps_QP, iq)
        Kq[iq] = np.diag(kq['sgn'][iq]) + _lambda_eff_q(
            b, dE, 0.0, kq['sgn'][iq])
    iq0 = int(kq['qidx'][0, 0])
    s0 = kq['sgn'][iq0]                          # metric of the q = 0 channel

    Lov0 = [kq['L'][k, k][:, :no, no:] for k in range(nk)]
    A = np.zeros((dim, dim), dtype=complex)
    B = None if tda else np.zeros((dim, dim), dtype=complex)
    for ki in range(nk):
        si = ki * no * nv
        for kj in range(nk):
            sj = kj * no * nv
            iq = int(kq['qidx'][ki, kj])
            K = Kq[iq]                          # bare + correlation W(q)
            Lfull = kq['L'][ki, kj]
            Lij = Lfull[:, :no, :no]
            Lab = Lfull[:, no:, no:]
            # A: 2 (a_ki i_ki | j_kj b_kj) - W_direct
            blk = 2.0 * np.einsum('X,Xia,Xjb->iajb', s0, Lov0[ki].conj(),
                                  Lov0[kj], optimize=True)
            blk -= np.einsum('Xab,XY,Yij->iajb', Lab, K, Lij.conj(),
                             optimize=True)
            A[si:si + no * nv, sj:sj + no * nv] = blk.reshape(no * nv,
                                                              no * nv)
            if not tda:
                Lvo = Lfull[:, no:, :no]        # L_aj (a at ki, j at kj)
                Lov = Lfull[:, :no, no:]        # L_ib (i at ki, b at kj)
                blkB = 2.0 * np.einsum('X,Xia,Xjb->iajb', s0,
                                       Lov0[ki].conj(), Lov0[kj].conj(),
                                       optimize=True)
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
