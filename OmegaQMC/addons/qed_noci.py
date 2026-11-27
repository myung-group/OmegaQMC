"""
QED-NOCI: nonorthogonal configuration interaction for a molecule coupled to
one cavity mode (Pauli-Fierz Hamiltonian, length gauge, dipole approximation).

    H = H_e + w b^+b - sqrt(w/2) (lam.d)(b + b^+) + 1/2 (lam.d)^2,
    d = -sum_i r_i + sum_A Z_A R_A          (total dipole, electron charge -1)

Basis functions are products of a single determinant and a displaced photon
number state, each determinant carrying its own coherent shift:

    |I n> = |Phi_I> (x) |z_I, n>,   |z, n> = U(z)|n>,   U(z) = exp(z (b^+ - b)),
    z_I = <lam.d>_I / sqrt(2 w)      (mean-field optimum of determinant I)

so the photon displacement follows the dipole of every determinant instead of
one common frame. The Hamiltonian and overlap in this basis factorize into an
electronic NOCI part and an analytic photon part:

    S_{Im,Jn} = S_IJ D_mn
    H_{Im,Jn} = <I|H_e'|J> D_mn + w S_IJ B_mn - sqrt(w/2) <I|lam.d|J> X_mn

    D_mn = <z_I,m|z_J,n> = <m|U(z_J - z_I)|n>
    X_mn = <z_I,m|b + b^+|z_J,n>,   B_mn = <z_I,m|b^+ b|z_J,n>

H_e' = H_e + 1/2 (lam.d)^2 is an ordinary electronic Hamiltonian with

    h'_pq    = h_pq + 1/2 q_pq + d0 d_pq
    (pq|rs)' = (pq|rs) + d_pq d_rs
    E0'      = E_nuc + 1/2 d0^2

    d_pq = -<p|lam.r|q>,   d0 = lam . sum_A Z_A R_A,   q_pq = <p|(lam.r)^2|q>

(``proper_dse=False`` replaces q by d S^-1 d, the operator-squared form that is
exact only in a complete basis; this is the same switch as in
:mod:`OmegaQMC.addons.qed_fci`). The electronic matrix elements use the
generalized Slater-Condon rules of Thom and Head-Gordon [J. Chem. Phys. 131,
124113 (2009)] for Lowdin-paired determinants; the rank-one DSE two-electron
term enters as an extra J/K pair built from d alone,

    J_dse[A].B = tr(d A) tr(d B),   K_dse(A, B) = tr(d B^T d A^T).

Gauge origin: the nuclear center of mass by default (``origin`` argument). The
spectrum is origin invariant (also for ions: a constant dipole shift is absorbed
by the photon displacement, exactly at every truncation when all determinants
share one frame, and at convergence in N otherwise); the values of <lam.d> and
z_I depend on the origin.

Determinants are given as ``(C_alpha_occ, C_beta_occ)`` in the AO basis (column
order defines the sign). Suitable determinants (coherent-state QED-RHF/ROHF,
maximum-overlap excited determinants and broken-symmetry M_s = 0 QED-ROHF
pairs) are produced by :mod:`OmegaQMC.addons.qed_noci_scf`; see
``examples/run_qed_noci_HeH+.py`` for a complete calculation.

Photon cutoff: n = 0..N_I per determinant (``nph`` may be a sequence), so the
Hamiltonian dimension is sum_I (N_I + 1).
"""

import numpy as np
import scipy.linalg
from scipy.special import eval_genlaguerre, gammaln
from pyscf import scf

ZERO_TOL = 1e-8   # Lowdin singular values below this are treated as exact zeros


# ---- Pauli-Fierz integrals -----------------------------------------------------
def center_of_mass(mol):
    """Nuclear center of mass (Bohr), the default gauge origin."""
    mass = mol.atom_mass_list(isotope_avg=True)
    return np.einsum('i,ix->x', mass, mol.atom_coords()) / mass.sum()


class PFIntegrals:
    """AO-basis Pauli-Fierz ingredients for one cavity mode.

    Args:
        mol: PySCF Mole.
        omega: photon energy (Hartree).
        coupling_vec: lambda vector (3,) in a.u.; a float means lam * e_z.
        origin: gauge origin (Bohr); default nuclear center of mass.
        proper_dse: True -> one-body DSE from <p|(lam.r)^2|q>; False -> d S^-1 d.

    Attributes (AO basis): ``d`` = -<u|lam.r|v>, ``q`` (one-body DSE matrix),
    ``d0`` (nuclear dipole), ``ovlp``, ``omega``, ``lam``, ``origin``.
    """

    def __init__(self, mol, omega, coupling_vec, origin=None, proper_dse=True):
        lam = np.asarray(coupling_vec, dtype=float)
        if lam.ndim == 0:
            lam = np.array([0., 0., float(lam)])
        assert lam.shape == (3,)
        self.mol = mol
        self.omega = float(omega)
        self.lam = lam
        self.proper_dse = bool(proper_dse)
        self.origin = center_of_mass(mol) if origin is None else np.asarray(origin, float)
        nao = mol.nao_nr()
        with mol.with_common_orig(self.origin):
            r = mol.intor_symmetric('int1e_r', comp=3)
            rr = mol.intor_symmetric('int1e_rr', comp=9).reshape(3, 3, nao, nao)
        self.ovlp = mol.intor_symmetric('int1e_ovlp')
        lr = np.einsum('k,kuv->uv', lam, r)
        self.d = -lr
        if proper_dse:
            self.q = np.einsum('k,l,kluv->uv', lam, lam, rr)
        else:
            self.q = lr @ np.linalg.solve(self.ovlp, lr)
        coords = mol.atom_coords() - self.origin
        self.d0 = float(lam @ np.einsum('a,ax->x', mol.atom_charges(), coords))

    def dipole(self, dm):
        """<lam.d> for a spin-summed (nao, nao) or spin-resolved (2, nao, nao) density."""
        dm = np.asarray(dm)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        return self.d0 + np.einsum('uv,vu->', self.d, dm).real

    def coherent_shift(self, dm):
        """Optimal coherent-state displacement z = <lam.d> / sqrt(2 w)."""
        return self.dipole(dm) / np.sqrt(2. * self.omega)


class QEDNOCIIntegrals:
    """AO integrals for QED-NOCI matrix elements (bare, untransformed frame).

    Two-electron terms use J/K builds for nonsymmetric codensities: in-core
    ERIs for ``nao <= incore_max_nao``, direct integrals otherwise.
    """

    def __init__(self, mol, omega, coupling_vec, origin=None, proper_dse=True,
                 incore_max_nao=160):
        self.mol = mol
        self.pf = PFIntegrals(mol, omega, coupling_vec, origin, proper_dse)
        self.omega = self.pf.omega
        self.ovlp = self.pf.ovlp
        self.d, self.d0 = self.pf.d, self.pf.d0
        hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
        self.hcore_dse = hcore + .5 * self.pf.q + self.d0 * self.d
        self.enuc_dse = mol.energy_nuc() + .5 * self.d0 ** 2
        self._eri = mol.intor('int2e', aosym='s8') if mol.nao_nr() <= incore_max_nao else None

    def jk(self, dm_j, dms_k):
        """J[dm_j] and K~[B] for B in dms_k, K~[B]_ms = sum_ln (mn|ls) B_ln,
        so that K(A, B) = sum(A * K~[B])."""
        dms = np.asarray([dm_j] + [b.T for b in dms_k])
        if self._eri is not None:
            vj, vk = scf.hf.dot_eri_dm(self._eri, dms, hermi=0)
        else:
            vj, vk = scf.hf.get_jk(self.mol, dms, hermi=0)
        return vj[0], vk[1:]

    def z_of(self, det):
        """Mean-field coherent shift z_I of a determinant (C_alpha_occ, C_beta_occ)."""
        return self.pf.coherent_shift(det[0] @ det[0].T + det[1] @ det[1].T)


# ---- electronic NOCI matrix elements -------------------------------------------
def pair_orbitals(ovlp, cw, cx):
    """Lowdin pairing of the occupied orbitals of one spin.
    Returns (cw_paired, cx_paired, s, phase) with <cw_i|cx_j> = s_i delta_ij."""
    if cw.shape[1] == 0:
        return cw, cx, np.zeros(0), 1.
    u, s, vh = np.linalg.svd(cw.T @ ovlp @ cx)
    return cw @ u, cx @ vh.T, s, np.linalg.det(u) * np.linalg.det(vh)


def elec_element(ints, detw, detx, zero_tol=ZERO_TOL):
    """(S_wx, <w|H_e'|x>, <w|lam.d|x>) between two determinants."""
    nao = ints.ovlp.shape[0]
    d = ints.d
    phase = 1.
    zeros = []                      # (spin, codensity P_k) for s_k ~ 0
    s_red = 1.
    m_spin = []
    for sigma in (0, 1):
        cw, cx, s, ph = pair_orbitals(ints.ovlp, detw[sigma], detx[sigma])
        phase *= ph
        m = np.zeros((nao, nao))
        for i, si in enumerate(s):
            p_i = np.outer(cw[:, i], cx[:, i])
            if abs(si) < zero_tol:
                zeros.append((sigma, p_i))
            else:
                s_red *= si
                m += p_i / si
        m_spin.append(m)
    s_red *= phase

    def k_dse(a, b):
        return np.sum((d @ b.T @ d) * a)

    if len(zeros) == 0:
        m_tot = m_spin[0] + m_spin[1]
        vj, vk = ints.jk(m_tot, m_spin)
        dm_ = np.sum(d * m_tot)
        e1 = np.sum(ints.hcore_dse * m_tot)
        e2 = .5 * np.sum(vj * m_tot) + .5 * dm_ ** 2
        for sig in (0, 1):
            e2 -= .5 * (np.sum(m_spin[sig] * vk[sig]) + k_dse(m_spin[sig], m_spin[sig]))
        return s_red, s_red * (e1 + e2 + ints.enuc_dse), s_red * (ints.d0 + dm_)

    if len(zeros) == 1:
        sigma, p_k = zeros[0]
        m_tot = m_spin[0] + m_spin[1]
        vj, vk = ints.jk(m_tot, [m_spin[sigma]])
        dp = np.sum(d * p_k)
        val = np.sum((ints.hcore_dse + vj) * p_k) - np.sum(p_k * vk[0])
        val += np.sum(d * m_tot) * dp - k_dse(p_k, m_spin[sigma])
        return 0., s_red * val, s_red * dp

    if len(zeros) == 2:
        (sig_k, p_k), (sig_l, p_l) = zeros
        vj, vk = ints.jk(p_k, [p_l])
        val = np.sum(vj * p_l) + np.sum(d * p_k) * np.sum(d * p_l)
        if sig_k == sig_l:
            val -= np.sum(p_k * vk[0]) + k_dse(p_k, p_l)
        return 0., s_red * val, 0.

    return 0., 0., 0.


def build_elec(ints, dets, zero_tol=ZERO_TOL):
    """Electronic S, H_e', lam.d matrices over the determinant list."""
    n = len(dets)
    s, h, dd = (np.zeros((n, n)) for _ in range(3))
    for w in range(n):
        for x in range(w, n):
            s[w, x], h[w, x], dd[w, x] = elec_element(ints, dets[w], dets[x], zero_tol)
            s[x, w], h[x, w], dd[x, w] = s[w, x], h[w, x], dd[w, x]
    return s, h, dd


def spin_flip(det):
    """Spin-inverted determinant: alpha and beta occupied orbitals exchanged."""
    return (det[1], det[0])


def solve(h, s, lindep=1e-10):
    """H C = S C E with canonical orthogonalization (drops S eigenvalues < lindep)."""
    sval, svec = scipy.linalg.eigh(s)
    keep = sval > lindep
    x = svec[:, keep] / np.sqrt(sval[keep])
    e, c = scipy.linalg.eigh(x.T @ h @ x)
    return e, x @ c


# ---- photon factors ------------------------------------------------------------
def displacement_matrix(delta, nmax):
    """D_mn = <m|U(delta)|n>, m, n = 0..nmax, real delta."""
    out = np.empty((nmax + 1, nmax + 1))
    x = delta * delta
    pref = np.exp(-x / 2)
    for m in range(nmax + 1):
        for n in range(nmax + 1):
            if m >= n:
                out[m, n] = (np.exp(.5 * (gammaln(n + 1) - gammaln(m + 1))) * pref
                             * delta ** (m - n) * eval_genlaguerre(n, m - n, x))
            else:
                out[m, n] = (np.exp(.5 * (gammaln(m + 1) - gammaln(n + 1))) * pref
                             * (-delta) ** (n - m) * eval_genlaguerre(m, n - m, x))
    return out


def photon_factors(zi, zj, nmi, nmj):
    """(D, X, B) blocks between |z_i, m> (m <= nmi) and |z_j, n> (n <= nmj)."""
    nmax = max(nmi, nmj) + 1
    dfull = displacement_matrix(zj - zi, nmax)
    dm = dfull[:nmi + 1, :nmj + 1]
    m = np.arange(nmi + 1)[:, None]
    n = np.arange(nmj + 1)[None, :]
    d_m_nm1 = np.where(n > 0, dfull[m, np.maximum(n - 1, 0)], 0.)       # D_{m,n-1}
    d_mm1_n = np.where(m > 0, dfull[np.maximum(m - 1, 0), n], 0.)       # D_{m-1,n}
    d_mm1_nm1 = np.where((m > 0) & (n > 0),
                         dfull[np.maximum(m - 1, 0), np.maximum(n - 1, 0)], 0.)
    x = np.sqrt(n) * d_m_nm1 + np.sqrt(m) * d_mm1_n + (zi + zj) * dm
    b = (np.sqrt(m * n) * d_mm1_nm1 + zj * np.sqrt(m) * d_mm1_n
         + zi * np.sqrt(n) * d_m_nm1 + zi * zj * dm)
    return dm, x, b


# ---- QED-NOCI ------------------------------------------------------------------
class QEDNOCI:
    """QED-NOCI over determinants ``dets`` with photon shifts ``zs`` and cutoff ``nph``.

    Args:
        ints: :class:`QEDNOCIIntegrals`.
        dets: list of (C_alpha_occ, C_beta_occ).
        zs: None (z_I of each determinant, default), a float (one common
            frame for all), or a sequence of per-determinant shifts.
        nph: int or sequence, photon cutoff N_I (n = 0..N_I).

    After :meth:`kernel`: ``e``, ``c`` (columns in the |I n> basis), ``dipole``
    = <lam.d>_k and ``nphoton`` = <b^+ b>_k (bare photon number) per root.
    """

    def __init__(self, ints, dets, zs=None, nph=1, zero_tol=ZERO_TOL):
        self.ints = ints
        self.dets = dets
        n = len(dets)
        if zs is None:
            zs = [ints.z_of(det) for det in dets]
        elif np.ndim(zs) == 0:
            zs = [float(zs)] * n
        self.zs = np.asarray(zs, dtype=float)
        self.nph = np.broadcast_to(np.asarray(nph, dtype=int), (n,)).copy()
        self.s_el, self.h_el, self.d_el = build_elec(ints, dets, zero_tol)
        self.offsets = np.concatenate([[0], np.cumsum(self.nph + 1)])

    @property
    def dim(self):
        return int(self.offsets[-1])

    def labels(self):
        """(I, n) for every basis function, in matrix order."""
        return [(i, m) for i in range(len(self.dets)) for m in range(self.nph[i] + 1)]

    def build(self):
        """Full (H, S) and the operators (lam.d, b^+b) in the product basis."""
        w = self.ints.omega
        h, s, ld, nb = (np.zeros((self.dim, self.dim)) for _ in range(4))
        for i in range(len(self.dets)):
            for j in range(len(self.dets)):
                dm, x, b = photon_factors(self.zs[i], self.zs[j], self.nph[i], self.nph[j])
                blk = (slice(self.offsets[i], self.offsets[i + 1]),
                       slice(self.offsets[j], self.offsets[j + 1]))
                s[blk] = self.s_el[i, j] * dm
                h[blk] = (self.h_el[i, j] * dm + w * self.s_el[i, j] * b
                          - np.sqrt(w / 2) * self.d_el[i, j] * x)
                ld[blk] = self.d_el[i, j] * dm
                nb[blk] = self.s_el[i, j] * b
        return h, s, ld, nb

    def spin_adapt(self, pairs, sign=+1):
        """Columns (full basis -> adapted basis) combining every spin-inversion pair
        (i, j), det_j = spin_flip(det_i), as |i n> + sign |j n> for all n; other
        determinants unchanged. With a restricted core the M_s = 0 pair spans exactly
        S = 0 and S = 1 and H commutes with spin inversion, so sign = +1 (singlet)
        block-diagonalizes H. The pair shares one density, hence one z; nph must match."""
        paired = {j for _, j in pairs}
        partner = dict(pairs)
        cols = []
        for i in range(len(self.dets)):
            if i in paired:
                continue
            for m in range(self.nph[i] + 1):
                v = np.zeros(self.dim)
                v[self.offsets[i] + m] = 1.
                if i in partner:
                    j = partner[i]
                    assert self.nph[j] == self.nph[i] and abs(self.zs[j] - self.zs[i]) < 1e-10
                    v[self.offsets[j] + m] = sign
                cols.append(v)
        return np.array(cols).T

    def kernel(self, lindep=1e-10, spin_pairs=None, sign=+1):
        """Diagonalize. spin_pairs: list of (i, j) spin-inversion partners -> keep only
        the sign = +1 (singlet) or -1 (M_s = 0 triplet) combination of each pair."""
        h, s, ld, nb = self.build()
        if spin_pairs:
            t = self.spin_adapt(spin_pairs, sign)
            e, c = solve(t.T @ h @ t, t.T @ s @ t, lindep)
            c = t @ c
        else:
            e, c = solve(h, s, lindep)
        self.e, self.c, self.s = e, c, s
        self.dipole = np.einsum('ik,ij,jk->k', c, ld, c)
        self.nphoton = np.einsum('ik,ij,jk->k', c, nb, c)
        return e, c

    def weights(self, k):
        """Chirgwin-Coulson weights c_In (S c)_In of root k, shape (ndet, max N + 1)."""
        cw = self.c[:, k] * (self.s @ self.c[:, k])
        out = np.zeros((len(self.dets), self.nph.max() + 1))
        for (i, m), v in zip(self.labels(), cw):
            out[i, m] = v
        return out


def run_qed_noci(mol, omega, coupling_vec, dets, nph_max=4, zs=None,
                 spin_pairs=None, sign=+1, origin=None, proper_dse=True,
                 lindep=1e-10):
    """QED-NOCI in the determinant basis ``dets`` (functional interface).

    Args:
        mol: PySCF Mole.
        omega: photon energy (Hartree).
        coupling_vec: lambda vector (3,) in a.u.
        dets: list of (C_alpha_occ, C_beta_occ), AO basis.
        nph_max: photon cutoff N (int, or one value per determinant).
        zs: photon shifts; None = per-determinant z_I (QED-NOCI), a float = one
            common frame.
        spin_pairs: list of (i, j) with dets[j] = spin_flip(dets[i]); only the
            ``sign`` combination of each pair is kept (+1 singlet, -1 triplet).
        origin: gauge origin (Bohr), default nuclear center of mass.
        proper_dse: see :class:`PFIntegrals`.

    Returns:
        dict with 'e_qed_noci' (lowest root), 'eigenvalues', 'eigenvectors'
        (|I n> basis), 'dipole' (<lam.d> per root), 'n_photon' (<b^+b> per root,
        bare photon number), 'z' (per-determinant shifts), 'ndet', 'ndim_total',
        'solver' (the :class:`QEDNOCI` object, for weights/labels).
    """
    ints = QEDNOCIIntegrals(mol, omega, coupling_vec, origin, proper_dse)
    q = QEDNOCI(ints, dets, zs=zs, nph=nph_max)
    e, c = q.kernel(lindep=lindep, spin_pairs=spin_pairs, sign=sign)
    return {
        'e_qed_noci': float(e[0]),
        'eigenvalues': e,
        'eigenvectors': c,
        'dipole': q.dipole,
        'n_photon': q.nphoton,
        'z': q.zs,
        'ndet': len(dets),
        'ndim_total': q.dim,
        'solver': q,
    }
