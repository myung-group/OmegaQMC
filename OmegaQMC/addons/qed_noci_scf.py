"""
Determinants for QED-NOCI: coherent-state QED-HF as PySCF SCF subclasses.

Unlike the stand-alone loop of :func:`OmegaQMC.addons.qed_hf.run_qed_hf`, these
classes plug into PySCF's SCF machinery, so occupations can be controlled
(maximum overlap, broken symmetry) to converge excited and charge-transfer
determinants in the cavity. Each converged determinant is used in
:mod:`OmegaQMC.addons.qed_noci` with its own photon shift z_I.

Coherent-state QED-HF: wavefunction |Phi> (x) |z, 0>, z = <lam.d>_Phi / sqrt(2w),

    E = E_nuc + tr(h D) + 1/2 tr[(J - 1/2 K) D] + 1/2 tr(q D) - 1/2 sum_s tr(d D_s d D_s)
    F_s = f_s^(e) + 1/2 q - d D_s d

(d, q as in :class:`OmegaQMC.addons.qed_noci.PFIntegrals`). The nuclear dipole
and the Coulomb-like DSE term cancel against the coherent shift, so E and F are
origin invariant, also for ions.

Classes
    QEDRHF, QEDROHF   coherent-state QED-RHF / QED-ROHF.
    QEDBSROHF         broken-symmetry M_s = 0 QED-ROHF |core a(alpha) b(beta)|.
    BSROHF            the same without cavity.
    imom_rhf(mf, c_ref)  initial-maximum-overlap occupations for a closed-shell
                      excited (e.g. ionic) determinant.

Broken-symmetry ROHF. PySCF ROHF puts every open shell in the alpha space, i.e.
converges the high-spin determinant. Here mo_occ is a (2, nmo) array with four
shells: core c (1, 1), open a (1, 0), open b (0, 1), virtual v (0, 0). The SCF
minimizes E[D_alpha, D_beta]; stationarity of the restricted orbitals requires

    g_pq = sum_s F^s_pq (n^s_q - n^s_p) = 0  for all pairs whose occupations differ,

i.e. the blocks ca: F^b, cb: F^a, cv: F^a + F^b, av: F^a, bv: F^b, ab: F^a - F^b.
The generalized Roothaan effective Fock (MO basis of the current orbitals)

             c      a      b      v
        c    Fc     Fb     Fa     Fc
        a    Fb     e_a    G      Fa
        b    Fa     G      e_b    Fb
        v    Fc     Fa     Fb     Fc

    Fc = (F^a + F^b)/2,  G = (F^a - F^b)_ab / 2,
    e_a = (F^a_aa + F^b_bb)/2,  e_b = (F^a_bb + F^b_aa)/2

carries these blocks off the diagonal; at convergence they vanish. Occupations
are assigned by maximum overlap with reference orbitals (IMOM) or with the
previous iteration (MOM), never by aufbau. A virtual level shift (default
0.25 Eh) is needed; without it the open shells and diffuse virtuals swap and the
iteration oscillates. DIIS uses the commutator of the effective Fock with a
shell-weighted density (c 1, a 2/3, b 1/3, v 0), which also sees the a-b block.

The spin-inversion partner (C_b, C_a) has the same density and z; spin-adapted
singlets are formed in ``QEDNOCI.kernel(spin_pairs=...)``.
"""

import numpy as np
import scipy.linalg
from pyscf import lib, scf
from pyscf.lib import logger

from OmegaQMC.addons.qed_noci import PFIntegrals


# ---- coherent-state QED-HF -----------------------------------------------------
class _QEDHFMixin:
    _keys = {'pf'}

    def _init_cavity(self, omega, coupling_vec, origin=None, proper_dse=True):
        self.pf = PFIntegrals(self.mol, omega, coupling_vec, origin, proper_dse)

    def get_hcore(self, mol=None):
        """h + 1/2 q."""
        return super().get_hcore(self.mol if mol is None else mol) + .5 * self.pf.q

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        """J and K + d D d. In the coherent-state frame the DSE two-body term acts
        as the exchange part of the rank-one ERI d_pq d_rs (its Coulomb part cancels
        against the photon shift). Folding it into K keeps get_veff, energy_elec
        and the ROHF/BS-ROHF effective Fock consistent.
        NOTE: post-HF codes that call mf.get_jk see this modified K."""
        vj, vk = super().get_jk(mol, dm, hermi, with_j, with_k, omega)
        if with_k and not omega:
            d = self.pf.d
            vk = vk + d @ np.asarray(dm) @ d
        return vj, vk

    def dipole_lambda(self, dm=None):
        """<lam.d> (electronic + nuclear)."""
        return self.pf.dipole(self.make_rdm1() if dm is None else dm)

    def coherent_shift(self, dm=None):
        return self.pf.coherent_shift(self.make_rdm1() if dm is None else dm)

    def _spin_dms(self, dm):
        dm = np.asarray(dm)
        if dm.ndim == 2:
            return (.5 * dm, .5 * dm)
        return (dm[0], dm[1])

    def energy_bare_frame(self, dm=None, z=None):
        """E(Phi, z) = <H_e'> + w z^2 - sqrt(2w) z <lam.d>, evaluated without the
        coherent-state transform. With z = z_opt it equals the QED-HF energy."""
        if dm is None:
            dm = self.make_rdm1()
        dms = self._spin_dms(dm)
        dtot = dms[0] + dms[1]
        mol, pf = self.mol, self.pf
        w = pf.omega
        if z is None:
            z = pf.coherent_shift(dtot)
        h = scf.hf.get_hcore(mol)
        vj, vk = scf.hf.SCF.get_jk(self, mol, np.asarray(dms))
        e_el = np.einsum('uv,vu->', h, dtot)
        e_el += .5 * np.einsum('uv,vu->', vj[0] + vj[1], dtot)
        e_el -= .5 * (np.einsum('uv,vu->', vk[0], dms[0]) + np.einsum('uv,vu->', vk[1], dms[1]))
        de = np.einsum('uv,vu->', pf.d, dtot)
        dse = .5 * pf.d0 ** 2 + pf.d0 * de + .5 * np.einsum('uv,vu->', pf.q, dtot) + .5 * de ** 2
        for D in dms:
            dse -= .5 * np.einsum('uv,vw,wx,xu->', pf.d, D, pf.d, D)
        e_ph = w * z ** 2 - np.sqrt(2 * w) * z * (pf.d0 + de)
        return (mol.energy_nuc() + e_el + dse + e_ph).real

    def nuc_grad_method(self):
        raise NotImplementedError


class QEDRHF(_QEDHFMixin, scf.hf.RHF):
    """Closed-shell coherent-state QED-HF.

    Args: mol, omega (Hartree), coupling_vec (3,), origin (default nuclear
    center of mass; affects only <lam.d> and z), proper_dse."""

    def __init__(self, mol, omega, coupling_vec, origin=None, proper_dse=True):
        scf.hf.RHF.__init__(self, mol)
        self._init_cavity(omega, coupling_vec, origin, proper_dse)


class QEDROHF(_QEDHFMixin, scf.rohf.ROHF):
    """Restricted open-shell coherent-state QED-HF (DSE exchange folded into the
    per-spin K, so PySCF's Roothaan effective Fock picks it up unchanged)."""

    def __init__(self, mol, omega, coupling_vec, origin=None, proper_dse=True):
        scf.rohf.ROHF.__init__(self, mol)
        self._init_cavity(omega, coupling_vec, origin, proper_dse)


def _proj(mf, c_ref, mo_coeff):
    o = c_ref.T @ mf.get_ovlp() @ mo_coeff
    return np.einsum('ij,ij->j', o, o)


def imom_rhf(mf, c_ref):
    """Make a (QED-)RHF object occupy, at every iteration, the orbitals with the
    largest overlap with the fixed reference orbitals c_ref (IMOM)."""
    def get_occ(mo_energy=None, mo_coeff=None):
        mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        occ = np.zeros(mo_coeff.shape[1])
        occ[np.argsort(-_proj(mf, c_ref, mo_coeff))[:c_ref.shape[1]]] = 2.
        return occ
    mf.get_occ = get_occ
    return mf


# ---- broken-symmetry ROHF ------------------------------------------------------
_W = {'c': 1., 'a': 2. / 3, 'b': 1. / 3, 'v': 0.}


def _shells(occ):
    na, nb = np.asarray(occ[0]) > .5, np.asarray(occ[1]) > .5
    return {'c': na & nb, 'a': na & ~nb, 'b': nb & ~na, 'v': ~na & ~nb}


def bs_fock_mo(fa, fb, occ, shift=0.):
    """Effective BS-ROHF Fock in the MO basis (fa, fb: alpha / beta Fock).
    shift > 0 adds a virtual level shift (does not move the fixed point)."""
    sh = _shells(occ)
    c, a, b, v = sh['c'], sh['a'], sh['b'], sh['v']
    assert a.sum() == 1 and b.sum() == 1, 'one alpha-only and one beta-only orbital expected'
    m = .5 * (fa + fb)
    for x, y, f in ((c, a, fb), (c, b, fa), (a, v, fa), (b, v, fb)):
        m[np.ix_(x, y)] = f[np.ix_(x, y)]
        m[np.ix_(y, x)] = f[np.ix_(y, x)]
    ia, ib = np.where(a)[0][0], np.where(b)[0][0]
    m[ia, ib] = m[ib, ia] = .5 * (fa[ia, ib] - fb[ia, ib])
    m[ia, ia] = .5 * (fa[ia, ia] + fb[ib, ib])
    m[ib, ib] = .5 * (fa[ib, ib] + fb[ia, ia])
    if shift:
        iv = np.where(v)[0]
        m[iv, iv] += shift
    return m


def bs_grad(fa, fb, occ):
    """Orbital gradient over pairs p > q whose occupations differ (MO-basis Fock)."""
    na, nb = np.asarray(occ[0], float), np.asarray(occ[1], float)
    p, q = np.tril_indices(len(na), -1)
    keep = (na[p] != na[q]) | (nb[p] != nb[q])
    p, q = p[keep], q[keep]
    return 2 * (fa[p, q] * (na[q] - na[p]) + fb[p, q] * (nb[q] - nb[p]))


class _BSROHFMixin:
    """Call bs_setup(c_core, c_a, c_b) before kernel(bs_guess())."""

    _keys = {'bs_ref', 'bs_mom'}
    level_shift = 0.25
    conv_check = False      # PySCF's final unshifted step would undo the damping

    def bs_setup(self, c_core, c_a, c_b, mom='imom'):
        """Reference orbitals (AO coefficients) for the core, the alpha-only orbital
        a and the beta-only orbital b. mom = 'imom': keep these references;
        'mom': follow the previous iteration."""
        assert mom in ('imom', 'mom')
        self.bs_ref = [np.asarray(c_core), np.asarray(c_a).reshape(-1, 1),
                       np.asarray(c_b).reshape(-1, 1)]
        self.bs_mom = mom
        return self

    def bs_guess(self, c0=None):
        """Density of [core, a, b] = bs_ref completed to an orthonormal set (or of c0
        with the occupations picked by overlap with bs_ref)."""
        if c0 is None:
            s = self.get_ovlp()
            cr = np.hstack(self.bs_ref)
            n = scipy.linalg.null_space(cr.T @ s)
            w, u = np.linalg.eigh(n.T @ s @ n)
            c0 = np.hstack([cr, n @ (u / np.sqrt(w)) @ u.T])
        return self.make_rdm1(c0, self.get_occ(None, c0))

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = mo_coeff.shape[1]
        c_core, c_a, c_b = self.bs_ref
        free = np.ones(nmo, bool)
        core = np.argsort(-_proj(self, c_core, mo_coeff))[:c_core.shape[1]]
        free[core] = False
        ia = int(np.argmax(np.where(free, _proj(self, c_a, mo_coeff), -1)))
        free[ia] = False
        ib = int(np.argmax(np.where(free, _proj(self, c_b, mo_coeff), -1)))
        occ = np.zeros((2, nmo))
        occ[:, core] = 1.
        occ[0, ia] = occ[1, ib] = 1.
        if self.bs_mom == 'mom':
            self.bs_ref = [mo_coeff[:, core], mo_coeff[:, [ia]], mo_coeff[:, [ib]]]
        return occ

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        return scf.rohf.make_rdm1(mo_coeff, np.asarray(mo_occ))

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                 fock_last=None):
        if h1e is None:
            h1e = self.get_hcore()
        if s1e is None:
            s1e = self.get_ovlp()
        if dm is None:
            dm = self.make_rdm1()
        if vhf is None:
            vhf = self.get_veff(self.mol, dm)
        c = getattr(dm, 'mo_coeff', None)
        occ = getattr(dm, 'mo_occ', None)
        if c is None:
            c, occ = self.mo_coeff, self.mo_occ
        focka, fockb = h1e + vhf[0], h1e + vhf[1]
        shift = 0.
        if cycle >= 0 or diis is not None:
            shift = self.level_shift if level_shift_factor is None else level_shift_factor
        m = bs_fock_mo(c.T @ focka @ c, c.T @ fockb @ c, occ, shift)
        sc = s1e @ c
        f = sc @ m @ sc.T
        if cycle >= 0 or diis is not None:
            if diis_start_cycle is None:
                diis_start_cycle = self.diis_start_cycle
            if diis and cycle >= diis_start_cycle:
                sh = _shells(occ)
                w = sum(_W[k] * sh[k] for k in sh)
                f = diis.update(s1e, (c * w) @ c.T, f, self, h1e, vhf, f_prev=fock_last)
        return lib.tag_array(f, focka=focka, fockb=fockb)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None or getattr(fock, 'focka', None) is None:
            fock = self.get_fock(dm=self.make_rdm1(mo_coeff, mo_occ))
        c = mo_coeff
        return bs_grad(c.T @ fock.focka @ c, c.T @ fock.fockb @ c, mo_occ)

    def det(self):
        """(C_alpha_occ, C_beta_occ) of the converged determinant; spin partner = reversed."""
        occ = np.asarray(self.mo_occ)
        return self.mo_coeff[:, occ[0] > .5], self.mo_coeff[:, occ[1] > .5]

    def spin_square(self, mo_coeff=None, s=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        ca, cb = (mo_coeff[:, np.asarray(self.mo_occ)[k] > .5] for k in (0, 1))
        return scf.uhf.spin_square((ca, cb), self.get_ovlp() if s is None else s)

    def analyze(self, *args, **kwargs):
        return None

    def _finalize(self):
        if self.converged:
            logger.note(self, 'converged BS-ROHF energy = %.15g', self.e_tot)
        else:
            logger.note(self, 'BS-ROHF not converged, energy = %.15g', self.e_tot)
        return self


class BSROHF(_BSROHFMixin, scf.rohf.ROHF):
    """Field-free M_s = 0 broken-symmetry ROHF."""


class QEDBSROHF(_BSROHFMixin, QEDROHF):
    """Coherent-state QED-BS-ROHF |core a(alpha) b(beta)|."""


def run_qed_bs_rohf(mol, omega, coupling_vec, c_core, c_a, c_b, origin=None,
                    proper_dse=True, conv_tol=1e-10, max_cycle=300):
    """Converge QED-BS-ROHF from reference orbitals; returns the converged object.
    ``mf.det()`` is the determinant, ``spin_flip(mf.det())`` its spin partner."""
    mf = QEDBSROHF(mol, omega, coupling_vec, origin, proper_dse)
    mf.verbose, mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 0, conv_tol, 1e-7, max_cycle
    mf.bs_setup(c_core, c_a, c_b)
    mf.kernel(mf.bs_guess())
    return mf
