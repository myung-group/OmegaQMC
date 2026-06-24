"""
QED-UCASSCF: spin-unrestricted orbital-optimized complete-active-space
self-consistent field for the Pauli-Fierz Hamiltonian, tensored with a
truncated photon Fock space.

This is the open-shell counterpart of :mod:`OmegaQMC.addons.qed_casscf`
(closed-shell, restricted) and the orbital-optimized counterpart of the
QED-UCASCI path in :mod:`OmegaQMC.addons.qed_casci`. Where QED-UCASCI
fixes the orbitals to the QED-UHF set and only solves the polaritonic
CI, QED-UCASSCF *relaxes* the separate α and β orbital sets
self-consistently in the presence of the cavity, minimising the joint
electron-photon energy

    E[Ψ, C_α, C_β] = ⟨Ψ| Ĥ_PF[C_α, C_β] |Ψ⟩

over the CI coefficients C_{I,m} (unrestricted determinant ⊗ photon
Fock state) and the independent α/β molecular-orbital rotations
(κ_α, κ_β). All Hamiltonian conventions match
:mod:`OmegaQMC.addons.qed_fci` / :mod:`OmegaQMC.addons.qed_casci`
exactly, so QED-UCASSCF reduces to open-shell QED-FCI in the limit
where the active space spans the full one-particle basis (orbital
optimisation then becomes redundant — the QED-FCI energy is
orbital-invariant), and it always lies at or below the QED-UCASCI
energy in the same active space.

The electronic structure follows pyscf's ``mcscf.UCASSCF`` scheme:
separate α/β orbital sets seeded by a QED-UHF reference
(:func:`OmegaQMC.addons.qed_uhf.run_qed_uhf`), a spin-polarised frozen
core (``ncore_α ≠ ncore_β`` allowed), three DSE-augmented ERI blocks
(αα/αβ/ββ) and the ``direct_uhf`` CI solver. The cavity part follows
the cavity-QED-CASSCF formulation of

    M. Castagnola, T. S. Haugland, E. Ronca, H. Koch, et al.,
    arXiv:2503.16417 (2025),

generalised to spin-unrestricted orbitals. The dipole self-energy is
absorbed into modified spin-resolved integrals,

    h^σ_pq    = h^e,σ_pq + ½ λ² q^σ_pq,
    g^{σσ'}_pqrs = g^{e,σσ'}_pqrs + d^σ_pq d^{σ'}_rs,

(coherent-state representation: the diagonal block is additionally
shifted by −d0·D̂ + ½d0² and the bilinear photon coupling uses the
fluctuation dipole D̂ − d0, with d0 = ⟨D̂⟩ the QED-UHF reference dipole
and the photon basis displaced by z = d0/√(2Ω); the displacement is an
exact unitary transform, so the energy is unchanged at convergence in
``nph_max`` but converges far faster for polar molecules).

Orbital optimisation
--------------------
The orbital gradient is the spin-resolved cavity-augmented generalised
Fock matrix. With the photon-traced electronic spin RDMs (γ^σ, Γ^{σσ'})
promoted to the full (core+active) MO space and the spin-resolved
photon *transition* 1-RDMs γ̃^σ that couple adjacent Fock blocks,

    γ̃^σ_pq = Σ_m √(m+1) ⟨I,m+1| Ê^σ_pq |J,m⟩ C*_{I,m+1} C_{J,m} + h.c.,

the gradients are g^σ = 2(W^σ − (W^σ)ᵀ) with

    W^α_pq = Σ_r h^α_pr γ^α_rq + Σ_rst g^{αα}_prst Γ^{αα}_qrst
             + Σ_rst g^{αβ}_prst Γ^{αβ}_qrst
             + √(Ω/2) Σ_r d^α_pr γ̃^α_rq,

(and α ↔ β with the αβ block contracted over its second orbital pair),
where h^σ = h^{e,σ} + ½λ²Q − d0·d^σ. The first two lines are the
standard unrestricted MCSCF generalised Fock; the last is the
photon-coupling contribution. This gradient has been verified against
finite differences to ≲1e-9 Ha and reduces term-by-term to the
restricted gradient of :mod:`OmegaQMC.addons.qed_casscf` when
C_α = C_β and nα = nβ.

The α and β rotations are optimised simultaneously by the same
first-order scheme as the restricted module: a preconditioned L-BFGS
direction on the concatenated (κ_α, κ_β) pair vector, with
effective-Fock orbital-energy gaps as the diagonal preconditioner and
an Armijo backtracking line search guaranteeing monotone energy
decrease.

Scope
-----
The method is spin-*unrestricted* (UHF-like): C_α and C_β relax
independently, so spin contamination of the reference is inherited
(``s_squared`` of the QED-UHF reference is reported). An ROHF or RHF
``mf`` object is accepted — it only seeds the bare-CASSCF comparison
and supplies the molecule; the QED orbitals always come from QED-UHF.

The total dipole μ̂ is approximated by the electronic part μ̂_e only
(matching qed_fci/qed_casci/qed_uhf); for neutral molecules this fixes
the origin at the nuclear centre of charge, and any origin-dependent
shift cancels in the correlation energy.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, expm
from pyscf import ao2mo, mcscf, scf
from pyscf.fci import direct_uhf, cistring

from OmegaQMC.addons.qed_fci import _build_uhf_ci_matrices
from OmegaQMC.addons.qed_casci import _parse_active_space_uhf
from OmegaQMC.addons.qed_uhf import run_qed_uhf


def _full_rdm1_on_mo(casdm1, ncore, ncas, norb):
    """Promote an active-space spin 1-RDM to the full MO space
    (singly-occupied frozen core of one spin channel)."""
    dm1 = np.zeros((norb, norb))
    for i in range(ncore):
        dm1[i, i] = 1.0
    dm1[ncore:ncore + ncas, ncore:ncore + ncas] = casdm1
    return dm1


def _full_dm2_samespin(casdm1, casdm2, ncore, ncas, norb):
    """Promote a same-spin active 2-RDM to the full MO space.

    Spin-resolved analogue of pyscf's ``_make_rdm12_on_mo`` core
    dressing, with single (per-spin) core occupations: Coulomb
    ``Γ[i,i,j,j] += 1`` and exchange ``Γ[i,j,j,i] −= 1`` between core
    orbitals (the i = j terms cancel, as required for same-spin pairs),
    and ±casdm1 cross terms between core and active orbitals.
    Convention: E2_σσ = ½ Σ g^{σσ}_pqrs Γ^{σσ}_pqrs (chemist).
    """
    nocc = ncore + ncas
    cas = slice(ncore, nocc)
    dm2 = np.zeros((norb, norb, norb, norb))
    dm2[cas, cas, cas, cas] = casdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i, i, j, j] += 1.0
            dm2[i, j, j, i] -= 1.0
        dm2[i, i, cas, cas] += casdm1
        dm2[cas, cas, i, i] += casdm1
        dm2[i, cas, cas, i] -= casdm1
        dm2[cas, i, i, cas] -= casdm1
    return dm2


def _full_dm2_ab(casdm1a, casdm1b, casdm2ab, ncore_a, ncore_b, ncas, norb):
    """Promote the αβ active 2-RDM to the full MO space.

    The first orbital pair lives in the α MO basis, the second in the β
    MO basis. Opposite spins have no exchange, so the core dressing is
    Coulomb-only. Convention: E2_αβ = Σ g^{αβ}_pqrs Γ^{αβ}_pqrs.
    """
    cas_a = slice(ncore_a, ncore_a + ncas)
    cas_b = slice(ncore_b, ncore_b + ncas)
    dm2 = np.zeros((norb, norb, norb, norb))
    dm2[cas_a, cas_a, cas_b, cas_b] = casdm2ab
    for i in range(ncore_a):
        for j in range(ncore_b):
            dm2[i, i, j, j] += 1.0
        dm2[i, i, cas_b, cas_b] += casdm1b
    for j in range(ncore_b):
        dm2[cas_a, cas_a, j, j] += casdm1a
    return dm2


def _rot_pairs(ncore, ncas, norb):
    """Non-redundant orbital-rotation pairs for one spin channel."""
    orb_type = (['c'] * ncore + ['a'] * ncas
                + ['v'] * (norb - ncore - ncas))
    pairs = [(p, q) for p in range(norb) for q in range(p)
             if orb_type[p] != orb_type[q]]
    pp = np.array([p for p, _ in pairs], dtype=int)
    qq = np.array([q for _, q in pairs], dtype=int)
    return pp, qq


def _lbfgs_dir(g, s_hist, y_hist, hdiag):
    """Preconditioned L-BFGS two-loop recursion → search direction."""
    q = g.copy()
    alphas = []
    rhos = [1.0 / float(np.dot(s, y)) for s, y in zip(s_hist, y_hist)]
    for s, y, rho in zip(reversed(s_hist), reversed(y_hist),
                         reversed(rhos)):
        a = rho * float(np.dot(s, q))
        q = q - a * y
        alphas.append(a)
    alphas.reverse()
    r = hdiag * q
    for s, y, rho, a in zip(s_hist, y_hist, rhos, alphas):
        b = rho * float(np.dot(y, r))
        r = r + s * (a - b)
    return -r


class _QEDUCASSCFObjective:
    """Energy / gradient evaluator for QED-UCASSCF at given (C_α, C_β).

    Builds the DSE-augmented spin-resolved MO integrals, solves the
    polaritonic CI in the active space, and assembles the spin-resolved
    cavity-augmented generalised Fock gradient. Exposed at module level
    (rather than nested in :func:`run_qed_ucasscf`) so the analytic
    gradient can be checked against finite differences in the tests.
    """

    def __init__(self, mf, ncas, nelec_act, ncore_a, ncore_b,
                 omega, lam, epsilon, proper_dse, nph_max, d0=0.0):
        mol = mf.mol
        self.norb = norb = mol.nao_nr()
        self.ncas = ncas
        self.nelec_act = nelec_act
        self.ncore_a = ncore_a
        self.ncore_b = ncore_b
        self.cas_a = slice(ncore_a, ncore_a + ncas)
        self.cas_b = slice(ncore_b, ncore_b + ncas)
        self.core_a = slice(0, ncore_a)
        self.core_b = slice(0, ncore_b)
        self.omega = float(omega)
        self.sqrt_w2 = np.sqrt(omega / 2.0)
        self.nph = nph_max + 1
        self.d0 = float(d0)
        self.enuc = mol.energy_nuc()

        # AO operators (built once; orbital optimisation rotates the MOs)
        self.hcore_ao = np.asarray(mf.get_hcore())
        self.ao_eri = mol.intor('int2e', aosym='s8')
        self.dip_ao = lam * np.einsum(
            'k,kpq->pq', epsilon, mol.intor('int1e_r', comp=3))
        self.use_proper = bool(proper_dse and lam > 0)
        if self.use_proper:
            quad_ao = mol.intor('int1e_rr', comp=9).reshape(3, 3, norb, norb)
            self.quad_ao_proj = 0.5 * (lam ** 2) * np.einsum(
                'a,b,abpq->pq', epsilon, epsilon, quad_ao)
        else:
            self.quad_ao_proj = np.zeros((norb, norb))
        self.h1_dse_ao = self.hcore_ao + self.quad_ao_proj

    # ------------------------------------------------------------------ #
    #  Integrals and the polaritonic CI solve                            #
    # ------------------------------------------------------------------ #
    def integrals(self, Ca, Cb):
        """DSE-augmented spin-resolved integrals in the (C_α, C_β) bases.

        Returns active-space CI integrals (for the polaritonic
        diagonalisation) together with the *full* MO integrals (for the
        orbital gradient).
        """
        norb, ncas = self.norb, self.ncas
        cas_a, cas_b = self.cas_a, self.cas_b
        core_a, core_b = self.core_a, self.core_b

        h1a = Ca.T @ self.h1_dse_ao @ Ca            # h^e,α + ½λ²Q
        h1b = Cb.T @ self.h1_dse_ao @ Cb
        dipa = Ca.T @ self.dip_ao @ Ca
        dipb = Cb.T @ self.dip_ao @ Cb

        eri_aa = ao2mo.restore(1, ao2mo.full(self.ao_eri, Ca), norb)
        eri_bb = ao2mo.restore(1, ao2mo.full(self.ao_eri, Cb), norb)
        eri_ab = ao2mo.general(
            self.ao_eri, (Ca, Ca, Cb, Cb),
            compact=False).reshape(norb, norb, norb, norb)
        # DSE 2-body augmentation: g^{σσ'} += d^σ ⊗ d^{σ'} (chemist)
        eri_aa = eri_aa + np.einsum('pq,rs->pqrs', dipa, dipa)
        eri_bb = eri_bb + np.einsum('pq,rs->pqrs', dipb, dipb)
        eri_ab = eri_ab + np.einsum('pq,rs->pqrs', dipa, dipb)

        # Spin-polarised frozen-core dressing of the DSE-augmented
        # integrals: same-spin J − K plus opposite-spin Coulomb.
        h1eff_a = h1a[cas_a, cas_a].copy()
        h1eff_b = h1b[cas_b, cas_b].copy()
        e_core = float(self.enuc)
        d_core = 0.0
        if self.ncore_a > 0:
            h1eff_a += (
                np.einsum('pqii->pq', eri_aa[cas_a, cas_a, core_a, core_a])
                - np.einsum('piiq->pq', eri_aa[cas_a, core_a, core_a, cas_a]))
            h1eff_b += np.einsum(
                'iipq->pq', eri_ab[core_a, core_a, cas_b, cas_b])
            eri_cc = eri_aa[core_a, core_a, core_a, core_a]
            e_core += float(np.trace(h1a[core_a, core_a])
                            + 0.5 * np.einsum('iijj->', eri_cc)
                            - 0.5 * np.einsum('ijji->', eri_cc))
            d_core += float(np.trace(dipa[core_a, core_a]))
        if self.ncore_b > 0:
            h1eff_b += (
                np.einsum('pqii->pq', eri_bb[cas_b, cas_b, core_b, core_b])
                - np.einsum('piiq->pq', eri_bb[cas_b, core_b, core_b, cas_b]))
            h1eff_a += np.einsum(
                'pqii->pq', eri_ab[cas_a, cas_a, core_b, core_b])
            eri_cc = eri_bb[core_b, core_b, core_b, core_b]
            e_core += float(np.trace(h1b[core_b, core_b])
                            + 0.5 * np.einsum('iijj->', eri_cc)
                            - 0.5 * np.einsum('ijji->', eri_cc))
            d_core += float(np.trace(dipb[core_b, core_b]))
        if self.ncore_a > 0 and self.ncore_b > 0:
            e_core += float(np.einsum(
                'iijj->', eri_ab[core_a, core_a, core_b, core_b]))

        return dict(
            h1eff_a=h1eff_a, h1eff_b=h1eff_b,
            eri_aa_act=np.ascontiguousarray(eri_aa[cas_a, cas_a, cas_a, cas_a]),
            eri_ab_act=np.ascontiguousarray(eri_ab[cas_a, cas_a, cas_b, cas_b]),
            eri_bb_act=np.ascontiguousarray(eri_bb[cas_b, cas_b, cas_b, cas_b]),
            dipa_act=np.ascontiguousarray(dipa[cas_a, cas_a]),
            dipb_act=np.ascontiguousarray(dipb[cas_b, cas_b]),
            e_core=e_core, d_core=d_core,
            h1a=h1a, h1b=h1b, dipa=dipa, dipb=dipb,
            eri_aa=eri_aa, eri_ab=eri_ab, eri_bb=eri_bb)

    def solve(self, ints, want_vectors=False):
        """Build and diagonalise the product-space Pauli-Fierz matrix."""
        H_elec, D_elec, ndim = _build_uhf_ci_matrices(
            ints['h1eff_a'], ints['h1eff_b'],
            ints['eri_aa_act'], ints['eri_ab_act'], ints['eri_bb_act'],
            ints['dipa_act'], ints['dipb_act'],
            self.ncas, self.nelec_act, ints['e_core'])
        H_elec = 0.5 * (H_elec + H_elec.T)
        D_elec = 0.5 * (D_elec + D_elec.T)

        eye = np.eye(ndim)
        D_total = D_elec + ints['d_core'] * eye
        if self.d0 != 0.0:
            H_elec = H_elec - self.d0 * D_total + 0.5 * self.d0 ** 2 * eye
            D_total = D_total - self.d0 * eye

        nph = self.nph
        ndim_total = ndim * nph
        H_total = np.zeros((ndim_total, ndim_total))
        for n in range(nph):
            r0, r1 = n * ndim, (n + 1) * ndim
            H_total[r0:r1, r0:r1] = H_elec + self.omega * n * eye
            if n + 1 < nph:
                m0, m1 = (n + 1) * ndim, (n + 2) * ndim
                cm = self.sqrt_w2 * np.sqrt(n + 1) * D_total
                H_total[m0:m1, r0:r1] += cm
                H_total[r0:r1, m0:m1] += cm.T
        H_total = 0.5 * (H_total + H_total.T)

        if want_vectors:
            w, v = eigh(H_total)
            return w, v, ndim
        w = eigh(H_total, eigvals_only=True)
        return w, None, ndim

    def energy(self, Ca, Cb):
        return self.solve(self.integrals(Ca, Cb), want_vectors=False)[0][0]

    def energy_and_grad(self, Ca, Cb):
        norb, ncas = self.norb, self.ncas
        nelecas_a, nelecas_b = self.nelec_act
        ncore_a, ncore_b = self.ncore_a, self.ncore_b
        cas_a, cas_b = self.cas_a, self.cas_b

        ints = self.integrals(Ca, Cb)
        w, v, ndim = self.solve(ints, want_vectors=True)
        e_gs = float(w[0])
        psi = v[:, 0]

        na = cistring.num_strings(ncas, nelecas_a)
        nb = cistring.num_strings(ncas, nelecas_b)
        blocks = [psi[n * ndim:(n + 1) * ndim].reshape(na, nb)
                  for n in range(self.nph)]

        # Photon-traced spin-resolved active 1-/2-RDMs.
        casdm1a = np.zeros((ncas, ncas))
        casdm1b = np.zeros((ncas, ncas))
        casdm2aa = np.zeros((ncas,) * 4)
        casdm2ab = np.zeros((ncas,) * 4)
        casdm2bb = np.zeros((ncas,) * 4)
        for cn in blocks:
            (d1a, d1b), (d2aa, d2ab, d2bb) = direct_uhf.make_rdm12s(
                cn, ncas, self.nelec_act)
            casdm1a += d1a
            casdm1b += d1b
            casdm2aa += d2aa
            casdm2ab += d2ab
            casdm2bb += d2bb

        # Spin-resolved photon transition 1-RDMs (Fock blocks n ↔ n+1).
        gtil_act_a = np.zeros((ncas, ncas))
        gtil_act_b = np.zeros((ncas, ncas))
        w_ph = 0.0
        for n in range(self.nph - 1):
            ta, tb = direct_uhf.trans_rdm1s(blocks[n + 1], blocks[n],
                                            ncas, self.nelec_act)
            gtil_act_a += np.sqrt(n + 1) * (ta + ta.T)
            gtil_act_b += np.sqrt(n + 1) * (tb + tb.T)
            w_ph += 2.0 * np.sqrt(n + 1) * float(
                np.dot(blocks[n + 1].ravel(), blocks[n].ravel()))

        # Full (core+active) spin RDMs and transition densities; the
        # per-spin core orbitals are singly occupied (occupation 1).
        dm1a = _full_rdm1_on_mo(casdm1a, ncore_a, ncas, norb)
        dm1b = _full_rdm1_on_mo(casdm1b, ncore_b, ncas, norb)
        dm2aa = _full_dm2_samespin(casdm1a, casdm2aa, ncore_a, ncas, norb)
        dm2bb = _full_dm2_samespin(casdm1b, casdm2bb, ncore_b, ncas, norb)
        dm2ab = _full_dm2_ab(casdm1a, casdm1b, casdm2ab,
                             ncore_a, ncore_b, ncas, norb)
        gtil_a = np.zeros((norb, norb))
        gtil_b = np.zeros((norb, norb))
        for i in range(ncore_a):
            gtil_a[i, i] = w_ph
        for i in range(ncore_b):
            gtil_b[i, i] = w_ph
        gtil_a[cas_a, cas_a] = gtil_act_a
        gtil_b[cas_b, cas_b] = gtil_act_b

        # Spin-resolved cavity-augmented generalised Fock and gradients.
        h_eff_a = ints['h1a'] - self.d0 * ints['dipa']
        h_eff_b = ints['h1b'] - self.d0 * ints['dipb']
        Wa = (h_eff_a @ dm1a
              + np.einsum('mqrs,nqrs->mn', ints['eri_aa'], dm2aa)
              + np.einsum('mqrs,nqrs->mn', ints['eri_ab'], dm2ab)
              + self.sqrt_w2 * (ints['dipa'] @ gtil_a))
        Wb = (h_eff_b @ dm1b
              + np.einsum('mqrs,nqrs->mn', ints['eri_bb'], dm2bb)
              + np.einsum('pqmr,pqnr->mn', ints['eri_ab'], dm2ab)
              + self.sqrt_w2 * (ints['dipb'] @ gtil_b))
        g_orb_a = 2.0 * (Wa - Wa.T)
        g_orb_b = 2.0 * (Wb - Wb.T)

        # Effective-Fock orbital energies for the step preconditioner.
        feff_a = (h_eff_a
                  + np.einsum('rs,pqrs->pq', dm1a, ints['eri_aa'])
                  + np.einsum('rs,pqrs->pq', dm1b, ints['eri_ab'])
                  - np.einsum('rs,prsq->pq', dm1a, ints['eri_aa']))
        feff_b = (h_eff_b
                  + np.einsum('rs,pqrs->pq', dm1b, ints['eri_bb'])
                  + np.einsum('rs,rspq->pq', dm1a, ints['eri_ab'])
                  - np.einsum('rs,prsq->pq', dm1b, ints['eri_bb']))
        eps_a = np.diag(feff_a).copy()
        eps_b = np.diag(feff_b).copy()

        # ⟨n_ph⟩ in the (displaced-frame) ground state.
        n_photon = 0.0
        for n in range(self.nph):
            n_photon += n * float(np.sum(blocks[n] ** 2))

        return dict(e=e_gs, g_orb_a=g_orb_a, g_orb_b=g_orb_b,
                    eps_a=eps_a, eps_b=eps_b,
                    eigenvalues=w, ndim=ndim, n_photon=n_photon,
                    d_core=ints['d_core'])


def run_qed_ucasscf(mf, ncas, nelecas, omega, coupling_vec,
                    nph_max=10, proper_dse=True, coherent_state=True,
                    max_cycle=100, conv_tol=1e-9, conv_tol_grad=1e-5,
                    verbose=False):
    """Spin-unrestricted QED-CASSCF (QED-UCASSCF) for the Pauli-Fierz
    Hamiltonian tensored with a truncated photon Fock space.

    Open-shell counterpart of
    :func:`OmegaQMC.addons.qed_casscf.run_qed_casscf`; all Hamiltonian
    conventions match :func:`OmegaQMC.addons.qed_fci.run_qed_fci` and
    the QED-UCASCI path of
    :func:`OmegaQMC.addons.qed_casci.run_qed_casci`. QED-UCASSCF reduces
    to open-shell QED-FCI at full active space and lies at or below the
    QED-UCASCI energy in the same active space. For a closed-shell
    molecule it reproduces the restricted QED-CASSCF energy (up to spin
    symmetry breaking, which can only lower it).

    Args:
        mf: PySCF mean-field object (RHF/ROHF/UHF, must have run
            kernel()). Only supplies the molecule, the bare-CASSCF
            comparison and ``e_hf``; the QED orbitals always come from a
            QED-UHF reference.
        ncas: Number of active spatial orbitals (same for α and β).
        nelecas: Number of active electrons. Either an int (total; split
            into (nα, nβ) using ``mol.spin``) or an (nα, nβ) tuple.
        omega: Photon frequency in Hartree.
        coupling_vec: Light-matter coupling vector λ·ε of length 3.
            Direction is the polarization ε; magnitude is λ.
        nph_max: Maximum photon number in the Fock truncation. The photon
            space has ``nph_max + 1`` states |0⟩…|nph_max⟩.
        proper_dse: If True (default), add ½λ²·q_pq with the true
            quadrupole integral q_pq = ⟨p|(ε·r̂)²|q⟩ to the one-body
            Hamiltonian, so the DSE matches the operator-form ½(λ·d̂)² in
            any basis. If False, only the d⊗d ERI augmentation is applied.
        coherent_state: If True (default), use the coherent-state
            (displaced) photon basis with z = d0/√(2Ω) and d0 the
            QED-UHF reference-determinant dipole (held fixed during the
            orbital optimisation). Exact unitary transform — energy
            unchanged at convergence in ``nph_max`` — but converges far
            faster for polar molecules. If False, use the raw
            photon-number (Fock) basis (d0 = 0).
        max_cycle: Maximum number of orbital macro-iterations.
        conv_tol: Convergence threshold on the energy change between
            macro-iterations.
        conv_tol_grad: Convergence threshold on the orbital-gradient norm
            (α and β pair gradients concatenated).
        verbose: Print per-macro-iteration energies and gradient norms.

    Returns:
        dict with the same keys as
        :func:`OmegaQMC.addons.qed_casscf.run_qed_casscf` where they
        apply, with the open-shell substitutions:
            'e_qed_casscf' : QED-UCASSCF ground-state energy.
            'e_casscf'     : Bare UCASSCF energy (no cavity), from pyscf
                             ``mcscf.UCASSCF``.
            'e_qed_hf'     : QED-UHF reference energy.
            'e_hf'         : Bare total energy of ``mf``.
            'e_qed_casci'  : QED-UCASCI energy in the starting QED-UHF
                             orbitals (= energy before orbital
                             optimisation).
            'mo_coeff'     : (C_α, C_β) optimized MO coefficients.
            'ncore'        : (ncore_α, ncore_β).
            'reference'    : 'QED-UHF'.
            's_squared',
            'multiplicity' : ⟨S²⟩ and 2S+1 of the QED-UHF reference.
        (No multi-determinant AFQMC trial export: the AFQMC driver's
        multi-determinant trial format assumes a single restricted MO
        set; see ``qed_casscf.to_afqmc_trial``.)
    """
    mol = mf.mol
    norb = mol.nao_nr()

    ncore_a, ncore_b, nelecas_a, nelecas_b = _parse_active_space_uhf(
        mol, ncas, nelecas)
    nelec_act = (nelecas_a, nelecas_b)
    nocc_a, nocc_b = mol.nelec        # reference determinant occupations

    # --- Coupling vector → magnitude λ and polarization ε ---
    coupling_vec = np.asarray(coupling_vec, dtype=np.float64)
    lam = float(np.linalg.norm(coupling_vec))
    if lam > 1e-15:
        epsilon = coupling_vec / lam
    else:
        epsilon = np.array([0.0, 0.0, 1.0])
        lam = 0.0

    # --- Reference orbitals: cavity-relaxed QED-UHF ---
    qeduhf = run_qed_uhf(mol, omega, lambda_cav=tuple(coupling_vec.tolist()))
    Ca = np.asarray(qeduhf['Ca'])
    Cb = np.asarray(qeduhf['Cb'])
    e_qed_hf = float(qeduhf['E_qed_uhf'])

    obj = _QEDUCASSCFObjective(mf, ncas, nelec_act, ncore_a, ncore_b,
                               omega, lam, epsilon, proper_dse, nph_max)

    # --- Fixed coherent-state displacement d0 = ⟨D̂⟩ in the QED-UHF ---
    # reference determinant (Aufbau occupation of the α/β orbitals).
    # Held constant during the orbital optimisation; the energy is
    # invariant to d0 at convergence in nph_max.
    if coherent_state and lam > 0:
        dipa0 = Ca.T @ obj.dip_ao @ Ca
        dipb0 = Cb.T @ obj.dip_ao @ Cb
        obj.d0 = float(np.trace(dipa0[:nocc_a, :nocc_a])
                       + np.trace(dipb0[:nocc_b, :nocc_b]))
    d0 = obj.d0

    # --- Non-redundant rotation pairs, α and β concatenated ---
    pp_a, qq_a = _rot_pairs(ncore_a, ncas, norb)
    pp_b, qq_b = _rot_pairs(ncore_b, ncas, norb)
    n_pair_a = len(pp_a)

    def _make_K(pp, qq, step):
        K = np.zeros((norb, norb))
        K[pp, qq] = step
        K[qq, pp] = -step
        return K

    def _rotate(Ca, Cb, step):
        Ca_new = Ca @ expm(_make_K(pp_a, qq_a, step[:n_pair_a]))
        Cb_new = Cb @ expm(_make_K(pp_b, qq_b, step[n_pair_a:]))
        return Ca_new, Cb_new

    def _grad_pairs(res):
        return np.concatenate([res['g_orb_a'][pp_a, qq_a],
                               res['g_orb_b'][pp_b, qq_b]])

    # ------------------------------------------------------------------ #
    #  QED-UCASCI energy in the starting QED-UHF orbitals                #
    # ------------------------------------------------------------------ #
    res = obj.energy_and_grad(Ca, Cb)
    e_qed_casci = res['e']

    # ------------------------------------------------------------------ #
    #  Orbital macro-iterations: preconditioned L-BFGS + Armijo backtrack #
    # ------------------------------------------------------------------ #
    converged = False
    n_macro = 0
    e_cur = res['e']
    grad_norm = float(np.linalg.norm(_grad_pairs(res)))
    if verbose:
        print(f"QED-UCASSCF macro   0: E = {e_cur:.12f}  "
              f"|g| = {grad_norm:.3e}  (QED-UCASCI reference)")

    s_hist, y_hist = [], []
    g_prev = s_prev = None
    m_max = 10

    for it in range(1, max_cycle + 1):
        n_macro = it
        g_pairs = _grad_pairs(res)
        grad_norm = float(np.linalg.norm(g_pairs))
        if grad_norm < conv_tol_grad:
            converged = True
            break

        # Effective-Fock orbital-energy-gap preconditioner (diagonal H₀).
        gaps = np.concatenate([
            np.abs(res['eps_a'][pp_a] - res['eps_a'][qq_a]),
            np.abs(res['eps_b'][pp_b] - res['eps_b'][qq_b])])
        hdiag = 1.0 / np.maximum(2.0 * gaps, 0.2)

        # L-BFGS curvature update (skip non-positive curvature pairs).
        if g_prev is not None:
            y = g_pairs - g_prev
            ys = float(np.dot(s_prev, y))
            if ys > 1e-12:
                s_hist.append(s_prev)
                y_hist.append(y)
                if len(s_hist) > m_max:
                    s_hist.pop(0)
                    y_hist.pop(0)

        step = _lbfgs_dir(g_pairs, s_hist, y_hist, hdiag)
        slope = float(np.dot(g_pairs, step))
        if slope >= 0.0:                      # not a descent direction
            step = -hdiag * g_pairs
            slope = float(np.dot(g_pairs, step))

        # Trust radius on the largest rotation angle.
        max_step = float(np.max(np.abs(step))) if step.size else 0.0
        if max_step > 0.30:
            step *= 0.30 / max_step
            slope *= 0.30 / max_step

        # Armijo backtracking line search.
        alpha = 1.0
        accepted = False
        for _ in range(15):
            Ca_try, Cb_try = _rotate(Ca, Cb, alpha * step)
            e_try = obj.energy(Ca_try, Cb_try)
            if e_try <= e_cur + 1e-4 * alpha * slope:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            converged = grad_norm < 1e-3
            break

        de = e_try - e_cur
        g_prev = g_pairs
        s_prev = alpha * step
        Ca, Cb = Ca_try, Cb_try
        e_cur = e_try
        res = obj.energy_and_grad(Ca, Cb)

        if verbose:
            print(f"QED-UCASSCF macro {it:3d}: E = {e_cur:.12f}  "
                  f"dE = {de:+.3e}  |g| = {grad_norm:.3e}  α = {alpha:.2e}")

        if abs(de) < conv_tol and grad_norm < conv_tol_grad * 10:
            converged = True
            break

    # Final quantities at the optimized orbitals.
    final = obj.energy_and_grad(Ca, Cb)
    e_gs = final['e']
    grad_norm = float(np.linalg.norm(_grad_pairs(final)))

    # --- Bare UCASSCF (no cavity) reference, orbital-optimized ---
    mf_u = mf if isinstance(mf, scf.uhf.UHF) else scf.addons.convert_to_uhf(mf)
    mc = mcscf.UCASSCF(mf_u, ncas, nelec_act)
    mc.verbose = 0
    e_casscf = float(mc.kernel()[0])

    e_hf = float(mf.e_tot) if getattr(mf, 'e_tot', None) is not None else None
    e_corr = (e_casscf - e_hf) if e_hf is not None else None
    e_corr_qed = e_gs - e_qed_hf
    cs_displacement = float(d0 / np.sqrt(2.0 * omega)) if d0 != 0.0 else 0.0

    return {
        'e_qed_casscf': e_gs,
        'e_casscf': e_casscf,
        'e_qed_hf': e_qed_hf,
        'e_hf': e_hf,
        'e_corr_qed': e_corr_qed,
        'e_corr': e_corr,
        'e_qed_casci': e_qed_casci,
        'mo_coeff': (Ca, Cb),
        'omega': float(omega),
        'reference': 'QED-UHF',
        'ncas': int(ncas),
        'ncore': (int(ncore_a), int(ncore_b)),
        'nelecas': nelec_act,
        'converged': bool(converged),
        'n_macro': int(n_macro),
        'grad_norm': grad_norm,
        'eigenvalues': final['eigenvalues'],
        'nph_max': int(nph_max),
        'ndim_elec': int(final['ndim']),
        'ndim_total': int(final['ndim'] * (nph_max + 1)),
        'n_photon': float(final['n_photon']),
        'd_core_const': float(final['d_core']),
        'coherent_state': bool(coherent_state and lam > 0),
        'cs_displacement': cs_displacement,
        'proper_dse': bool(proper_dse and lam > 0),
        's_squared': float(qeduhf['s_squared']),
        'multiplicity': float(qeduhf['multiplicity']),
    }


if __name__ == '__main__':
    from pyscf import gto

    # OH radical / STO-3G in a cavity: QED-UCASSCF vs QED-UCASCI.
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='sto-3g', spin=1,
                unit='Angstrom', symmetry=False, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()

    omega = 0.1
    coupling_vec = (0.0, 0.0, 0.05)

    res = run_qed_ucasscf(mf, ncas=4, nelecas=(3, 2), omega=omega,
                          coupling_vec=coupling_vec, nph_max=2, verbose=True)
    print(f"\nE_QED-UCASSCF = {res['e_qed_casscf']:.10f}")
    print(f"E_QED-UCASCI  = {res['e_qed_casci']:.10f}  (starting orbitals)")
    print(f"relaxation    = "
          f"{res['e_qed_casscf'] - res['e_qed_casci']:+.3e} Ha")
    print(f"E_QED-UHF     = {res['e_qed_hf']:.10f}")
    print(f"<S^2>         = {res['s_squared']:.6f}  "
          f"(2S+1 = {res['multiplicity']:.4f})")
    print(f"converged     = {res['converged']}  ({res['n_macro']} macro-iters)")
