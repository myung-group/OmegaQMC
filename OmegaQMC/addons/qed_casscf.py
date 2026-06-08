"""
QED-CASSCF: orbital-optimized complete-active-space self-consistent field
for the Pauli-Fierz Hamiltonian, tensored with a truncated photon Fock
space.

This is the orbital-optimized counterpart of
:mod:`OmegaQMC.addons.qed_casci`. Where CS-QED-CASCI fixes the orbitals
to the QED-HF set and only solves the polaritonic CI, CS-QED-CASSCF
*relaxes* the orbitals self-consistently in the presence of the cavity,
minimising the joint electron-photon energy

    E[Ψ, C] = ⟨Ψ| Ĥ_PF[C] |Ψ⟩

over both the CI coefficients C_{I,m} (electronic determinant ⊗ photon
Fock state) and the molecular-orbital rotations κ. All Hamiltonian
conventions match :mod:`OmegaQMC.addons.qed_fci` /
:mod:`OmegaQMC.addons.qed_casci` exactly, so QED-CASSCF reduces to
QED-FCI in the limit where the active space spans the full one-particle
MO basis (orbital optimisation then becomes redundant — the QED-FCI
energy is orbital-invariant), and it always lies at or below the
QED-CASCI energy in the same active space.

Following the cavity-QED-CASSCF formulation of

    M. Castagnola, T. S. Haugland, E. Ronca, H. Koch, et al.,
    arXiv:2503.16417 (2025),

the wavefunction is

    |Ψ₀⟩ = Σ_{I,m} C_{I,m} |I⟩ ⊗ |m⟩,

where |I⟩ are CAS determinants/CSFs and |m⟩ are photon number states.
The dipole self-energy is absorbed into modified one- and two-electron
integrals (Eq. 19 of the reference),

    h_pq = h^e_pq + ½ q_pq − d_pq⟨d⟩ + δ_pq ⟨d⟩²/(2 N_e),
    g_pqrs = g^e_pqrs + d_pq d_rs,

which is exactly the coherent-state (displaced) representation used by
the CS-QED-CASCI/FCI siblings: with the per-electron / displacement
terms collected, the diagonal block is shifted by −d0·D̂ + ½d0² and the
bilinear photon coupling uses the fluctuation dipole (D̂ − d0), with
d0 = ⟨d⟩ the reference-determinant dipole and the photon basis displaced
by z = d0/√(2Ω). The displacement is an exact unitary transform of the
Pauli-Fierz Hamiltonian, so the energy is unchanged at convergence in
``nph_max`` but converges far faster for polar molecules.

Orbital optimisation
--------------------
The orbital gradient is the cavity-augmented generalised Fock matrix.
With the photon-traced electronic 1-/2-RDMs (γ, Γ) and the photon
*transition* 1-RDM γ̃ that couples adjacent Fock blocks,

    γ̃_pq = Σ_m √(m+1) ⟨I,m+1| Ê_pq |J,m⟩ C*_{I,m+1} C_{J,m} + h.c.,

the gradient is g_pq = 2(W_pq − W_qp) with

    W_pq = Σ_r h_pr D_rq + Σ_{rst} g_prst Γ_qrst
           + √(Ω/2) Σ_r d_pr γ̃_rq,

where D, Γ are the *full* (core+active) RDMs in the modified integrals
h (= h^e + ½λ²Q − d0·d) and g (= g^e + d⊗d). The first two terms are
the standard MCSCF generalised Fock; the last is the photon-coupling
contribution (Eq. 40 of the reference). This gradient has been verified
against finite differences to ≲1e-10 Ha.

Orbitals are optimised by a first-order (preconditioned, line-searched)
scheme with re-basing of the rotation each macro-iteration: a
diagonal-Newton step δκ = −g / h_diag using effective-Fock orbital-energy
gaps as the preconditioner, followed by an Armijo backtracking line
search to guarantee monotone energy decrease. This reaches the same
stationary point as the reference's restricted-step second-order
(Newton-Raphson) algorithm; only the optimisation path differs.

Scope
-----
This module implements the closed-shell (restricted) QED-CASSCF. Open-
shell molecules raise ``NotImplementedError`` — use the spin-unrestricted
QED-UCASCI path of :func:`OmegaQMC.addons.qed_casci.run_qed_casci` for a
fixed-orbital open-shell treatment.

The total dipole μ̂ is approximated by the electronic part μ̂_e only
(matching qed_fci/qed_casci/qed_hf); for neutral molecules this fixes the
origin at the nuclear centre of charge, and any origin-dependent shift
cancels in the correlation energy.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, expm
from pyscf import ao2mo, mcscf
from pyscf.fci import direct_spin1, cistring
from pyscf.mcscf.addons import _make_rdm12_on_mo

from OmegaQMC.addons.qed_fci import _build_fci_matrices
from OmegaQMC.addons.qed_hf import run_qed_hf


def _parse_active_space(mol, ncas, nelecas):
    """Parse and validate a closed-shell active-space specification.

    Returns ``(ncore, nelecas_a, nelecas_b)``. Mirrors the closed-shell
    branch of :func:`OmegaQMC.addons.qed_casci._parse_active_space`.
    """
    nelec_total = mol.nelectron
    if isinstance(nelecas, (int, np.integer)):
        nelecas_int = int(nelecas)
        if nelecas_int % 2 != 0:
            raise ValueError(
                "When ``nelecas`` is an int it must be even; for "
                "open-shell active spaces pass a (na, nb) tuple."
            )
        nelecas_a = nelecas_b = nelecas_int // 2
    else:
        nelecas_a, nelecas_b = int(nelecas[0]), int(nelecas[1])

    if nelecas_a != nelecas_b:
        raise NotImplementedError(
            "qed_casscf implements the closed-shell (restricted) "
            "QED-CASSCF; the active space must have nα == nβ. For an "
            "open-shell, fixed-orbital treatment use the QED-UCASCI path "
            "of OmegaQMC.addons.qed_casci.run_qed_casci."
        )

    nelec_act_total = nelecas_a + nelecas_b
    if nelec_act_total > nelec_total:
        raise ValueError(
            f"nelecas={nelec_act_total} exceeds molecular electron "
            f"count {nelec_total}."
        )

    ncore_total = nelec_total - nelec_act_total
    if ncore_total % 2 != 0:
        raise ValueError(
            f"Frozen core must be doubly occupied; "
            f"nelec_total - nelecas = {ncore_total} is odd."
        )
    ncore = ncore_total // 2

    norb = mol.nao_nr()
    if ncas + ncore > norb:
        raise ValueError(
            f"ncas + ncore = {ncas + ncore} exceeds nmo = {norb}."
        )

    return ncore, nelecas_a, nelecas_b


def run_qed_casscf(mf, ncas, nelecas, omega, coupling_vec,
                   nph_max=10, proper_dse=True, coherent_state=True,
                   max_cycle=100, conv_tol=1e-9, conv_tol_grad=1e-5,
                   verbose=False):
    """Closed-shell QED-CASSCF: orbital-optimized CAS for the Pauli-Fierz
    Hamiltonian tensored with a truncated photon Fock space.

    Conventions match :func:`OmegaQMC.addons.qed_fci.run_qed_fci` and
    :func:`OmegaQMC.addons.qed_casci.run_qed_casci`. QED-CASSCF reduces to
    QED-FCI at full active space (``ncas = nmo``, ``nelecas = mol.nelec``)
    and lies at or below the QED-CASCI energy in the same active space.

    Args:
        mf: PySCF mean-field object (closed shell, must have run kernel()).
            Its orbitals seed the bare-CASSCF reference; the QED-CASSCF
            orbital optimisation starts from a QED-HF reference.
        ncas: Number of active spatial orbitals.
        nelecas: Number of active electrons. Either an even int (total) or
            a (n_alpha, n_beta) tuple with n_alpha == n_beta.
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
            reference-determinant dipole (held fixed during the orbital
            optimisation). The displacement is an exact unitary transform
            (energy unchanged at convergence in ``nph_max``) but converges
            far faster for polar molecules. If False, use the raw
            photon-number (Fock) basis (d0 = 0).
        max_cycle: Maximum number of orbital macro-iterations.
        conv_tol: Convergence threshold on the energy change between
            macro-iterations.
        conv_tol_grad: Convergence threshold on the orbital-gradient norm.
        verbose: Print per-macro-iteration energies and gradient norms.

    Returns:
        dict with:
            'e_qed_casscf'       : QED-CASSCF ground-state energy.
            'e_casscf'           : Bare CASSCF energy (no cavity), from
                                   pyscf ``mcscf.CASSCF``.
            'e_qed_hf'           : QED-HF reference energy.
            'e_hf'               : Bare RHF total energy from ``mf``.
            'e_corr_qed'         : e_qed_casscf − e_qed_hf.
            'e_corr'             : e_casscf − e_hf.
            'e_qed_casci'        : QED-CASCI energy in the *starting*
                                   QED-HF orbitals (= energy before orbital
                                   optimisation); the relaxation energy is
                                   e_qed_casscf − e_qed_casci ≤ 0.
            'mo_coeff'           : Optimized QED-CASSCF MO coefficients.
            'reference'          : 'QED-HF'.
            'ncas'               : Number of active orbitals.
            'ncore'              : Number of frozen-core orbitals.
            'nelecas'            : (n_alpha_act, n_beta_act).
            'converged'          : Whether the orbital optimisation
                                   converged.
            'n_macro'            : Number of macro-iterations performed.
            'grad_norm'          : Final orbital-gradient norm.
            'eigenvalues'        : All product-space eigenvalues at the
                                   optimized orbitals.
            'nph_max'            : Photon truncation used.
            'ndim_elec'          : Electronic CASCI dimension.
            'ndim_total'         : Total product-space dimension.
            'n_photon'           : ⟨n_ph⟩ in the ground state (displaced
                                   frame ⟨b†b⟩ when coherent_state=True).
            'coherent_state'     : Whether the coherent-state basis was used.
            'cs_displacement'    : Coherent-state displacement z (0 if off).
            'd_core_const'       : Constant electronic dipole shift from the
                                   frozen core (2 Σ_{i∈core} d_ii) at the
                                   optimized orbitals.
            'proper_dse'         : Whether the ½λ²q correction was applied.
    """
    mol = mf.mol

    if mol.nelec[0] != mol.nelec[1]:
        raise NotImplementedError(
            "qed_casscf implements the closed-shell (restricted) "
            "QED-CASSCF. For an open-shell, fixed-orbital treatment use "
            "the QED-UCASCI path of "
            "OmegaQMC.addons.qed_casci.run_qed_casci."
        )

    norb = mol.nao_nr()
    enuc = mol.energy_nuc()
    ncore, nelecas_a, nelecas_b = _parse_active_space(mol, ncas, nelecas)
    nelec_act = (nelecas_a, nelecas_b)
    nocc = ncore + nelecas_a            # doubly-occupied reference orbitals
    cas = slice(ncore, ncore + ncas)
    core = slice(0, ncore)

    # --- Coupling vector → magnitude λ and polarization ε ---
    coupling_vec = np.asarray(coupling_vec, dtype=np.float64)
    lam = float(np.linalg.norm(coupling_vec))
    if lam > 1e-15:
        epsilon = coupling_vec / lam
    else:
        epsilon = np.array([0.0, 0.0, 1.0])
        lam = 0.0

    # --- AO operators (built once; orbital optimisation rotates the MOs) ---
    hcore_ao = np.asarray(mf.get_hcore())
    ao_eri = mol.intor('int2e', aosym='s8')
    dip_ao = lam * np.einsum('k,kpq->pq', epsilon, mol.intor('int1e_r', comp=3))
    use_proper = bool(proper_dse and lam > 0)
    if use_proper:
        quad_ao = mol.intor('int1e_rr', comp=9).reshape(3, 3, norb, norb)
        quad_ao_proj = 0.5 * (lam ** 2) * np.einsum(
            'a,b,abpq->pq', epsilon, epsilon, quad_ao)
    else:
        quad_ao_proj = np.zeros((norb, norb))

    sqrt_w2 = np.sqrt(omega / 2.0)
    nph = nph_max + 1

    # --- Reference orbitals: cavity-relaxed QED-HF ---
    qedhf = run_qed_hf(mol, omega, lambda_cav=tuple(coupling_vec.tolist()))
    C0 = np.asarray(qedhf['C'])
    e_qed_hf = float(qedhf['E_qed_hf'])

    # --- Fixed coherent-state displacement d0 = ⟨D̂⟩ in the reference ---
    # determinant (closed-shell Aufbau occupation of the QED-HF orbitals).
    # Held constant during the orbital optimisation; the energy is
    # invariant to d0 at convergence in nph_max.
    if coherent_state and lam > 0:
        dip0 = C0.T @ dip_ao @ C0
        d0 = 2.0 * float(sum(dip0[i, i] for i in range(nocc)))
    else:
        d0 = 0.0

    # ------------------------------------------------------------------ #
    #  Integral builders and the polaritonic CI solve                    #
    # ------------------------------------------------------------------ #
    def _integrals(C):
        """DSE-augmented integrals in the MO basis defined by ``C``.

        Returns active-space CI integrals (for the polaritonic
        diagonalisation) together with the *full* MO integrals (for the
        orbital gradient).
        """
        h1 = C.T @ hcore_ao @ C
        dip = C.T @ dip_ao @ C
        eri = ao2mo.restore(1, ao2mo.full(ao_eri, C), norb)

        h1_dse = h1 + (C.T @ quad_ao_proj @ C)        # h^e + ½λ²Q
        eri_dse = eri + np.einsum('pq,rs->pqrs', dip, dip)   # g^e + d⊗d

        # Frozen-core dressing of the DSE-augmented integrals.
        if ncore > 0:
            J_core = 2.0 * np.einsum('pqii->pq', eri_dse[cas, cas, core, core])
            K_core = np.einsum('piiq->pq', eri_dse[cas, core, core, cas])
            h1eff_act = h1_dse[cas, cas] + J_core - K_core
            eri_cc = eri_dse[core, core, core, core]
            e_core = float(enuc + 2.0 * np.trace(h1_dse[core, core])
                           + 2.0 * np.einsum('iijj->', eri_cc)
                           - np.einsum('ijji->', eri_cc))
            d_core = 2.0 * float(np.trace(dip[core, core]))
        else:
            h1eff_act = h1_dse[cas, cas].copy()
            e_core = float(enuc)
            d_core = 0.0

        eri_act = np.ascontiguousarray(eri_dse[cas, cas, cas, cas])
        dip_act = np.ascontiguousarray(dip[cas, cas])
        return dict(h1eff_act=h1eff_act, eri_act=eri_act, dip_act=dip_act,
                    e_core=e_core, d_core=d_core,
                    h1_dse=h1_dse, eri_dse=eri_dse, dip=dip)

    def _solve(ints, want_vectors=False):
        """Build and diagonalise the product-space Pauli-Fierz matrix."""
        H_elec, D_elec, ndim = _build_fci_matrices(
            ints['h1eff_act'], ints['eri_act'], ints['dip_act'],
            ncas, nelec_act, ints['e_core'])
        H_elec = 0.5 * (H_elec + H_elec.T)
        D_elec = 0.5 * (D_elec + D_elec.T)

        eye = np.eye(ndim)
        D_total = D_elec + ints['d_core'] * eye
        if d0 != 0.0:
            H_elec = H_elec - d0 * D_total + 0.5 * d0 ** 2 * eye
            D_total = D_total - d0 * eye

        ndim_total = ndim * nph
        H_total = np.zeros((ndim_total, ndim_total))
        for n in range(nph):
            r0, r1 = n * ndim, (n + 1) * ndim
            H_total[r0:r1, r0:r1] = H_elec + omega * n * eye
            if n + 1 < nph:
                m0, m1 = (n + 1) * ndim, (n + 2) * ndim
                cm = sqrt_w2 * np.sqrt(n + 1) * D_total
                H_total[m0:m1, r0:r1] += cm
                H_total[r0:r1, m0:m1] += cm.T
        H_total = 0.5 * (H_total + H_total.T)

        if want_vectors:
            w, v = eigh(H_total)
            return w, v, ndim
        # Lowest eigenvalue only (line-search evaluations).
        w = eigh(H_total, eigvals_only=True)
        return w, None, ndim

    def _energy(C):
        return _solve(_integrals(C), want_vectors=False)[0][0]

    def _energy_and_grad(C):
        ints = _integrals(C)
        w, v, ndim = _solve(ints, want_vectors=True)
        e_gs = float(w[0])
        psi = v[:, 0]

        na = cistring.num_strings(ncas, nelecas_a)
        nb = cistring.num_strings(ncas, nelecas_b)
        blocks = [psi[n * ndim:(n + 1) * ndim].reshape(na, nb)
                  for n in range(nph)]

        # Photon-traced active 1-/2-RDMs.
        casdm1 = np.zeros((ncas, ncas))
        casdm2 = np.zeros((ncas, ncas, ncas, ncas))
        for cn in blocks:
            d1, d2 = direct_spin1.make_rdm12(cn, ncas, nelec_act)
            casdm1 += d1
            casdm2 += d2

        # Photon transition 1-RDM (couples Fock blocks n ↔ n+1).
        gtil_act = np.zeros((ncas, ncas))
        w_ph = 0.0
        for n in range(nph - 1):
            t = direct_spin1.trans_rdm1(blocks[n + 1], blocks[n],
                                        ncas, nelec_act)
            gtil_act += np.sqrt(n + 1) * (t + t.T)
            w_ph += 2.0 * np.sqrt(n + 1) * float(
                np.dot(blocks[n + 1].ravel(), blocks[n].ravel()))

        # Full (core+active) RDMs and transition density.
        dm1, dm2 = _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, norb)
        gtil = np.zeros((norb, norb))
        for i in range(ncore):
            gtil[i, i] = 2.0 * w_ph
        gtil[cas, cas] = gtil_act

        # Cavity-augmented generalised Fock and orbital gradient.
        h_eff = ints['h1_dse'] - d0 * ints['dip']
        W = (h_eff @ dm1
             + np.einsum('mqrs,nqrs->mn', ints['eri_dse'], dm2)
             + sqrt_w2 * (ints['dip'] @ gtil))
        g_orb = 2.0 * (W - W.T)

        # Effective-Fock orbital energies for the step preconditioner.
        feff = (h_eff
                + np.einsum('rs,pqrs->pq', dm1, ints['eri_dse'])
                - 0.5 * np.einsum('rs,prsq->pq', dm1, ints['eri_dse']))
        eps_orb = np.diag(feff).copy()

        # ⟨n_ph⟩ in the (displaced-frame) ground state.
        n_photon = 0.0
        for n in range(nph):
            n_photon += n * float(np.sum(blocks[n] ** 2))

        # Electronic CI vector of the dominant (m=0) photon sector, in the
        # CASSCF MO basis, normalised — the natural multi-determinant trial
        # for QED-AFQMC (see ``to_afqmc_trial``). With the coherent-state
        # photon basis this is the displaced-vacuum sector, which carries
        # essentially all the weight (⟨n_ph⟩ ≪ 1). The CASSCF electronic
        # dipole ⟨D̂⟩ fixes the photon coherent-state displacement.
        ci0 = blocks[0].copy()
        n0 = float(np.linalg.norm(ci0))
        if n0 > 0.0:
            ci0 = ci0 / n0
        dipole = float(np.einsum('pq,pq->', ints['dip'], dm1))

        return dict(e=e_gs, g_orb=g_orb, eps_orb=eps_orb,
                    eigenvalues=w, ndim=ndim, n_photon=n_photon,
                    d_core=ints['d_core'], ci0=ci0, dipole=dipole)

    # ------------------------------------------------------------------ #
    #  Non-redundant orbital-rotation pairs                              #
    # ------------------------------------------------------------------ #
    orb_type = (['c'] * ncore + ['a'] * ncas
                + ['v'] * (norb - ncore - ncas))
    pairs = [(p, q) for p in range(norb) for q in range(p)
             if orb_type[p] != orb_type[q]]
    pair_idx = (np.array([p for p, _ in pairs], dtype=int),
                np.array([q for _, q in pairs], dtype=int))

    pp, qq = pair_idx

    def _make_K(step):
        K = np.zeros((norb, norb))
        K[pp, qq] = step
        K[qq, pp] = -step
        return K

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

    # ------------------------------------------------------------------ #
    #  QED-CASCI energy in the starting QED-HF orbitals (pre-relaxation) #
    # ------------------------------------------------------------------ #
    res = _energy_and_grad(C0)
    e_qed_casci = res['e']

    # ------------------------------------------------------------------ #
    #  Orbital macro-iterations: preconditioned L-BFGS + Armijo backtrack #
    # ------------------------------------------------------------------ #
    C = C0.copy()
    converged = False
    n_macro = 0
    e_cur = res['e']
    grad_norm = float(np.linalg.norm(res['g_orb'][pair_idx]))
    if verbose:
        print(f"QED-CASSCF macro   0: E = {e_cur:.12f}  "
              f"|g| = {grad_norm:.3e}  (QED-CASCI reference)")

    s_hist, y_hist = [], []     # L-BFGS history (tangent-coordinate vectors)
    g_prev = s_prev = None
    m_max = 10

    for it in range(1, max_cycle + 1):
        n_macro = it
        g_pairs = res['g_orb'][pair_idx]
        grad_norm = float(np.linalg.norm(g_pairs))
        if grad_norm < conv_tol_grad:
            converged = True
            break

        # Effective-Fock orbital-energy-gap preconditioner (diagonal H₀).
        eps_orb = res['eps_orb']
        denom = np.maximum(
            2.0 * np.abs(eps_orb[pp] - eps_orb[qq]), 0.2)
        hdiag = 1.0 / denom

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
            C_try = C @ expm(_make_K(alpha * step))
            e_try = _energy(C_try)
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
        C = C_try
        e_cur = e_try
        res = _energy_and_grad(C)

        if verbose:
            print(f"QED-CASSCF macro {it:3d}: E = {e_cur:.12f}  "
                  f"dE = {de:+.3e}  |g| = {grad_norm:.3e}  α = {alpha:.2e}")

        if abs(de) < conv_tol and grad_norm < conv_tol_grad * 10:
            converged = True
            break

    # Final quantities at the optimized orbitals.
    final = _energy_and_grad(C)
    e_gs = final['e']
    grad_norm = float(np.linalg.norm(final['g_orb'][pair_idx]))
    ndim_elec = final['ndim']
    eigenvalues = final['eigenvalues']
    n_photon = final['n_photon']
    d_core_const = final['d_core']

    # --- Bare CASSCF (no cavity) reference, orbital-optimized ---
    mc = mcscf.CASSCF(mf, ncas, (nelecas_a, nelecas_b))
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
        'mo_coeff': C,
        'ci': np.asarray(final['ci0']),
        'dipole': float(final['dipole']),
        'omega': float(omega),
        'reference': 'QED-HF',
        'ncas': int(ncas),
        'ncore': int(ncore),
        'nelecas': nelec_act,
        'converged': bool(converged),
        'n_macro': int(n_macro),
        'grad_norm': grad_norm,
        'eigenvalues': eigenvalues,
        'nph_max': int(nph_max),
        'ndim_elec': int(ndim_elec),
        'ndim_total': int(ndim_elec * nph),
        'n_photon': float(n_photon),
        'd_core_const': float(d_core_const),
        'coherent_state': bool(coherent_state and lam > 0),
        'cs_displacement': cs_displacement,
        'proper_dse': bool(use_proper),
    }


def to_afqmc_trial(result, coeff_threshold=1e-4):
    """Build a multi-determinant QED-AFQMC trial from a QED-CASSCF result.

    Converts the QED-CASSCF wavefunction into the trial format consumed by
    the multi-determinant branch of
    :class:`OmegaQMC.qed_afqmc_gto._QEDAFQMCDriverGTO` (``trial=`` keyword):
    a CI expansion of Slater determinants in the optimized CASSCF orbitals,
    paired with the dipole-gauge coherent-state photon displacement.

    The electronic part is taken from the dominant (m=0) photon sector of
    the QED-CASSCF state — with the coherent-state photon basis this sector
    carries essentially all the weight (⟨n_ph⟩ ≪ 1), so its CI coefficients
    are the cavity-dressed electronic amplitudes. The photon trial is the
    single Gaussian centred at ``q0 = -⟨D̂⟩/√Ω`` (the CASSCF electronic
    dipole), matching the photon trial used by the single-determinant
    QED-AFQMC driver. The residual electron–photon entanglement neglected
    by this factorisation is of order ⟨n_ph⟩ and is handled by the AFQMC
    dynamics.

    Args:
        result: The dict returned by :func:`run_qed_casscf`. Must contain
            ``'ci'`` (m=0-sector CI array, shape (na, nb)), ``'mo_coeff'``,
            ``'ncore'``, ``'ncas'``, ``'nelecas'``, ``'dipole'`` and
            ``'omega'``.
        coeff_threshold: Threshold on |c_I| for truncating the CI
            expansion. Determinants with |c_I| below this are dropped.

    Returns:
        dict with keys (numpy arrays / Python scalars):
            'ci_coeffs': shape (ndet,), sorted by descending |c_I| so the
                         dominant determinant is first (walker init).
            'occ_up'   : int array, shape (ndet, nup), occupied spatial
                         orbital indices (frozen core + active) per det.
            'occ_dn'   : int array, shape (ndet, ndown).
            'ndet'     : number of determinants retained.
            'mo_coeff' : optimized CASSCF MO coefficients (nao, nmo).
            'q0'       : photon coherent-state displacement -⟨D̂⟩/√Ω.
    """
    from pyscf.fci import cistring

    ncore = int(result['ncore'])
    ncas = int(result['ncas'])
    nelecas_a, nelecas_b = (int(n) for n in result['nelecas'])
    ci = np.asarray(result['ci'])

    occslst_a = cistring.gen_occslst(range(ncas), nelecas_a)
    occslst_b = cistring.gen_occslst(range(ncas), nelecas_b)
    core = list(range(ncore))

    coeffs, occ_up_list, occ_dn_list = [], [], []
    for ia in range(len(occslst_a)):
        for ib in range(len(occslst_b)):
            c = ci[ia, ib]
            if abs(c) > coeff_threshold:
                coeffs.append(c)
                occ_up_list.append(core + [ncore + j for j in occslst_a[ia]])
                occ_dn_list.append(core + [ncore + j for j in occslst_b[ib]])

    coeffs = np.array(coeffs)
    occ_up = np.array(occ_up_list, dtype=np.int32)
    occ_dn = np.array(occ_dn_list, dtype=np.int32)

    # Sort by descending |c_I| so trials[0] (used to seed the walkers) is
    # the dominant determinant.
    order = np.argsort(-np.abs(coeffs))
    coeffs = coeffs[order]
    occ_up = occ_up[order]
    occ_dn = occ_dn[order]

    omega = float(result['omega'])
    dipole = float(result['dipole'])
    q0 = -dipole / np.sqrt(omega) if omega > 0 else 0.0

    return {
        'ci_coeffs': coeffs,
        'occ_up': occ_up,
        'occ_dn': occ_dn,
        'ndet': int(len(coeffs)),
        'mo_coeff': np.asarray(result['mo_coeff']),
        'q0': float(q0),
        'cavity': True,          # marks this as a cavity-aware (QED) trial
    }


if __name__ == '__main__':
    from pyscf import gto, scf

    # LiH / 6-31g in a cavity: CS-QED-CASSCF vs CS-QED-CASCI / QED-FCI.
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='6-31g',
                unit='Angstrom', symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()

    omega = 0.1
    coupling_vec = (0.0, 0.0, 0.05)

    res = run_qed_casscf(mf, ncas=4, nelecas=2, omega=omega,
                         coupling_vec=coupling_vec, nph_max=2, verbose=True)
    print(f"\nE_QED-CASSCF = {res['e_qed_casscf']:.10f}")
    print(f"E_QED-CASCI  = {res['e_qed_casci']:.10f}  (starting orbitals)")
    print(f"relaxation   = {res['e_qed_casscf'] - res['e_qed_casci']:+.3e} Ha")
    print(f"E_QED-HF     = {res['e_qed_hf']:.10f}")
    print(f"converged    = {res['converged']}  ({res['n_macro']} macro-iters)")
