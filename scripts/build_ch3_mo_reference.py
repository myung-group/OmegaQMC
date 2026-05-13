"""Build a PySCF reference for CH3· at the same geometry used by NN-VMC,
extract MO coefficients, identify the 1e' symmetry-adapted orbital pair
(m_l = ±1 around the C3 axis), and save everything to a .npz file for
downstream 1-RDM analysis.

This is the basis we'll project the NN-VMC walker positions against to
extract natural orbital occupations of the e' shell.

Output: scripts/ch3_mo_reference.npz with:
  - ao_basis_label  list of strings (one per AO)
  - mo_coeff_a, mo_coeff_b   (nao, nmo) UHF MO coefficients
  - mo_energy_a, mo_energy_b
  - mo_occ_a, mo_occ_b
  - e_prime_plus_coef   (nao,) complex   <-- 1e'_+ orbital (m_l=+1)
  - e_prime_minus_coef  (nao,) complex   <-- 1e'_- orbital (m_l=-1)
  - lz_ao  (nao, nao)   real    AO matrix elements of -i L_z (real after factoring out i)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pyscf import gto, scf


def build_ch3():
    coords = [["C", (0.0, 0.0, 0.0)]]
    r_ch = 2.039
    for k in range(3):
        theta = 2 * math.pi * k / 3
        coords.append([
            "H", (r_ch * math.cos(theta), r_ch * math.sin(theta), 0.0),
        ])
    return gto.M(
        atom=coords, basis="cc-pVDZ",
        spin=1, unit="Bohr", symmetry=True, verbose=4,
    )


def identify_e_prime_orbitals(mol, mf):
    """Find the doubly-occupied 1e' pair (ROHF: single set of MOs)."""
    from pyscf.symm import label_orb_symm

    irreps = label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff,
    )
    print("\nMO table:")
    for i, (e, occ, irr) in enumerate(zip(
        mf.mo_energy, mf.mo_occ, irreps,
    )):
        if e > 0.5:
            break   # only print occupied + a few virtual
        flag = "  <-- e'" if irr in ("E'", "E'x", "E'y") else ""
        marker = "2" if occ > 1.5 else ("1" if occ > 0.5 else "0")
        print(f"  MO {i:>2} occ={marker} ε={e:+.4f} sym={irr}{flag}")

    # The 1e' D3h pair shows up under C2v as (A1, B1) with degenerate
    # energy. Identify the degenerate doubly-occupied pair by energy
    # proximity rather than irrep label.
    occ_idx = [i for i in range(len(mf.mo_occ)) if mf.mo_occ[i] > 1.5]
    energies = [(i, mf.mo_energy[i]) for i in occ_idx]
    energies.sort(key=lambda t: t[1])
    e_idx = None
    for k in range(len(energies) - 1):
        i, j = energies[k][0], energies[k + 1][0]
        if abs(mf.mo_energy[i] - mf.mo_energy[j]) < 1e-3:
            e_idx = [i, j]
            print(f"\nDegenerate doubly-occupied pair found at "
                  f"ε={mf.mo_energy[i]:+.4f}: MOs {i} ({irreps[i]}) "
                  f"and {j} ({irreps[j]})")
            break
    if e_idx is None:
        raise ValueError(
            "no degenerate doubly-occupied pair found "
            f"(occ MOs: {occ_idx}, energies: "
            f"{[mf.mo_energy[i] for i in occ_idx]})"
        )
    return e_idx, irreps


def build_mplus_minus(mf, e_prime_indices):
    i_x, i_y = e_prime_indices
    cx = mf.mo_coeff[:, i_x]
    cy = mf.mo_coeff[:, i_y]
    e_plus  = (cx + 1j * cy) / np.sqrt(2.0)
    e_minus = (cx - 1j * cy) / np.sqrt(2.0)
    return e_plus, e_minus


def main():
    mol = build_ch3()
    print(f"D3h irreps: {mol.irrep_name}")
    print(f"nao = {mol.nao}, n_elec = ({mol.nelec[0]}, {mol.nelec[1]})")

    # Use ROHF: preserves spatial symmetry (UHF would split D3h -> C2v
    # via differential alpha/beta polarization at the SOMO).
    mf = scf.ROHF(mol)
    mf.kernel()
    print(f"ROHF E = {mf.e_tot:.6f} Ha")
    print(f"detected symmetry: {mol.groupname}")

    e_idx, irreps = identify_e_prime_orbitals(mol, mf)
    e_plus, e_minus = build_mplus_minus(mf, e_idx)

    # L_z AO integrals (PySCF: int1e_cg_irxp returns (Lx, Ly, Lz) as REAL
    # arrays representing the IMAGINARY part of <i| L_a |j>. Since L_a is
    # antisymmetric, the matrix elements are pure imaginary; the returned
    # real number is multiplied by i to get the actual matrix element.)
    L_xyz = mol.intor("int1e_cg_irxp")
    Lz_ao = L_xyz[2]

    # Test: verify e_plus and e_minus are eigenfunctions of L_z
    Lz_plus = e_plus.conj() @ (1j * Lz_ao) @ e_plus
    Lz_minus = e_minus.conj() @ (1j * Lz_ao) @ e_minus
    print(f"\nSymmetry check (should give ±1 ℏ):")
    print(f"  <e'_+|L_z|e'_+> = {Lz_plus:+.4f}")
    print(f"  <e'_-|L_z|e'_-> = {Lz_minus:+.4f}")

    # Norms
    s = mol.intor("int1e_ovlp")
    print(f"\nNormalization check (should be 1):")
    print(f"  <e'_+|e'_+> = {(e_plus.conj() @ s @ e_plus).real:.4f}")
    print(f"  <e'_-|e'_-> = {(e_minus.conj() @ s @ e_minus).real:.4f}")
    print(f"  <e'_+|e'_-> = {(e_plus.conj() @ s @ e_minus):+.4e}")

    # Save
    out = Path(__file__).parent / "ch3_mo_reference.npz"
    np.savez_compressed(
        out,
        ao_basis_label=np.array(mol.ao_labels(), dtype=object),
        mo_coeff=mf.mo_coeff,
        mo_energy=mf.mo_energy,
        mo_occ=mf.mo_occ,
        e_prime_indices=np.array(e_idx),
        e_prime_plus_coef=e_plus,
        e_prime_minus_coef=e_minus,
        Lz_ao=Lz_ao,
        ovlp_ao=s,
        ch3_geometry=np.array([list(mol.atom_coord(i))
                               for i in range(mol.natm)]),
    )
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
