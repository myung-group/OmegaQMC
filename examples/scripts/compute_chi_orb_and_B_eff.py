"""Compute orbital paramagnetic susceptibility chi_orb for CH3 in vacuum,
then convert our cavity-induced <L_z> to equivalent magnetic field B_eff.

chi_orb(zz)  =  -2 (μ_B / hbar)^2  Σ_{i in occ, a in virt}  |<a|L_z|i>|^2 / (ε_a - ε_i)

where L_z = -i hbar (x ∂_y - y ∂_x).  In atomic units (μ_B = 1/2, hbar = 1):
chi_orb [a.u. of magnetic moment per a.u. of B] = -(1/2)
                                                 Σ_{ia} |M_ai|^2 / (ε_a - ε_i)

The induced ⟨L_z⟩ from an applied static B_z is:
<L_z>_external  =  chi_orb * B_z / (-μ_B)    (sign convention check)
                =  -2 chi_orb B_z  in atomic units

So given the cavity-induced <L_z>_cavity, the equivalent static field:
B_eff [a.u.] = <L_z>_cavity / (-2 chi_orb)

Convert to Tesla:  B_eff [T] = B_eff [a.u.] * 2.35051756758e5
"""
from __future__ import annotations

import math

import numpy as np
import pyscf
from pyscf import gto, scf

LZ_FROM_CAVITY = 0.0530          # ⟨L_z⟩ from CH3 sigma+ λ=0.5 (from prior run)
LZ_FROM_CAVITY_SERR = 0.0018
HARTREE_TO_EV = 27.211386245988
AU_BFIELD_TO_TESLA = 2.35051756758e5   # 1 a.u. of B = 2.35e5 T


def build_ch3():
    """Planar D3h CH3, r_CH = 2.039 Bohr."""
    coords = [["C", (0.0, 0.0, 0.0)]]
    r_ch = 2.039
    for k in range(3):
        theta = 2 * math.pi * k / 3
        coords.append([
            "H", (r_ch * math.cos(theta), r_ch * math.sin(theta), 0.0),
        ])
    mol = gto.M(
        atom=coords, basis="cc-pVDZ", spin=1, unit="Bohr",
        verbose=0,
    )
    return mol


def compute_chi_orb_zz(mol, mf):
    """Compute orbital paramagnetic susceptibility chi_orb_zz [a.u.]
    via uncoupled HF (UCHF) sum-over-states formula.

    UHF MO coefficients & energies for both spin channels.
    Returns chi_orb in atomic units.
    """
    # AO integrals of L_z = -i (x py - y px), in a.u., shape (nao, nao)
    # PySCF returns (Lx, Ly, Lz) from int1e_cg_irxp (with -i prefactor implicit)
    # The integrals are pure imaginary; PySCF returns them with -i factored out
    # (i.e. you get real numbers that you multiply by i later if needed).
    # For our chi_orb, we need |<a|L_z|i>|^2, so sign and i factor square out.
    L_int = mol.intor("int1e_cg_irxp")    # shape (3, nao, nao)
    Lz_ao = L_int[2]                      # the z-component

    chi_orb = 0.0
    for spin in (0, 1):                   # alpha, beta
        mo = mf.mo_coeff[spin]            # (nao, nmo)
        mo_e = mf.mo_energy[spin]
        mo_occ = mf.mo_occ[spin]
        # Transform to MO basis
        Lz_mo = mo.T @ Lz_ao @ mo
        n_occ_idx = np.where(mo_occ > 0.5)[0]
        n_vir_idx = np.where(mo_occ < 0.5)[0]
        for i in n_occ_idx:
            for a in n_vir_idx:
                M = Lz_mo[a, i]
                de = mo_e[a] - mo_e[i]
                if de > 1e-8:
                    chi_orb += np.abs(M) ** 2 / de
    # chi_orb so far = Σ |M_ai|^2 / (ε_a - ε_i), summed over both spins.
    # In standard convention: chi_orb_para [a.u.] = -alpha²/(2 c²) * Σ ...
    # but for the linear-response <L_z>/B_z definition (in a.u., μ_B=1/2):
    # <L_z>_induced = (μ_B B) * 2 * Σ_ia |M_ai|^2 / (ε_a - ε_i)
    # So d<L_z>/dB = 2 μ_B Σ... = Σ... (since μ_B = 1/2 in a.u.)
    # Hence χ_orb [a.u.] = chi_orb (the bare sum)
    return chi_orb


def main():
    print("=" * 60)
    print("CH3· vacuum: compute orbital paramagnetic susceptibility")
    print("=" * 60)

    mol = build_ch3()
    print(f"basis: {mol.basis}, nao = {mol.nao}, "
          f"n_elec = ({mol.nelec[0]}, {mol.nelec[1]})")

    mf = scf.UHF(mol).run()
    print(f"UHF energy: {mf.e_tot:.6f} Ha")

    chi = compute_chi_orb_zz(mol, mf)
    print(f"\nchi_orb_zz (paramagnetic, UHF) = {chi:.6f} a.u.")
    print(f"  in SI: chi_orb = {chi:.4e} hbar^2 / Ha")

    # B_eff = <L_z>_cavity / chi  (in a.u.)
    print("\n" + "=" * 60)
    print("B_eff conversion: cavity-induced <L_z> -> equivalent static B")
    print("=" * 60)
    print(f"chi_orb_zz = {chi:.4f} a.u. (UHF, cc-pVDZ)")
    print()
    print(f"{'lambda':>8} {'hand':>5} {'<L_z>':>10} {'B_eff (T)':>15}"
          f" {'vs strongest pulsed':>22}")
    print(f"{'='*8} {'='*5} {'='*10} {'='*15} {'='*22}")
    data = [
        (0.30,  1, 0.0206, 0.0018),
        (0.50,  1, 0.0530, 0.0018),
        (0.50, -1, -0.0369, 0.0016),
        (0.70,  1, 0.0533, 0.0023),
    ]
    PULSED_MAX = 1200.0
    for lam, hand, lz, lz_serr in data:
        B_T = (lz / chi) * AU_BFIELD_TO_TESLA
        B_T_err = (lz_serr / chi) * AU_BFIELD_TO_TESLA
        sign = "σ+" if hand == 1 else "σ-"
        print(f"{lam:>8.2f} {sign:>5} {lz:>+10.4f} "
              f"{B_T:>+10.0f} ± {B_T_err:<3.0f} "
              f"{B_T/PULSED_MAX:>+18.2f}×")

    print("\nContext:")
    print(f"  NMR ¹H 500MHz magnet:    11.7 T")
    print(f"  MRI:                     3 T")
    print(f"  Strongest steady lab:    45 T (NHMFL)")
    print(f"  Strongest pulsed lab:    ~1200 T (Los Alamos, μs)")
    print(f"  White-dwarf surface:     ~10⁸ T")
    print(f"  Neutron star surface:    ~10¹¹ T")


if __name__ == "__main__":
    main()
