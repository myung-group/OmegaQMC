"""Predict the cavity-induced NMR chemical shift on H nuclei of CH3·.

Physics
-------
The cavity induces a ground-state orbital magnetic moment
   mu_orb = -g_orb * mu_B * <L_z>

This moment sits primarily on the carbon center (where the 1e' shell
lives). It produces a dipolar magnetic field at the surrounding H
nuclei:

   B_induced(r_H) = (mu_0 / 4pi) [3(mu . r_hat) r_hat - mu] / r^3

For a moment along z (perpendicular to CH3 molecular plane) and H atoms
in the xy plane (r_hat perpendicular to mu), this simplifies to:

   B_induced(H) = -(mu_0 / 4pi) * mu_z / r_CH^3   along z

This is the standard ring-current shielding formula (related to the
McConnell point-dipole approximation).

The resulting NMR chemical shift in ppm, for an external NMR field
of strength B_ext along z:

   sigma_cavity = -B_induced_z / B_ext (dimensionless, x 10^6 for ppm)

Or equivalently, the cavity-induced ABSOLUTE shielding contribution
(independent of B_ext):

   delta_B = -B_induced_z / B_ext_unit

Numbers for CH3 sigma+ at lambda=0.5:
  <L_z> = +0.053 hbar
  r_CH  = 2.039 Bohr = 1.079 A
  mu_orb = +0.053 mu_B
"""
from __future__ import annotations

import numpy as np


# Physical constants (SI)
MU_0    = 4 * np.pi * 1e-7         # vacuum permeability [T·m/A]
MU_B    = 9.2740100783e-24         # Bohr magneton [J/T]
A_BOHR  = 5.29177210903e-11        # Bohr radius [m]


def cavity_nmr_shift(L_z, r_CH_bohr, g_orb=1.0):
    """Cavity-induced NMR chemical shift at H nuclei from CH3 ring current.

    Returns the predicted ABSOLUTE shielding sigma (dimensionless, x 1e6 ppm).
    The shift is reported as -B_induced/B_ext (sign convention: positive
    shift = downfield).

    Args:
        L_z: cavity-induced <L_z> in units of hbar (signed).
        r_CH_bohr: C-H distance in Bohr (CH3 default ~2.04).
        g_orb: orbital g-factor (=1 for pure orbital).

    Returns:
        dict with B_induced_T (field at H in Tesla), shift_ppm
        (chemical shift in parts per million).
    """
    # Magnetic moment in J/T
    mu = -g_orb * MU_B * L_z   # SI

    # H position relative to C: in xy plane, at distance r_CH along x
    r_m = r_CH_bohr * A_BOHR   # SI

    # Point-dipole field at H, with mu along z and r along x:
    #   B_z(H) = -(mu_0 / 4pi) * mu / r^3
    B_z = -(MU_0 / (4 * np.pi)) * mu / r_m ** 3

    # NMR shift in ppm (absolute shielding due to cavity)
    # Convention: at typical lab field B_ext (e.g., 10 T), the relative
    # shift fraction is B_induced/B_ext. We report sigma = -B_induced
    # (so that downfield = positive ppm) PER unit B_ext.
    # Actually for "absolute" shift independent of B_ext, ppm is simply
    # -B_induced / B_ext * 1e6 where B_ext is any reference.
    # We use a generic B_ext = 1 Tesla, so shift = -B_z * 1e6 (per Tesla).
    shift_per_T_ppm = -B_z * 1e6   # ppm per Tesla

    return dict(
        L_z=L_z, r_CH_bohr=r_CH_bohr,
        mu_orb_JpT=mu,
        B_at_H_Tesla=B_z,
        shift_per_T_ppm=shift_per_T_ppm,
    )


def main():
    print("=" * 70)
    print("Cavity-induced NMR shift prediction — CH3· radical")
    print("=" * 70)

    runs = [
        (0.10, +1, +0.0107),
        (0.30, +1, +0.0206),
        (0.50, +1, +0.0530),
        (0.70, +1, +0.0533),
        (0.50, -1, -0.0369),
    ]
    r_CH_bohr = 2.039
    B_ext_NMR = 11.74    # 500 MHz 1H magnet, in Tesla

    print(f"\nGeometry: r_CH = {r_CH_bohr} Bohr = "
          f"{r_CH_bohr * 0.52918:.3f} A")
    print(f"NMR reference field: B_ext = {B_ext_NMR} T (500 MHz 1H magnet)\n")
    print(f"{'λ':>5} {'hand':>5} {'<L_z>':>10} {'B_at_H (mT)':>14} "
          f"{'shift (ppm)':>14} {'shift (Hz @ 500MHz)':>22}")
    print(f"{'='*5} {'='*5} {'='*10} {'='*14} {'='*14} {'='*22}")
    for lam, hand, lz in runs:
        r = cavity_nmr_shift(lz, r_CH_bohr)
        sign = "σ+" if hand == +1 else "σ-"
        B_mT = r['B_at_H_Tesla'] * 1000
        shift_ppm = -r['B_at_H_Tesla'] / B_ext_NMR * 1e6
        # frequency shift = chemical shift in ppm × γ_H × B_ext / (2π)
        # γ_H = 42.577 MHz/T → at B_ext = 11.74 T, frequency = 500 MHz.
        # shift in Hz = shift_ppm × 500 MHz × 1e-6 = shift_ppm × 500
        shift_Hz = shift_ppm * 500.0
        print(f"{lam:>5.2f} {sign:>5} {lz:>+10.4f} "
              f"{B_mT:>+13.2f}   "
              f"{shift_ppm:>+13.1f}   "
              f"{shift_Hz:>+22.0f}")

    print()
    print("Notes:")
    print("  - B_at_H is the INDUCED magnetic field at each H nucleus from")
    print("    the cavity-driven ring current on the C center.")
    print("  - 'shift (ppm)' is the resulting chemical shift on H, sign:")
    print("    +ppm = downfield (deshielded), conventional NMR convention.")
    print("  - Point-dipole approximation -- actual ring current is more")
    print("    distributed, so this likely overestimates by ~2-5x.")
    print()
    print("Comparison context:")
    print("  - Typical 1H NMR chemical shift range:   0-12 ppm")
    print("  - 1H NMR spectrometer resolution:        0.001 ppm")
    print("  - Solvent-induced shifts:                0.01-0.1 ppm")
    print("  - Predicted cavity shift at λ=0.5:       ~ -3300 ppm")
    print()
    print("=> The predicted shift is ENORMOUS compared to typical chemical")
    print("   shifts. Even accounting for ~5x overestimation from point-")
    print("   dipole approximation, the shift would be ~600 ppm -- still")
    print("   ~100x bigger than the entire 1H chemical-shift range.")
    print()
    print("   The cavity-induced ring current would shift the H NMR peak")
    print("   COMPLETELY OUT of the usual H NMR window -- a unique,")
    print("   unambiguous experimental signature of strong-coupling")
    print("   cavity QED on radicals.")


if __name__ == "__main__":
    main()
