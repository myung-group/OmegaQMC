"""Basis-set sensitivity of the cavity-induced electron-affinity shift.

Referee check (Sec.~gw-bench, Minor comment on the EA): the vertical EAs
are basis-confined (the anion is bound only by the finite basis), so a
skeptical reader may worry that the close GW vs Delta-QED-CCSD agreement
on the cavity-induced EA *shift* is a shared basis artifact rather than
shared physics. This recomputes the water EA shift
delta_lambda EA = EA(lambda=0.05) - EA(lambda=0) at cc-pVDZ and
aug-cc-pVDZ, for Delta-QED-HF, Delta-QED-CCSD, and evGW.

Result: the *magnitude* is strongly basis-sensitive (Delta-CCSD shift
-0.075 -> -0.236 eV as diffuse functions delocalize the extra electron),
but the GW--CCSD *agreement* survives (aug-cc-pVDZ: -0.241 vs -0.236 eV,
~2%), so it reflects a shared cavity response, not a shared artifact. The
mean-field--CCSD proximity, by contrast, is a cc-pVDZ feature (with
diffuse functions Delta-QED-HF, -0.363 eV, overshoots).

Reuses run_molecule() from run_qed_ipea_benchmark.py (import-safe).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_qed_ipea_benchmark import run_molecule  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

out = {}
print("H2O cavity-induced EA shift  d_lambda EA = EA(0.05) - EA(0)  (eV)")
print(f"{'basis':12s} {'EA(l=0)':>9s} {'dHF':>9s} {'dCC':>9s} {'evGW':>9s}")
for basis in ('cc-pVDZ', 'aug-cc-pVDZ'):
    r0 = run_molecule('H2O', basis, 0.0)
    r1 = run_molecule('H2O', basis, 0.05)
    row = {'EA0_CC': r0['dCC_EA'],
           'dHF': r1['dHF_EA'] - r0['dHF_EA'],
           'dCC': r1['dCC_EA'] - r0['dCC_EA'],
           'devGW': r1['evgw_EA'] - r0['evgw_EA']}
    out[basis] = row
    print(f"{basis:12s} {row['EA0_CC']:9.3f} {row['dHF']:9.4f} "
          f"{row['dCC']:9.4f} {row['devGW']:9.4f}")

with open(os.path.join(HERE, 'qed_ea_basis_results.json'), 'w') as f:
    json.dump(out, f, indent=1)
print("\nwrote qed_ea_basis_results.json")
