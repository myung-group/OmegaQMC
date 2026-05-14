"""Validation figure: 3-panel comparison of NN-VMC vs literature."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

CSV = 'logs/validation/results.csv'

def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

rows = load()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# === Panel 1: T1 — bare H2 dissoc curve ===
ax = axes[0]
nnvmc_T1 = [r for r in rows if r['tier'] == 'T1']
kw_T1 = [r for r in rows if r['tier'] == 'T1_ref']
R_nn = [float(r['R_bohr']) for r in nnvmc_T1]
E_nn = [float(r['E_Ha']) for r in nnvmc_T1]
err_nn = [float(r['E_err_Ha']) for r in nnvmc_T1]
R_kw = [float(r['R_bohr']) for r in kw_T1]
E_kw = [float(r['E_Ha']) for r in kw_T1]
ax.errorbar(R_nn, E_nn, yerr=err_nn, fmt='o-', label='NN-VMC (this work)', markersize=7)
ax.plot(R_kw, E_kw, 's--', label='Kolos-Wolniewicz exact', markersize=7, alpha=0.7)
ax.set_xlabel('R (Bohr)')
ax.set_ylabel('E (Ha)')
ax.set_title('T1: Bare H$_2$ dissociation curve')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

# Inset: residuals
inset_ax = ax.inset_axes([0.55, 0.55, 0.4, 0.35])
diff_mHa = [(e_nn - e_kw) * 1000 for e_nn, e_kw in zip(E_nn, E_kw)]
inset_ax.plot(R_nn, diff_mHa, 'ko-', markersize=5)
inset_ax.axhline(0, color='gray', linestyle=':')
inset_ax.set_xlabel('R (Bohr)', fontsize=8)
inset_ax.set_ylabel('NN-VMC$-$KW (mHa)', fontsize=8)
inset_ax.tick_params(labelsize=8)
inset_ax.grid(alpha=0.3)

# === Panel 2: T2 — Riera benchmark ===
ax = axes[1]
methods = ['NN-VMC\n(this work)', 'QED-FCI\n(aug-cc-pVDZ)', 'Riera 2024\nAFQMC (Fig 1)']
dE_vals = [6.61, 5.64, 9.5]
colors = ['C0', 'C1', 'C2']
bars = ax.bar(methods, dE_vals, color=colors, alpha=0.8)
for b, v in zip(bars, dE_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f'{v:+.2f}',
            ha='center', fontsize=10)
ax.axhline(0, color='gray', linestyle=':')
ax.set_ylabel('$\\Delta E$ (mHa)')
ax.set_title('T2: Riera setting\n($R=1.4$, $\\omega=0.3$ Ha, $\\lambda=0.1$)')
ax.grid(alpha=0.3, axis='y')

# === Panel 3: T3 — Weight benchmark ===
ax = axes[2]
A0 = [0.2, 0.5, 0.8]
nnvmc = [14.87, 74.21, 159.06]
fci   = [12.97, 71.45, 158.90]
ax.plot(A0, nnvmc, 'o-', label='NN-VMC (this work)', markersize=8)
ax.plot(A0, fci,   's--', label='QED-FCI (aug-cc-pVDZ)', markersize=8, alpha=0.7)
# Annotate residuals
for a, n, f in zip(A0, nnvmc, fci):
    ax.annotate(f'{n - f:+.2f}', xy=(a, n), xytext=(a + 0.03, n),
                fontsize=8, color='darkblue')
ax.set_xlabel('$A_0$ (a.u.)')
ax.set_ylabel('$\\Delta E$ (mHa)')
ax.set_title('T3: Weight setting\n($R=2.8$, $\\omega_c=5$ eV)')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
ax.text(0.95, 0.05, 'scQED-CCSD\nfails this\nregime',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8, style='italic', color='gray',
        bbox=dict(facecolor='white', edgecolor='lightgray'))

plt.tight_layout()
out = 'logs/validation/figure_validation.pdf'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
print(f'wrote {out} and .png')
