"""H2O excitation-energy ladder: NES-VMC + CS + NEVPT2 vs traditional QC methods.

Compares the bright A^1B_1 vertical transition across:
- CIS / TDHF / EOM-CCSD (cc-pVDZ; PySCF benchmark)
- Pfau-NES K=2/K=3/K=4 + CS + NEVPT2 at progressively larger CAS and basis
- Experimental value (7.4 eV)
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("papers/cs_recovery/data/h2o_traditional_benchmark.json") as f:
    bench = json.load(f)["cc-pvdz"]

# Use bright-state values from the manuscript table tab:h2o_ccpvdz_comparison
methods = [
    ("CIS",                 bench["CIS"]["dE_eV"][0],     "#888"),
    ("TDHF",                bench["TDHF"]["dE_eV"][0],    "#888"),
    ("EOM-CCSD",            bench["EOM-CCSD"]["dE_eV"][0],"#444"),
    ("Pfau K=2 (cc-pVDZ)",  16.70, "#d97757"),
    ("Pfau K=3 (cc-pVDZ)",  10.16, "#c66"),
    ("Pfau K=4 (cc-pVDZ)",  10.16, "#a66"),
    ("Pfau K=3 (aug, FN+J, CAS12)", 9.63, "#3a3"),
    ("Experiment",          7.40, "#06c"),
]

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ypos = list(range(len(methods)))
labels = [m[0] for m in methods]
values = [m[1] for m in methods]
colors = [m[2] for m in methods]

bars = ax.barh(ypos, values, color=colors, edgecolor="black", linewidth=0.6, height=0.7)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel(r"Vertical excitation $\Delta E_{0\to1}$ (eV) for $\tilde{A}\,^1B_1$ of H$_2$O", fontsize=10)
ax.axvline(7.4, color="#06c", linestyle="--", linewidth=1, alpha=0.6)
ax.set_xlim(0, 18)
for b, v in zip(bars, values):
    ax.text(v + 0.15, b.get_y() + b.get_height()/2, f"{v:.2f}",
            va="center", fontsize=8)

# Group labels on the right
ax.text(17.6, -0.4, "Traditional QC", fontsize=8, ha="right", color="#444", style="italic")
ax.text(17.6, 5.6, "This work (NES-VMC + CS bridge)", fontsize=8, ha="right", color="#c66", style="italic")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=9)

plt.tight_layout()
plt.savefig("papers/cs_recovery/figs/h2o_excitation_ladder.pdf", bbox_inches="tight")
print("wrote papers/cs_recovery/figs/h2o_excitation_ladder.pdf")
