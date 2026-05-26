"""CAS-vocabulary convergence diagram (placeholder + completed points).

X-axis: CAS active-space size (8, 12, 16, 20)
Y-axis (left): match-overlap of the recovered c^(exc) with the closest CAS eigvec
Y-axis (right): |E_PT2^exc| in m E_h
"""
import matplotlib.pyplot as plt
import numpy as np

# From tab:cas_size_convergence (CAS 16/20 TBD until SLURM 5275/5276 finish; placeholder)
cas_sizes = np.array([8, 12, 16, 20])
match_overlap = np.array([0.48, 0.48, np.nan, np.nan])  # known points
pt2_mE         = np.array([235,  201,  np.nan, np.nan])
dE_CASPT2_eV   = np.array([9.97, 9.63, np.nan, np.nan])

fig, ax = plt.subplots(figsize=(6.4, 4.0))

color_m = "#3a3"
color_e = "#c66"
ax.plot(cas_sizes, match_overlap, "o-", color=color_m, markersize=7,
        markerfacecolor=color_m, markeredgecolor="black",
        markeredgewidth=0.6, label=r"match-overlap $|\langle\widehat{c}^{(\mathrm{exc})}|v^{\mathrm{CAS}}\rangle|$")
ax.set_xlabel("CAS active-space size (orbitals)", fontsize=10)
ax.set_ylabel(r"match-overlap with closest CAS eigenstate", fontsize=10, color=color_m)
ax.tick_params(axis="y", labelcolor=color_m)
ax.set_xticks(cas_sizes)
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color=color_m, linestyle=":", linewidth=0.7, alpha=0.6)
ax.text(20, 1.02, "perfect translation", color=color_m, fontsize=8, ha="right", va="bottom")

ax2 = ax.twinx()
ax2.plot(cas_sizes, dE_CASPT2_eV, "s--", color=color_e, markersize=7,
         markerfacecolor=color_e, markeredgecolor="black",
         markeredgewidth=0.6, label=r"$\Delta E_{\mathrm{CAS+PT2}}$ (matched root)")
ax2.set_ylabel(r"$\Delta E_{\mathrm{CAS+PT2}}$ at matched root (eV)", fontsize=10, color=color_e)
ax2.tick_params(axis="y", labelcolor=color_e)
ax2.axhline(7.4, color="#06c", linestyle=":", linewidth=0.8)
ax2.text(20, 7.55, "exp 7.4 eV", color="#06c", fontsize=8, ha="right")

# TBD markers
for cas, y_overlap in zip(cas_sizes, match_overlap):
    if np.isnan(y_overlap):
        ax.annotate("TBD", (cas, 0.65), ha="center", fontsize=9, color="#888", style="italic")

ax.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False,
          loc="center right", bbox_to_anchor=(1.0, 0.4))

ax.set_title("CAS-vocabulary convergence (FermiNet+J K=4, aug-cc-pVDZ)", fontsize=10)
plt.tight_layout()
plt.savefig("papers/cs_recovery/figs/cas_vocab_convergence.pdf", bbox_inches="tight")
print("wrote papers/cs_recovery/figs/cas_vocab_convergence.pdf")
