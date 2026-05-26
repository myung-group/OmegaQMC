"""Linear H4 / cc-pVDZ top-8 determinant recovery: c-hat vs FCI."""
import matplotlib.pyplot as plt
import numpy as np

# From tab:h4_topdets in main.tex (existing manuscript data)
labels = [
    "(01,01)", "(02,02)", "(12,12)", "(12,03)",
    "(03,12)", "(13,13)", "(03,03)", "(23,01)",
]
c_hat = np.array([+0.854, -0.148, -0.052, -0.050, -0.039, -0.034, -0.030, +0.039])
se    = np.array([ 0.002,  0.006,  0.006,  0.006,  0.006,  0.005,  0.006,  0.006])
c_fci = np.array([+0.969, -0.160, -0.069, -0.065, -0.065, -0.047, -0.042, +0.032])

x = np.arange(len(labels))
w = 0.35

fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.bar(x - w/2, c_fci, w, label="FCI (cc-pVDZ)", color="#777", edgecolor="black", linewidth=0.4)
ax.bar(x + w/2, c_hat, w, yerr=se, label=r"Recovered $\widehat{c}$  (CS pipeline)",
       color="#3a3", edgecolor="black", linewidth=0.4, error_kw={"linewidth": 0.8, "capsize": 2})
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=0)
ax.set_xlabel(r"Determinant $(\mathrm{occ}^\alpha,\mathrm{occ}^\beta)$ (NO basis)", fontsize=10)
ax.set_ylabel(r"CI coefficient", fontsize=10)
ax.set_title(r"Linear H$_4$ / cc-pVDZ, $R = 1.0$\,\AA: top-8 determinant recovery", fontsize=10)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig("papers/cs_recovery/figs/h4_recovery_top8.pdf", bbox_inches="tight")
print("wrote papers/cs_recovery/figs/h4_recovery_top8.pdf")
