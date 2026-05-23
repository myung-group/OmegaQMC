"""Aggregate 6 chunk outputs of run_qed_l5_tabc.py into one summary.

Usage:
    python scripts/_aggregate_tabc_chunks.py runs/tabc_v11_g050
        (looks for runs/tabc_v11_g050_c{0..5}/tabc_summary.json)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 2:
    print("usage: _aggregate_tabc_chunks.py <project-prefix>")
    sys.exit(1)
project = Path(sys.argv[1])
prefix = project.name

# Locate chunks
chunks = []
for c in range(6):
    p = project.parent / f"{prefix}_c{c}" / "tabc_summary.json"
    if not p.exists():
        print(f"  missing: {p}")
        continue
    with open(p) as f:
        chunks.append((c, json.load(f)))
if not chunks:
    print("no chunks found"); sys.exit(1)

twists, energies, sems = [], [], []
chkpt_e_per_e = None
for c, data in chunks:
    twists.extend(data["twists"])
    energies.extend(data["per_twist_E_per_e_ha"])
    sems.extend(data["per_twist_SEM_ha"])
    if chkpt_e_per_e is None:
        chkpt_e_per_e = data["chkpt_E_per_e_ha"]
    elif abs(chkpt_e_per_e - data["chkpt_E_per_e_ha"]) > 1e-12:
        print(f"WARNING: chunk {c} has chkpt_E_per_e_ha differing — "
              f"are all chunks from same chkpt?")

twists = np.asarray(twists)
energies = np.asarray(energies)
sems = np.asarray(sems)
mean_E = float(energies.mean())
sem_twist = float(energies.std(ddof=1) / np.sqrt(len(energies)))

print(f"\n=== Aggregate TABC ({len(energies)} twists across {len(chunks)} chunks) ===")
print(f"  chunks present: {[c for c, _ in chunks]}")
print(f"  Γ-only E/N            = {chkpt_e_per_e*1000:+.4f} mHa/e")
print(f"  Twist-avg E/N         = {mean_E*1000:+.4f} ± {sem_twist*1000:.4f} mHa/e")
print(f"  Per-twist mean SEM    = {sems.mean()*1000:.4f} mHa/e")
print(f"  TABC – Γ shift        = {(mean_E - chkpt_e_per_e)*1000:+.4f} mHa/e")
print(f"  twist E range         = "
      f"[{energies.min()*1000:+.4f}, {energies.max()*1000:+.4f}] mHa/e")

out = {
    "n_chunks_loaded": len(chunks),
    "chunks_loaded": [c for c, _ in chunks],
    "n_twists_total": len(energies),
    "chkpt_E_per_e_ha": chkpt_e_per_e,
    "twists": twists.tolist(),
    "per_twist_E_per_e_ha": energies.tolist(),
    "per_twist_SEM_ha": sems.tolist(),
    "twist_avg_E_per_e_ha": mean_E,
    "twist_avg_SEM_ha": sem_twist,
}
out_path = project.parent / f"{prefix}_aggregate_summary.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"  saved: {out_path}")
