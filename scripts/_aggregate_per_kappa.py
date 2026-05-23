"""Aggregate per-κ training chunks into a single TABC summary."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 2:
    print("usage: _aggregate_per_kappa.py <project-prefix>")
    sys.exit(1)
project = Path(sys.argv[1])
prefix = project.name

chunks = []
for c in range(6):
    p = project.parent / f"{prefix}_c{c}" / "per_kappa_summary.json"
    if not p.exists():
        print(f"  missing: {p}")
        continue
    with open(p) as f:
        chunks.append((c, json.load(f)))
if not chunks:
    print("no chunks"); sys.exit(1)

all_results = []
for c, d in chunks:
    all_results.extend(d["results"])
all_results.sort(key=lambda x: x["twist_idx"])

n = len(all_results)
e_init = np.asarray([r["E_initial_ha"] or 0.0 for r in all_results])
e_final = np.asarray([r["E_final_ha"] for r in all_results])
kappas = np.asarray([r["kappa"] for r in all_results])
kabs = np.linalg.norm(kappas, axis=1)

mean_E_final = float(e_final.mean())
sem_E_final = float(e_final.std(ddof=1) / np.sqrt(n))
mean_E_init = float(e_init.mean())
delta_mean = mean_E_final - mean_E_init

print(f"=== Per-κ Fine-tune TABC ({n} twists across {len(chunks)} chunks) ===")
print(f"  chunks present: {[c for c, _ in chunks]}")
print(f"  V11 γ=0.5 Γ baseline = +137.10 mHa/e")
print(f"  V11_on TABC (no fine-tune)= +160.25 mHa/e")
print(f"  Per-κ initial E (V11 transferred) avg = "
      f"{mean_E_init * 1000:+.4f} mHa/e")
print(f"  Per-κ fine-tuned E avg = "
      f"{mean_E_final * 1000:+.4f} ± {sem_E_final * 1000:.4f} mHa/e")
print(f"  Per-κ mean improvement  = "
      f"{delta_mean * 1000:+.4f} mHa/e per twist")
print(f"  Final E range = [{e_final.min() * 1000:+.4f}, "
      f"{e_final.max() * 1000:+.4f}] mHa/e")
print(f"  correlation(|κ|, E_final) = "
      f"{float(np.corrcoef(kabs, e_final)[0,1]):+.3f}")

out_path = project.parent / f"{prefix}_aggregate.json"
with open(out_path, "w") as f:
    json.dump({
        "n_twists": n,
        "chunks_loaded": [c for c, _ in chunks],
        "twists": kappas.tolist(),
        "E_initial_ha": e_init.tolist(),
        "E_final_ha": e_final.tolist(),
        "tabc_avg_initial_ha": mean_E_init,
        "tabc_avg_final_ha": mean_E_final,
        "tabc_avg_final_sem_ha": sem_E_final,
        "mean_improvement_ha": delta_mean,
    }, f, indent=2, default=float)
print(f"  saved: {out_path}")
