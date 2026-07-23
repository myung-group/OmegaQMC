"""Generate + run all test molecules through g16, save parsed data."""
import pickle
import numpy as np
from molecules import MOLECULES
from g16io import write_com, run_g16, parse_log

results = {}
for name, fn in MOLECULES.items():
    syms, coords = fn()
    write_com(name, syms, coords)
    print(f"running {name} ({len(syms)} atoms) ...", flush=True)
    log = run_g16(name)
    d = parse_log(log)
    d['syms'] = syms
    results[name] = d
    fw = d['framework']
    pg = d['pointgroup']
    ok = d['standard'] is not None
    print(f"  framework={fw} pg={pg} std={'ok' if ok else 'MISSING'}")

with open("g16_results.pkl", "wb") as fh:
    pickle.dump(results, fh)
print("saved g16_results.pkl")
