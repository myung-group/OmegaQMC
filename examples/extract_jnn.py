"""Extract the NN Jastrow J_NN(R_k) on a bank's walkers -> npz for ANOVA
distillation. J_NN = log|Psi_full| - log|Phi_slater| (jastrow-off sibling).

  python examples/extract_jnn.py --system h2
  python examples/extract_jnn.py --system beh2
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cs_properties import build_mol
from run_cs_tc_gate0 import SYSTEMS, ANSATZ
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.jastrow_extract import nn_jastrow_on_walkers
from OmegaQMC.vmc_nn import get_vmc_nn_func

p = argparse.ArgumentParser()
p.add_argument("--system", required=True, choices=list(SYSTEMS))
p.add_argument("--maxk", type=int, default=50000)
args = p.parse_args()

s = SYSTEMS[args.system]
mol = build_mol(s["geometry"], s["R"], s["basis"], s["unit"], s["tag"])
cell = Path(s["cell_dir"])
walkers, _, _ = load_walker_bank(str(cell / f"{cell.name}_walkers.h5"),
                                 max_K_s=args.maxk)
key = jax.random.split(jax.random.key(42), 4)[1]
drv = get_vmc_nn_func(mol, ANSATZ, key, prefix=str(cell / cell.name))
drv.load_checkpoint(str(cell / f"{cell.name}.chk.h5"))
nuc = np.asarray(mol.atom_coords())
J = nn_jastrow_on_walkers(ANSATZ, mol, key, drv.params, walkers, nuc,
                          batch_size=4096)
out = f"{args.system}_jnn.npz"
np.savez(out, walkers=np.asarray(walkers, dtype=np.float32),
         jnn=np.asarray(J, dtype=np.float32), nuc=nuc,
         na=s["n_alpha"], nb=s["n_beta"])
print(f"saved {out}: walkers {walkers.shape}, ({s['n_alpha']},{s['n_beta']})e-, "
      f"J range [{J.min():+.4f}, {J.max():+.4f}] mean {J.mean():+.4f}")
