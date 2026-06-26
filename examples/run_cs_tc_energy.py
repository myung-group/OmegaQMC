"""
Transcorrelated energy on a converged NN bank: does folding the Jastrow into
the operator recover the beyond-finite-basis correlation that bare CI loses?

The point is NOT CI-vector leakage (inherent to projecting a real-space
backflow wavefunction) but *information loss from the finite basis*. The
finite-basis ceiling is FCI(basis): bare CI in the Gaussian basis cannot go
below it. The full real-space Psi_NN sits below it (captures correlation
outside the basis). The question: does a transcorrelated, finite-basis
estimator recover that?

We use the reference-projected (mixed) energy

    E_proj^(tau) = <f_ref . E_L> / <f_ref>,   f_ref = D_ref e^{-tau J} / Psi_NN,

with E_L = H Psi_NN / Psi_NN the VMC local energy and D_ref a single in-basis
(HF/dominant) determinant. Key property: there is NO finite-basis truncation
of Psi_NN -- if Psi_NN were exact this returns E_0 in *any* basis, so it
carries the beyond-basis correlation. tau = 0 is the bare mixed estimator;
tau = +1 (with J = learned Jastrow or cusp-only) is the transcorrelated one.

Reports E_VMC (target), FCI(basis) (ceiling), and E_proj at tau = 0 / +1,
with jackknife errors and the estimator variance (the zero-variance angle).

    python examples/run_cs_tc_energy.py --system h4_linear --j-source nn
    python examples/run_cs_tc_energy.py --system h4_linear --j-source cusp --cusp-b 1.0
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_cs_properties import build_mol
from run_cs_h4_scaling import evaluate_signed_psi

from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import (
    evaluate_orbitals_on_walkers,
    evaluate_ci_wavefunction,
)
from OmegaQMC.cs.jastrow_extract import nn_jastrow_on_walkers, kato_cusp_jastrow
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func

# Reuse the system registry from the gate-0 driver.
from run_cs_tc_gate0 import SYSTEMS, ANSATZ


def _local_energy(drv, walkers, batch=2048):
    """E_L with the LOADED params. (drv._local_energy_batch is a closure
    bound to the *initial* params at construction, so it must NOT be used
    after load_checkpoint; we rebuild from energy_ee/en/ke + drv.params.)"""
    import jax
    import jax.numpy as jnp
    ee, en, ke = drv.energy_ee, drv.energy_en, drv.energy_ke
    enr = float(np.asarray(drv.enr_nn))
    p = drv.params

    @jax.jit
    def batch_el(w):
        return (jax.vmap(ee)(w) + jax.vmap(en)(w)
                + jax.vmap(ke, in_axes=(0, None))(w, p) + enr)

    out = np.empty(walkers.shape[0], dtype=float)
    for s in range(0, walkers.shape[0], batch):
        b = jnp.asarray(walkers[s:s + batch])
        out[s:s + b.shape[0]] = np.asarray(batch_el(b))
    return out


def _proj_energy(f_ref, E_L, n_blocks=20):
    """Ratio estimator <f E_L>/<f> with a jackknife error over blocks."""
    f = np.asarray(f_ref); e = np.asarray(E_L)
    full = float(np.mean(f * e) / np.mean(f))
    K = len(f); bs = K // n_blocks
    jk = []
    for i in range(n_blocks):
        mask = np.ones(K, bool); mask[i * bs:(i + 1) * bs] = False
        jk.append(np.mean(f[mask] * e[mask]) / np.mean(f[mask]))
    jk = np.array(jk)
    err = float(np.sqrt((n_blocks - 1) * np.mean((jk - jk.mean()) ** 2)))
    return full, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True, choices=list(SYSTEMS))
    p.add_argument("--j-source", default="nn", choices=["nn", "cusp"])
    p.add_argument("--cusp-b", type=float, default=1.0)
    p.add_argument("--ansatz", default=ANSATZ)
    p.add_argument("--candidate-tol", type=float, default=1e-4)
    p.add_argument("--psi-batch", type=int, default=2048)
    args = p.parse_args()

    s = SYSTEMS[args.system]
    na, nb = s["n_alpha"], s["n_beta"]
    print(f"== TC energy :: {args.system} :: {jax.devices()} ==")
    mol = build_mol(s["geometry"], s["R"], s["basis"], s["unit"], s["tag"])
    fci = compute_fci_reference(mol, n_alpha=na, n_beta=nb,
                                candidate_tol=args.candidate_tol)
    E_FCI, E_HF = fci["E_FCI"], fci["E_HF"]
    pool = fci["candidate_set"]
    c_fci = np.array([fci["ci_dict"].get(I, 0.0) for I in pool])
    ref_det = pool[int(np.argmax(np.abs(c_fci)))]
    print(f"[1] basis={s['basis']}  E_HF={E_HF:.6f}  FCI(basis)={E_FCI:.6f}")
    print(f"    reference determinant = {ref_det}")

    cell = Path(s["cell_dir"])
    walkers, _, _ = load_walker_bank(str(cell / f"{cell.name}_walkers.h5"))
    key = jax.random.split(jax.random.key(42), 4)[1]
    drv = get_vmc_nn_func(mol, args.ansatz, key, prefix=str(cell / cell.name))
    drv.load_checkpoint(str(cell / f"{cell.name}.chk.h5"))
    log_psi, _, _, _ = make_nn_log_psi(args.ansatz, mol, key)
    nuc = np.asarray(mol.atom_coords())

    print("[2] local energy E_L on the bank")
    E_L = _local_energy(drv, walkers, batch=args.psi_batch)
    E_VMC = float(np.mean(E_L))
    E_VMC_err = float(np.std(E_L) / np.sqrt(len(E_L)))
    print(f"    E_VMC = <E_L> = {E_VMC:.6f} +/- {E_VMC_err:.6f}  "
          f"(target; below FCI(basis) by {(E_VMC-E_FCI)*1e3:+.2f} mE_h)")

    psi = evaluate_signed_psi(walkers, nuc, drv.params, log_psi,
                              batch_size=args.psi_batch)
    orb = evaluate_orbitals_on_walkers(mol, walkers, fci["no_coeff_ao"],
                                       convention="interleaved",
                                       n_alpha=na, n_beta=nb)
    D_ref = evaluate_ci_wavefunction(orb, [ref_det], np.array([1.0]), na, nb)

    if args.j_source == "cusp":
        J = kato_cusp_jastrow(walkers, na, nb, b=args.cusp_b,
                              layout="interleaved")
        jlabel = f"cusp(b={args.cusp_b})"
    else:
        J = nn_jastrow_on_walkers(args.ansatz, mol, key, drv.params, walkers,
                                  nuc, batch_size=args.psi_batch)
        jlabel = "learned-J"

    f0 = D_ref / psi                    # tau = 0  (bare mixed estimator)
    f1 = D_ref * np.exp(-J) / psi       # tau = +1 (transcorrelated)

    E0, E0e = _proj_energy(f0, E_L)
    E1, E1e = _proj_energy(f1, E_L)
    # estimator-noise proxy: relative spread of the (signed) weights
    relvar0 = float(np.std(f0) / np.abs(np.mean(f0)))
    relvar1 = float(np.std(f1) / np.abs(np.mean(f1)))

    print("\n=== energies (Ha) ===")
    print(f"  E_HF                         = {E_HF:.6f}")
    print(f"  FCI(basis) ceiling           = {E_FCI:.6f}")
    print(f"  E_VMC  (full Psi_NN, target) = {E_VMC:.6f} +/- {E_VMC_err:.6f}")
    print(f"  E_proj  tau=0  (bare mixed)  = {E0:.6f} +/- {E0e:.6f}   "
          f"(rel.wt.spread {relvar0:.1f})")
    print(f"  E_proj  tau=+1 [{jlabel}] (TC)= {E1:.6f} +/- {E1e:.6f}   "
          f"(rel.wt.spread {relvar1:.1f})")
    print("\n  vs FCI(basis):")
    print(f"    E_proj(tau=0)  - FCI = {(E0-E_FCI)*1e3:+.2f} mE_h")
    print(f"    E_proj(tau=+1) - FCI = {(E1-E_FCI)*1e3:+.2f} mE_h")
    print(f"    E_VMC          - FCI = {(E_VMC-E_FCI)*1e3:+.2f} mE_h "
          f"(the beyond-basis correlation to recover)")


if __name__ == "__main__":
    main()
