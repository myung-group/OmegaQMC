"""OMP vs Lasso-CS vs identity baseline on H2, H4 linear, BeH2.

For each system: pull the trained NN-VMC trial, load walker bank,
run all three decoders. For OMP, sweep m and rel_tol to see how
the support grows.
"""
from __future__ import annotations
import argparse, json, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
import jax

sys.path.insert(0, "examples")

from OmegaQMC.utils import Mole_custom
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import evaluate_orbitals_on_walkers, f_I_matrix
from OmegaQMC.cs.compressed import compressed_sensing_decode, omp_decode
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.vmc_nn import get_vmc_nn_func
from run_cs_properties import build_mol
from run_cs_h4_scaling import evaluate_signed_psi


def enumerate_pool(n_orb, n_alpha, n_beta):
    """Full enumeration for small molecules."""
    a = list(combinations(range(n_orb), n_alpha))
    b = list(combinations(range(n_orb), n_beta))
    return [(x, y) for x in a for y in b]


def run_one(name, geometry, R, basis, unit, n_alpha, n_beta,
             cell_dir, ansatz, pool_filter=False, candidate_tol=1e-10,
             m_values=(20, 40, 80),
             omp_rel_tols=(1e-3, 1e-2, 5e-2),
             omp_max_support=None, geometry_tag=""):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    mol = build_mol(geometry, R, basis, unit, geometry_tag)
    n_orb = int(mol.nao)
    print(f"  mol: {n_orb} AOs, ({n_alpha},{n_beta}) electrons, basis={basis}")

    fci = compute_fci_reference(mol, n_alpha=n_alpha, n_beta=n_beta,
                                candidate_tol=candidate_tol)
    full_pool = enumerate_pool(n_orb, n_alpha, n_beta)
    if pool_filter:
        pool = fci["candidate_set"]
        print(f"  pool: filtered FCI, |pool|={len(pool)} "
              f"(full enum would be {len(full_pool)})")
    else:
        pool = full_pool
        print(f"  pool: full enumeration, |pool|={len(pool)}")
    c_fci = np.array([fci["ci_dict"].get(I, 0.0) for I in pool])
    c_fci = c_fci / np.linalg.norm(c_fci)
    n_det = len(pool)

    # Number of "important" FCI dets (above 1e-2)
    n_signal = int(np.sum(np.abs(c_fci) > 1e-2))
    print(f"  FCI: {n_signal} determinants with |c| > 1e-2 (signal); "
          f"{n_det - n_signal} below")
    print(f"  E_HF={fci['E_HF']:.6f}, E_FCI={fci['E_FCI']:.6f}")

    # Walker bank + psi
    walkers, _, _ = load_walker_bank(str(cell_dir / f"{cell_dir.name}_walkers.h5"))
    key = jax.random.split(jax.random.key(42), 4)[1]
    drv = get_vmc_nn_func(mol, ansatz, key, prefix=str(cell_dir / cell_dir.name))
    drv.load_checkpoint(str(cell_dir / f"{cell_dir.name}.chk.h5"))
    log_psi, _, _, _ = make_nn_log_psi(ansatz, mol, key)
    psi = evaluate_signed_psi(walkers, np.asarray(mol.atom_coords()),
                              drv.params, log_psi, batch_size=2048)
    K_s = walkers.shape[0]
    no_coeff = fci["no_coeff_ao"]
    print(f"  walkers: K_s = {K_s}")

    out = dict(name=name, n_orb=n_orb, n_det=n_det, K_s=K_s,
               n_signal=n_signal, c_fci_max=float(np.max(np.abs(c_fci))))

    # ── Identity baseline ──
    t0 = time.time()
    orb = evaluate_orbitals_on_walkers(mol, walkers, no_coeff,
                                        convention="interleaved",
                                        n_alpha=n_alpha, n_beta=n_beta)
    f_I = f_I_matrix(orb, pool, psi, n_alpha, n_beta)
    c_id = np.asarray(f_I).mean(axis=1)
    c_id = c_id / np.linalg.norm(c_id)
    if c_id[0] * c_fci[0] < 0: c_id = -c_id
    err_id = float(np.linalg.norm(c_id - c_fci))
    ov_id = float(np.dot(c_id, c_fci))
    print(f"  [Identity baseline]      err={err_id:.4f}  <c|c_FCI>={ov_id:.5f}  "
          f"({time.time()-t0:.1f}s)")
    out["identity"] = dict(err=err_id, ov=ov_id, n_kept=n_det)

    # ── Lasso-CS sweep ──
    out["lasso_cs"] = []
    for m in m_values:
        if m > n_det: continue
        errs, ovs = [], []
        for s in range(3):
            try:
                c = compressed_sensing_decode(mol, walkers, psi, no_coeff, pool,
                                               n_alpha, n_beta,
                                               m=m, lam=-0.1, seed=s)
                if c[0] * c_fci[0] < 0: c = -c
                errs.append(float(np.linalg.norm(c - c_fci)))
                ovs.append(float(np.dot(c, c_fci)))
            except Exception as ex:
                errs.append(float("nan")); ovs.append(float("nan"))
        e_mean, e_std = float(np.mean(errs)), float(np.std(errs))
        o_mean = float(np.mean(ovs))
        print(f"  [Lasso-CS m={m:>4} lam=-0.1]      err={e_mean:.4f}+/-{e_std:.4f}  "
              f"<c|c_FCI>={o_mean:.5f}")
        out["lasso_cs"].append(dict(m=m, err_mean=e_mean, err_std=e_std,
                                     ov_mean=o_mean))

    # ── OMP sweep ──
    out["omp"] = []
    if omp_max_support is None:
        omp_max_support = min(n_det, 3 * max(n_signal, 10))
    for m in m_values:
        if m > n_det: continue
        for rel_tol in omp_rel_tols:
            errs, ovs, sups = [], [], []
            for s in range(3):
                try:
                    sup, coeffs, diag = omp_decode(
                        mol, walkers, psi, no_coeff, pool,
                        n_alpha, n_beta, m=m,
                        rel_residual_tol=rel_tol,
                        max_support=omp_max_support,
                        seed=s, return_diagnostics=True,
                    )
                    # Embed into full c
                    c = np.zeros(n_det)
                    idx_map = {tuple((tuple(I[0]), tuple(I[1]))): k
                               for k, I in enumerate(pool)}
                    for I, v in zip(sup, coeffs):
                        c[idx_map[tuple((tuple(I[0]), tuple(I[1])))]] = v
                    if c[0] * c_fci[0] < 0: c = -c
                    errs.append(float(np.linalg.norm(c - c_fci)))
                    ovs.append(float(np.dot(c, c_fci)))
                    sups.append(len(sup))
                except Exception as ex:
                    errs.append(float("nan"))
            e_mean, e_std = float(np.mean(errs)), float(np.std(errs))
            o_mean = float(np.mean(ovs))
            s_mean = float(np.mean(sups))
            print(f"  [OMP m={m:>4} rel_tol={rel_tol:.0e}]  err={e_mean:.4f}+/-{e_std:.4f}  "
                  f"<c|c_FCI>={o_mean:.5f}  |S|={s_mean:.0f}/{n_signal}*")
            out["omp"].append(dict(m=m, rel_tol=rel_tol,
                                    err_mean=e_mean, err_std=e_std,
                                    ov_mean=o_mean, support_size_mean=s_mean))
    return out


def main():
    results = {}

    # H2
    results["H2"] = run_one(
        name="H2 / cc-pVDZ R=2.5 a0 (PsiFormer-small)",
        geometry="h2", R=2.5, basis="cc-pvdz", unit="Bohr",
        n_alpha=1, n_beta=1,
        cell_dir=Path("cs_h2_validation/h2_R2p500_cc-pvdz"),
        ansatz="examples/inputs/psiformer_small.yaml",
        pool_filter=False,
        m_values=(20, 40, 80),
        omp_rel_tols=(1e-3, 1e-2, 5e-2),
    )

    # H4 linear
    results["H4_linear"] = run_one(
        name="H4 linear / cc-pVDZ R=1.0 A (FermiNet+J)",
        geometry="h4", R=1.0, basis="cc-pvdz", unit="Angstrom",
        geometry_tag="linear",
        n_alpha=2, n_beta=2,
        cell_dir=Path("cs_h4_results/h4_linear_ferminet_jastrow_R1p000_cc-pvdz"),
        ansatz="OmegaQMC/psi/nn/conf/ferminet_jastrow.yaml",
        pool_filter=True, candidate_tol=1e-4,
        m_values=(100, 300, 600),
        omp_rel_tols=(1e-3, 1e-2),
    )

    # BeH2 converged
    results["BeH2"] = run_one(
        name="BeH2 / cc-pVDZ converged (FermiNet+J, 5000 iter)",
        geometry="beh2", R=1.33, basis="cc-pvdz", unit="Angstrom",
        n_alpha=3, n_beta=3,
        cell_dir=Path("cs_pilot_results_converged/beh2_ferminet_jastrow_R1p330_cc-pvdz"),
        ansatz="OmegaQMC/psi/nn/conf/ferminet_jastrow.yaml",
        pool_filter=True, candidate_tol=1e-4,
        m_values=(200, 500, 1000),
        omp_rel_tols=(1e-3, 1e-2),
    )

    with open("papers/cs_recovery/data/omp_decoder_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote papers/cs_recovery/data/omp_decoder_sweep.json")


def _adjust_geom_tag(args):
    pass

if __name__ == "__main__":
    main()
