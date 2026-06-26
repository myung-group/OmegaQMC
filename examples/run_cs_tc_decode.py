"""
Gate-0 driver: transcorrelated two-sided decode on a trained NN-VMC bank.

Runs the cusp-removed decode (:mod:`OmegaQMC.cs.transcorrelated`) on an
existing walker bank + trained ansatz and prints the Gate-0 decision
artifact:

  * leakage collapse  M_spurious(tau=0)  ->  M_spurious(tau=+1)
        the spurious squared mass on determinants FCI calls negligible;
        the headline test of whether folding in the Jastrow removes the
        basis-incompleteness leakage.
  * biorthogonal overlap rho = <c_L|c_R>/(||c_L|| ||c_R||)
        conditioning / health of the non-Hermitian TC pair.
  * top-K amplitude preservation against the FCI reference.

The Jastrow J(R) per walker is read from the bank when present (schema
>= 1.1.0) or re-evaluated from the trained ansatz (retrofit path).

Example
-------
    python examples/run_cs_tc_decode.py \
        --mol h4_linear --basis cc-pvdz \
        --bank runs/h4/h4_walkers.h5 \
        --ansatz examples/inputs/ferminet_jastrow.yaml \
        --chk runs/h4/h4.chk.h5
"""

import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from OmegaQMC.cs.walkers import load_walker_bank
from OmegaQMC.cs.estimators import evaluate_orbitals_on_walkers
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.transcorrelated import decode_two_sided, print_gate0_report
from OmegaQMC.cs.jastrow_extract import nn_jastrow_on_walkers
from OmegaQMC.psi.nn.adapter import make_nn_log_psi


def _evaluate_signed_psi(walkers, nuc, params, log_psi, batch_size=2048,
                         jit=True):
    signed_v = jax.vmap(log_psi.signed, in_axes=(0, None, None))
    signed_fn = jax.jit(signed_v) if jit else signed_v
    w = jnp.asarray(walkers)
    nuc_j = jnp.asarray(nuc)
    out = np.empty(w.shape[0], dtype=float)
    for s in range(0, w.shape[0], batch_size):
        b = w[s:s + batch_size]
        sign_b, log_b = signed_fn(b, nuc_j, params)
        out[s:s + b.shape[0]] = np.asarray(sign_b) * np.exp(np.asarray(log_b))
    return out


def _build_mol(args):
    # The NN builder needs OmegaQMC's Mole_custom (adds n_up/n_down);
    # a raw pyscf Mole lacks those attributes.
    from OmegaQMC.utils import Mole_custom
    if not args.atom:
        raise SystemExit("provide --atom \"H 0 0 0; H 0 0 1.0; ...\"")
    mol = Mole_custom()
    mol.build(atom=args.atom, basis=args.basis, spin=args.spin,
              charge=args.charge, unit=args.unit, verbose=0)
    return mol


def _get_jastrow(bank_meta, walker_bank, mol, args, params, init_key):
    if "jastrow" in bank_meta:
        print("    J source: bank /jastrow dataset (schema >= 1.1.0)")
        return np.asarray(bank_meta["jastrow"])
    print("    J source: retrofit re-evaluation (jastrow-off sibling)")
    return nn_jastrow_on_walkers(
        args.ansatz, mol, init_key, params, walker_bank,
        np.asarray(mol.atom_coords()), batch_size=args.psi_batch,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atom", required=True,
                   help='PySCF atom string, e.g. "H 0 0 0; H 0 0 1.0"')
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--unit", default="Angstrom")
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--n-alpha", type=int, required=True)
    p.add_argument("--n-beta", type=int, required=True)
    p.add_argument("--bank", required=True, help="walker-bank HDF5 path")
    p.add_argument("--ansatz", required=True, help="ansatz YAML")
    p.add_argument("--chk", required=True, help="trained checkpoint HDF5")
    p.add_argument("--prefix", default=None,
                   help="vmc driver prefix (defaults to chk stem)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--psi-batch", type=int, default=2048)
    p.add_argument("--leak-thresh", type=float, default=1e-3)
    p.add_argument("--lam-mult", type=float, default=0.5)
    p.add_argument("--no-lasso", action="store_true",
                   help="plain L2 decode (no soft-threshold)")
    p.add_argument("--top-k", type=int, default=8)
    args = p.parse_args()

    na, nb = args.n_alpha, args.n_beta
    mol = _build_mol(args)

    # [1] FCI reference (candidate set + c_true for leakage / overlap).
    print("[1] FCI reference")
    fci_ref = compute_fci_reference(mol, n_alpha=na, n_beta=nb,
                                    candidate_tol=0.0)
    candidate = fci_ref["candidate_set"]
    c_true = np.array([fci_ref["ci_dict"][k] for k in candidate])
    ref_sign = float(np.sign(c_true[0])) if c_true[0] != 0 else 1.0
    print(f"    |candidate set| = {len(candidate)}  "
          f"E_FCI = {fci_ref['E_FCI']:.6f} Ha")

    # [2] Load bank + signed Psi.
    print("[2] walker bank + trained ansatz")
    walker_bank, _lp, bank_meta = load_walker_bank(args.bank)
    print(f"    bank shape={walker_bank.shape}  "
          f"schema={bank_meta.get('schema_version', '?')}")
    init_key = jax.random.key(args.seed)
    log_psi, _p0, _g, _ = make_nn_log_psi(args.ansatz, mol, init_key)

    # Load trained params via the vmc driver (mirrors validate_cs_h2).
    from OmegaQMC.vmc_nn import get_vmc_nn_func
    prefix = args.prefix or str(Path(args.chk).with_suffix(""))
    driver = get_vmc_nn_func(mol, args.ansatz, init_key, prefix=prefix)
    driver.load_checkpoint(args.chk)
    params = driver.params

    psi_vals = _evaluate_signed_psi(
        walker_bank, np.asarray(mol.atom_coords()), params, log_psi,
        batch_size=args.psi_batch)

    # [3] Jastrow per walker.
    print("[3] Jastrow extraction")
    jastrow_vals = _get_jastrow(bank_meta, walker_bank, mol, args, params,
                                init_key)
    print(f"    J range [{jastrow_vals.min():+.4f}, {jastrow_vals.max():+.4f}]"
          f"  mean {jastrow_vals.mean():+.4f}")

    # [4] Orbitals (interleaved NN layout) + two-sided decode.
    print("[4] two-sided TC decode (tau = 0, +1, -1)")
    orb = evaluate_orbitals_on_walkers(
        mol, walker_bank, fci_ref["no_coeff_ao"],
        convention="interleaved", n_alpha=na, n_beta=nb)
    out = decode_two_sided(
        orb, candidate, psi_vals, jastrow_vals,
        n_alpha=na, n_beta=nb, reference_sign=ref_sign,
        lam_mult=args.lam_mult, use_lasso=not args.no_lasso)

    # [5] Gate-0 decision artifact.
    print_gate0_report(candidate, c_true, out,
                       leak_thresh=args.leak_thresh, top_k=args.top_k)


if __name__ == "__main__":
    main()
