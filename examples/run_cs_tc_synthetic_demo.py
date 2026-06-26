"""
Synthetic end-to-end Gate-0 demo (no NN training, no GPU).

Constructs a *controlled* trial  Psi_NN(R) = Phi_det(R) * exp(J(R))  where
Phi_det is the exact FCI vector of a small molecule (so we know the answer)
and J is a bounded pairwise log-Jastrow that sprays amplitude onto the
determinant tail -- the controlled stand-in for basis-incompleteness leakage.
It then:

  1. Metropolis-samples |Psi_NN|^2,
  2. writes a *real* walker bank via WalkerDumper including the /jastrow
     dataset (schema 1.1.0) -- the exact on-disk format the NN driver emits,
  3. reads the bank back, evaluates orbitals + signed Psi + reads J,
  4. runs the two-sided TC decode and prints the Gate-0 report.

Expected: the tau=+1 (TC right) decode recovers the sparse FCI pattern
(leakage ~ 0, overlap ~ 1) while the tau=0 Hermitian decode carries the
e^{J} leakage -- i.e. a clean leakage collapse, on a file in the real
bank format. Use it to eyeball the output before plugging real NN data
into examples/run_cs_tc_decode.py.

    python examples/run_cs_tc_synthetic_demo.py --R 1.2 --basis 6-31g
"""

import argparse
from pathlib import Path

import numpy as np
from pyscf import gto

from OmegaQMC.cs.walkers import WalkerDumper, load_walker_bank
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.estimators import (
    evaluate_ci_wavefunction,
    evaluate_orbitals_on_walkers,
)
from OmegaQMC.cs.transcorrelated import decode_two_sided, print_gate0_report


def jastrow(walkers, amp):
    """Bounded smooth pairwise log-Jastrow J(R) = -amp * sum_{i<j} exp(-r_ij^2/2)."""
    w = np.asarray(walkers)
    K, N, _ = w.shape
    J = np.zeros(K)
    for i in range(N):
        for j in range(i + 1, N):
            rij2 = np.sum((w[:, i, :] - w[:, j, :]) ** 2, axis=1)
            J += np.exp(-0.5 * rij2)
    return -amp * J


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=1.2,
                   help="H-H bond length in Angstrom")
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--jastrow-amp", type=float, default=0.6)
    p.add_argument("--n-walkers", type=int, default=4000)
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--step-size", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="cs_tc_synth/synth_walkers.h5")
    args = p.parse_args()

    # H2: 1 alpha + 1 beta. Small basis keeps the candidate set tiny but
    # gives a virtual-orbital tail for the leakage to land on.
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {args.R}", basis=args.basis,
                unit="Angstrom", verbose=0)
    na = nb = 1

    print(f"[1] FCI reference  H2/{args.basis}  R={args.R} A")
    fci = compute_fci_reference(mol, n_alpha=na, n_beta=nb, candidate_tol=0.0)
    candidate = fci["candidate_set"]
    c_true = np.array([fci["ci_dict"][k] for k in candidate])
    no_coeff = fci["no_coeff_ao"]
    n_orb = no_coeff.shape[1]
    print(f"    |candidate set| = {len(candidate)}  n_orb = {n_orb}  "
          f"E_FCI = {fci['E_FCI']:.6f} Ha")

    # Phi_det(R) = exact FCI; Psi_NN = Phi_det * e^{J}.
    def phi_det(R):
        orb = evaluate_orbitals_on_walkers(mol, np.asarray(R), no_coeff)
        return evaluate_ci_wavefunction(orb, candidate, c_true, na, nb)

    def psi_nn(R):
        return phi_det(R) * np.exp(jastrow(R, args.jastrow_amp))

    # [2] Metropolis sample |Psi_NN|^2 (grouped layout: elec 0 = a, 1 = b).
    print(f"[2] sampling |Psi_NN|^2  "
          f"({args.n_blocks} blocks x {args.n_walkers} walkers)")
    rng = np.random.default_rng(args.seed)
    R = rng.normal(scale=1.0, size=(args.n_walkers, na + nb, 3))
    psi = psi_nn(R)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    burn = 200
    with WalkerDumper(str(out_path), num_blocks=args.n_blocks,
                      num_walkers=args.n_walkers, nelec=na + nb,
                      mc_timestep=args.step_size) as dumper:
        step = 0
        for blk in range(args.n_blocks + 1):  # +1 warmup block, not written
            for _ in range(burn if blk == 0 else args.n_steps):
                for e in range(na + nb):
                    prop = R.copy()
                    prop[:, e, :] += rng.normal(scale=args.step_size,
                                                size=(args.n_walkers, 3))
                    psi_new = psi_nn(prop)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(np.abs(psi) > 1e-30,
                                         (psi_new / psi) ** 2, 1.0)
                    acc = rng.uniform(size=args.n_walkers) < ratio
                    R = np.where(acc[:, None, None], prop, R)
                    psi = np.where(acc, psi_new, psi)
                    step += 1
            if blk == 0:
                continue  # discard warmup
            log_psi = np.log(np.abs(psi) + 1e-300)
            J = jastrow(R, args.jastrow_amp)
            dumper.write_block(R.astype("f4"), log_psi, jastrow=J)
    print(f"    wrote {out_path}")

    # [3] Read the bank back (real on-disk round trip).
    walker_bank, _lp, meta = load_walker_bank(str(out_path))
    print(f"[3] reloaded bank  shape={walker_bank.shape}  "
          f"schema={meta.get('schema_version')}  "
          f"has /jastrow: {'jastrow' in meta}")
    jastrow_vals = np.asarray(meta["jastrow"])

    # Signed Psi at the bank walkers (analytic for the synthetic trial).
    psi_vals = psi_nn(walker_bank)
    orb = evaluate_orbitals_on_walkers(mol, walker_bank, no_coeff)

    # [4] Two-sided decode + Gate-0 report.
    ref_sign = float(np.sign(c_true[0])) if c_true[0] != 0 else 1.0
    out = decode_two_sided(
        orb, candidate, psi_vals, jastrow_vals,
        n_alpha=na, n_beta=nb, reference_sign=ref_sign, use_lasso=False)
    print_gate0_report(candidate, c_true, out, top_k=min(8, len(candidate)))


if __name__ == "__main__":
    main()
