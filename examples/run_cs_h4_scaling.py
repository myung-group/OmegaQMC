"""H4 square compressed-sensing scaling experiment driver.

Loops over (R, basis) cells. Per cell: FCI reference, optional NN-VMC
training, optional VMC sampling with the dump_walkers hook, evaluation
of signed Psi at the dumped walkers, and a K_s sweep producing cells
and aux records that match the frozen
:mod:`OmegaQMC.cs.analysis` schema. After the grid finishes the script
runs :func:`run_analysis` and saves the result.

Stages
------
Per (R, basis):
    [1] FCI reference (PySCF, fast).
    [2] NN-VMC training (vmcopt_nn_sr).  Skipped if a checkpoint exists
        unless ``--retrain`` is passed.
    [3] VMC sampling with ``dump_walkers_path``.  Skipped if a walker
        bank exists unless ``--resample`` is passed.
    [4] Signed-Psi evaluation at the walker bank via
        ``log_psi.signed`` (vmapped+jit).
    [5] :func:`OmegaQMC.cs.run_sweep` -> cells + aux.

Final stage:
    [6] Concatenate cells/aux across cells, run
        :func:`OmegaQMC.cs.run_analysis`, save JSON.

Usage
-----
Pilot smoke test (H4/STO-3G, single R, fast):
    python examples/run_cs_h4_scaling.py --R 1.8 --basis sto-3g \
        --train-iters 50 --sample-blocks 20 --sample-walkers 64 \
        --K-s-sweep 50 100 200 400 --n-seeds 3

Pilot scale (H4 cc-pVDZ, full R grid):
    python examples/run_cs_h4_scaling.py \
        --R 1.0 1.4 1.8 2.2 2.6 --basis cc-pvdz \
        --train-iters 2000 --sample-blocks 200 --sample-walkers 256 \
        --K-s-sweep 100 200 400 800 1600 3200 6400 12800 \
        --n-seeds 20
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import numpy as np
from pyscf.data.elements import ELEMENTS

from OmegaQMC.utils import Mole_custom
from OmegaQMC.vmc_nn import get_vmc_nn_func
from OmegaQMC.vmcopt_nn_sr import get_vmcopt_nn_func
from OmegaQMC.psi.nn.adapter import make_nn_log_psi

from OmegaQMC.cs.analysis import run_analysis, validate_schema
from OmegaQMC.cs.reference import compute_fci_reference
from OmegaQMC.cs.scaling import run_sweep
from OmegaQMC.cs.walkers import load_walker_bank


def build_h4_mol(R_angstrom: float, basis: str, geometry: str) -> Mole_custom:
    """H4 molecule in either square or linear-chain geometry.

    For the scaling experiment the square geometry is the headline target;
    the linear chain is useful as a single-reference smoke test (square H4
    is intrinsically multi-reference at every R because of D_4h symmetry).
    """
    if geometry == "square":
        coords = [
            [0.0, 0.0, 0.0],
            [R_angstrom, 0.0, 0.0],
            [R_angstrom, R_angstrom, 0.0],
            [0.0, R_angstrom, 0.0],
        ]
    elif geometry == "linear":
        coords = [
            [0.0, 0.0, k * R_angstrom] for k in range(4)
        ]
    else:
        raise ValueError(f"geometry must be 'square' or 'linear', got {geometry!r}")
    atom_list = [("H", c) for c in coords]
    mol = Mole_custom()
    mol.build(
        atom=atom_list,
        basis=basis,
        spin=0,
        charge=0,
        unit="Angstrom",
        verbose=0,
    )
    return mol


def build_h4_square_mol(R_angstrom: float, basis: str) -> Mole_custom:
    """Backward-compatible wrapper for the square geometry."""
    return build_h4_mol(R_angstrom, basis, "square")


def evaluate_signed_psi(
    walkers: np.ndarray,
    nuc_crds: np.ndarray,
    params,
    log_psi,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute signed Psi(R) at every walker via ``log_psi.signed``.

    Returns a numpy array of shape ``(K_s,)``. The trial wavefunction
    may have node-crossings, so the sign component is essential — we
    cannot reconstruct from ``log|Psi|`` alone.
    """
    signed_per_walker = jax.vmap(
        log_psi.signed, in_axes=(0, None, None),
    )
    signed_jit = jax.jit(signed_per_walker)

    K_s = walkers.shape[0]
    nuc_j = jnp.asarray(nuc_crds)
    out = np.empty(K_s, dtype=np.float64)
    for start in range(0, K_s, batch_size):
        end = min(start + batch_size, K_s)
        batch = jnp.asarray(walkers[start:end])
        sign_b, log_amp_b = signed_jit(batch, nuc_j, params)
        psi_b = sign_b * jnp.exp(log_amp_b)
        out[start:end] = np.asarray(psi_b, dtype=np.float64)
    return out


def cell_prefix(R: float, basis: str, ansatz_tag: str, geometry: str = "square") -> str:
    geom_tag = "" if geometry == "square" else f"_{geometry}"
    return f"h4{geom_tag}_{ansatz_tag}_R{R:.3f}_{basis}".replace(".", "p")


def read_vmc_eval_from_checkpoint(path: str):
    """Return ``(E_mean, E_serr)`` from the most recent VMC eval written to
    the checkpoint by ``append_vmc_results``, or ``(nan, nan)`` if missing.

    The per-iter ``meta['energy']`` written by the SR trainer is a
    single-block noisy estimate (we saw ~50 mE_h difference vs the
    block-averaged VMC eval). The convergence gate needs the latter.
    """
    import h5py
    try:
        with h5py.File(path, "r") as f:
            if "vmc" not in f:
                return float("nan"), float("nan")
            vg = f["vmc"]
            return float(vg.attrs["E_mean"]), float(vg.attrs["E_serr"])
    except (OSError, KeyError):
        return float("nan"), float("nan")


def cells_aux_paths(out_dir: Path, prefix: str):
    return out_dir / f"{prefix}_cells.json", out_dir / f"{prefix}_aux.json"


def process_cell(
    R: float,
    basis: str,
    args,
    out_dir: Path,
) -> tuple:
    """Build refs, train if needed, sample if needed, and run the sweep.

    Returns ``(cells, aux)`` for this (R, basis) cell.
    """
    prefix = cell_prefix(R, basis, args.ansatz_tag, args.geometry)
    work_dir = out_dir / prefix
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = work_dir / f"{prefix}.chk.h5"
    bank_path = work_dir / f"{prefix}_walkers.h5"

    print(f"\n==== cell R={R} A, basis={basis}, geometry={args.geometry} ====")
    print(f"  work_dir   = {work_dir}")

    # [1] FCI reference
    mol = build_h4_mol(R, basis, args.geometry)
    print(f"  mol: {mol.nao} AOs, {mol.nelec} electrons")
    fci_ref = compute_fci_reference(
        mol, n_alpha=2, n_beta=2,
        candidate_tol=args.candidate_tol,
    )
    print(f"  E_HF  = {fci_ref['E_HF']: .8f} Ha")
    print(f"  E_FCI = {fci_ref['E_FCI']: .8f} Ha")
    print(f"  |candidate set| = {len(fci_ref['candidate_set'])}")

    # [2] NN-VMC training
    rng = jax.random.key(args.seed)
    rng, init_key, opt_key, smp_key = jax.random.split(rng, 4)
    need_train = args.retrain or not checkpoint.exists()
    if need_train:
        print(f"  [train] vmcopt_nn_sr  iters={args.train_iters}  "
              f"walkers={args.train_walkers}")
        opt = get_vmcopt_nn_func(mol, args.ansatz, init_key)
        opt(
            opt_key,
            num_iters=args.train_iters,
            num_walkers=args.train_walkers,
            lr=args.lr,
            mc_timestep=args.mc_timestep,
            jac_batch_size=args.jac_batch,
            prefix=str(work_dir / prefix),
            verbose=1,
        )
    else:
        print(f"  [train] reusing checkpoint {checkpoint.name}")

    # [3] VMC sampling with dump_walkers
    driver = get_vmc_nn_func(mol, args.ansatz, init_key, prefix=str(work_dir / prefix))
    if checkpoint.exists():
        meta = driver.load_checkpoint(str(checkpoint))
        e_train_noisy = float(meta.get("energy", float("nan")))
        print(f"  [load ] checkpoint epoch={int(meta.get('epoch', -1))}  "
              f"last-iter E={e_train_noisy:.6f} Ha (noisy, single block)")
    else:
        e_train_noisy = float("nan")
        print(f"  [load ] no checkpoint; running with random init")

    need_sample = args.resample or not bank_path.exists()
    if need_sample:
        print(f"  [sample] blocks={args.sample_blocks}  "
              f"walkers={args.sample_walkers}  "
              f"bank size={args.sample_blocks * args.sample_walkers}")
        vmc_result = driver(
            smp_key,
            num_walkers=args.sample_walkers,
            num_steps_per_block=args.sample_steps_per_block,
            num_blocks=args.sample_blocks,
            num_blocks_equil=args.sample_equil_blocks,
            mc_timestep=args.mc_timestep,
            compute_gradients=False,
            dump_walkers_path=str(bank_path),
            verbose=1,
        )
        e_train = float(vmc_result["E_mean"])
        e_serr = float(vmc_result["E_serr"])
    else:
        print(f"  [sample] reusing bank {bank_path.name}")
        # No fresh sample this invocation; read the VMC E_mean that was
        # appended to the checkpoint by the last completed sampling run.
        e_train, e_serr = read_vmc_eval_from_checkpoint(str(checkpoint))

    if np.isnan(e_train):
        # Last resort: fall back to the noisy single-iter estimate.
        e_train = e_train_noisy
        e_serr = float("nan")
        e_source = "last-iter (noisy)"
    else:
        e_source = f"VMC eval ({args.sample_blocks} blocks)"

    psi_nn_energy_error_mEh = (
        (e_train - fci_ref["E_FCI"]) * 1000.0
        if not np.isnan(e_train) else float("inf")
    )
    serr_mEh = e_serr * 1000.0 if not np.isnan(e_serr) else float("nan")
    print(f"  E_NN = {e_train:.6f} +/- {e_serr if not np.isnan(e_serr) else float('nan'):.6f} Ha "
          f"({e_source})")
    print(f"  Delta E (NN vs FCI) = {psi_nn_energy_error_mEh:+.3f} mE_h "
          f"(+/- {serr_mEh:.3f} mE_h)")

    # [4] Signed Psi at the bank
    walker_bank, _log_psi_bank, meta = load_walker_bank(str(bank_path))
    print(f"  loaded bank: walkers {walker_bank.shape}, "
          f"schema_version={meta.get('schema_version', '?')}")
    log_psi, _params0, _gdef, _ = make_nn_log_psi(args.ansatz, mol, init_key)
    psi_vals = evaluate_signed_psi(
        walker_bank, np.asarray(mol.atom_coords()),
        driver.params, log_psi,
        batch_size=args.psi_batch,
    )
    print(f"  signed Psi: |Psi| range "
          f"[{np.abs(psi_vals).min():.2e}, {np.abs(psi_vals).max():.2e}]  "
          f"sign mean = {np.mean(np.sign(psi_vals)):+.3f}")

    # [5] Sweep
    K_s_sweep = [k for k in args.K_s_sweep if k <= walker_bank.shape[0]]
    if not K_s_sweep:
        raise RuntimeError(
            f"every K_s in {args.K_s_sweep} exceeds bank size "
            f"{walker_bank.shape[0]}; increase --sample-blocks/--sample-walkers"
        )
    cells, aux = run_sweep(
        mol, fci_ref,
        walker_bank, psi_vals,
        R=R, basis=basis,
        K_s_sweep=K_s_sweep, etas=args.etas, n_seeds=args.n_seeds,
        psi_nn_energy_error=psi_nn_energy_error_mEh,
        lambda_coef=args.lambda_coef,
        det_chunk_size=args.det_chunk,
        walker_convention="interleaved",
    )
    print(f"  emitted {len(cells)} cells")
    return cells, aux


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", nargs="+", type=float,
                   default=[1.0, 1.4, 1.8, 2.2, 2.6],
                   help="H-H side lengths in Angstrom")
    p.add_argument("--basis", nargs="+", default=["cc-pvdz"],
                   help="Gaussian basis sets")
    p.add_argument("--geometry", default="square", choices=["square", "linear"],
                   help=("H4 geometry: 'square' (default, headline pilot) "
                         "or 'linear' (single-reference smoke test; square "
                         "is intrinsically multi-reference at all R)."))
    p.add_argument(
        "--ansatz",
        default=str(Path(__file__).parents[1] / "OmegaQMC" / "psi" /
                    "nn" / "conf" / "ferminet_jastrow.yaml"),
        help=(
            "NN ansatz config (builtin name like 'ferminet' or YAML "
            "path). Default: OmegaQMC/psi/nn/conf/ferminet_jastrow.yaml — "
            "FermiNet GNN backbone with a Smith-style deep Jastrow "
            "scalar head (matches Tang's PauliNet2 baseline)."
        ),
    )
    p.add_argument("--ansatz-tag", default=None,
                   help="Short tag used in filenames; defaults to ansatz stem")
    p.add_argument("--out-dir", default="cs_h4_results")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train-iters", type=int, default=2000)
    p.add_argument("--train-walkers", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--mc-timestep", type=float, default=0.1)
    p.add_argument("--jac-batch", type=int, default=4)
    p.add_argument("--retrain", action="store_true",
                   help="Force retraining even when a checkpoint exists")

    p.add_argument("--sample-blocks", type=int, default=200)
    p.add_argument("--sample-walkers", type=int, default=256)
    p.add_argument("--sample-steps-per-block", type=int, default=20)
    p.add_argument("--sample-equil-blocks", type=int, default=5)
    p.add_argument("--psi-batch", type=int, default=4096,
                   help="Batch size for vmapped signed-Psi evaluation")
    p.add_argument("--resample", action="store_true",
                   help="Force re-sampling even when a walker bank exists")

    p.add_argument("--K-s-sweep", nargs="+", type=int,
                   default=[100, 200, 400, 800, 1600, 3200, 6400, 12800])
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--etas", nargs="+", type=float,
                   default=[1e-2, 1e-3, 1e-4])
    p.add_argument("--lambda-coef", type=float, default=0.5)
    p.add_argument("--candidate-tol", type=float, default=1e-10)
    p.add_argument("--det-chunk", type=int, default=200)

    p.add_argument("--skip-analysis", action="store_true")
    args = p.parse_args()

    if args.ansatz_tag is None:
        args.ansatz_tag = os.path.splitext(
            os.path.basename(args.ansatz)
        )[0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir.resolve()}")

    all_cells: list = []
    all_aux: list = []
    for basis in args.basis:
        for R in args.R:
            cells, aux = process_cell(R, basis, args, out_dir)
            all_cells.extend(cells)
            all_aux.append(aux)
            prefix = cell_prefix(R, basis, args.ansatz_tag, args.geometry)
            cpath, apath = cells_aux_paths(out_dir, prefix)
            with open(cpath, "w") as f:
                json.dump(cells, f)
            with open(apath, "w") as f:
                json.dump(aux, f)

    combined_cells = out_dir / "all_cells.json"
    combined_aux = out_dir / "all_aux.json"
    with open(combined_cells, "w") as f:
        json.dump(all_cells, f)
    with open(combined_aux, "w") as f:
        json.dump(all_aux, f)
    print(f"\nWrote {len(all_cells)} cells -> {combined_cells}")
    print(f"Wrote {len(all_aux)} aux    -> {combined_aux}")

    if args.skip_analysis:
        print("--skip-analysis set; not running run_analysis")
        return

    print("\n==== running pre-registered analysis ====")
    validate_schema(all_cells, all_aux)
    result = run_analysis(all_cells, all_aux, n_bootstrap=2000)

    print(f"  alpha = {result.alpha:.3f}   CI95 {result.alpha_ci}")
    print(f"  beta  = {result.beta:.3f}    CI95 {result.beta_ci}")
    print(f"  gamma = {result.gamma:.3f}")
    print(f"  R^2   = {result.r_squared:.3f}")
    print(f"  cells used = {result.n_cells}")
    print(f"  flat-sparsity slope = {result.flat_sparsity_slope:.3f}")
    print(f"  gated geometries    = {result.gated_geometries}")
    print(f"  >>> regime = {result.regime} <<<")

    summary = dict(
        alpha=result.alpha, alpha_ci=list(result.alpha_ci),
        beta=result.beta, beta_ci=list(result.beta_ci),
        gamma=result.gamma, r_squared=result.r_squared,
        n_cells=result.n_cells, regime=result.regime,
        flat_sparsity_slope=result.flat_sparsity_slope,
        gated_geometries=result.gated_geometries,
    )
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {out_dir / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
