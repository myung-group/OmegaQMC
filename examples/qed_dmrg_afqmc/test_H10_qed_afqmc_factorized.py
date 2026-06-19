"""
H10 QED-AFQMC with the DMRG-sourced FACTORIZED trial.

Copy this script (and ``qed_dmrg2omegaqmc_factorized.py``) into each λ
subdirectory along with ``orbitals.molden`` and ``edga_result.dat``.
Edit ``LAMBDA`` (and any other top-of-file constant if needed), then run:

    python test_H10_qed_afqmc_factorized.py            # both steps
    python test_H10_qed_afqmc_factorized.py --step 1   # HF trial only
    python test_H10_qed_afqmc_factorized.py --step 2   # MSD trial only

stdout is tee'd to ``h10_qed_afqmc_step{N}_{timestamp}.log`` in cwd.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime
import numpy as np

# ============================================================
#  EDIT THESE PER DIRECTORY
# ============================================================
LAMBDA      = 0.05       
POL_AXIS    = 'x'        
OMEGA       = 0.02516026 
MOLDEN_PATH = 'orbitals.molden'
EDGA_PATH   = 'edga_result.dat'

# ---- System (rarely changes for H10 1D chain) ----
BASIS       = 'cc-pvdz'
NCAS        = 10
NELECAS     = (5, 5)
N_CORE      = 0
H10_GEOM    = ';'.join(f"H {2*i - 9:.6f} 0.0 0.0" for i in range(10))

# ---- AFQMC parameters ----
DT                  = 0.005
CHOL_CUT            = 1e-6
NUM_WALKERS         = 1600
NUM_BLOCKS          = 100
NUM_STEPS_PER_BLOCK = 200
STABILIZE_FREQ      = 5
POP_CONTROL_FREQ    = 5
COEFF_THRESHOLD     = 1e-4
MAX_DET             = 256
DET_CHUNK_SIZE      = 10
WALKER_CHUNK_SIZE   = None
SEED                = 42


# ============================================================
#  Setup
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def _find_omegaqmc_root():
    def _ok(root):
        return os.path.isfile(os.path.join(
            root, 'OmegaQMC', 'addons', 'qed_casscf.py'))
    env = os.environ.get('OMEGAQMC_DIR')
    if env and _ok(env):
        return env
    for c in ('OmegaQMC-th5', 'OmegaQMC-sh-new', 'OmegaQMC-main',
              'OmegaQMC'):
        cur = _HERE
        for _ in range(8):
            cand = os.path.join(cur, c)
            if _ok(cand):
                return cand
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return None


_OMEGAQMC_ROOT = _find_omegaqmc_root()
if _OMEGAQMC_ROOT is None:
    raise RuntimeError(
        "Set $OMEGAQMC_DIR to the OmegaQMC root containing "
        "OmegaQMC/qed_afqmc_gto.py (e.g. /home/park/prog/OmegaQMC-th5).")
sys.path.insert(0, _OMEGAQMC_ROOT)


# ============================================================
#  Logging (tee stdout to a timestamped log file)
# ============================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def _enable_logging(step):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = f"{step}_{ts}.log"
    log_handle = open(log_name, 'w', buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_handle)
    print(f"[log] tee -> {log_name}")
    return log_handle


# ============================================================
#  Helpers
# ============================================================
def _load_localized_mo(path):
    from pyscf.tools import molden as mt
    _, _, c, occ, _, _ = mt.load(path)
    if isinstance(c, tuple):
        ca, cb = np.asarray(c[0]), np.asarray(c[1])
        d = float(np.max(np.abs(ca - cb)))
        if d > 1e-6:
            print(f"  [molden] WARNING: |C_a - C_b|_max = {d:.3e} (>1e-6); "
                  f"localized RHF should give 0. Using alpha block.")
        mo = ca.astype(np.float64)
        oc = np.asarray(occ[0], dtype=np.float64)
    else:
        mo = np.asarray(c, dtype=np.float64)
        oc = np.asarray(occ, dtype=np.float64)
    print(f"  [molden] {path}: shape {mo.shape}, sum(occ)={oc.sum():.4f}")
    return mo, oc


def _run_qed_afqmc(mf, omega, coupling_vec, trial, label):
    import jax
    from OmegaQMC import get_qed_afqmc_func

    driver = get_qed_afqmc_func(
        mf, omega, coupling_vec,
        dt=DT, chol_cut=CHOL_CUT,
        s=1.0, q0='auto',
        verbose=True,
        use_qed_hf=(trial is None),
        trial=trial,
        det_chunk_size=DET_CHUNK_SIZE,
    )
    print(f"\n[{label}] starting QED-AFQMC...")
    t0 = time.time()
    result = driver(
        rng_key=jax.random.key(SEED),
        num_walkers=NUM_WALKERS,
        num_blocks=NUM_BLOCKS,
        num_steps_per_block=NUM_STEPS_PER_BLOCK,
        stabilize_freq=STABILIZE_FREQ,
        pop_control_freq=POP_CONTROL_FREQ,
        num_blocks_equil=max(NUM_BLOCKS // 10, 10),
        walker_chunk_size=WALKER_CHUNK_SIZE,
    )
    dt_min = (time.time() - t0) / 60.0
    print(f"[{label}] E = {result['energy_mean']:+.10f} "
          f"+/- {result['energy_err']:.10f}    "
          f"<q> = {result['q_mean']:+.6f}    "
          f"E_ph = {result['e_photon_mean']:+.6f}    "
          f"({dt_min:.2f} min)")
    return result


def _setup_mf_with_localized_mo():
    """PySCF Mol + RHF + override mo_coeff from psi4 molden."""
    from pyscf import gto, scf
    mol = gto.M(atom=H10_GEOM, basis=BASIS, unit='Angstrom',
                symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    print(f"  PySCF nao={mol.nao}, nelec={mol.nelectron}, "
          f"E_RHF={mf.e_tot:.10f}")
    mo_loc, occ_loc = _load_localized_mo(MOLDEN_PATH)
    if mo_loc.shape != mf.mo_coeff.shape:
        raise RuntimeError(
            f"MO shape mismatch: molden {mo_loc.shape} vs "
            f"PySCF {mf.mo_coeff.shape}.")
    mf.mo_coeff = mo_loc
    mf.mo_occ = occ_loc
    return mol, mf


# ============================================================
#  Step 1: HF trial * Gaussian photon
# ============================================================
def run_hf_trial(coupling_vec):
    print("=" * 70)
    print("Step 1: H10 QED-AFQMC with single-det HF trial")
    print("=" * 70)
    mol, mf = _setup_mf_with_localized_mo()
    return _run_qed_afqmc(mf, OMEGA, coupling_vec, trial=None, label='HF')


# ============================================================
#  Step 2: MSD (DMRG-factorized) trial
# ============================================================
def run_msd_trial(coupling_vec):
    print("=" * 70)
    print("Step 2: H10 QED-AFQMC with DMRG-factorized MSD trial")
    print("=" * 70)
    mol, mf = _setup_mf_with_localized_mo()
    from qed_dmrg2omegaqmc_factorized import (
        build_factorized_trial_from_qed_dmrg_file,
    )
    trial = build_factorized_trial_from_qed_dmrg_file(
        mol=mol, mo_coeff=mf.mo_coeff,
        omega=OMEGA, coupling_vec=coupling_vec,
        n_core=N_CORE, ncas=NCAS, nelecas=NELECAS,
        edga_path=EDGA_PATH,
        coeff_threshold=COEFF_THRESHOLD,
        max_det=MAX_DET,
        verbose=True,
    )
    r = _run_qed_afqmc(mf, OMEGA, coupling_vec, trial=trial, label='DMRG-fact')
    return r, trial


# ============================================================
#  Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--step', type=int, choices=(0, 1, 2), default=0,
                   help='0=both (default), 1=HF only, 2=MSD only')
    args = p.parse_args()
    step = args.step

    _enable_logging(step)
    print(f"[setup] OmegaQMC root: {_OMEGAQMC_ROOT}")

    pol_vec = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}[POL_AXIS]
    coupling_vec = tuple(LAMBDA * c for c in pol_vec)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print("=" * 70)
    print(f"H10 QED-AFQMC (factorized trial)   step={step}   {ts}")
    print("=" * 70)
    print(f"  lambda={LAMBDA}, pol={POL_AXIS}, omega={OMEGA:.8f}")
    print(f"  coupling_vec={coupling_vec}")
    print(f"  CAS{NELECAS},{NCAS}; basis={BASIS}")
    print(f"  AFQMC: dt={DT}, walkers={NUM_WALKERS}, blocks={NUM_BLOCKS}, "
          f"seed={SEED}")
    print(f"  molden={MOLDEN_PATH}")
    print(f"  edga  ={EDGA_PATH}")
    for need_edga, p_in in ((step in (0, 2), EDGA_PATH),
                            (True,           MOLDEN_PATH)):
        if need_edga and not os.path.isfile(p_in):
            raise FileNotFoundError(f"Missing input file: {p_in}")

    results = {}
    trial = None
    if step in (0, 1):
        results['HF'] = run_hf_trial(coupling_vec)
    if step in (0, 2):
        results['DMRG'], trial = run_msd_trial(coupling_vec)

    # Summary
    print("\n" + "=" * 70)
    print(f"H10 QED-AFQMC summary (lambda={LAMBDA}, pol={POL_AXIS}, "
          f"step={step})")
    print("=" * 70)
    for k, r in results.items():
        print(f"  {k:10s}  E = {r['energy_mean']:+.10f} "
              f"+/- {r['energy_err']:.10f}    "
              f"<q> = {r['q_mean']:+.6f}    "
              f"E_ph = {r['e_photon_mean']:+.6f}")

    # λ=0 sanity gates
    if LAMBDA == 0.0 and trial is not None:
        d = trial['_diagnostics']
        print()
        print(f"  [lambda=0 sanity]")
        print(f"    q0 from converter   = {d['q0']:+.3e}   "
              f"(must be exactly 0)")
        print(f"    EDGA m=0 sector wt  = {d['edga_sum_c2_m0']:.10f}   "
              f"(must equal fidelity^2 from summary.dat)")
        if abs(d['q0']) > 1e-12:
            print("    -> q0 != 0 at lambda=0: lambda scaling broken")
        if 'HF' in results and 'DMRG' in results:
            de = results['DMRG']['energy_mean'] - results['HF']['energy_mean']
            se = (results['HF']['energy_err']**2
                  + results['DMRG']['energy_err']**2) ** 0.5
            print(f"    DMRG - HF           = {de*1000:+.3f} mHa "
                  f"(combined σ = {se*1000:.3f} mHa) — should be NEGATIVE")
    print('=' * 70)


if __name__ == '__main__':
    main()

