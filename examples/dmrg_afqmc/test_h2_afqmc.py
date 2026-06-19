"""
Benzene (non-QED) AFQMC using **OmegaQMC-th4** (= OmegaQMC-main + the
multi-det AFQMC features from OmegaQMC-th + single-det local_energy
walker chunking).

Configuration:
- Step 1: AFQMC with single-det HF (RHF) trial (baseline)
- Step 2: AFQMC with EDGA multi-det trial

Usage:
  python3 th4_benzene_afqmc.py --step 1
  python3 th4_benzene_afqmc.py --step 2
  python3 th4_benzene_afqmc.py            # both + comparison

OmegaQMC-th4 highlights:
- Multi-det estimator from OmegaQMC-th: chunked-scan Green's function
  + on-the-fly half-rotation, no (ndet, naux, nocc, nbasis) per-det
  rchol storage. Safe for naphthalene-scale CAS expansions.
- Single-det local_energy now walker-chunked so the
  (naux, nwalkers, nocc, nocc) exchange tensor in local_energy_2body
  does not OOM at large nwalkers (set AFQMC_WALKER_CHUNK_SIZE).
- All OmegaQMC-main memory features preserved: DiskChol (chol_h5_path),
  walker x g VHS chunking, build_propagator g-chunking.
"""

import sys
import os
from datetime import datetime

# Use OmegaQMC-th4 (OmegaQMC-main + th's multi-det AFQMC + local_energy chunking)
_local_omegaqmc = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'OmegaQMC-th4')
if os.path.isdir(_local_omegaqmc):
    sys.path.insert(0, _local_omegaqmc)

# edga2omegaqmc lives next to this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0')

import time
import numpy as np
import jax

from pyscf import gto, scf


class _Tee:
    """Mirror stdout writes to an extra file handle (tee-style)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _enable_logging(step):
    """Tee stdout to {step}_{timestamp}.log in cwd."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = f"{step}_{ts}.log"
    log_handle = open(log_name, 'w', buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_handle)
    print(f"[log] tee -> {log_name}")
    return log_handle


print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
n_dev = len(jax.devices())

# ============================================================
# Configuration
# ============================================================
MOLDEN_FILE = os.path.expanduser("./dump.molden.input")
EDGA_FILE = os.path.expanduser("./edga_result.dat")

# AFQMC parameters
AFQMC_DT = 0.005
AFQMC_CHOL_CUT = 1e-6
AFQMC_NUM_WALKERS = 1600        # None -> auto-tune via _autotune_afqmc_walkers
AFQMC_NUM_BLOCKS = 100
AFQMC_NUM_STEPS_PER_BLOCK = 200
AFQMC_STABILIZE_FREQ = 5
AFQMC_POP_CONTROL_FREQ = 5
AFQMC_NUM_BLOCKS_EQUIL = 10    # main-style equilibration blocks

# Multi-det trial parameters (Step 2 only)
AFQMC_MAX_DET = 222
AFQMC_COEFF_THRESHOLD = 1e-4
AFQMC_DET_CHUNK_SIZE = 6       # dets per chunked-scan step (th-style)

# OmegaQMC-th4 memory chunking knobs.
# walker_chunk_size applies to BOTH propagate_walkers (VHS slab) AND
# the single-det end-of-block local_energy (exchange-tensor slab).
# None = no chunking; set a value like 500 if 5000+ walkers OOM.
AFQMC_WALKER_CHUNK_SIZE = None  # walkers per chunk
AFQMC_CHOL_CHUNK_G = None       # g-axis slab size (main default)
AFQMC_CHOL_H5_PATH = None       # disk-backed Cholesky HDF5 file

# Benzene geometry (Angstrom)
GEOM = '''
H -9.000000  0.000000  0.000000
H -7.000000  0.000000  0.000000
H -5.000000  0.000000  0.000000
H -3.000000  0.000000  0.000000
H -1.000000  0.000000  0.000000
H  1.000000  0.000000  0.000000
H  3.000000  0.000000  0.000000
H  5.000000  0.000000  0.000000
H  7.000000  0.000000  0.000000
H  9.000000  0.000000  0.000000
'''


def _resolve_num_walkers(driver):
    if AFQMC_NUM_WALKERS is not None:
        return AFQMC_NUM_WALKERS
    try:
        from OmegaQMC.afqmc_gto import _autotune_afqmc_walkers
        from OmegaQMC.vmcopt_gto_linear import _get_free_gpu_mb
        free_mb = _get_free_gpu_mb()
    except Exception:
        print("  WARNING: auto-tune unavailable, using 200 walkers")
        return 200
    n_rec, bpw = _autotune_afqmc_walkers(driver, free_mb)
    n_rec = max(n_dev, (n_rec // n_dev) * n_dev)
    print(f"  Auto-recommended walkers: {n_rec} "
          f"({bpw / 1e6:.2f} MB/walker, {free_mb:.0f} MiB free)")
    return n_rec


def _run_afqmc_driver(driver):
    num_walkers = _resolve_num_walkers(driver)
    print(f"\n  AFQMC parameters (OmegaQMC-th4):")
    print(f"  dt = {AFQMC_DT}, chol_cut = {AFQMC_CHOL_CUT}")
    print(f"  num_walkers = {num_walkers}, num_blocks = {AFQMC_NUM_BLOCKS}")
    print(f"  num_steps_per_block = {AFQMC_NUM_STEPS_PER_BLOCK}")
    print(f"  num_blocks_equil = {AFQMC_NUM_BLOCKS_EQUIL}")
    if AFQMC_WALKER_CHUNK_SIZE is not None:
        print(f"  walker_chunk_size = {AFQMC_WALKER_CHUNK_SIZE} "
              f"(applies to VHS apply AND end-of-block local_energy)")
    if AFQMC_CHOL_CHUNK_G is not None:
        print(f"  chol_chunk_g = {AFQMC_CHOL_CHUNK_G} "
              f"(g-axis slab size)")
    if AFQMC_CHOL_H5_PATH is not None:
        print(f"  chol on disk: {AFQMC_CHOL_H5_PATH}")
    t_start = time.time()
    result = driver(
        rng_key=jax.random.key(42),
        num_walkers=num_walkers,
        num_blocks=AFQMC_NUM_BLOCKS,
        num_steps_per_block=AFQMC_NUM_STEPS_PER_BLOCK,
        stabilize_freq=AFQMC_STABILIZE_FREQ,
        pop_control_freq=AFQMC_POP_CONTROL_FREQ,
        num_blocks_equil=AFQMC_NUM_BLOCKS_EQUIL,
        walker_chunk_size=AFQMC_WALKER_CHUNK_SIZE,
    )
    elapsed = time.time() - t_start
    return result, elapsed


def _print_results(label, result, elapsed, extra_lines=None):
    print(f"\n{'=' * 70}")
    print(f"{label} Results:")
    print(f"  E_HF    = {result['ehf']:.10f}")
    print(f"  E_AFQMC = {result['energy_mean']:.10f} "
          f"+/- {result['energy_err']:.10f}")
    print(f"  E_corr  = {result['energy_mean'] - result['ehf']:.10f}")
    if extra_lines:
        for line in extra_lines:
            print(f"  {line}")
    print(f"  Elapsed: {elapsed:.1f} s ({elapsed/60:.2f} min)")
    print('=' * 70)


# ============================================================
# Step 1: HF trial (baseline)
# ============================================================
def run_hf_afqmc():
    from OmegaQMC import get_afqmc_func

    print("=" * 70)
    print("Step 1: Benzene AFQMC with HF trial (OmegaQMC-th4)")
    print("=" * 70)

    mol = gto.M(atom=GEOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    print(f"  E_HF (PySCF) = {mf.e_tot:.10f}")
    print(f"  nao = {mol.nao}, nelectron = {mol.nelectron}")

    driver = get_afqmc_func(
        mf, dt=AFQMC_DT, chol_cut=AFQMC_CHOL_CUT,
        verbose=True,
        chol_h5_path=AFQMC_CHOL_H5_PATH,
        chol_chunk_g=AFQMC_CHOL_CHUNK_G,
        # trial=None -> single-det HF
    )
    result, elapsed = _run_afqmc_driver(driver)
    _print_results("Step 1 (HF)", result, elapsed)
    return result


# ============================================================
# Step 2: EDGA multi-det trial
# ============================================================
def run_edga_afqmc():
    from OmegaQMC import get_afqmc_func
    from pyscf.tools import molden as molden_tools
    from edga2omegaqmc import load_edga_result, make_omegaqmc_trial

    coeff_threshold = AFQMC_COEFF_THRESHOLD
    max_det = AFQMC_MAX_DET

    print("=" * 70)
    print("Step 2: Benzene AFQMC with EDGA multi-det trial (OmegaQMC-th4)")
    print(f"  coeff_threshold = {coeff_threshold}")
    if max_det:
        print(f"  max_det = {max_det}")
    print("=" * 70)

    mol = gto.M(atom=GEOM, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    print(f"  E_HF (PySCF) = {mf.e_tot:.10f}")

    print(f"  Loading ORCA MOs from: {MOLDEN_FILE}")
    molden_data = molden_tools.load(MOLDEN_FILE)
    if len(molden_data) >= 3:
        mo_coeff_orca = molden_data[2]
    else:
        raise RuntimeError("Cannot extract MO coefficients from molden file")
    print(f"  ORCA MO shape: {mo_coeff_orca.shape}")
    if mo_coeff_orca.shape != mf.mo_coeff.shape:
        print(f"  WARNING: MO shapes differ! ORCA={mo_coeff_orca.shape}, "
              f"PySCF={mf.mo_coeff.shape}")
    mf.mo_coeff = mo_coeff_orca

    print(f"  Loading EDGA results from: {EDGA_FILE}")
    dets, coeffs, meta = load_edga_result(
        EDGA_FILE, coeff_threshold=coeff_threshold)
    if max_det and len(dets) > max_det:
        order = np.argsort(-np.abs(np.array(coeffs)))[:max_det]
        dets = [dets[i] for i in order]
        coeffs = [coeffs[i] for i in order]

    print(f"  Determinants loaded: {len(dets)}")
    if meta:
        for k, v in meta.items():
            print(f"  {k} = {v}")
    sum_c2 = sum(c**2 for c in coeffs)
    print(f"  sum(c^2) = {sum_c2:.6f}")

    n_active_orbs = len(dets[0]['occ'])
    n_active_els = len(dets[0]['alpha']) + len(dets[0]['beta'])
    n_core = (mol.nelectron - n_active_els) // 2
    print(f"  n_core = {n_core}, n_active_orbs = {n_active_orbs}, "
          f"n_active_els = {n_active_els}")

    trial = make_omegaqmc_trial(coeffs, dets, mf.mo_coeff, n_core=n_core)
    print(f"  Trial: ndet={trial['ndet']}, "
          f"occ_up shape={trial['occ_up'].shape}, "
          f"occ_dn shape={trial['occ_dn'].shape}")
    print(f"  Det 0 occ_up: {trial['occ_up'][0]}")
    print(f"  Det 0 occ_dn: {trial['occ_dn'][0]}")

    # OmegaQMC-th4 uses the chunked-scan multi-det estimator (th-style):
    # per-det Ghalf is never stored, half-rotated Cholesky vectors are
    # built on-the-fly inside the scan. Memory scales as
    # det_chunk_size * naux * nocc * nbasis instead of
    # ndet * naux * nocc * nbasis.
    #
    # NOTE: chol_h5_path (DiskChol) is incompatible with multi-det
    # because the on-the-fly half-rotation uses direct einsum on chol.
    # For multi-det runs, keep AFQMC_CHOL_H5_PATH = None.
    if AFQMC_CHOL_H5_PATH is not None:
        print("  WARNING: chol_h5_path is set but ignored for multi-det "
              "(falling back to in-memory chol).")
    driver = get_afqmc_func(
        mf, dt=AFQMC_DT, chol_cut=AFQMC_CHOL_CUT,
        verbose=True, trial=trial,
        det_chunk_size=AFQMC_DET_CHUNK_SIZE,
        chol_h5_path=None,  # multi-det + DiskChol not supported
        chol_chunk_g=AFQMC_CHOL_CHUNK_G,
    )
    result, elapsed = _run_afqmc_driver(driver)
    extra = [f"Trial: {len(dets)} dets, sum(c^2) = {sum_c2:.6f}"]
    _print_results("Step 2 (EDGA MSD)", result, elapsed, extra_lines=extra)
    return result


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    step = 0
    for arg in sys.argv[1:]:
        if arg == '--step':
            idx = sys.argv.index('--step')
            step = int(sys.argv[idx + 1])
        elif arg.startswith('--step='):
            step = int(arg.split('=')[1])

    _log_handle = _enable_logging(step)

    if step == 0 or step == 1:
        result_hf = run_hf_afqmc()

    if step == 0 or step == 2:
        if not os.path.exists(MOLDEN_FILE):
            print(f"ERROR: Molden file not found: {MOLDEN_FILE}")
            sys.exit(1)
        if not os.path.exists(EDGA_FILE):
            print(f"ERROR: EDGA result file not found: {EDGA_FILE}")
            sys.exit(1)
        result_edga = run_edga_afqmc()

    if step == 0:
        print("\n" + "=" * 70)
        print("Comparison (OmegaQMC-th4): HF trial vs EDGA multi-det trial")
        print("=" * 70)
        print(f"  E_AFQMC (HF):   {result_hf['energy_mean']:.10f} "
              f"+/- {result_hf['energy_err']:.10f}")
        print(f"  E_AFQMC (EDGA): {result_edga['energy_mean']:.10f} "
              f"+/- {result_edga['energy_err']:.10f}")
        diff = result_edga['energy_mean'] - result_hf['energy_mean']
        sigma = (result_hf['energy_err'] ** 2
                 + result_edga['energy_err'] ** 2) ** 0.5
        print(f"  Diff (EDGA - HF):  {diff*1000:.3f} mHa  "
              f"(combined sigma = {sigma*1000:.3f} mHa)")
        print("=" * 70)

