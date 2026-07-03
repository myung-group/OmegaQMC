"""
Unified QED-CCSD driver.

Reference: https://pubs.acs.org/doi/10.1021/jacs.1c13201

This module is a thin front end: :func:`run_qed_ccsd` inspects the
reference dict and dispatches to the spin-adapted backend that matches it —

* restricted QED-HF references (dicts from
  :func:`OmegaQMC.addons.qed_hf.run_qed_hf`, key ``'C'``) go to
  :mod:`OmegaQMC.addons.qed_ccsd_rhf` (closed shell, spatial orbitals);
* unrestricted QED-UHF references (dicts from
  :func:`OmegaQMC.addons.qed_uhf.run_qed_uhf`, key ``'Ca'``) go to
  :mod:`OmegaQMC.addons.qed_ccsd_uhf` (alpha/beta spin blocks).

Both backends work with density-fitted (3-index) integrals throughout; a
dense reference (``run_qed_hf`` without ``auxbasis``) is factorised
exactly, so dense-integral results are reproduced to machine precision.
See the backend modules for the working equations.

The return dict carries the same energy/convergence keys for both
references; amplitude keys differ — spatial-orbital arrays (``'t1_10'``,
``'t2_20'``, ...) for closed shell, per-spin blocks (``'t1_10_a'``,
``'t2_20_ab'``, ...) for open shell. For QED-UHF references the reference
energies are additionally aliased under the restricted key names
(``'E_qed_hf'``, ``'E_rhf'``) so downstream code can read one set of keys.
"""

from pyscf import gto

from .qed_hf import run_qed_hf
from . import qed_ccsd_rhf
from . import qed_ccsd_uhf


def run_qed_ccsd(qedhf, do_t1_01=True, do_t2_11=True, do_t2_21=True,
                 do_t2_02=False, do_t2_12=False, do_t2_22=False,
                 frozen=0, max_iter=50, tol=1e-8, tol_amp=1e-7,
                 max_diis=8, diis_on_disk=False, verbose=True):
    """DIIS-accelerated QED-CCSD on a QED-HF or QED-UHF reference.

    The set of active photonic amplitudes selects the flavour:

    * none of do_t1_01..do_t2_22 → conventional CCSD
    * do_t1_01, do_t2_11, do_t2_21 → QED-CCSD-1 / Deprince (QED-CCSD-21)
    * do_t1_01, do_t2_11, do_t2_02, do_t2_12 → QED-CCSD-12 / White
    * all flags True → QED-CCSD-22 (full)

    Convergence requires *both* the energy change between iterations to
    drop below ``tol`` and the amplitude-step norm to drop below
    ``tol_amp``.

    Args:
        qedhf: reference dict from :func:`run_qed_hf` (restricted) or
            :func:`OmegaQMC.addons.qed_uhf.run_qed_uhf` (unrestricted);
            the backend is chosen automatically.
        do_*: enable individual photonic excitation classes.
        frozen: drop the ``frozen`` lowest (core) spatial orbitals from
            the correlation treatment.
        max_iter: max CCSD iterations.
        tol: energy convergence threshold.
        tol_amp: amplitude-step-norm convergence threshold.
        max_diis: DIIS history depth.
        diis_on_disk: keep the DIIS history as ``.npy`` files on disk
            (large systems are then not RAM-limited by the history).
        verbose: print per-iteration progress.

    Returns:
        dict with the correlation and total QED-CCSD energy, the
        reference energies, the converged amplitudes and a
        ``'converged'`` flag.
    """
    kwargs = dict(
        do_t1_01=do_t1_01, do_t2_11=do_t2_11, do_t2_21=do_t2_21,
        do_t2_02=do_t2_02, do_t2_12=do_t2_12, do_t2_22=do_t2_22,
        frozen=frozen, max_iter=max_iter, tol=tol, tol_amp=tol_amp,
        max_diis=max_diis, diis_on_disk=diis_on_disk, verbose=verbose)
    if 'Ca' in qedhf:                 # QED-UHF (unrestricted) reference
        out = qed_ccsd_uhf.run_qed_ccsd(qedhf, **kwargs)
        out.setdefault('E_qed_hf', out['E_qed_uhf'])
        out.setdefault('E_rhf', out['E_uhf'])
        return out
    return qed_ccsd_rhf.run_qed_ccsd(qedhf, **kwargs)


# ---------------------------------------------------------------------------
# Demo: glycolaldehyde / STO-3G QED-CCSD-21, ω = 3 eV, λ = (0, 0, 0.1).
# Complete-equations value: -262.416985787 Ha (the published
# -262.416986187 lacks the quartic t2_21/t2_22 terms).
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    mol = gto.M(
        atom="""
            C   0.00000000   0.00000000   0.00000000
            O   0.00000000   1.23456800   0.00000000
            H   0.97075033  -0.54577032   0.00000000
            C  -1.21509881  -0.80991169   0.00000000
            H  -1.15288176  -1.89931439   0.00000000
            C  -2.43440063  -0.19144555   0.00000000
            H  -3.37262777  -0.75937214   0.00000000
            O  -2.62194056   1.12501165   0.00000000
            H  -1.71446384   1.51627790   0.00000000
        """,
        basis='STO-3G', unit='Angstrom', symmetry=False, verbose=0)
    omega = 3.0 / 27.211386245988
    qedhf = run_qed_hf(mol, omega, (0.0, 0.0, 0.1), verbose=False)
    print(f"E_QED_HF = {qedhf['E_qed_hf']:.12f}")
    result = run_qed_ccsd(qedhf, do_t1_01=True, do_t2_11=True,
                          do_t2_21=True, verbose=True)
    print(f"\nE_QED_CCSD total = {result['E_qed_ccsd_total']:.12f}")
    print("Complete-equations QED-CCSD-21 reference: -262.416985787002")
