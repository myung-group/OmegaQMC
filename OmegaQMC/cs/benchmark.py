"""Traditional excited-state methods for side-by-side comparison with
the Pfau-NES + CS + NEVPT2 hybrid.

Runs CIS, TDHF, TDA-TDHF, and EOM-CCSD on the same molecule using
PySCF and returns excitation energies, transition dipoles, and
oscillator strengths in a uniform dict schema so they can be dropped
into the same comparison table as the Pfau-NES result.

The motivation for the §VI.B comparison: in the basis-incomplete
regime, the CS-recovered NEVPT2 from a Pfau-NES reference should be
competitive with EOM-CCSD or better, because the NN ansatz captures
correlation that single-reference CC misses. CIS / TDHF are
methodological baselines (no dynamic correlation); EOM-CCSD is the
single-reference gold standard.
"""

from typing import Mapping

import numpy as np


HARTREE_TO_EV = 27.211386245988
AU_TO_DEBYE = 2.541746229


def _absolute_energies(mf_energy, dE_au):
    """Given mf.e_tot (HF or CC ground) and excitation energies, return
    absolute total energies for each root."""
    return [float(mf_energy)] + [
        float(mf_energy) + float(e) for e in dE_au
    ]


def run_cis(mol, nstates: int = 4, return_oscillator: bool = True) -> dict:
    """Configuration-interaction singles (CIS) for the lowest ``nstates``
    excited states.

    Returns a dict with:
      ``method``         = "CIS"
      ``E_ground``       = HF total energy (atomic units)
      ``E_excited_au``   = list of ``nstates`` absolute excited-state
                           energies (atomic units)
      ``dE_au``          = list of CIS excitation energies (au)
      ``dE_eV``          = list of excitation energies (eV)
      ``oscillator``     = list of length-gauge oscillator strengths
      ``transition_dipole_debye`` = list of 3-vectors (Debye)
    """
    from pyscf import scf, tdscf

    mf = scf.RHF(mol).run(verbose=0)
    td = tdscf.TDA(mf)
    td.nstates = nstates
    td.verbose = 0
    td.kernel()
    dE_au = np.asarray(td.e)
    out = dict(
        method="CIS (TDA-HF)",
        E_ground=float(mf.e_tot),
        E_excited_au=_absolute_energies(mf.e_tot, dE_au)[1:],
        dE_au=dE_au.tolist(),
        dE_eV=(dE_au * HARTREE_TO_EV).tolist(),
    )
    if return_oscillator:
        mu = td.transition_dipole()
        f = td.oscillator_strength()
        out["transition_dipole_debye"] = [
            (np.asarray(m) * AU_TO_DEBYE).tolist() for m in mu
        ]
        out["oscillator"] = [float(x) for x in f]
    return out


def run_tdhf(mol, nstates: int = 4, return_oscillator: bool = True) -> dict:
    """Random-phase approximation / TDHF (full Tamm-Dancoff off)."""
    from pyscf import scf, tdscf

    mf = scf.RHF(mol).run(verbose=0)
    td = tdscf.TDHF(mf)
    td.nstates = nstates
    td.verbose = 0
    td.kernel()
    dE_au = np.asarray(td.e)
    out = dict(
        method="TDHF (RPA)",
        E_ground=float(mf.e_tot),
        E_excited_au=_absolute_energies(mf.e_tot, dE_au)[1:],
        dE_au=dE_au.tolist(),
        dE_eV=(dE_au * HARTREE_TO_EV).tolist(),
    )
    if return_oscillator:
        mu = td.transition_dipole()
        f = td.oscillator_strength()
        out["transition_dipole_debye"] = [
            (np.asarray(m) * AU_TO_DEBYE).tolist() for m in mu
        ]
        out["oscillator"] = [float(x) for x in f]
    return out


def run_eom_ccsd(mol, nstates: int = 4) -> dict:
    """Equation-of-motion CCSD for the lowest ``nstates`` excited states.

    Single-reference gold standard for excited states with single-
    reference ground.
    """
    from pyscf import scf, cc

    mf = scf.RHF(mol).run(verbose=0)
    mycc = cc.RCCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    eom_ee = mycc.eomee_ccsd_singlet(nroots=nstates)
    if isinstance(eom_ee, tuple):
        dE_au = np.asarray(eom_ee[0])
    else:
        dE_au = np.asarray(eom_ee)
    out = dict(
        method="EOM-CCSD",
        E_ground=float(mycc.e_tot),
        E_excited_au=_absolute_energies(mycc.e_tot, dE_au)[1:],
        dE_au=dE_au.tolist(),
        dE_eV=(dE_au * HARTREE_TO_EV).tolist(),
    )
    return out


def run_cisd(mol, nstates: int = 4) -> dict:
    """Single-reference CISD for the lowest ``nstates`` states."""
    from pyscf import scf, ci

    mf = scf.RHF(mol).run(verbose=0)
    myci = ci.RCISD(mf)
    myci.nroots = nstates
    myci.verbose = 0
    myci.kernel()
    # myci.e is array of nstates total energies; convert to excitation
    # energies relative to ground (which is myci.e[0])
    e_arr = np.asarray(myci.e)
    if e_arr.ndim == 0:
        e_arr = np.array([float(e_arr)])
    if len(e_arr) < 2:
        raise RuntimeError(
            f"CISD returned only {len(e_arr)} roots; need >= 2"
        )
    dE_au = e_arr[1:] - e_arr[0]
    out = dict(
        method="CISD",
        E_ground=float(e_arr[0]),
        E_excited_au=e_arr[1:].tolist(),
        dE_au=dE_au.tolist(),
        dE_eV=(dE_au * HARTREE_TO_EV).tolist(),
    )
    return out


def benchmark_all(
    mol,
    nstates: int = 4,
    methods: tuple = ("CIS", "TDHF", "EOM-CCSD"),
) -> dict:
    """Run a battery of traditional excited-state methods on ``mol``.

    Returns ``{method_label: result_dict}`` for use in side-by-side
    tables. Methods that raise (e.g. CCSD non-convergence) are silently
    skipped with a stub ``{"error": str(exc)}``.
    """
    results = {}
    for m in methods:
        try:
            if m == "CIS":
                r = run_cis(mol, nstates=nstates)
            elif m == "TDHF":
                r = run_tdhf(mol, nstates=nstates)
            elif m == "EOM-CCSD":
                r = run_eom_ccsd(mol, nstates=nstates)
            elif m == "CISD":
                r = run_cisd(mol, nstates=nstates)
            else:
                raise ValueError(f"unknown method {m!r}")
        except Exception as exc:
            r = dict(method=m, error=f"{type(exc).__name__}: {exc}")
        results[m] = r
    return results


def print_benchmark_summary(results: Mapping) -> None:
    """One-paragraph summary suitable for the paper Table caption."""
    print("=== Traditional excited-state methods ===")
    print(f"  {'method':>14s}  {'E_ground (Ha)':>14s}  "
          f"{'dE_1 (eV)':>10s}  {'dE_2 (eV)':>10s}  "
          f"{'f_1':>7s}  {'f_2':>7s}")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:>14s}  {'(failed)':>14s}  {r['error']}")
            continue
        line = f"  {name:>14s}  {r['E_ground']:>14.6f}"
        for i in range(2):
            if i < len(r.get("dE_eV", [])):
                line += f"  {r['dE_eV'][i]:>10.4f}"
            else:
                line += f"  {'---':>10s}"
        for i in range(2):
            if i < len(r.get("oscillator", [])):
                line += f"  {r['oscillator'][i]:>7.4f}"
            else:
                line += f"  {'---':>7s}"
        print(line)
