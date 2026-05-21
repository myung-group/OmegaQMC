"""Smoke test: multi-Slater-determinant trial support
in all _VMCOptDriverGTO_* classes.

Builds a tiny H₂/6-31G CASCI(2,2) trial via
``extract_casscf_trial`` and runs each optimizer for a
handful of epochs.  Asserts only that:

* the driver constructs and runs without exception when
  ``trial`` is passed,
* the returned Jastrow params are finite,
* the recorded energies are finite.

Not a numerical regression — just plumbing.  The
single-det (``trial=None``) path is left to the existing
optimizer regressions; this test exists to confirm the
multi-det path is wired correctly through to
``_PsiGTO``'s multi-det Slater kernels.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf, mcscf

from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.psi.gto import extract_casscf_trial


def _build_h2_msd_trial():
    """H₂/6-31G with a CASCI(2,2) trial.  Returns
    (modrv, trial)."""
    L = 1.4010  # bohr
    atoms_string = (
        "H 0.0 0.0 {:.6f} 1\n"
        "H 0.0 0.0 {:.6f} 1\n"
    ).format(-L / 2, L / 2)

    modrv = generate_molecular_orbitals(
        atoms_string, units="Bohr", basis="6-31G",
    )

    # Build a PySCF CASCI on top of the same RHF
    # solution to extract a multi-det trial.
    mc = mcscf.CASCI(modrv, ncas=2, nelecas=(1, 1))
    mc.verbose = 0
    mc.kernel()
    trial = extract_casscf_trial(mc, coeff_threshold=1e-3)
    return modrv, trial


def _common_params_jastrow():
    return {
        "J1_pade": {
            "H": jnp.array([-0.05574627, 0.08272289])
        },
        "J2_pade": {
            "like": jnp.array([0.25, 0.6046799]),
            "unlike": jnp.array([0.5, 0.38077791]),
        },
    }


def _assert_finite(obj, ctx="value"):
    """Recursively assert every leaf array is finite.

    Walks through dicts, lists, and tuples; treats
    every leaf as an array via ``np.asarray``.
    Non-numeric leaves (e.g. strings or ``None``) are
    skipped.
    """
    if obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite(v, f"{ctx}[{k!r}]")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_finite(v, f"{ctx}[{i}]")
        return
    try:
        arr = np.asarray(obj)
    except (TypeError, ValueError):
        return
    if arr.dtype.kind not in ('f', 'c', 'i', 'u'):
        return
    assert np.all(np.isfinite(arr)), (
        f"non-finite leaf at {ctx}: {arr}"
    )


def test_vmcopt_msd_irsgd():
    from OmegaQMC.vmcopt_gto_irsgd import (
        get_vmcopt_gto_func,
    )
    modrv, trial = _build_h2_msd_trial()
    assert trial['ndet'] >= 1
    drv = get_vmcopt_gto_func(modrv, trial=trial)
    params_final, E_data = drv(
        jax.random.key(0),
        params_corr_init=_common_params_jastrow(),
        num_iters=1,
        num_epochs=2,
        num_walkers=128,
        num_steps_per_block=20,
        num_sample_blocks=2,
        num_blocks_equil=2,
        batch_size=64,
        verbose=0,
    )
    _assert_finite(params_final, "params_final")
    _assert_finite(E_data, "E_data")


def test_vmcopt_msd_linear():
    from OmegaQMC.vmcopt_gto_linear import (
        get_vmcopt_gto_func,
    )
    modrv, trial = _build_h2_msd_trial()
    drv = get_vmcopt_gto_func(modrv, trial=trial)
    params_final, E_data = drv(
        jax.random.key(0),
        params_corr_init=_common_params_jastrow(),
        frozen_keys={
            "J2_pade": {"like": [0], "unlike": [0]}
        },
        num_epochs=2,
        num_walkers=128,
        num_steps_per_block=50,
        num_blocks_equil=2,
        deriv_batch_size=32,
        verbose=0,
    )
    _assert_finite(params_final, "params_final")
    _assert_finite(E_data, "E_data")


def test_vmcopt_msd_naive():
    from OmegaQMC.vmcopt_gto_naive import (
        get_vmcopt_gto_func,
    )
    modrv, trial = _build_h2_msd_trial()
    drv = get_vmcopt_gto_func(modrv, trial=trial)
    params_final, E_data = drv(
        jax.random.key(0),
        params_corr_init=_common_params_jastrow(),
        num_epochs=2,
        num_walkers=128,
        num_steps_per_block=50,
        num_blocks=2,
        num_blocks_equil=2,
        verbose=0,
    )
    _assert_finite(params_final, "params_final")
    _assert_finite(E_data, "E_data")


if __name__ == '__main__':
    test_vmcopt_msd_irsgd()
    test_vmcopt_msd_linear()
    test_vmcopt_msd_naive()
    print("All MSD opt-driver smoke tests passed.")
