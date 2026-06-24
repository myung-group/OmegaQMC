"""End-to-end SWCT regression: v6 vs frozen v5 baseline.

The v6 rewrite of ``vmc_gto_gradients`` swaps the
``jax.jacobian(rescale_fn)`` call for a closed-form
value-and-electron-diagonal-gradient pair (see
:mod:`OmegaQMC.observables.force` and the v6 plan section
in ``atomic-frolicking-reddy.md``).  Mathematically the
two paths are identical up to where the downstream
``'beneK->bnK'`` einsum traces only the electron-diagonal
of the rescale Jacobian — which off-diagonal entries are
zero by chain rule, so the trace is the full sum.

This test pins the new bundled output of
``vmc_gto_gradients`` to the **frozen v5** baseline at
1e-10 in float64 across a batch of random walker
configurations on H₂O / 6-31G, for both schemes.

The frozen baseline lives at ``tests/force_v5_frozen.py``
(byte-equivalent copy of ``OmegaQMC/observables/force.py``
at commit ``ad3fb3d``).  If ``force.py`` is rewritten
again in the future, the frozen copy still tests v5↔now
equivalence — do **not** edit it.
"""
import sys
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.psi.gto import get_psi_fun
from OmegaQMC.observables.force import (
    vmc_gto_gradients as vmc_gto_gradients_v6,
)
from OmegaQMC.vmc_gto import _build_cusp_params


def _load_v5_frozen():
    """Load the frozen v5 force kernel under its own name."""
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "_force_v5_frozen", here / "force_v5_frozen.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_force_v5_frozen"] = mod
    spec.loader.exec_module(mod)
    return mod.vmc_gto_gradients


def _h2o_atoms():
    return (
        "O   0.000  0.000  0.000\n"
        "H   0.757  0.586  0.000\n"
        "H  -0.757  0.586  0.000\n"
    )


def _build_pieces(atoms_string):
    """Build the wavefunction + energy callables once, then
    hand them to v5 and v6 separately."""
    mf = generate_molecular_orbitals(
        atoms_string, units="Ang", basis="6-31G",
    )
    params_cusp = _build_cusp_params(
        mf, "Quady2025", mf.mol.natm,
    )
    log_trial, local_energy, mo_fns, _C_fns = get_psi_fun(
        mf, params_cusp=params_cusp, trial=None,
        jastrow_config=None,
    )
    get_psi_mo, get_psi_mo_partition_vg = mo_fns
    le_ee, _le_nn, le_en, le_ke = local_energy
    nuc_crds = jnp.array(mf.mol.atom_coords())
    nelec = mf.mol.nelectron
    eps = float(jnp.finfo(nuc_crds.dtype).eps)
    return dict(
        log_trial=log_trial,
        le_ee=le_ee, le_en=le_en, le_ke=le_ke,
        get_psi_mo=get_psi_mo,
        get_psi_mo_partition_vg=get_psi_mo_partition_vg,
        nuc_crds=nuc_crds, eps=eps, nelec=nelec,
    )


def _compare(scheme, p, batch, tol=1e-10):
    v5 = _load_v5_frozen()
    params_corr = {}
    fn_v5 = v5(
        p["le_ee"], p["le_en"], p["le_ke"], p["log_trial"],
        p["nuc_crds"], params_corr, p["get_psi_mo"], p["eps"],
        scheme,
    )
    fn_v6 = vmc_gto_gradients_v6(
        p["le_ee"], p["le_en"], p["le_ke"], p["log_trial"],
        p["nuc_crds"], params_corr,
        p["get_psi_mo_partition_vg"], p["eps"], scheme,
    )
    out_v5 = fn_v5(batch)
    out_v6 = fn_v6(batch)
    assert len(out_v5) == len(out_v6)
    for label, a, b in zip(
        ("grd_ee", "grd_en", "grd_ke", "grd_logpsi"),
        out_v5, out_v6,
    ):
        err = float(jnp.max(jnp.abs(a - b)))
        print(f"  [{scheme}] {label:<10s} |Δ|={err:.3e}")
        assert err < tol, (scheme, label, err)


def test_h2o_scheme1_regression():
    p = _build_pieces(_h2o_atoms())
    rng = np.random.default_rng(11)
    batch = jnp.array(
        rng.normal(size=(8, p["nelec"], 3)) * 0.7,
    )
    _compare("scheme1", p, batch)


def test_h2o_scheme2_regression():
    p = _build_pieces(_h2o_atoms())
    rng = np.random.default_rng(13)
    batch = jnp.array(
        rng.normal(size=(8, p["nelec"], 3)) * 0.7,
    )
    _compare("scheme2", p, batch)


if __name__ == "__main__":
    test_h2o_scheme1_regression()
    test_h2o_scheme2_regression()
    print("OK")
