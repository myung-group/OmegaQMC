"""End-to-end regression: analytical vs Hessian KE with bsplines.

Confirms that the new analytical bspline path inside
``local_energy_ke`` matches the prior
``_local_energy_ke_hessian`` kernel to high precision on
configurations that include B-spline Jastrows alongside (or
instead of) Padé.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.psi.gto import _PsiGTO
from OmegaQMC.psi.cusp import get_cusp_params


def _make_psi(atoms, basis="6-31G"):
    modrv = generate_molecular_orbitals(
        atoms, units="Bohr", basis=basis,
    )
    params_cusp = {}
    for i in range(modrv.mol.natm):
        sym = modrv.mol.atom_symbol(i)
        if sym not in params_cusp:
            p = get_cusp_params(sym, basis)
            params_cusp[sym] = p[sym]
    return _PsiGTO(
        modrv, params_cusp=params_cusp, trial=None,
        jastrow_config=None,
    )


def _h2o_atoms():
    return (
        "O   0.000  0.000  0.000\n"
        "H   0.757  0.586  0.000\n"
        "H  -0.757  0.586  0.000\n"
    )


def _bspline_only_params(rng, sym_list):
    # Small amplitudes so all evaluations stay smoothly inside
    # the cutoff; default r_cut is 10 bohr.
    n_j2 = 8
    out = {
        "J2_bspline": {
            "like": jnp.array(
                rng.normal(size=n_j2) * 0.05
            ),
            "unlike": jnp.array(
                rng.normal(size=n_j2) * 0.05
            ),
        },
        "J1_bspline": {
            sym: jnp.array(
                rng.normal(size=8) * 0.05
            ) for sym in sym_list
        },
    }
    return out


def _pade_params(sym_list):
    return {
        "J1_pade": {
            sym: jnp.array([-0.05, 0.4]) for sym in sym_list
        },
        "J2_pade": {
            "like": jnp.array([0.25, 0.6]),
            "unlike": jnp.array([0.5, 0.4]),
        },
    }


def _check(psi, params, label, tol=1e-9):
    rng = np.random.default_rng(91)
    nuc = jnp.array(psi.mf.mol.atom_coords())
    elec = jnp.array(
        rng.normal(size=(psi.nelec, 3)) * 0.7
    )
    e_an = float(psi.local_energy_ke(elec, nuc, params))
    e_he = float(
        psi._local_energy_ke_hessian(elec, nuc, params)
    )
    err = abs(e_an - e_he)
    rel = err / max(abs(e_he), 1.0)
    print(
        f"  {label:<40s}"
        f" analytical={e_an:.10f},"
        f" hessian={e_he:.10f},"
        f" |Δ|={err:.3e}"
    )
    assert err < tol, (label, err)


def test_h2o_bspline_only():
    psi = _make_psi(_h2o_atoms())
    rng = np.random.default_rng(3)
    params = _bspline_only_params(rng, ["O", "H"])
    _check(psi, params, "H₂O: J1+J2 bspline only")


def test_h2o_bspline_plus_pade():
    psi = _make_psi(_h2o_atoms())
    rng = np.random.default_rng(5)
    p = _bspline_only_params(rng, ["O", "H"])
    p.update(_pade_params(["O", "H"]))
    _check(psi, p, "H₂O: bspline + Padé (analytical mix)")


def test_h2o_bspline_only_pade_only_match():
    """Sanity: Padé-only path is unaffected by the bspline
    rewiring (regression check on the existing code path)."""
    psi = _make_psi(_h2o_atoms())
    p = _pade_params(["O", "H"])
    _check(psi, p, "H₂O: Padé only (pre-existing path)")


if __name__ == "__main__":
    test_h2o_bspline_only_pade_only_match()
    test_h2o_bspline_only()
    test_h2o_bspline_plus_pade()
    print("OK")
