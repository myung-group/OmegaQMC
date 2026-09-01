"""End-to-end regression: analytical vs Hessian KE with J3_eeI.

Confirms that the analytical J3_eeI path inside
``local_energy_ke`` matches the prior
``_local_energy_ke_hessian`` kernel to high precision on
configurations that include J3_eeI alone or in combination with
Padé and B-spline Jastrows.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.psi.gto import _PsiGTO, _build_eeI_constraint_map
from OmegaQMC.psi.cusp import get_cusp_params


def _h2o_atoms():
    return (
        "O   0.000  0.000  0.000\n"
        "H   0.757  0.586  0.000\n"
        "H  -0.757  0.586  0.000\n"
    )


def _make_psi(N_eI=3, N_ee=3, r_cut=5.0, basis="6-31G"):
    modrv = generate_molecular_orbitals(
        _h2o_atoms(), units="Bohr", basis=basis,
    )
    params_cusp = {
        sym: get_cusp_params(sym, basis)[sym]
        for sym in ("O", "H")
    }
    # Provide eeI plus default-cutoff bspline configs so the
    # bspline VGL closures are also wired up.
    jcfg = {
        "eeI": {
            "N_eI": N_eI, "N_ee": N_ee, "r_cut": r_cut,
        },
        "J2": {"r_cut": 10.0},
        "J1": {sym: {"r_cut": 10.0} for sym in ("O", "H")},
    }
    psi = _PsiGTO(
        modrv, params_cusp=params_cusp, trial=None,
        jastrow_config=jcfg,
    )
    A, _ls, _ms, _ns = _build_eeI_constraint_map(
        N_eI, N_ee, r_cut,
    )
    return psi, A.shape[1]


def _eeI_params(NumGamma, rng, syms=("O", "H")):
    out = {}
    for sym in syms:
        for prefix in ("like+", "unlike+"):
            out[prefix + sym] = jnp.array(
                rng.normal(size=NumGamma) * 0.05,
            )
    return out


def _bspline_params(rng, sym_list):
    n_j2 = 8
    return {
        "J2_bspline": {
            "like": jnp.array(
                rng.normal(size=n_j2) * 0.05,
            ),
            "unlike": jnp.array(
                rng.normal(size=n_j2) * 0.05,
            ),
        },
        "J1_bspline": {
            sym: jnp.array(
                rng.normal(size=8) * 0.05,
            ) for sym in sym_list
        },
    }


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
        rng.normal(size=(psi.nelec, 3)) * 0.7,
    )
    e_an = float(psi.local_energy_ke(elec, nuc, params))
    e_he = float(
        psi._local_energy_ke_hessian(elec, nuc, params),
    )
    err = abs(e_an - e_he)
    print(
        f"  {label:<46s}"
        f" analytical={e_an:.10f},"
        f" hessian={e_he:.10f},"
        f" |Δ|={err:.3e}"
    )
    assert err < tol, (label, err)


def test_h2o_eeI_only():
    psi, NumGamma = _make_psi()
    rng = np.random.default_rng(3)
    params = {"J3_eeI": _eeI_params(NumGamma, rng)}
    _check(psi, params, "H₂O: J3_eeI only")


def test_h2o_eeI_plus_pade():
    psi, NumGamma = _make_psi()
    rng = np.random.default_rng(5)
    params = _pade_params(["O", "H"])
    params["J3_eeI"] = _eeI_params(NumGamma, rng)
    _check(psi, params, "H₂O: J3_eeI + Padé")


def test_h2o_eeI_plus_bspline():
    psi, NumGamma = _make_psi()
    rng = np.random.default_rng(7)
    params = _bspline_params(rng, ["O", "H"])
    params["J3_eeI"] = _eeI_params(NumGamma, rng)
    _check(psi, params, "H₂O: J3_eeI + bspline")


def test_h2o_eeI_plus_pade_plus_bspline():
    psi, NumGamma = _make_psi()
    rng = np.random.default_rng(11)
    params = _pade_params(["O", "H"])
    params.update(_bspline_params(rng, ["O", "H"]))
    params["J3_eeI"] = _eeI_params(NumGamma, rng)
    _check(
        psi, params,
        "H₂O: J3_eeI + Padé + bspline (all analytical)",
    )


if __name__ == "__main__":
    test_h2o_eeI_only()
    test_h2o_eeI_plus_pade()
    test_h2o_eeI_plus_bspline()
    test_h2o_eeI_plus_pade_plus_bspline()
    print("OK")
