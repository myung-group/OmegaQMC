"""End-to-end regression: analytical vs Hessian KE on multi-det.

Confirms the new ``log_slater_multidet_analytic`` path inside
``local_energy_ke`` matches the prior
``_local_energy_ke_hessian`` kernel to high precision on a
small CASSCF case, alone and combined with each Jastrow type.
"""
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import mcscf

jax.config.update("jax_enable_x64", True)

from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.psi.gto import (
    _PsiGTO,
    _build_eeI_constraint_map,
    extract_casscf_trial,
)


def _h2_atoms():
    return """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.4
    """


def _make_psi(jastrow_config=None, threshold=1e-4):
    modrv = generate_molecular_orbitals(
        _h2_atoms(), units="Bohr", basis="6-31G",
    )
    mc = mcscf.CASSCF(modrv, 2, 2)
    mc.kernel()
    trial = extract_casscf_trial(
        mc, coeff_threshold=threshold,
    )
    psi = _PsiGTO(
        modrv, params_cusp=None, trial=trial,
        jastrow_config=jastrow_config,
    )
    return psi, trial


def _pade_params():
    return {
        "J1_pade": {"H": jnp.array(0.4)},
        "J2_pade": {
            "like": jnp.array([0.25, 0.6]),
            "unlike": jnp.array([0.5, 0.4]),
        },
    }


def _bspline_params(rng):
    return {
        "J2_bspline": {
            "like": jnp.array(rng.normal(size=8) * 0.05),
            "unlike": jnp.array(rng.normal(size=8) * 0.05),
        },
        "J1_bspline": {
            "H": jnp.array(rng.normal(size=8) * 0.05),
        },
    }


def _eeI_params(NumGamma, rng, syms=("H",)):
    out = {}
    for sym in syms:
        for prefix in ("like+", "unlike+"):
            out[prefix + sym] = jnp.array(
                rng.normal(size=NumGamma) * 0.05,
            )
    return out


def _check(psi, params, label, tol=1e-8):
    rng = np.random.default_rng(91)
    nuc = jnp.array(psi.mf.mol.atom_coords())
    elec = jnp.array(
        rng.normal(size=(psi.nelec, 3)) * 0.6,
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


def test_h2_multidet_only():
    psi, _ = _make_psi()
    _check(psi, {}, "H₂ CAS(2,2): multidet only")


def test_h2_multidet_plus_pade():
    psi, _ = _make_psi()
    p = _pade_params()
    _check(psi, p, "H₂ CAS(2,2): multidet + Padé")


def test_h2_multidet_plus_bspline():
    psi, _ = _make_psi(
        jastrow_config={
            "J2": {"r_cut": 10.0},
            "J1": {"H": {"r_cut": 10.0}},
        },
    )
    rng = np.random.default_rng(7)
    p = _bspline_params(rng)
    _check(psi, p, "H₂ CAS(2,2): multidet + bspline")


def test_h2_multidet_plus_pade_plus_bspline_plus_eeI():
    N_eI, N_ee, r_cut = 3, 3, 5.0
    psi, _ = _make_psi(
        jastrow_config={
            "J2": {"r_cut": 10.0},
            "J1": {"H": {"r_cut": 10.0}},
            "eeI": {
                "N_eI": N_eI, "N_ee": N_ee, "r_cut": r_cut,
            },
        },
    )
    rng = np.random.default_rng(11)
    A, _ls, _ms, _ns = _build_eeI_constraint_map(
        N_eI, N_ee, r_cut,
    )
    p = _pade_params()
    p.update(_bspline_params(rng))
    p["J3_eeI"] = _eeI_params(A.shape[1], rng)
    _check(
        psi, p,
        "H₂ CAS(2,2): all-Jastrow + multidet (analytical)",
    )


if __name__ == "__main__":
    test_h2_multidet_only()
    test_h2_multidet_plus_pade()
    test_h2_multidet_plus_bspline()
    test_h2_multidet_plus_pade_plus_bspline_plus_eeI()
    print("OK")
