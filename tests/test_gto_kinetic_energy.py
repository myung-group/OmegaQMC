"""Analytical kinetic energy of GTO Slater-Jastrow trials.

The analytical local kinetic energy (and its electron and nuclear
derivatives) against finite differences, the Hessian reference and
autodiff, for B-spline, J3_eeI and multi-determinant trials, plus its
ingredients: multi-determinant Slater (value, gradient, Laplacian),
the Cartesian angular gradients and the quintic radial spline of the
AOs.
"""
import jax
import jax.numpy as jnp
import numpy as np
from pyscf import mcscf

jax.config.update('jax_enable_x64', True)

from OmegaQMC import generate_molecular_orbitals  # noqa: E402
from OmegaQMC.psi.gto import (  # noqa: E402
    get_psi_fun,
    _PsiGTO,
    _build_eeI_constraint_map,
    extract_casscf_trial,
    _angular_cartesian_poly,
    _angular_cartesian_vgl,
    _radial_vgl,
    _radial_val_spline,
    _radial_vgl_spline,
    _build_quintic_radial_spline,
)
from OmegaQMC.psi.cusp import get_cusp_params  # noqa: E402


# ---------------------------------------------------------------------------
# From test_local_energy_ke_ad.py
#
# FD vs AD regression for the new analytical local_energy_ke.
#
# Verifies that ``jax.grad(local_energy_ke, argnums=(0, 1))``
# matches central finite differences through the new analytical
# kernel.  This is the most sensitive test of the AD path.
# ---------------------------------------------------------------------------


def _build(mol_atoms, basis, params_jastrow):
    modrv = generate_molecular_orbitals(
        mol_atoms, units="Bohr", basis=basis
    )
    params_cusp = {}
    for i in range(modrv.mol.natm):
        sym = modrv.mol.atom_symbol(i)
        if sym not in params_cusp:
            p = get_cusp_params(sym, basis)
            params_cusp[sym] = p[sym]

    _, (_ee, _nn, _en, ke), _mo, _ = get_psi_fun(
        modrv, params_cusp=params_cusp,
    )
    nuc = jnp.array(modrv.mol.atom_coords())
    nelec = modrv.mol.nelec[0] + modrv.mol.nelec[1]
    return ke, nuc, nelec, params_jastrow


def _fd(fn, x, eps=1e-5):
    """Central FD of a scalar-valued fn over flat x."""
    x = np.asarray(x)
    g = np.zeros_like(x)
    flat = x.ravel()
    out = np.zeros_like(flat)
    for k in range(flat.size):
        xp = flat.copy(); xp[k] += eps
        xm = flat.copy(); xm[k] -= eps
        out[k] = (
            float(fn(xp.reshape(x.shape)))
            - float(fn(xm.reshape(x.shape)))
        ) / (2.0 * eps)
    return out.reshape(x.shape)


def test_H2_grad_ke_vs_fd():
    L = 1.4010
    atoms = (
        f"H 0 0 {-L/2:.6f} 1\n"
        f"H 0 0 { L/2:.6f} 1\n"
    )
    pj = {
        "J1_pade": {"H": jnp.array([-0.056, 0.083])},
        "J2_pade": {
            "like": jnp.array([0.25, 0.6047]),
            "unlike": jnp.array([0.5, 0.3808]),
        },
    }
    ke, nuc, nelec, pj = _build(atoms, "6-31G", pj)

    rng = np.random.default_rng(0)
    elec = jnp.array(rng.normal(size=(nelec, 3)) * 0.5)

    f_r = lambda r: ke(r, nuc, pj)
    f_R = lambda R: ke(elec, R, pj)
    gr_ad, gR_ad = jax.grad(ke, argnums=(0, 1))(elec, nuc, pj)
    gr_fd = _fd(f_r, elec)
    gR_fd = _fd(f_R, nuc)

    err_r = float(jnp.max(jnp.abs(gr_ad - gr_fd)))
    err_R = float(jnp.max(jnp.abs(gR_ad - gR_fd)))
    print(f"H2 grad_r  max|AD-FD|={err_r:.3e}")
    print(f"H2 grad_R  max|AD-FD|={err_R:.3e}")
    assert err_r < 1e-6, err_r
    assert err_R < 1e-6, err_R


# ---------------------------------------------------------------------------
# From test_local_energy_ke_bspline.py
#
# End-to-end regression: analytical vs Hessian KE with bsplines.
#
# Confirms that the new analytical bspline path inside
# ``local_energy_ke`` matches the prior
# ``_local_energy_ke_hessian`` kernel to high precision on
# configurations that include B-spline Jastrows alongside (or
# instead of) Padé.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# From test_local_energy_ke_eeI.py
#
# End-to-end regression: analytical vs Hessian KE with J3_eeI.
#
# Confirms that the analytical J3_eeI path inside
# ``local_energy_ke`` matches the prior
# ``_local_energy_ke_hessian`` kernel to high precision on
# configurations that include J3_eeI alone or in combination with
# Padé and B-spline Jastrows.
# ---------------------------------------------------------------------------


def _make_psi_eei(N_eI=3, N_ee=3, r_cut=5.0, basis="6-31G"):
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


def _check_eei(psi, params, label, tol=1e-9):
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
    psi, NumGamma = _make_psi_eei()
    rng = np.random.default_rng(3)
    params = {"J3_eeI": _eeI_params(NumGamma, rng)}
    _check_eei(psi, params, "H₂O: J3_eeI only")


def test_h2o_eeI_plus_pade():
    psi, NumGamma = _make_psi_eei()
    rng = np.random.default_rng(5)
    params = _pade_params(["O", "H"])
    params["J3_eeI"] = _eeI_params(NumGamma, rng)
    _check_eei(psi, params, "H₂O: J3_eeI + Padé")


def test_h2o_eeI_plus_bspline():
    psi, NumGamma = _make_psi_eei()
    rng = np.random.default_rng(7)
    params = _bspline_params(rng, ["O", "H"])
    params["J3_eeI"] = _eeI_params(NumGamma, rng)
    _check_eei(psi, params, "H₂O: J3_eeI + bspline")


def test_h2o_eeI_plus_pade_plus_bspline():
    psi, NumGamma = _make_psi_eei()
    rng = np.random.default_rng(11)
    params = _pade_params(["O", "H"])
    params.update(_bspline_params(rng, ["O", "H"]))
    params["J3_eeI"] = _eeI_params(NumGamma, rng)
    _check_eei(
        psi, params,
        "H₂O: J3_eeI + Padé + bspline (all analytical)",
    )


# ---------------------------------------------------------------------------
# From test_local_energy_ke_multidet.py
#
# End-to-end regression: analytical vs Hessian KE on multi-det.
#
# Confirms the new ``log_slater_multidet_analytic`` path inside
# ``local_energy_ke`` matches the prior
# ``_local_energy_ke_hessian`` kernel to high precision on a
# small CASSCF case, alone and combined with each Jastrow type.
# ---------------------------------------------------------------------------


def _h2_atoms():
    return """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.4
    """


def _make_psi_msd(jastrow_config=None, threshold=1e-4):
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


def _pade_params_msd():
    return {
        "J1_pade": {"H": jnp.array(0.4)},
        "J2_pade": {
            "like": jnp.array([0.25, 0.6]),
            "unlike": jnp.array([0.5, 0.4]),
        },
    }


def _bspline_params_msd(rng):
    return {
        "J2_bspline": {
            "like": jnp.array(rng.normal(size=8) * 0.05),
            "unlike": jnp.array(rng.normal(size=8) * 0.05),
        },
        "J1_bspline": {
            "H": jnp.array(rng.normal(size=8) * 0.05),
        },
    }


def _eeI_params_msd(NumGamma, rng, syms=("H",)):
    out = {}
    for sym in syms:
        for prefix in ("like+", "unlike+"):
            out[prefix + sym] = jnp.array(
                rng.normal(size=NumGamma) * 0.05,
            )
    return out


def _check_msd(psi, params, label, tol=1e-8):
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
    psi, _ = _make_psi_msd()
    _check_msd(psi, {}, "H₂ CAS(2,2): multidet only")


def test_h2_multidet_plus_pade():
    psi, _ = _make_psi_msd()
    p = _pade_params_msd()
    _check_msd(psi, p, "H₂ CAS(2,2): multidet + Padé")


def test_h2_multidet_plus_bspline():
    psi, _ = _make_psi_msd(
        jastrow_config={
            "J2": {"r_cut": 10.0},
            "J1": {"H": {"r_cut": 10.0}},
        },
    )
    rng = np.random.default_rng(7)
    p = _bspline_params_msd(rng)
    _check_msd(psi, p, "H₂ CAS(2,2): multidet + bspline")


def test_h2_multidet_plus_pade_plus_bspline_plus_eeI():
    N_eI, N_ee, r_cut = 3, 3, 5.0
    psi, _ = _make_psi_msd(
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
    p = _pade_params_msd()
    p.update(_bspline_params_msd(rng))
    p["J3_eeI"] = _eeI_params_msd(A.shape[1], rng)
    _check_msd(
        psi, p,
        "H₂ CAS(2,2): all-Jastrow + multidet (analytical)",
    )


# ---------------------------------------------------------------------------
# From test_slater_multidet_vgl.py
#
# Analytical multi-determinant Slater (val, grad, lap) vs autodiff.
#
# The single-det path already routes through
# ``_slater_det_assemble``.  This test exercises the multi-det
# twin ``log_slater_multidet_analytic`` that replaces the
# ``laplacian_linearize`` fallback when ``trial is not None``.
# ---------------------------------------------------------------------------


def _h2_psi(threshold=1e-4):
    atoms = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.4
    """
    modrv = generate_molecular_orbitals(
        atoms, units="Bohr", basis="6-31G",
    )
    mc = mcscf.CASSCF(modrv, 2, 2)
    mc.kernel()
    trial = extract_casscf_trial(
        mc, coeff_threshold=threshold,
    )
    psi = _PsiGTO(
        modrv, params_cusp=None, trial=trial,
        jastrow_config=None,
    )
    return psi, trial


def _h4_psi(threshold=1e-3):
    """A slightly larger CAS to exercise multiple alpha and
    beta excitations."""
    atoms = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.4
    H 0.0 0.0 2.8
    H 0.0 0.0 4.2
    """
    modrv = generate_molecular_orbitals(
        atoms, units="Bohr", basis="6-31G",
    )
    mc = mcscf.CASSCF(modrv, 4, 4)
    mc.kernel()
    trial = extract_casscf_trial(
        mc, coeff_threshold=threshold,
    )
    psi = _PsiGTO(
        modrv, params_cusp=None, trial=trial,
        jastrow_config=None,
    )
    return psi, trial


def _check_against_autodiff(psi, elec, label, tol=1e-9):
    nuc = jnp.array(psi.mf.mol.atom_coords())
    n_e = psi.nelec
    val_an, g_an, lap_an = psi._log_slater_analytic(elec, nuc)
    f = lambda e: psi._log_slater(e, nuc)
    val_ad = float(f(elec))
    g_ad = jax.grad(f)(elec)
    H_ad = jax.hessian(f)(elec)
    H_resh = H_ad.reshape(n_e, 3, n_e, 3)
    lap_ad = jnp.einsum("eaea->e", H_resh)
    err_v = abs(float(val_an) - val_ad)
    err_g = float(jnp.max(jnp.abs(g_an - g_ad)))
    err_L = float(jnp.max(jnp.abs(lap_an - lap_ad)))
    print(
        f"  {label:<30s}"
        f" |Δval|={err_v:.3e},"
        f" |Δgrad|={err_g:.3e},"
        f" |Δlap|={err_L:.3e}"
    )
    assert err_v < tol
    assert err_g < tol
    assert err_L < tol


def test_h2_cas22_multidet_analytic():
    psi, trial = _h2_psi()
    print(f"H₂ CAS(2,2) ndet = {trial['ndet']}")
    rng = np.random.default_rng(7)
    n_e = psi.nelec
    for seed in (1, 23, 99):
        rng = np.random.default_rng(seed)
        elec = jnp.array(
            rng.normal(size=(n_e, 3)) * 0.5,
        )
        _check_against_autodiff(
            psi, elec, f"H₂ CAS(2,2) seed={seed}",
        )


def test_h4_cas44_multidet_analytic():
    psi, trial = _h4_psi()
    print(f"H₄ CAS(4,4) ndet = {trial['ndet']}")
    n_e = psi.nelec
    for seed in (3, 31):
        rng = np.random.default_rng(seed)
        elec = jnp.array(
            rng.normal(size=(n_e, 3)) * 0.6,
        )
        _check_against_autodiff(
            psi, elec, f"H₄ CAS(4,4) seed={seed}",
        )


# ---------------------------------------------------------------------------
# From test_angular_cartesian_grad.py
#
# Hand-rolled vs autodiff gradients of the harmonic polynomial.
#
# ``_angular_cartesian_vgl`` returns hand-rolled gradients for
# ``am ∈ {0, 1, 2}`` and falls back to ``jax.jacobian`` for
# ``am ≥ 3``.  This test pins the hand-rolled branches to the
# autodiff reference at machine precision.
# ---------------------------------------------------------------------------


def _autodiff_grad(am, dr):
    return jax.jacobian(
        lambda d: _angular_cartesian_poly(am, d)
    )(dr)


def _check_am(am, tol=1e-14):
    rng = np.random.default_rng(am * 17 + 3)
    max_err_v = 0.0
    max_err_g = 0.0
    for _ in range(8):
        dr = jnp.array(rng.normal(size=3))
        Y_an, g_an = _angular_cartesian_vgl(am, dr)
        Y_ref = _angular_cartesian_poly(am, dr)
        g_ref = _autodiff_grad(am, dr)
        max_err_v = max(
            max_err_v, float(jnp.max(jnp.abs(Y_an - Y_ref)))
        )
        max_err_g = max(
            max_err_g, float(jnp.max(jnp.abs(g_an - g_ref)))
        )
    print(
        f"  am={am}:"
        f" |ΔY|={max_err_v:.3e},"
        f" |Δgrad|={max_err_g:.3e}"
    )
    assert max_err_v < tol, (am, max_err_v)
    assert max_err_g < tol, (am, max_err_g)


def test_am0_grad_hand_rolled():
    _check_am(0)


def test_am1_grad_hand_rolled():
    _check_am(1)


def test_am2_grad_hand_rolled():
    _check_am(2)


def test_am3_grad_autodiff_fallback():
    # am=3 still routes through jax.jacobian; this just
    # confirms the fallback path returns finite, correct shape.
    rng = np.random.default_rng(31)
    dr = jnp.array(rng.normal(size=3))
    Y, g = _angular_cartesian_vgl(3, dr)
    assert Y.shape == (7,)
    assert g.shape == (7, 3)
    assert bool(jnp.all(jnp.isfinite(Y)))
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# From test_ao_radial_spline_vgl.py
#
# Regression tests for the AO->quintic-spline radial transform.
#
# The opt-in transform (QMCPACK NMO idea) replaces each non-cusp GTO
# shell's analytic contracted-Gaussian radial with a log-grid quintic
# spline in ``s = r^2`` (see ``_build_quintic_radial_spline`` /
# ``_radial_vgl_spline`` in ``OmegaQMC.psi.gto``).  These tests confirm:
#
# 1. the spline radial VGL reproduces the analytic ``_radial_vgl`` triple
#    ``(R, R'/r, R'' + 2 R'/r)`` to spline tolerance, and
# 2. an end-to-end GTO trial evaluates the same ``log|psi|`` and kinetic
#    local energy with the transform on vs off (cusp disabled, so every
#    shell is splined).
# ---------------------------------------------------------------------------


_SHELLS = {
    "O_1s": ([130.70932, 23.808861, 6.4436083],
             [0.15432897, 0.53532814, 0.44463454]),
    "O_2s": ([5.0331513, 1.1695961, 0.380389],
             [-0.09996723, 0.39951283, 0.70011547]),
    "H_1s": ([3.42525091, 0.62391373, 0.16885540],
             [0.15432897, 0.53532814, 0.44463454]),
    "diffuse": ([0.08], [1.0]),
    "d_prim": ([1.185], [1.0]),
    "tight": ([500.0, 80.0], [0.2, 0.6]),
}


def test_radial_vgl_spline_matches_analytic():
    r_test = np.concatenate([
        [0.0, 1e-6, 1e-4, 1e-3, 1e-2],
        np.linspace(0.02, 5.0, 300),
    ])
    for name, (alpha, norm) in _SHELLS.items():
        alpha = jnp.array(alpha)
        norm = jnp.array(norm)
        knots, coefs = _build_quintic_radial_spline(alpha, norm)
        for r in r_test:
            r = float(r)
            r2 = jnp.array(r * r)
            rj = jnp.array(r)
            Ra, Rpa, Rla = _radial_vgl(alpha, norm, r2, rj)
            Rs, Rps, Rls = _radial_vgl_spline(knots, coefs, r2, rj)
            val = _radial_val_spline(knots, coefs, r2, rj)
            # Value matches tightly; derivatives to spline tolerance.
            assert abs(float(Ra) - float(Rs)) < 1e-6, name
            assert abs(float(Ra) - float(val)) < 1e-6, name
            assert abs(float(Rpa) - float(Rps)) < 1e-4, name
            assert abs(float(Rla) - float(Rls)) < 5e-3, name


def test_spline_zero_beyond_cutoff():
    alpha = jnp.array([1.0])
    norm = jnp.array([1.0])
    knots, coefs = _build_quintic_radial_spline(alpha, norm)
    r = float(jnp.sqrt(knots[-1])) + 2.0  # s = r^2 past the last knot
    r2 = jnp.array(r * r)
    R, Rp, Rl = _radial_vgl_spline(knots, coefs, r2, jnp.array(r))
    assert float(R) == 0.0 and float(Rp) == 0.0 and float(Rl) == 0.0


def _trial(mol, transform, basis="6-31g"):
    mf = generate_molecular_orbitals(
        mol, units="Bohr", basis=basis,
        ao_splines=transform, dm0_pkl=None,
    )
    return get_psi_fun(
        mf, params_cusp=None, trial=None, jastrow_config=None,
    ), jnp.array(mf.mol.atom_coords())


def test_trial_logpsi_and_ke_match_transform_on_off():
    fa, nuc = _trial("H 0 0 0; H 0 0 1.4", None)
    fs, _ = _trial("H 0 0 0; H 0 0 1.4", True)
    log_a, (_, _, _, ke_a), *_ = fa
    log_s, (_, _, _, ke_s), *_ = fs

    key = jax.random.PRNGKey(1)
    crds = jax.random.normal(key, (300, 2, 3)) * 0.9
    p = {}
    lpa = jax.vmap(lambda r: log_a(r, nuc, p))(crds)
    lps = jax.vmap(lambda r: log_s(r, nuc, p))(crds)
    kea = jax.vmap(lambda r: ke_a(r, nuc, p))(crds)
    kes = jax.vmap(lambda r: ke_s(r, nuc, p))(crds)

    assert float(jnp.max(jnp.abs(lpa - lps))) < 1e-8
    assert float(jnp.max(jnp.abs(kea - kes))) < 1e-5


def test_transform_off_is_analytic_by_default():
    # No ao_transform kwarg -> no shell is splined.
    (fa, _), = [_trial("H 0 0 0; H 0 0 1.4", None)]
    obj_fns = fa
    # Rebuild the _PsiGTO directly to inspect shell flags.
    mf = generate_molecular_orbitals(
        "H 0 0 0; H 0 0 1.4", units="Bohr", basis="6-31g",
        dm0_pkl=None,
    )
    from OmegaQMC.psi.gto import _PsiGTO
    psi = _PsiGTO(mf, params_cusp=None)
    assert not any(sh.use_spline for sh in psi.shell_list)
