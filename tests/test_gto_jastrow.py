"""Jastrow factors of GTO trials.

Analytical gradients and Laplacians of the B-spline (including knot
boundaries) and J3_eeI Jastrows against autodiff, and the Pade J1
without cusp-corrected orbitals.
"""
import jax
import jax.numpy as jnp
import numpy as np
import warnings
import pytest

jax.config.update('jax_enable_x64', True)

from OmegaQMC import generate_molecular_orbitals  # noqa: E402
from OmegaQMC.psi.gto import (  # noqa: E402
    _PsiGTO,
    _bspline_eval,
    _bspline_eval_vgl,
    _build_bspline_coefs,
    _eval_eeI_poly,
    _eval_eeI_poly_vgl,
    _build_eeI_constraint_map,
)
from OmegaQMC.psi.cusp import get_cusp_params  # noqa: E402


# ---------------------------------------------------------------------------
# From test_jastrow_bspline_vgl.py
#
# Analytical (grad, lap) of B-spline Jastrows vs autodiff.
#
# Compares ``_J{1,2}_bspline*_vgl`` against
# ``jax.grad`` / trace-of-``jax.hessian`` of the value-only
# ``J{1,2}_bspline*`` closures.
# ---------------------------------------------------------------------------


def _make_psi(atoms, basis="6-31G", jastrow_config=None):
    modrv = generate_molecular_orbitals(
        atoms, units="Bohr", basis=basis,
    )
    return _PsiGTO(
        modrv, params_cusp=None, trial=None,
        jastrow_config=jastrow_config,
    )


def _per_e_grad_lap(fn, elec, nuc=None):
    """(grad_e, lap_e) of a scalar fn(elec, [nuc, ...]).

    fn is a closure over curr_params already; we differentiate
    only w.r.t. ``elec``.  Returns ``(grad_e, lap_e)`` of shapes
    ``(n_e, 3)`` and ``(n_e,)`` so we can compare against the
    analytical VGL convention directly.
    """
    if nuc is None:
        f = lambda e: fn(e)
    else:
        f = lambda e: fn(e, nuc)
    g = jax.grad(f)(elec)
    H = jax.hessian(f)(elec)
    n_e = elec.shape[0]
    H_resh = H.reshape(n_e, 3, n_e, 3)
    lap = jnp.einsum("eaea->e", H_resh)
    return g, lap


# ----------------------------------------------------------------
# 1-D _bspline_eval_vgl: agreement with autodiff
# ----------------------------------------------------------------
def test_bspline_eval_vgl_matches_autodiff():
    np.random.seed(0)
    n_params = 8
    delta_r = 0.5
    delta_r_inv = 1.0 / delta_r
    max_index = n_params
    rng = np.random.default_rng(42)
    params = jnp.array(rng.normal(size=n_params))
    coefs = _build_bspline_coefs(params, delta_r, 0.5)

    # interior points
    rs = jnp.linspace(0.05, n_params * delta_r - 0.05, 17)

    def f(r):
        return _bspline_eval(
            r[None], coefs, delta_r_inv, max_index,
        )[0]

    g_auto = jax.vmap(jax.grad(f))(rs)
    h_auto = jax.vmap(jax.grad(jax.grad(f)))(rs)
    val_v, d1_v, d2_v = _bspline_eval_vgl(
        rs, coefs, delta_r_inv, max_index,
    )

    err_v = float(jnp.max(jnp.abs(
        val_v - jax.vmap(f)(rs)
    )))
    err_d1 = float(jnp.max(jnp.abs(d1_v - g_auto)))
    err_d2 = float(jnp.max(jnp.abs(d2_v - h_auto)))
    print(
        f"bspline_eval_vgl interior:"
        f" |val|={err_v:.3e}, |d1|={err_d1:.3e},"
        f" |d2|={err_d2:.3e}"
    )
    assert err_v < 1e-12
    assert err_d1 < 1e-10
    assert err_d2 < 1e-10


def test_bspline_eval_vgl_at_boundaries():
    """At r=r_cut: u, u', u'' all vanish (C² boundary)."""
    n_params = 8
    delta_r = 0.5
    r_cut = (n_params + 1) * delta_r
    delta_r_inv = 1.0 / delta_r
    max_index = n_params
    rng = np.random.default_rng(7)
    params = jnp.array(rng.normal(size=n_params))
    coefs = _build_bspline_coefs(params, delta_r, 0.5)

    r_b = jnp.array([r_cut, r_cut - 1e-12])
    val, d1, d2 = _bspline_eval_vgl(
        r_b, coefs, delta_r_inv, max_index,
    )
    err = max(
        float(jnp.max(jnp.abs(val))),
        float(jnp.max(jnp.abs(d1))),
        float(jnp.max(jnp.abs(d2))),
    )
    print(
        f"bspline_eval_vgl at r=r_cut: max(|val,d1,d2|)"
        f"={err:.3e}"
    )
    # C² in math, but t=1 lands as 1+ε after floating-point;
    # the polynomial leak is ~|c[n]|·ε, well under 1e-10.
    assert err < 1e-10


# ----------------------------------------------------------------
# Per-electron VGL of full Jastrows: H₂O / 6-31G config
# ----------------------------------------------------------------
def _h2o_atoms():
    return (
        "O   0.000  0.000  0.000\n"
        "H   0.757  0.586  0.000\n"
        "H  -0.757  0.586  0.000\n"
    )


def _h2o_params(rng):
    n_j2 = 8
    n_j1_O = 8
    n_j1_H = 6
    return {
        "J2_bspline": {
            "like": jnp.array(
                rng.normal(size=n_j2) * 0.05
            ),
            "unlike": jnp.array(
                rng.normal(size=n_j2) * 0.05
            ),
        },
        "J1_bspline": {
            "O": jnp.array(rng.normal(size=n_j1_O) * 0.05),
            "H": jnp.array(rng.normal(size=n_j1_H) * 0.05),
        },
    }


def test_J2_bspline_aa_vgl_matches_autodiff_h2o():
    rng = np.random.default_rng(11)
    psi = _make_psi(_h2o_atoms())
    params = _h2o_params(rng)
    n_e = psi.nelec
    elec = jnp.array(
        rng.normal(size=(n_e, 3)) * 0.7
    )

    f = lambda e: psi.J2_bspline_aa(
        e, params["J2_bspline"]
    )
    g_ad, lap_ad = _per_e_grad_lap(f, elec)
    g_an, lap_an = psi._J2_bspline_aa_vgl(
        elec, params["J2_bspline"]
    )
    err_g = float(jnp.max(jnp.abs(g_an - g_ad)))
    err_L = float(jnp.max(jnp.abs(lap_an - lap_ad)))
    print(
        f"J2_bspline_aa  H₂O:"
        f" |Δgrad|={err_g:.3e}, |Δlap|={err_L:.3e}"
    )
    assert err_g < 1e-10
    assert err_L < 1e-10


def test_J2_bspline_ab_vgl_matches_autodiff_h2o():
    rng = np.random.default_rng(13)
    psi = _make_psi(_h2o_atoms())
    params = _h2o_params(rng)
    n_e = psi.nelec
    elec = jnp.array(
        rng.normal(size=(n_e, 3)) * 0.7
    )

    f = lambda e: psi.J2_bspline_ab(
        e, params["J2_bspline"]
    )
    g_ad, lap_ad = _per_e_grad_lap(f, elec)
    g_an, lap_an = psi._J2_bspline_ab_vgl(
        elec, params["J2_bspline"]
    )
    err_g = float(jnp.max(jnp.abs(g_an - g_ad)))
    err_L = float(jnp.max(jnp.abs(lap_an - lap_ad)))
    print(
        f"J2_bspline_ab  H₂O:"
        f" |Δgrad|={err_g:.3e}, |Δlap|={err_L:.3e}"
    )
    assert err_g < 1e-10
    assert err_L < 1e-10


def test_J1_bspline_vgl_matches_autodiff_h2o():
    rng = np.random.default_rng(17)
    psi = _make_psi(_h2o_atoms())
    params = _h2o_params(rng)
    n_e = psi.nelec
    nuc = jnp.array(psi.mf.mol.atom_coords())
    elec = jnp.array(
        rng.normal(size=(n_e, 3)) * 0.7
    )

    f = lambda e, n: psi.J1_bspline_fn(
        e, n, params["J1_bspline"]
    )
    g_ad, lap_ad = _per_e_grad_lap(f, elec, nuc)
    g_an, lap_an = psi._J1_bspline_vgl(
        elec, nuc, params["J1_bspline"]
    )
    err_g = float(jnp.max(jnp.abs(g_an - g_ad)))
    err_L = float(jnp.max(jnp.abs(lap_an - lap_ad)))
    print(
        f"J1_bspline    H₂O:"
        f" |Δgrad|={err_g:.3e}, |Δlap|={err_L:.3e}"
    )
    assert err_g < 1e-10
    assert err_L < 1e-10


# ---------------------------------------------------------------------------
# From test_jastrow_eeI_vgl.py
#
# Analytical (grad, lap) of J3_eeI vs autodiff.
#
# Two layers of comparison:
#
#   * ``_eval_eeI_poly_vgl`` returns nine partials that agree
#     with ``jax.grad`` / ``jax.hessian`` of the value-only
#     :func:`_eval_eeI_poly` to 1e-12.
#   * ``_J3_eeI_vgl`` per-electron grad / lap matches
#     ``jax.grad`` / trace-of-``jax.hessian`` of
#     :func:`J3_eeI_fn` on H₂O / 6-31G with both ``like+`` and
#     ``unlike+`` blocks populated.
# ---------------------------------------------------------------------------


# ----------------------------------------------------------------
# 1) Per-triplet polynomial × envelope: nine partials
# ----------------------------------------------------------------
def test_eval_eeI_poly_vgl_matches_autodiff():
    N_eI, N_ee, L = 3, 3, 2.5
    rng = np.random.default_rng(42)
    T = 11
    r_1I = jnp.array(rng.uniform(0.1, 2.0, T))
    r_2I = jnp.array(rng.uniform(0.1, 2.0, T))
    r_12 = jnp.array(rng.uniform(0.1, 4.0, T))
    gamma = jnp.array(
        rng.normal(size=(N_eI + 1, N_eI + 1, N_ee + 1))
        * 0.1
    )

    (u, u_d1I, u_d2I, u_d12,
     u_d1I_1I, u_d2I_2I, u_d12_12,
     u_d1I_12, u_d2I_12) = _eval_eeI_poly_vgl(
        r_12, r_1I, r_2I, gamma, L, N_eI, N_ee,
    )

    def f_scalar(r12_s, r1I_s, r2I_s):
        return _eval_eeI_poly(
            jnp.array([r12_s]),
            jnp.array([r1I_s]),
            jnp.array([r2I_s]),
            gamma, L, N_eI, N_ee,
        )[0]

    err = {k: 0.0 for k in (
        "val", "d1I", "d2I", "d12",
        "d1I_1I", "d2I_2I", "d12_12",
        "d1I_12", "d2I_12",
    )}
    for t in range(T):
        r12_t = float(r_12[t])
        r1I_t = float(r_1I[t])
        r2I_t = float(r_2I[t])
        val_ad = float(f_scalar(r12_t, r1I_t, r2I_t))
        g_ad = jax.grad(
            f_scalar, argnums=(0, 1, 2)
        )(r12_t, r1I_t, r2I_t)
        h_ad = jax.hessian(
            f_scalar, argnums=(0, 1, 2)
        )(r12_t, r1I_t, r2I_t)
        err["val"] = max(
            err["val"], abs(val_ad - float(u[t]))
        )
        err["d12"] = max(
            err["d12"], abs(float(g_ad[0]) - float(u_d12[t]))
        )
        err["d1I"] = max(
            err["d1I"], abs(float(g_ad[1]) - float(u_d1I[t]))
        )
        err["d2I"] = max(
            err["d2I"], abs(float(g_ad[2]) - float(u_d2I[t]))
        )
        err["d12_12"] = max(
            err["d12_12"],
            abs(float(h_ad[0][0]) - float(u_d12_12[t])),
        )
        err["d1I_1I"] = max(
            err["d1I_1I"],
            abs(float(h_ad[1][1]) - float(u_d1I_1I[t])),
        )
        err["d2I_2I"] = max(
            err["d2I_2I"],
            abs(float(h_ad[2][2]) - float(u_d2I_2I[t])),
        )
        err["d1I_12"] = max(
            err["d1I_12"],
            abs(float(h_ad[1][0]) - float(u_d1I_12[t])),
        )
        err["d2I_12"] = max(
            err["d2I_12"],
            abs(float(h_ad[2][0]) - float(u_d2I_12[t])),
        )
    print("_eval_eeI_poly_vgl partials max err:")
    for k, v in err.items():
        print(f"  {k:<8s} {v:.3e}")
    for k, v in err.items():
        assert v < 1e-12, (k, v)


def _make_psi_with_eeI(N_eI=3, N_ee=3, r_cut=5.0):
    atoms = _h2o_atoms()
    modrv = generate_molecular_orbitals(
        atoms, units="Bohr", basis="6-31G",
    )
    params_cusp = {
        sym: get_cusp_params(sym, "6-31G")[sym]
        for sym in ("O", "H")
    }
    psi = _PsiGTO(
        modrv, params_cusp=params_cusp, trial=None,
        jastrow_config={
            "eeI": {
                "N_eI": N_eI, "N_ee": N_ee, "r_cut": r_cut,
            },
        },
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
                rng.normal(size=NumGamma) * 0.05
            )
    return out


def test_J3_eeI_vgl_matches_autodiff_h2o():
    rng = np.random.default_rng(13)
    psi, NumGamma = _make_psi_with_eeI()
    n_e = psi.nelec
    nuc = jnp.array(psi.mf.mol.atom_coords())
    elec = jnp.array(rng.normal(size=(n_e, 3)) * 0.7)
    params = _eeI_params(NumGamma, rng)

    f = lambda e: psi.J3_eeI_fn(e, nuc, params)
    g_ad = jax.grad(f)(elec)
    H_ad = jax.hessian(f)(elec)
    H_resh = H_ad.reshape(n_e, 3, n_e, 3)
    lap_ad = jnp.einsum("eaea->e", H_resh)

    g_an, lap_an = psi._J3_eeI_vgl(elec, nuc, params)
    err_g = float(jnp.max(jnp.abs(g_an - g_ad)))
    err_L = float(jnp.max(jnp.abs(lap_an - lap_ad)))
    print(
        f"J3_eeI H₂O: |Δgrad|={err_g:.3e},"
        f" |Δlap|={err_L:.3e}"
    )
    assert err_g < 1e-10
    assert err_L < 1e-10


def test_J3_eeI_vgl_only_one_spin_block_h2o():
    """Only ``like+`` populated — sanity for the Python-side
    branch in :func:`J3_eeI_vgl` that skips missing keys.
    """
    rng = np.random.default_rng(19)
    psi, NumGamma = _make_psi_with_eeI()
    n_e = psi.nelec
    nuc = jnp.array(psi.mf.mol.atom_coords())
    elec = jnp.array(rng.normal(size=(n_e, 3)) * 0.7)
    params = {
        "like+O": jnp.array(
            rng.normal(size=NumGamma) * 0.05,
        ),
        "like+H": jnp.array(
            rng.normal(size=NumGamma) * 0.05,
        ),
    }
    f = lambda e: psi.J3_eeI_fn(e, nuc, params)
    g_ad = jax.grad(f)(elec)
    H_ad = jax.hessian(f)(elec)
    H_resh = H_ad.reshape(n_e, 3, n_e, 3)
    lap_ad = jnp.einsum("eaea->e", H_resh)

    g_an, lap_an = psi._J3_eeI_vgl(elec, nuc, params)
    err_g = float(jnp.max(jnp.abs(g_an - g_ad)))
    err_L = float(jnp.max(jnp.abs(lap_an - lap_ad)))
    print(
        f"J3_eeI H₂O (like+ only):"
        f" |Δgrad|={err_g:.3e}, |Δlap|={err_L:.3e}"
    )
    assert err_g < 1e-10
    assert err_L < 1e-10


# ---------------------------------------------------------------------------
# From test_pade_j1_nocusp.py
#
# Padé J1 without cusp-corrected orbitals (``params_cusp=None``).
#
# Without the cusp correction, J1 fixes ``a = -Z`` so that it carries
# the electron-nucleus cusp, and each per-element entry is ``b``.  An
# ``[a, b]`` entry (the format used with cusp correction) used to fail
# with a broadcasting error for any molecule; it is now accepted, with
# a warning, and its ``a`` ignored.  Uses H2O so that two elements with
# different ``Z`` are present.
# ---------------------------------------------------------------------------


B = {"O": 0.9, "H": 0.4}
J2 = {
    "like": jnp.array([0.25, 0.6]),
    "unlike": jnp.array([0.5, 0.4]),
}


def _h2o_atoms_pade():
    return """
    O  0.000000  0.000000  0.221665
    H  0.000000  1.430901 -0.886659
    H  0.000000 -1.430901 -0.886659
    """


@pytest.fixture(scope="module")
def psi():
    mf = generate_molecular_orbitals(
        _h2o_atoms_pade(), units="Bohr", basis="6-31G", dm0_pkl=None,
    )
    return _PsiGTO(mf, params_cusp=None, trial=None)


def _coords(psi):
    rng = np.random.default_rng(5)
    nuc = jnp.array(psi.mf.mol.atom_coords())
    Z = psi.mf.mol.atom_charges().astype(int)
    idx = np.repeat(np.arange(len(Z)), Z)
    elec = jnp.array(
        np.asarray(nuc)[idx] + 0.6 * rng.normal(size=(len(idx), 3))
    )
    return elec, nuc


def _params(j1):
    return {"J1_pade": j1, "J2_pade": J2}


def test_scalar_b_matches_formula(psi):
    elec, nuc = _coords(psi)
    p_b = _params({s: jnp.array(b) for s, b in B.items()})
    j1 = psi.J1(elec, nuc, p_b["J1_pade"])
    Z = psi.mf.mol.atom_charges()
    b = np.array([B[s] for s in psi.mf.mol.elements])
    r = np.linalg.norm(
        np.asarray(elec)[None] - np.asarray(nuc)[:, None], axis=-1,
    )
    ref = np.sum(-Z[:, None] * r / (1.0 + b[:, None] * r))
    assert abs(float(j1) - ref) < 1e-12


def test_ab_equals_scalar_b(psi):
    elec, nuc = _coords(psi)
    p_b = _params({s: jnp.array(b) for s, b in B.items()})
    p_ab = _params({s: jnp.array([-0.05, b]) for s, b in B.items()})
    with pytest.warns(UserWarning, match="a is fixed at -Z"):
        lp_ab = float(psi.log_trial_wavefunction(elec, nuc, p_ab))
    lp_b = float(psi.log_trial_wavefunction(elec, nuc, p_b))
    assert abs(lp_ab - lp_b) < 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ke_ab = float(psi.local_energy_ke(elec, nuc, p_ab))
    ke_b = float(psi.local_energy_ke(elec, nuc, p_b))
    assert abs(ke_ab - ke_b) < 1e-10


@pytest.mark.parametrize("fmt", ["b", "ab"])
def test_analytic_ke_matches_hessian(psi, fmt):
    elec, nuc = _coords(psi)
    if fmt == "b":
        j1 = {s: jnp.array(b) for s, b in B.items()}
    else:
        j1 = {s: jnp.array([-0.05, b]) for s, b in B.items()}
    p = _params(j1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e_an = float(psi.local_energy_ke(elec, nuc, p))
        e_he = float(psi._local_energy_ke_hessian(elec, nuc, p))
    assert abs(e_an - e_he) < 1e-8, (e_an, e_he)


def test_bad_length_raises(psi):
    elec, nuc = _coords(psi)
    p = _params({s: jnp.array([0.1, 0.2, b]) for s, b in B.items()})
    with pytest.raises(ValueError, match="expected b or"):
        psi.log_trial_wavefunction(elec, nuc, p)
