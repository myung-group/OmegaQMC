"""Analytical (grad, lap) of B-spline Jastrows vs autodiff.

Compares ``_J{1,2}_bspline*_vgl`` against
``jax.grad`` / trace-of-``jax.hessian`` of the value-only
``J{1,2}_bspline*`` closures.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.psi.gto import (
    _PsiGTO,
    _bspline_eval,
    _bspline_eval_vgl,
    _build_bspline_coefs,
)


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


if __name__ == "__main__":
    test_bspline_eval_vgl_matches_autodiff()
    test_bspline_eval_vgl_at_boundaries()
    test_J2_bspline_aa_vgl_matches_autodiff_h2o()
    test_J2_bspline_ab_vgl_matches_autodiff_h2o()
    test_J1_bspline_vgl_matches_autodiff_h2o()
    print("OK")
