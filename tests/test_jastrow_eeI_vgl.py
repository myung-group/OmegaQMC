"""Analytical (grad, lap) of J3_eeI vs autodiff.

Two layers of comparison:

  * ``_eval_eeI_poly_vgl`` returns nine partials that agree
    with ``jax.grad`` / ``jax.hessian`` of the value-only
    :func:`_eval_eeI_poly` to 1e-12.
  * ``_J3_eeI_vgl`` per-electron grad / lap matches
    ``jax.grad`` / trace-of-``jax.hessian`` of
    :func:`J3_eeI_fn` on H₂O / 6-31G with both ``like+`` and
    ``unlike+`` blocks populated.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.psi.gto import (
    _PsiGTO,
    _eval_eeI_poly,
    _eval_eeI_poly_vgl,
    _build_eeI_constraint_map,
)
from OmegaQMC.psi.cusp import get_cusp_params


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


# ----------------------------------------------------------------
# 2) Per-electron VGL of full J3_eeI on H₂O / 6-31G
# ----------------------------------------------------------------

def _h2o_atoms():
    return (
        "O   0.000  0.000  0.000\n"
        "H   0.757  0.586  0.000\n"
        "H  -0.757  0.586  0.000\n"
    )


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


if __name__ == "__main__":
    test_eval_eeI_poly_vgl_matches_autodiff()
    test_J3_eeI_vgl_matches_autodiff_h2o()
    test_J3_eeI_vgl_only_one_spin_block_h2o()
    print("OK")
