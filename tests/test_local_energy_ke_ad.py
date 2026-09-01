"""FD vs AD regression for the new analytical local_energy_ke.

Verifies that ``jax.grad(local_energy_ke, argnums=(0, 1))``
matches central finite differences through the new analytical
kernel.  This is the most sensitive test of the AD path.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.psi.gto import get_psi_fun
from OmegaQMC.psi.cusp import get_cusp_params


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


if __name__ == "__main__":
    test_H2_grad_ke_vs_fd()
    print("OK")
