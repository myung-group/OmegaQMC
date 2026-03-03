import numpy as np
import jax.numpy as jnp
from pyscf import hessian


def compute_orbital_response(mf):
    """Compute CPHF orbital response dC/dR.

    Returns:
        mo1s_jnp: jnp.ndarray of shape (natm, 3, nao, nocc)
            The orbital response dC/dR_{ia,K} for each atom ia
            and Cartesian direction K.
    """
    nocc = np.count_nonzero(mf.mo_occ > 0)
    hobj = mf.Hessian()
    h1ao = hobj.make_h1(mf.mo_coeff, mf.mo_occ)
    mo1s, _ = hessian.rhf.solve_mo1(
        mf, mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao)

    natm = mf.mol.natm
    nao = mf.mol.nao
    mo1s_np = np.zeros((natm, 3, nao, nocc))
    for ia in range(natm):
        mo1s_np[ia] = mo1s[ia]

    return jnp.array(mo1s_np)
