"""Initial envelope parameters of :class:`ExponentialEnvelopes`.

``init_to_ones=True`` starts the envelopes at unit scale.  For the
anisotropic envelope (``isotropic=False``) that means identity 3x3
decay matrices: the exponent ``|zeta @ d|`` is then ``|d|``, the same
envelope as the isotropic one with ``zeta = 1``.  (All-ones matrices,
the earlier initialisation, are rank 1 and do not decay perpendicular
to (1, 1, 1).)  The isotropic case and ``init_to_ones=False`` keep
their earlier values exactly.
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.psi.nn.compat import param_value  # noqa: E402
from OmegaQMC.psi.nn.env import ExponentialEnvelopes  # noqa: E402
from OmegaQMC.psi.nn.types import PhysicalConfiguration  # noqa: E402

N_DET = 2


def _mol():
    # LiH, open-shell occupation so the two spin blocks differ in size.
    return Mole_custom.from_arrays(
        charges=[3, 1], coords=[[0, 0, 0], [0.1, 0, 3.0]],
        n_up=3, n_down=1,
    )


def _env(mol, **kw):
    opts = dict(isotropic=True, per_shell=True, per_orbital_exponent=True,
                spin_restricted=False, init_to_ones=True,
                softplus_zeta=False)
    opts.update(kw)
    return ExponentialEnvelopes(mol, N_DET, **opts)


def _zeta_names(spin_restricted):
    return ['zetas'] if spin_restricted else ['zetas_up', 'zetas_down']


def _old_zetas(mol, isotropic, per_shell, per_orbital_exponent,
               init_to_ones):
    """The initial zetas as built before the anisotropic fix."""
    z0 = []
    for z, n_sh in zip(mol.charges, mol.mol_shells):
        for k in range(n_sh if per_shell else 1):
            z0.append(z / (k + 1))
    z0 = jnp.array(z0)
    n_orb = N_DET * (mol.n_up + mol.n_down)
    if per_orbital_exponent:
        z0 = jnp.tile(z0[None], (n_orb, 1))
    if not isotropic:
        z0 = z0[..., None, None] * jnp.eye(3)
    return jnp.ones(z0.shape) if init_to_ones else jnp.copy(z0)


CASES = list(itertools.product(
    [True, False],   # isotropic
    [True, False],   # per_shell
    [True, False],   # per_orbital_exponent
    [True, False],   # spin_restricted
    [True, False],   # init_to_ones
))


@pytest.mark.parametrize(
    "isotropic,per_shell,per_orbital_exponent,spin_restricted,init_to_ones",
    CASES,
)
def test_initial_zetas(isotropic, per_shell, per_orbital_exponent,
                       spin_restricted, init_to_ones):
    mol = _mol()
    env = _env(mol, isotropic=isotropic, per_shell=per_shell,
               per_orbital_exponent=per_orbital_exponent,
               spin_restricted=spin_restricted, init_to_ones=init_to_ones)
    old = _old_zetas(mol, isotropic, per_shell, per_orbital_exponent,
                     init_to_ones)
    for name in _zeta_names(spin_restricted):
        zeta = param_value(getattr(env, name))
        assert zeta.shape == old.shape and zeta.dtype == old.dtype
        if init_to_ones and not isotropic:
            # identity decay matrices, unit scale
            eye = jnp.broadcast_to(jnp.eye(3), old.shape)
            np.testing.assert_array_equal(zeta, eye)
        else:
            # unchanged, bit for bit
            np.testing.assert_array_equal(zeta, old)


@pytest.mark.parametrize("per_orbital_exponent", [True, False])
@pytest.mark.parametrize("spin_restricted", [True, False])
def test_anisotropic_ones_equals_isotropic_ones(per_orbital_exponent,
                                                spin_restricted):
    """At init the anisotropic envelope is the isotropic one."""
    mol = _mol()
    kw = dict(per_orbital_exponent=per_orbital_exponent,
              spin_restricted=spin_restricted, init_to_ones=True)
    iso = _env(mol, isotropic=True, **kw)
    aniso = _env(mol, isotropic=False, **kw)
    n_e = mol.n_up + mol.n_down
    r = 1.5 * jax.random.normal(jax.random.key(0), (n_e, 3))
    conf = PhysicalConfiguration(
        R=jnp.asarray(mol.coords, dtype=jnp.float64), r=r,
        mol_idx=jnp.array(0),
    )
    o_iso = iso(conf, None)
    o_aniso = aniso(conf, None)
    assert o_iso.shape == o_aniso.shape
    assert np.all(np.isfinite(o_aniso))
    np.testing.assert_allclose(o_aniso, o_iso, rtol=1e-12)
    # An electron displaced from Li perpendicular to (1, 1, 1) must see
    # a decaying envelope (all-ones matrices gave exponent 0 there).
    d = jnp.array([1.0, -1.0, 0.0]) * 2.0
    conf_perp = PhysicalConfiguration(
        R=conf.R, r=r.at[0].set(conf.R[0] + d), mol_idx=jnp.array(0),
    )
    np.testing.assert_allclose(aniso(conf_perp, None),
                               iso(conf_perp, None), rtol=1e-12)
