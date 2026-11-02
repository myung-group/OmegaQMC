"""Tests for the SWCT nuclear-force estimator of NN trials.

``force_warp='swct'`` for NN trials is Eq. 14 of Qian, Fu, Ren and
Chen, J. Chem. Phys. 157, 164104 (2022): the local energy is
differentiated along the space warp ``r_i -> r_i + dR_A w_A(r_i)``.
These tests check

* every per-sample component against brute-force autodiff (full
  Hessian Laplacian, reverse-mode derivatives), on the FermiNet
  (linearize Laplacian) and PsiFormer (forward-Laplacian) paths;
* the per-sample translational sum rule on an asymmetric molecule;
* the zero-variance limit of a single atom;
* agreement with the fast-warp estimator on the same samples through
  the driver (PGCS-free).
"""
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.psi.nn.adapter import make_nn_log_psi  # noqa: E402
from OmegaQMC.vmc_nn import get_vmc_nn_func  # noqa: E402
from OmegaQMC.observables.force import (  # noqa: E402
    FORCE_WARPS_NN,
    _warp_weights_distance_vg,
    postproc_h5_pgcs,
    vmc_nn_gradients_fast_warp,
    vmc_nn_gradients_swct,
)

L_H2 = 1.4010
EPS = float(jnp.finfo(jnp.float64).eps)


def _rel(a, b):
    return float(
        jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-300)
    )


def _h2_mol(shift=0.0):
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[[0.0, 0.0, -L_H2 / 2], [shift, 0.0, L_H2 / 2]],
        n_up=1, n_down=1,
    )


def _lih_mol():
    # Asymmetric heteronuclear test system (4 electrons).
    return Mole_custom.from_arrays(
        charges=[3, 1],
        coords=[[0.0, 0.0, 0.0], [0.2, -0.1, 3.0]],
        n_up=2, n_down=2,
    )


def _walkers(mol, n, seed):
    rng = np.random.default_rng(seed)
    nuc = np.asarray(mol.coords)
    nelec = mol.n_up + mol.n_down
    ctr = nuc[np.arange(nelec) % len(nuc)]
    return jnp.asarray(ctr[None] + 0.7 * rng.normal(size=(n, nelec, 3)))


def _brute_components(log_psi, params, nuc, charges, x):
    """SWCT components of one walker by brute-force autodiff."""
    nelec = x.shape[0]
    i_e, j_e = jnp.triu_indices(nelec, k=1)

    def t_loc(r, R):
        def f(xf):
            return log_psi(xf.reshape(nelec, 3), R, params)
        xf = r.reshape(-1)
        g = jax.grad(f)(xf)
        return -0.5 * (jnp.trace(jax.hessian(f)(xf)) + jnp.dot(g, g))

    def v_pot(r, R):
        d_en = jnp.linalg.norm(r[:, None] - R[None], axis=-1)
        v = -jnp.sum(charges[None] / d_en)
        if nelec > 1:
            v = v + jnp.sum(
                1.0 / jnp.linalg.norm(r[i_e] - r[j_e], axis=-1)
            )
        return v

    w, dw = _warp_weights_distance_vg(x, nuc, EPS)
    dt_dr, dt_dR = jax.grad(t_loc, argnums=(0, 1))(x, nuc)
    dv_dr, dv_dR = jax.grad(v_pot, argnums=(0, 1))(x, nuc)
    g, dlp_dR = jax.grad(log_psi, argnums=(0, 1))(x, nuc, params)
    ke = dt_dR + jnp.einsum('in,ik->nk', w, dt_dr)
    ee_en = dv_dR + jnp.einsum('in,ik->nk', w, dv_dr)
    p = dlp_dR + jnp.einsum('in,ik->nk', w, g) + 0.5 * dw.sum(0)
    return ee_en, ke, p


def test_swct_registered():
    assert 'swct' in FORCE_WARPS_NN


# ---------------------------------------------------------------------
# 1. Exact algebra (float64)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("config", ['ferminet', 'psiformer'])
@pytest.mark.parametrize("mol_fn", [lambda: _h2_mol(shift=0.1), _lih_mol])
def test_swct_components_exact(config, mol_fn):
    mol = mol_fn()
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    charges = jnp.asarray(mol.charges, dtype=jnp.float64)
    nelec = mol.n_up + mol.n_down
    log_psi, params, _, lap_grad = make_nn_log_psi(
        config, mol, jax.random.key(0), compute_dtype=None,
    )
    sw = vmc_nn_gradients_swct(
        log_psi, nuc, charges, nelec, params, lap_grad=lap_grad,
    )
    x = _walkers(mol, 3, seed=1)
    ee_en, ke, p = sw(x)
    for b in range(x.shape[0]):
        ref = _brute_components(log_psi, params, nuc, charges, x[b])
        assert _rel(ee_en[b], ref[0]) < 1e-10
        assert _rel(ke[b], ref[1]) < 1e-8
        assert _rel(p[b], ref[2]) < 1e-10

    # Without lap_grad the linearize Laplacian gives the same result.
    sw0 = vmc_nn_gradients_swct(log_psi, nuc, charges, nelec, params)
    assert _rel(sw0(x)[1], ke) < 1e-9

    # The Pulay vector is the same as the fast-warp one.
    fw = vmc_nn_gradients_fast_warp(log_psi, nuc, charges, nelec, params)
    assert _rel(fw(x)[2], p) < 1e-10
    assert _rel(fw(x)[0], ee_en) < 1e-12


# ---------------------------------------------------------------------
# 2. Translational sum rule, per sample
# ---------------------------------------------------------------------
@pytest.mark.parametrize("config", ['ferminet', 'psiformer'])
def test_swct_translation_sum_rule(config):
    mol = _lih_mol()
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    charges = jnp.asarray(mol.charges, dtype=jnp.float64)
    log_psi, params, _, lap_grad = make_nn_log_psi(
        config, mol, jax.random.key(3), compute_dtype=None,
    )
    sw = vmc_nn_gradients_swct(
        log_psi, nuc, charges, 4, params, lap_grad=lap_grad,
    )
    ee_en, ke, p = sw(_walkers(mol, 6, seed=2))
    for comp in (ee_en, ke, p):
        scale = float(jnp.abs(comp).max())
        assert float(jnp.abs(comp.sum(axis=1)).max()) < 1e-8 * scale


# ---------------------------------------------------------------------
# 3. Zero-variance limit of a single atom
# ---------------------------------------------------------------------
_HE_CTR = np.array([0.3, -0.2, 0.1])


@pytest.mark.parametrize("config", ['ferminet', 'psiformer'])
def test_single_atom_zero_variance(config):
    mol = Mole_custom.from_arrays(
        charges=[2], coords=[_HE_CTR.tolist()], n_up=1, n_down=1,
    )
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    log_psi, params, _, lap_grad = make_nn_log_psi(
        config, mol, jax.random.key(0), compute_dtype=None,
    )
    sw = vmc_nn_gradients_swct(
        log_psi, nuc, jnp.array([2.0]), 2, params, lap_grad=lap_grad,
    )
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(8, 2, 3)) * 0.7 + _HE_CTR)
    for comp in sw(x):
        assert float(jnp.abs(comp).max()) < 1e-8


# ---------------------------------------------------------------------
# 4. Driver: SWCT vs fast warp on the same samples
# ---------------------------------------------------------------------
def test_nn_h2_driver_swct_vs_fast_warp(tmp_path):
    res = {}
    for warp in ('swct', 'fast_warp'):
        pre = str(tmp_path / f'nn_h2_{warp}')
        drv = get_vmc_nn_func(
            _h2_mol(), 'ferminet', jax.random.key(99), prefix=pre,
            force_warp=warp,
        )
        drv(jax.random.key(1), num_walkers=100, num_steps_per_block=4,
            num_steps_decorr=5, num_blocks=16, num_steps_equil=300,
            compute_gradients=True, verbose=0,
            fname_log=pre + '.log')
        with h5py.File(pre + '.grd.h5') as f:
            assert f.attrs['force_warp'] == warp
            assert f.attrs['warp_weights'] == 'distance'
        out = postproc_h5_pgcs(prefix=pre)
        res[warp] = (np.asarray(out[0]), np.asarray(out[1]))
    (fa, ea), (fb, eb) = res['swct'], res['fast_warp']
    bound = 3.0 * np.sqrt(ea ** 2 + eb ** 2)
    assert np.all(np.abs(fa - fb) <= bound)
