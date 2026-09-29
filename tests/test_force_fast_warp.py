"""Tests for the fast-warp nuclear-force estimator.

Fast-warp (Qian, Li and Chen, Faraday Discuss. 254, 529 (2024), SI
Eq. 19) applies the SWCT space warp but removes every kinetic-energy
derivative by Hermiticity.  These tests check

* the helpers and :func:`_fast_warp_terms` against brute-force
  autodiff, including the commutator the warp terms replace;
* the zero-variance limit of a single atom, where the warp weights are
  identically 1 and translation invariance makes every per-sample
  component vanish (NN and GTO, both weight schemes);
* agreement with the existing estimator on the same samples (GTO SWCT
  on H2; NN no-warp ZVZB on H2, through the driver, PGCS-free);
* optionally, the efficiency gain on the 2H2O PsiFormer checkpoint
  (skipped when the checkpoint is absent).
"""
import time
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func  # noqa
from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.psi.nn.adapter import make_nn_log_psi  # noqa: E402
from OmegaQMC.vmc_nn import get_vmc_nn_func, _chunk_calls  # noqa: E402
from OmegaQMC.observables.force import (  # noqa: E402
    _fast_warp_terms,
    _logpsi_grad_hessian_blocks,
    _warp_weight_laplacian,
    _warp_weights_distance_vg,
    postproc_h5_pgcs,
    vmc_nn_gradients_fast_warp,
    vmc_nn_gradients_zvzb,
)

L_H2 = 1.4010
EPS = float(jnp.finfo(jnp.float64).eps)
REPO = Path(__file__).resolve().parents[1]
CHK_2H2O = REPO / '2H2O-psiformer' / '2h2o_kfac.chk.h5'


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


# ---------------------------------------------------------------------
# 1. Exact algebra (float64 NN H2)
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def nn_h2_f64():
    mol = _h2_mol(shift=0.1)
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    log_psi, params, _, _ = make_nn_log_psi(
        'psiformer', mol, jax.random.key(0), compute_dtype=None,
    )
    return nuc, log_psi, params


@pytest.mark.parametrize("seed", [1, 2])
def test_helpers_and_warp_terms_exact(nn_h2_f64, seed):
    nuc, log_psi, params = nn_h2_f64
    n_e = 2
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(n_e, 3)) * 0.8 + [0.0, 0.0, 0.3])

    def wvg(r):
        return _warp_weights_distance_vg(r, nuc, EPS)

    w, dw = wvg(x)
    lap_w = _warp_weight_laplacian(wvg, x)

    # Weight gradient and Laplacian, one electron at a time.
    def w_one(r1, i):
        return wvg(x.at[i].set(r1))[0][i]

    dw_ad = jnp.stack([jax.jacfwd(w_one)(x[i], i) for i in range(n_e)])
    lap_ad = jnp.stack([
        jnp.trace(jax.hessian(w_one)(x[i], i), axis1=-2, axis2=-1)
        for i in range(n_e)
    ])
    assert _rel(dw, dw_ad) < 1e-12
    assert _rel(lap_w, lap_ad) < 1e-12

    # Closed form of the unnormalised weight: lap d^-4 = 12 d^-6.
    def g_one(r1, n):
        return jnp.sum((r1 - nuc[n]) ** 2) ** -2

    d = jnp.linalg.norm(x[:, None] - nuc[None], axis=-1)
    lap_g = jnp.array([
        [jnp.trace(jax.hessian(g_one)(x[i], n)) for n in range(2)]
        for i in range(n_e)
    ])
    assert _rel(lap_g, 12.0 * d ** -6) < 1e-12

    # Same-electron Hessian blocks vs per-electron jax.hessian.
    def lp_flat(xf):
        return log_psi(xf.reshape(n_e, 3), nuc, params)

    g, hblk = _logpsi_grad_hessian_blocks(lp_flat, x)

    def lp_one(r1, i):
        return log_psi(x.at[i].set(r1), nuc, params)

    hblk_ad = jnp.stack([
        jax.hessian(lp_one)(x[i], i) for i in range(n_e)
    ])
    g_ad = jax.grad(lambda y: log_psi(y, nuc, params))(x)
    assert _rel(hblk, hblk_ad) < 1e-12
    assert _rel(g, g_ad) < 1e-12

    # Warp terms vs the commutator they replace,
    # sum_i [w T_i f - T_i(w f)] / psi with T_i = -1/2 nabla_i^2 and
    # f = d psi / d r_ik (psi normalised at x to avoid overflow).
    lp0 = log_psi(x, nuc, params)

    def psi(y):
        return jnp.exp(log_psi(y, nuc, params) - lp0)

    def f_ik(y, i, k):
        return jax.grad(psi)(y)[i, k]

    def lap_i(fun, y, i):
        return jnp.trace(jax.hessian(lambda r1: fun(y.at[i].set(r1)))(y[i]))

    brute = np.zeros((2, 3))
    for n in range(2):
        for k in range(3):
            tot = 0.0
            for i in range(n_e):
                def wf(y, i=i, n=n, k=k):
                    return wvg(y)[0][i, n] * f_ik(y, i, k)
                t_f = -0.5 * lap_i(lambda y: f_ik(y, i, k), x, i)
                t_wf = -0.5 * lap_i(wf, x, i)
                tot += float(w[i, n] * t_f - t_wf)
            brute[n, k] = tot / float(psi(x))

    grd_warp, p_warp = _fast_warp_terms(w, dw, lap_w, g, hblk)
    assert _rel(grd_warp, jnp.asarray(brute)) < 1e-9
    p_ref = jnp.einsum('in,ik->nk', w, g_ad) + 0.5 * dw_ad.sum(0)
    assert _rel(p_warp, p_ref) < 1e-12


# ---------------------------------------------------------------------
# 2. Zero-variance limit of a single atom
# ---------------------------------------------------------------------
_HE_CTR = np.array([0.3, -0.2, 0.1])


def _he_walkers():
    rng = np.random.default_rng(3)
    return jnp.asarray(
        rng.normal(size=(8, 2, 3)) * 0.7 + [0.2, -0.1, 0.3] + _HE_CTR
    )


@pytest.mark.parametrize("nn_dtype", [None, 'float32'])
def test_single_atom_zero_variance_nn(nn_dtype):
    mol = Mole_custom.from_arrays(
        charges=[2], coords=[_HE_CTR.tolist()], n_up=1, n_down=1,
    )
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    log_psi, params, _, _ = make_nn_log_psi(
        'psiformer', mol, jax.random.key(0), compute_dtype=nn_dtype,
    )
    fw = vmc_nn_gradients_fast_warp(
        log_psi, nuc, jnp.array([2.0]), 2, params,
    )
    x = _he_walkers()
    grd_ee_en, grd_ke, grd_logpsi = fw(x)
    assert float(jnp.abs(grd_ee_en).max()) < 1e-10
    assert float(jnp.abs(grd_ke).max()) < 1e-10
    # P = d ln|psi|/dR + sum_i d ln|psi|/dr_i cancels only to the
    # precision of the network (~1e-3 of |d ln|psi|/dR| in float32).
    scale = float(jnp.abs(jax.vmap(
        lambda r: jax.grad(log_psi, argnums=1)(r, nuc, params)
    )(x)).max())
    tol = 1e-10 if nn_dtype is None else 5e-3 * scale
    assert float(jnp.abs(grd_logpsi).max()) < tol


@pytest.mark.parametrize("gr_scheme", ['scheme1', 'scheme2'])
def test_single_atom_zero_variance_gto(tmp_path, gr_scheme):
    x, y, z = _HE_CTR
    mf = generate_molecular_orbitals(
        f'He {x} {y} {z} 1', units='Bohr', basis='cc-pvdz',
    )
    # No cusp correction: this checks the estimator only (and skips
    # generating cusp data for He).
    drv = get_vmc_gto_func(
        mf, None, cusp_scheme=None, gr_scheme=gr_scheme,
        force_warp='fast_warp', prefix=str(tmp_path / 'he'),
    )
    ctr = jnp.asarray(mf.mol.atom_coords()[0])
    walkers = _he_walkers() - jnp.asarray(_HE_CTR) + ctr
    grd_ee, grd_en, grd_ke, grd_logpsi = drv.vmc_gradient_batch(walkers)
    assert float(jnp.abs(grd_ee + grd_en).max()) < 1e-10
    assert float(jnp.abs(grd_ke).max()) < 1e-10
    assert float(jnp.abs(grd_logpsi).max()) < 1e-10


# ---------------------------------------------------------------------
# 3. Agreement with the existing estimators on the same samples
# ---------------------------------------------------------------------
def _assert_forces_agree(res_a, res_b):
    (fa, ea), (fb, eb) = res_a, res_b
    # Same samples, so the difference is far below this bound.
    bound = 3.0 * np.sqrt(np.asarray(ea) ** 2 + np.asarray(eb) ** 2)
    assert np.all(np.abs(np.asarray(fa) - np.asarray(fb)) <= bound)


def test_gto_h2_swct_vs_fast_warp(tmp_path):
    params_jastrow = {
        "J1_pade": {"H": jnp.array([-0.05574627, 0.08272289])},
        "J2_pade": {"like": jnp.array([0.25, 0.6046799]),
                    "unlike": jnp.array([0.5, 0.38077791])},
    }
    atoms = f'H 0 0 {-L_H2 / 2:.6f} 1\nH 0 0 {L_H2 / 2:.6f} 1\n'
    mf = generate_molecular_orbitals(atoms, units="Bohr", basis="6-31G")
    res = {}
    for warp in ('swct', 'fast_warp'):
        pre = str(tmp_path / f'h2_{warp}')
        drv = get_vmc_gto_func(
            mf, params_jastrow, cusp_scheme='Quady2025',
            gr_scheme='scheme1', force_warp=warp, prefix=pre,
        )
        drv(jax.random.key(888), num_walkers=200,
            num_steps_per_block=100, num_blocks=20, num_blocks_equil=5,
            mc_timestep=0.1, compute_gradients=True)
        with h5py.File(pre + '.grd.h5') as f:
            assert f.attrs['force_warp'] == warp
        out = postproc_h5_pgcs(prefix=pre)
        res[warp] = (out[0], out[1])
    _assert_forces_agree(res['swct'], res['fast_warp'])


def test_nn_h2_driver_zvzb_vs_fast_warp(tmp_path):
    res = {}
    for warp in (None, 'fast_warp'):
        pre = str(tmp_path / f'nn_h2_{warp}')
        drv = get_vmc_nn_func(
            _h2_mol(), 'psiformer', jax.random.key(99), prefix=pre,
            force_warp=warp,
        )
        drv(jax.random.key(1), num_walkers=100, num_steps_per_block=4,
            num_steps_decorr=5, num_blocks=16, num_steps_equil=300,
            compute_gradients=True, verbose=0,
            fname_log=pre + '.log')
        with h5py.File(pre + '.grd.h5') as f:
            assert f.attrs['force_warp'] == (warp or 'none')
            if warp is not None:
                assert f.attrs['warp_weights'] == 'distance'
        out = postproc_h5_pgcs(prefix=pre)
        res[warp] = (out[0], out[1])
    _assert_forces_agree(res[None], res['fast_warp'])


def test_force_warp_validation(tmp_path):
    with pytest.raises(ValueError):
        get_vmc_nn_func(
            _h2_mol(), 'psiformer', jax.random.key(0),
            prefix=str(tmp_path / 'bad'), force_warp='swct',
        )


# ---------------------------------------------------------------------
# 4. Efficiency on the 2H2O PsiFormer (optional)
# ---------------------------------------------------------------------
@pytest.mark.skipif(not CHK_2H2O.exists(),
                    reason="2H2O PsiFormer checkpoint not available")
def test_2h2o_fast_warp_cheaper_and_less_noisy(tmp_path):
    from OmegaQMC.vmcopt_nn_kfac import get_vmcopt_nn_func
    from OmegaQMC.psi.nn.checkpoint import load_nn_checkpoint

    mol = Mole_custom()
    mol.build(atom=str(REPO / '2H2O-psiformer' / 'geo_ini.xyz'),
              unit='Angstrom', basis='sto-3g', spin=0, charge=0,
              verbose=0)
    nelec = mol.n_up + mol.n_down

    # Equilibrated walkers for the checkpointed parameters.  The
    # walker count is a multiple of the chunk size, so the timed
    # pass does not compile a ragged last chunk.
    chunk = 8
    opt = get_vmcopt_nn_func(mol, 'psiformer', jax.random.key(99))
    params, _ = load_nn_checkpoint(str(CHK_2H2O), opt.init_params)
    w = opt.initialize_walkers(jax.random.key(1), 64 * chunk)
    (_, w, _, _), _ = opt.decorr_scan(
        jax.random.key(6), w, jnp.asarray(0.05), params, 1000,
    )

    drv = get_vmc_nn_func(mol, 'psiformer', jax.random.key(99),
                          prefix=str(tmp_path / 'w2'))
    drv.load_checkpoint(str(CHK_2H2O))
    n = w.shape[0]
    e_loc = jnp.concatenate([
        drv._local_energy_batch_p(w[i:i + 100], drv.params)
        for i in range(0, n, 100)
    ])
    d_enr = (e_loc - e_loc.mean())[:, None, None]

    tot, cost = {}, {}
    for name, builder in (('none', vmc_nn_gradients_zvzb),
                          ('fast_warp', vmc_nn_gradients_fast_warp)):
        kw = {'lap_grad': drv.lap_grad} if name == 'none' else {}
        g = _chunk_calls(builder(drv.log_psi, drv.nuc_crds, drv.charges,
                                 nelec, drv.params, **kw), chunk)
        jax.block_until_ready(g(w[:chunk]))
        t0 = time.perf_counter()
        grd_ee_en, grd_ke, grd_logpsi = jax.block_until_ready(g(w))
        cost[name] = (time.perf_counter() - t0) / n
        tot[name] = np.asarray(grd_ee_en + grd_ke + 2.0 * d_enr * grd_logpsi)

    a, b = tot['none'], tot['fast_warp']
    ratio = np.median(a.var(0) / b.var(0))
    d = b - a
    t = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(n))
    print(f"cost {cost['none'] * 1e3:.1f} -> {cost['fast_warp'] * 1e3:.1f}"
          f" ms/sample; median variance ratio {ratio:.1f};"
          f" max paired |t| {np.abs(t).max():.2f}")
    assert cost['fast_warp'] < cost['none']
    assert ratio > 2.0
    assert np.abs(t).max() < 4.0
