"""Space-warp nuclear-force estimators of NN trials: SWCT and fast warp.

Per-sample terms against brute-force autodiff, the translational sum
rule, zero variance for a single atom, and agreement of SWCT, ZVZB and
fast warp through the drivers.
"""
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

jax.config.update('jax_enable_x64', True)

from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.psi.nn.adapter import make_nn_log_psi  # noqa: E402
from OmegaQMC.vmc_nn import get_vmc_nn_func  # noqa: E402
from OmegaQMC.observables.force import (  # noqa: E402
    FORCE_WARPS_NN,
    _warp_weights_distance_vg,
    postproc_h5_pgcs,
    vmc_nn_gradients_fast_warp,
    vmc_nn_gradients_swct,
    _fast_warp_terms,
    _logpsi_grad_hessian_blocks,
    _warp_weight_laplacian,
)
from OmegaQMC import (  # noqa: E402
    generate_molecular_orbitals,
    get_vmc_gto_func,
)


# ---------------------------------------------------------------------------
# From test_force_swct_nn.py
#
# Tests for the SWCT nuclear-force estimator of NN trials.
#
# ``force_warp='swct'`` for NN trials is Eq. 14 of Qian, Fu, Ren and
# Chen, J. Chem. Phys. 157, 164104 (2022): the local energy is
# differentiated along the space warp ``r_i -> r_i + dR_A w_A(r_i)``.
# These tests check
#
# * every per-sample component against brute-force autodiff (full
#   Hessian Laplacian, reverse-mode derivatives), on the FermiNet
#   (linearize Laplacian) and PsiFormer (forward-Laplacian) paths;
# * the per-sample translational sum rule on an asymmetric molecule;
# * the zero-variance limit of a single atom;
# * agreement with the fast-warp estimator on the same samples through
#   the driver (PGCS-free).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# From test_force_fast_warp.py
#
# Tests for the fast-warp nuclear-force estimator.
#
# Fast-warp (Qian, Li and Chen, Faraday Discuss. 254, 529 (2024), SI
# Eq. 19) applies the SWCT space warp but removes every kinetic-energy
# derivative by Hermiticity.  These tests check
#
# * the helpers and :func:`_fast_warp_terms` against brute-force
#   autodiff, including the commutator the warp terms replace;
# * the zero-variance limit of a single atom, where the warp weights are
#   identically 1 and translation invariance makes every per-sample
#   component vanish (NN and GTO, both weight schemes);
# * agreement with the existing estimator on the same samples (GTO SWCT
#   on H2; NN no-warp ZVZB on H2, through the driver, PGCS-free);
# * optionally, the efficiency gain on the 2H2O PsiFormer checkpoint
#   (skipped when the checkpoint is absent).
# ---------------------------------------------------------------------------


REPO = Path(__file__).resolve().parents[1]
CHK_2H2O = REPO / '2H2O-psiformer' / '2h2o_kfac.chk.h5'


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
            prefix=str(tmp_path / 'bad'), force_warp='bogus',
        )
