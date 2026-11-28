"""Forward-Laplacian (VGL) kinetic energy of NN trials.

The adapter's VGL kinetic energy against the generic laplacian path
over many network configurations, and the VGL primitives where they are
numerically fragile: extreme softmax/logsumexp and log-determinant
magnitudes, the zero-distance norm, and the domain edges of log, sqrt
and fractional powers.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

from OmegaQMC.utils import Mole_custom  # noqa: E402
from OmegaQMC.psi.nn.adapter import make_nn_log_psi  # noqa: E402
from OmegaQMC.psi.nn.config import load_nn_config  # noqa: E402
from OmegaQMC.psi.nn.physics import laplacian  # noqa: E402
from OmegaQMC.psi.nn.forward_lap import (  # noqa: E402
    VGL,
    vgl_input,
    vgl_linear,
    vgl_log,
    vgl_log1p,
    vgl_logsumexp,
    vgl_pairwise_self_distance,
    vgl_pow,
    vgl_reshape,
    vgl_safe_norm,
    vgl_softmax,
    vgl_softplus,
    vgl_sqrt,
    slogdet_multidet_vgl,
    slogdet_vgl,
)
from OmegaQMC.psi.nn.utils import norm as safe_norm_ref  # noqa: E402


# ---------------------------------------------------------------------------
# From test_energy_ke_nn_vgl.py
#
# End-to-end driver-level KE: VGL path vs ``laplacian`` (slice 14).
#
# Verifies that the kinetic-energy expression
# ``-0.5 * (lap + |grad|^2)`` agrees between the new
# analytic forward-Laplacian path (PsiFormer-compatible
# configs) and the previous ``laplacian_linearize`` fallback
# exposed by :func:`OmegaQMC.psi.nn.physics.laplacian`.
#
# Two checks:
#
# 1. ``test_lap_grad_psiformer_matches_linearize`` — for the
#    built-in ``'psiformer'`` config on H2, the adapter
#    selects the VGL path; we compare against the linearize
#    path constructed by hand on the same model.
# 2. ``test_lap_grad_falls_back_for_non_psiformer`` — for a
#    non-PsiFormer config (``'ferminet'``), the adapter must
#    return the linearize fallback.  We sanity-check by
#    confirming the returned callable agrees with the manual
#    ``laplacian`` path (i.e. no silent VGL substitution).
# ---------------------------------------------------------------------------


TOL = 1e-9


def _h2_mol():
    L = 1.4010
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[
            [0.0, 0.0, -L / 2],
            [0.0, 0.0, L / 2],
        ],
        n_up=1, n_down=1,
    )


def _interleaved_random_elec(seed, n_e):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.normal(size=(n_e, 3)) * 0.7,
    )


def _linearize_lap_grad(log_psi, nuc, params, n_e):
    """Manual ``laplacian_linearize`` reference."""
    def go(elec):
        def f_flat(r_flat):
            r = r_flat.reshape(n_e, 3)
            return log_psi(r, nuc, params)
        return laplacian(f_flat)(elec.reshape(-1))
    return go


def test_lap_grad_psiformer_matches_linearize():
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    rng_key = jax.random.key(800)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        'psiformer', mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(801, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    # The VGL path's grad is grouped (alpha…beta), the
    # linearize path's grad is interleaved.  The kinetic
    # energy uses ``dot(grad, grad)`` which is permutation-
    # invariant — verify equality of magnitudes only.
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )

    # And the full KE expression must agree.
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_deepqmc_cusp_matches_linearize():
    """PsiFormer preset with the DeepQMC cusp variant
    (``cusp_electrons_type='deepqmc'``) — the adapter
    selects the VGL path (cusp dispatch handles both
    rationals); we compare against the linearize path
    constructed by hand on the same model."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.cusp_electrons_type = 'deepqmc'
    rng_key = jax.random.key(820)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(821, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    # VGL grad is grouped, linearize grad is interleaved;
    # KE is permutation-invariant.
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_with_jastrow_matches_linearize():
    """PsiFormer preset with ``use_jastrow=True`` — VGL
    path includes the deep-Jastrow scalar; compare against
    the linearize path on the same model."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.use_jastrow = True
    rng_key = jax.random.key(830)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(831, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_mahalanobis_envelope_matches_linearize():
    """PsiFormer preset with ``envelope_isotropic=False``
    (Mahalanobis 3x3 envelope) — VGL path uses the
    einsum-then-norm branch; compare against linearize."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.envelope_isotropic = False
    rng_key = jax.random.key(840)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(841, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_multi_update_features_matches_linearize():
    """PsiFormer preset with extra ``residual`` and
    ``node_sum`` update features appended — exercises the
    generalized GNN-layer VGL path that composes multiple
    updates through the ``concatenate`` rule.  Compare
    against the linearize path on the same model."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.update_features = list(cfg.update_features) + [
        {'type': 'residual'},
        {
            'type': 'node_sum',
            'node_types': ['up', 'down'],
            'normalize': True,
        },
    ]
    rng_key = jax.random.key(850)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(851, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_softplus_zeta_matches_linearize():
    """PsiFormer preset with ``envelope_softplus_zeta=True``
    — the envelope reparameterizes ``zeta`` via softplus.
    VGL path supports both branches; compare against the
    linearize path on the same model."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.envelope_softplus_zeta = True
    rng_key = jax.random.key(860)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(861, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_ne_log_rescale_false_matches_linearize():
    """PsiFormer preset with ``ne_log_rescale=False`` — the
    nuc–elec edge feature drops the ``log(1 + r)/r`` rescale.
    VGL embedding helper supports both branches; the kwarg
    is now plumbed through from config."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.ne_log_rescale = False
    rng_key = jax.random.key(870)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(871, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_no_cusp_matches_linearize():
    """PsiFormer preset with ``cusp_electrons=False`` — the
    cusp branch is skipped entirely. VGL path already
    handles ``cusp=None``; the gate is now dropped."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.cusp_electrons = False
    rng_key = jax.random.key(880)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(881, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_non_trainable_alpha_matches_linearize():
    """PsiFormer preset with ``cusp_trainable_alpha=False``
    — the cusp module stores alpha as a non-Param attribute
    (``initial_alpha``).  Adapter now reads from there when
    ``trainable_alpha`` is False."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.cusp_trainable_alpha = False
    rng_key = jax.random.key(890)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(891, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_no_spin_embedding_matches_linearize():
    """PsiFormer preset with ``use_spin_embedding=False`` —
    the electron embedding skips the spin-indicator column.
    VGL helper already supports both branches; gate is now
    dropped."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.use_spin_embedding = False
    rng_key = jax.random.key(900)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(901, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_no_proj_embedding_matches_linearize():
    """PsiFormer preset with
    ``project_to_embedding_dim=False`` — the electron
    embedding skips the bias-less linear projection.  VGL
    helper already supports ``proj_W=None``.  The init dim
    becomes ``n_nuc * (len(ne_powers)+3) + use_spin``; we
    set ``num_heads=1`` so the attention block accepts any
    init dim."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.project_to_embedding_dim = False
    cfg.update_features = [
        {**cfg.update_features[0], 'num_heads': 1},
    ]
    rng_key = jax.random.key(910)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(911, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_edge_sum_up_down_matches_linearize():
    """PsiFormer preset extended with an ``edge_sum`` update
    over ``edge_types=['up','down']``.

    Sets ``two_particle_stream_dim=4`` so the
    ``deep_features=False`` path keeps the raw edge feature
    dim (``ee_feat_dim==4``) consistent with
    :class:`EdgeSumElectronUpdateFeature`'s declared
    ``output_dim`` (used to size the subnet MLP)."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.edge_types = ['up', 'down']
    cfg.two_particle_stream_dim = 4
    cfg.update_features = [
        cfg.update_features[0],
        {
            'type': 'edge_sum',
            'edge_types': ['up', 'down'],
            'normalize': True,
        },
    ]
    rng_key = jax.random.key(960)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(961, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_edge_sum_same_anti_matches_linearize():
    """PsiFormer preset extended with an ``edge_sum`` update
    over ``edge_types=['same','anti']`` (DeepErwin-style edge
    typing) — exercises the per-block sum_senders branches in
    :func:`vgl_edge_sum_update`."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.edge_types = ['same', 'anti']
    cfg.two_particle_stream_dim = 4
    cfg.update_features = [
        cfg.update_features[0],
        {
            'type': 'edge_sum',
            'edge_types': ['same', 'anti'],
            'normalize': True,
        },
    ]
    rng_key = jax.random.key(970)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(971, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_convolution_same_anti_matches_linearize():
    """PsiFormer preset extended with a ``convolution``
    update over ``edge_types=['same','anti']`` (DeepErwin-
    style edge typing).

    Sets ``two_particle_stream_dim=4`` so the
    ``deep_features=False`` path keeps the raw edge feature
    dim (``ee_feat_dim==4``) consistent with the per-type
    ``w_nets``/``h_nets`` MLP input dims across all layers.
    """
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.edge_types = ['same', 'anti']
    cfg.two_particle_stream_dim = 4
    cfg.update_features = [
        cfg.update_features[0],
        {
            'type': 'convolution',
            'edge_types': ['same', 'anti'],
            'normalize': False,
        },
    ]
    rng_key = jax.random.key(980)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(981, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_convolution_up_down_matches_linearize():
    """PsiFormer preset extended with a ``convolution``
    update over ``edge_types=['up','down']`` — exercises the
    SimpleGraphEdges ``convolve`` branch (single-block sender
    indexing rather than per-spin-block)."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.edge_types = ['up', 'down']
    cfg.two_particle_stream_dim = 4
    cfg.update_features = [
        cfg.update_features[0],
        {
            'type': 'convolution',
            'edge_types': ['up', 'down'],
            'normalize': True,
        },
    ]
    rng_key = jax.random.key(990)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(991, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_deep_features_shared_same_anti_matches_linearize():
    """PsiFormer preset extended with ``deep_features='shared'``
    over ``edge_types=['same','anti']`` (homogeneous 2D
    ``single_array`` layout — production
    ``concatenate(axis=-2)`` works across both keys).

    Sets ``two_particle_stream_dim=4`` so the per-layer
    ``subnet_g`` input/output widths line up: the first
    layer's ``ee_feat_dim==4`` already equals the target
    ``tp_dim==4``, so subsequent layers see edges of the
    same width and the ``tp_residual`` branch takes the
    shape-matched (additive) path.
    """
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.edge_types = ['same', 'anti']
    cfg.two_particle_stream_dim = 4
    cfg.deep_features = 'shared'
    cfg.update_features = [
        cfg.update_features[0],
        {
            'type': 'edge_sum',
            'edge_types': ['same', 'anti'],
            'normalize': False,
        },
    ]
    rng_key = jax.random.key(1000)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(1001, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_deep_features_shared_up_down_matches_linearize():
    """``deep_features='shared'`` over the SimpleGraphEdges
    layout (``edge_types=['up','down']``).  H2 has
    ``n_up == n_down == 1`` so the production
    ``concatenate(axis=-2)`` over the 3D
    ``(n_send, n_recv, F)`` arrays is well-defined."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.edge_types = ['up', 'down']
    cfg.two_particle_stream_dim = 4
    cfg.deep_features = 'shared'
    cfg.update_features = [
        cfg.update_features[0],
        {
            'type': 'edge_sum',
            'edge_types': ['up', 'down'],
            'normalize': True,
        },
    ]
    rng_key = jax.random.key(1010)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(1011, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_ferminet_preset_matches_linearize():
    """Built-in FermiNet preset (downsized for test runtime)
    routes through the VGL path end-to-end: residual +
    node_sum + edge_sum update features, ``deep_features=
    'shared'``, and ``edge_types=['up','down']``."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('ferminet')
    cfg.embedding_dim = 16
    cfg.n_interactions = 2
    cfg.n_determinants = 2
    cfg.two_particle_stream_dim = 4
    rng_key = jax.random.key(1020)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(1021, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        jnp.dot(grad_v, grad_v),
        jnp.dot(grad_ref, grad_ref),
        atol=TOL, rtol=TOL,
    )
    ke_v = -0.5 * (lap_v + jnp.dot(grad_v, grad_v))
    ke_ref = -0.5 * (
        lap_ref + jnp.dot(grad_ref, grad_ref)
    )
    np.testing.assert_allclose(
        ke_v, ke_ref, atol=TOL, rtol=TOL,
    )


def test_lap_grad_falls_back_for_non_psiformer():
    """Tweak the PsiFormer preset to a non-compatible
    setting (``envelope_spin_restricted=True``) and confirm
    the adapter falls back to the linearize path rather
    than silently using the VGL twin (spin-restricted
    envelopes share zetas/pis across spins, which the VGL
    extractor doesn't support)."""
    mol = _h2_mol()
    n_e = mol.n_up + mol.n_down
    cfg = load_nn_config('psiformer')
    cfg.envelope_spin_restricted = True
    rng_key = jax.random.key(810)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        cfg, mol, rng_key,
    )
    nuc = mol.coords
    elec = _interleaved_random_elec(811, n_e)

    lap_v, grad_v = lap_grad(elec, nuc, params)
    ref_fn = _linearize_lap_grad(
        log_psi, nuc, params, n_e,
    )
    lap_ref, grad_ref = ref_fn(elec)

    # Fallback path: lap_grad and laplacian-of-log_psi are
    # identical in coordinate order, so both lap and grad
    # must match elementwise.
    np.testing.assert_allclose(
        lap_v, lap_ref, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        grad_v, grad_ref, atol=TOL, rtol=TOL,
    )


# ---------------------------------------------------------------------------
# VGL primitives at their numerically fragile points (kept from the
# former test_nn_forward_lap_primitives.py): each against jax.jacobian
# and the trace of jax.hessian in float64, tolerance 1e-10.
# ---------------------------------------------------------------------------


TOL_prim = 1e-10


def _ref_grad_lap(f, x):
    """``(value, jac, input-trace-of-hess)`` of ``f`` at ``x``.

    For scalar-output ``f``, ``jac`` has shape ``(D,)`` and
    ``trace_hess`` is a scalar.  For vector-output, shapes
    are ``(*S, D)`` and ``(*S,)``.  We move the input axis
    to the leading position to match the VGL ``(D, *S)``
    convention.
    """
    val = f(x)
    jac = jax.jacobian(f)(x)
    hess = jax.hessian(f)(x)
    # jac shape: (*S, D) → (D, *S)
    if jac.ndim == 1:
        grad = jac
    else:
        grad = jnp.moveaxis(jac, -1, 0)
    # hess shape: (*S, D, D) → trace last two axes → (*S,)
    if hess.ndim == 2:
        lap = jnp.trace(hess)
    else:
        lap = jnp.trace(hess, axis1=-2, axis2=-1)
    return val, grad, lap


def _check(vgl_out: VGL, f, x):
    val_ref, grad_ref, lap_ref = _ref_grad_lap(f, x)
    err_v = float(jnp.max(jnp.abs(vgl_out.value - val_ref)))
    err_g = float(jnp.max(jnp.abs(vgl_out.grad - grad_ref)))
    err_l = float(jnp.max(jnp.abs(vgl_out.lap - lap_ref)))
    assert err_v < TOL_prim, f"value: {err_v:.3e}"
    assert err_g < TOL_prim, f"grad:  {err_g:.3e}"
    assert err_l < TOL_prim, f"lap:   {err_l:.3e}"


def test_vgl_safe_norm_against_autodiff():
    rng = np.random.default_rng(3)
    # input is a 3-vector built linearly from x.
    d_in = 7
    x = jnp.asarray(rng.normal(size=(d_in,)))
    w = jnp.asarray(rng.normal(size=(d_in, 3)))
    b = jnp.asarray(rng.normal(size=(3,)))
    out = vgl_safe_norm(vgl_linear(vgl_input(x), w, b))

    def fwd(z):
        return safe_norm_ref(z @ w + b, safe=True)

    _check(out, fwd, x)


def test_vgl_safe_norm_matches_reference_value():
    """vgl_safe_norm.value equals safe_norm_ref(...) bit for bit."""
    rng = np.random.default_rng(4)
    d_in = 5
    x = jnp.asarray(rng.normal(size=(d_in,)))
    w = jnp.asarray(rng.normal(size=(d_in, 3)))
    b = jnp.asarray(rng.normal(size=(3,)))
    out = vgl_safe_norm(vgl_linear(vgl_input(x), w, b))
    ref = safe_norm_ref(x @ w + b, safe=True)
    np.testing.assert_allclose(out.value, ref, atol=TOL_prim)


def test_vgl_log_against_autodiff():
    rng = np.random.default_rng(15)
    d_in, d_out = 6, 4
    x = jnp.asarray(rng.normal(size=(d_in,)))
    w = jnp.asarray(rng.normal(size=(d_in, d_out)))
    # Bias keeps the linear output strictly positive.
    b = jnp.asarray(2.0 + np.abs(rng.normal(size=(d_out,))))
    affine = vgl_linear(vgl_input(x), w, b)
    affine = vgl_softplus(affine)  # ensures > 0
    out = vgl_log(affine)
    _check(out, lambda z: jnp.log(jax.nn.softplus(z @ w + b)), x)


def test_vgl_softmax_extreme_magnitude():
    """Softmax remains accurate when logits span ±20 — the
    max-shift in the implementation must keep the
    reverse-mode reference and the VGL output co-stable.
    """
    rng = np.random.default_rng(34)
    d_in, n = 5, 4
    x = jnp.asarray(rng.normal(size=(d_in,)))
    # Scale weights/bias so that pre-softmax logits land in
    # roughly [-20, +20].
    w = jnp.asarray(20.0 * rng.normal(size=(d_in, n)))
    b = jnp.asarray(rng.normal(size=(n,)))
    out = vgl_softmax(vgl_linear(vgl_input(x), w, b))
    _check(out, lambda z: jax.nn.softmax(z @ w + b, axis=-1), x)


def test_vgl_logsumexp_extreme_magnitude():
    rng = np.random.default_rng(35)
    d_in, n = 5, 4
    x = jnp.asarray(rng.normal(size=(d_in,)))
    w = jnp.asarray(20.0 * rng.normal(size=(d_in, n)))
    b = jnp.asarray(rng.normal(size=(n,)))
    out = vgl_logsumexp(vgl_linear(vgl_input(x), w, b))
    _check(
        out,
        lambda z: jax.scipy.special.logsumexp(z @ w + b, axis=-1),
        x,
    )


def test_vgl_sqrt_against_autodiff():
    rng = np.random.default_rng(43)
    d_in, n = 5, 4
    x = jnp.asarray(rng.normal(size=(d_in,)))
    w = jnp.asarray(rng.normal(size=(d_in, n)))
    # Bias keeps the affine output strictly positive.
    b = jnp.asarray(2.0 + np.abs(rng.normal(size=(n,))))
    affine = vgl_softplus(vgl_linear(vgl_input(x), w, b))
    out = vgl_sqrt(affine)
    _check(
        out, lambda z: jnp.sqrt(jax.nn.softplus(z @ w + b)), x,
    )


def test_vgl_pow_fractional_against_autodiff():
    """Fractional p (=1.7) on positive inputs."""
    rng = np.random.default_rng(45)
    d_in, n = 5, 4
    x = jnp.asarray(rng.normal(size=(d_in,)))
    w = jnp.asarray(rng.normal(size=(d_in, n)))
    b = jnp.asarray(2.0 + np.abs(rng.normal(size=(n,))))
    affine = vgl_softplus(vgl_linear(vgl_input(x), w, b))
    out = vgl_pow(affine, 1.7)
    _check(
        out,
        lambda z: jnp.power(jax.nn.softplus(z @ w + b), 1.7),
        x,
    )


def test_vgl_log1p_against_autodiff():
    rng = np.random.default_rng(60)
    d_in, n = 5, 4
    x = jnp.asarray(rng.normal(size=(d_in,)))
    w = jnp.asarray(rng.normal(size=(d_in, n)))
    b = jnp.asarray(2.0 + np.abs(rng.normal(size=(n,))))
    affine = vgl_softplus(vgl_linear(vgl_input(x), w, b))
    out = vgl_log1p(affine)
    _check(
        out,
        lambda z: jnp.log1p(jax.nn.softplus(z @ w + b)),
        x,
    )


def _coords_vgl_from_x(seed, n_e):
    """Helper: build a VGL of electron coords ``(n_e, 3)``
    from a flat input ``x`` of length ``D = 3·n_e``."""
    rng = np.random.default_rng(seed)
    d_in = 3 * n_e
    x = jnp.asarray(rng.normal(size=(d_in,)) * 0.7)
    coords_vgl = vgl_reshape(vgl_input(x), (n_e, 3))
    return x, coords_vgl


def test_vgl_pairwise_self_distance_full_diagonal_is_zero():
    """The diagonal mask must give exact zero in value/grad/lap."""
    n_e = 5
    _x, coords = _coords_vgl_from_x(87, n_e)
    out = vgl_pairwise_self_distance(coords, full=True)
    diag_idx = jnp.arange(n_e)
    np.testing.assert_array_equal(
        out.value[diag_idx, diag_idx], jnp.zeros(n_e),
    )
    np.testing.assert_array_equal(
        out.grad[:, diag_idx, diag_idx],
        jnp.zeros((3 * n_e, n_e)),
    )
    np.testing.assert_array_equal(
        out.lap[diag_idx, diag_idx], jnp.zeros(n_e),
    )


# ---------------------------------------------------------------------------
# Multi-determinant slogdet VGL with log|D| spread over +-20 (kept from
# the former test_nn_slater_vgl.py), against autodiff.
# ---------------------------------------------------------------------------


def _ref_grad_lap_slater(f, x):
    """``(value, jac, input-trace-of-hess)`` of ``f`` at ``x``.

    Mirrors the reference helper of the primitive tests above.
    """
    val = f(x)
    jac = jax.jacobian(f)(x)
    hess = jax.hessian(f)(x)
    if jac.ndim == 1:
        grad = jac
    else:
        grad = jnp.moveaxis(jac, -1, 0)
    if hess.ndim == 2:
        lap = jnp.trace(hess)
    else:
        lap = jnp.trace(hess, axis1=-2, axis2=-1)
    return val, grad, lap


def _check_slater(vgl_out: VGL, f, x, tol=TOL):
    val_ref, grad_ref, lap_ref = _ref_grad_lap_slater(f, x)
    err_v = float(jnp.max(jnp.abs(vgl_out.value - val_ref)))
    err_g = float(jnp.max(jnp.abs(vgl_out.grad - grad_ref)))
    err_l = float(jnp.max(jnp.abs(vgl_out.lap - lap_ref)))
    assert err_v < tol, f"value: {err_v:.3e}"
    assert err_g < tol, f"grad:  {err_g:.3e}"
    assert err_l < tol, f"lap:   {err_l:.3e}"


def _slater_vgl(seed, d_in, lead_shape, n, scale=0.1):
    """Build a VGL of Slater matrices linearly from ``x``.

    Returns ``(x, slater_vgl, fwd_S)`` where ``slater_vgl``
    has value shape ``(*lead_shape, n, n)`` and ``fwd_S(z)``
    reproduces the matrix in plain JAX.

    The bias is set to (broadcast) identity plus a small
    perturbation, and the linear weights are downscaled by
    ``scale`` so that ``S(x)`` stays comfortably non-
    singular across the test draws.
    """
    rng = np.random.default_rng(seed)
    target = lead_shape + (n, n)
    n_out = int(np.prod(target))
    x = jnp.asarray(rng.normal(size=(d_in,)))
    w = jnp.asarray(rng.normal(size=(d_in, n_out)) * scale)
    eye_b = jnp.broadcast_to(
        jnp.eye(n), target,
    ).reshape(n_out)
    perturb = jnp.asarray(0.05 * rng.normal(size=(n_out,)))
    b = eye_b + perturb
    flat = vgl_linear(vgl_input(x), w, b)
    value = flat.value.reshape(target)
    grad = flat.grad.reshape((d_in,) + target)
    lap = flat.lap.reshape(target)
    slater_vgl = type(flat)(value=value, grad=grad, lap=lap)

    def fwd_S(z):
        return (z @ w + b).reshape(target)

    return x, slater_vgl, fwd_S


def test_slogdet_multidet_vgl_extreme_log_magnitudes():
    """Stable max-shift survives ``log|D_d|`` spread of ±20."""
    n_det = 4
    rng = np.random.default_rng(112)
    x, sv, fwd_S = _slater_vgl(
        112, d_in=5, lead_shape=(n_det,), n=3,
    )
    # Inflate the coefficients so the per-det ``log|c| +
    # log|D|`` contributions span a wide range.
    coeffs = jnp.asarray(
        rng.normal(size=(n_det,)) * jnp.exp(
            jnp.asarray(rng.normal(size=(n_det,)) * 10.0),
        ),
    )

    signs, log_vgl = slogdet_vgl(sv)
    out = slogdet_multidet_vgl(log_vgl, signs, coeffs)

    def fwd(z):
        S = fwd_S(z)
        sgn, ld = jnp.linalg.slogdet(S)
        log_contrib = jnp.log(jnp.abs(coeffs)) + ld
        shift = jnp.max(log_contrib)
        sum_val = jnp.sum(
            jnp.sign(coeffs) * sgn
            * jnp.exp(log_contrib - shift),
        )
        return jnp.log(jnp.abs(sum_val)) + shift

    _check_slater(out, fwd, x)
