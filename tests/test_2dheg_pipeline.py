"""End-to-end pipeline tests for the 2D HEG PsiFormer.

These verify that the 2D-aware PsiFormer + Ewald + system-builder
pipeline produces the analytical Hartree-Fock energy at initialisation
when:

* the Jastrow / backflow / cusp / deep-Jastrow / pair-Jastrow are all
  disabled, leaving only the plane-wave envelope (the Slater
  determinant of plane waves at the closed-shell Fermi sea), and
* the local energy is computed by the same Laplacian + Ewald path the
  production VMC drivers use.

The thermodynamic-limit HF (Stern 1973) is

    E_HF/N = 1/(2 r_s^2) - 4 sqrt(2) / (3 pi r_s)   (unpolarized)

and the analytical finite-N HF closed-form lives in
:func:`OmegaQMC.psi.heg_2d.hf_energy_2d_finite`.  The test checks the
PsiFormer's E_HF computed via the production code path matches the
analytical finite-N HF to ~1 mHa/electron (statistical noise from the
walker average).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from OmegaQMC.psi.heg_2d import (
    build_2deg_system,
    hf_energy_2d_finite,
    hf_energy_2d_td,
)
from OmegaQMC.observables.ewald_2d import (
    build_ewald_2d_tables,
    ewald_2d_pair_energy,
)
from OmegaQMC.psi.nn.heg_wf_module import build_heg_psiformer_wf
from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig


def _build_hf_only_psiformer(rs, N, polarization='unpolarized'):
    """Build a 2D PsiFormer with everything OFF except the envelope.

    Returns ``(wf, sys_info, params, graphdef, other)`` ready for
    JAX-traced evaluation.
    """
    sys_info = build_2deg_system(rs, N_elec=N, polarization=polarization)
    config = HEGPsiFormerConfig(
        n_up=sys_info['nup'], n_down=sys_info['ndown'],
        L=sys_info['L'], n_det=1,
        embedding_dim=16, n_interactions=1,
        two_particle_stream_dim=8, n_attention_heads=1,
        n_virt_pw=0, det_jitter=0.0,
        # Everything off — pure plane-wave Slater.
        use_backflow=False,
        use_cusp=False,
        use_deep_jastrow=False,
        use_pair_jastrow=False,
        use_ghost_atom=False,
        dim=2,
    )
    rngs = nnx.Rngs(0)
    wf = build_heg_psiformer_wf(config, rngs)
    graphdef, params, other = nnx.split(wf, nnx.Param, ...)
    return wf, sys_info, params, graphdef, other


def _local_kinetic_per_walker(graphdef, params, other, walker, dim, n_elec):
    """Replicate the production driver's kinetic energy computation.

    -1/2 * (Laplacian + |grad|^2) of log psi w.r.t. all electron
    coordinates, evaluated at one walker.
    """
    from OmegaQMC.psi.nn.physics import laplacian

    def log_psi_flat(r_flat):
        mdl = nnx.merge(graphdef, params, other)
        return mdl(r_flat.reshape(n_elec, dim)).log

    lap_val, grad_val = laplacian(log_psi_flat)(walker.reshape(-1))
    return -0.5 * (lap_val + jnp.dot(grad_val, grad_val))


@pytest.mark.parametrize('rs,N', [(1.0, 10), (2.0, 10), (5.0, 10)])
def test_2d_psiformer_HF_init_kinetic_matches_analytic(rs, N):
    """Test 0: kinetic-only.  At HF init (envelope-only PsiFormer),
    averaged kinetic energy over walkers matches analytical
    finite-N HF kinetic to ~1% (limited by walker statistics)."""
    wf, sys_info, params, graphdef, other = _build_hf_only_psiformer(
        rs, N,
    )
    L = sys_info['L']
    n_elec = sys_info['N_elec']

    # Analytical reference
    hf_analytic = hf_energy_2d_finite(sys_info)
    T_analytic_per_elec = hf_analytic['kinetic']

    # Sample uniformly distributed walkers in the cell — for the
    # plane-wave HF envelope the kinetic energy is *constant* at the
    # exact HF value at every walker (because the kinetic operator
    # acting on a plane-wave determinant gives sum |k|^2/2 per
    # electron, independent of position).  So a small number of
    # walkers suffices for a near-zero-variance check.
    rng = np.random.default_rng(0)
    n_walkers = 4
    walkers = jnp.asarray(
        rng.uniform(0, L, size=(n_walkers, n_elec, 2)),
    )
    kin_fn = jax.jit(jax.vmap(
        lambda w: _local_kinetic_per_walker(
            graphdef, params, other, w, dim=2, n_elec=n_elec,
        ),
    ))
    kin_per_walker = kin_fn(walkers)
    T_psiformer_per_elec = float(jnp.mean(kin_per_walker)) / n_elec
    # Walker-to-walker variance should be ~0 for the plane-wave
    # determinant.
    walker_spread = float(jnp.std(kin_per_walker / n_elec))
    assert walker_spread < 1e-6, (
        f"rs={rs}, N={N}: walkers should give identical kinetic "
        f"for plane-wave HF, but spread = {walker_spread:.2e}"
    )
    # Mean must match the analytical HF kinetic to numerical precision.
    assert abs(T_psiformer_per_elec - T_analytic_per_elec) < 1e-8, (
        f"rs={rs}, N={N}: PsiFormer T/N {T_psiformer_per_elec:.6f} "
        f"!= analytic {T_analytic_per_elec:.6f}"
    )


def _mcmc_sample_hf(graphdef, params, other, L, n_elec, dim,
                    n_walkers, n_equil, n_steps, seed):
    """Sample walkers from |psi_HF|^2 via Metropolis (uniform-ball
    proposals + accept/reject on |psi(r')/psi(r)|^2).  Returns the
    final equilibrated walker array."""
    from OmegaQMC.psi.nn.periodic import (
        make_cubic_lattice, make_square_lattice, wrap_to_cell,
    )
    if dim == 3:
        lattice = make_cubic_lattice(L)
    else:
        lattice = make_square_lattice(L)

    def log_psi(r, p):
        return nnx.merge(graphdef, p, other)(r).log

    @jax.jit
    def step(rng, walkers, step_size):
        kp, ka = jax.random.split(rng)
        proposed = walkers + step_size * jax.random.normal(
            kp, walkers.shape,
        )
        # Wrap each walker.
        wrapped = jax.vmap(lambda x: wrap_to_cell(x, lattice))(proposed)
        lp_old = jax.vmap(lambda r: log_psi(r, params))(walkers)
        lp_new = jax.vmap(lambda r: log_psi(r, params))(wrapped)
        u = jax.random.uniform(ka, (n_walkers,))
        accept = jnp.log(u) < 2.0 * (lp_new - lp_old)
        new = jnp.where(accept[:, None, None], wrapped, walkers)
        return new, accept.mean()

    rng = jax.random.key(seed)
    rng, kw = jax.random.split(rng)
    walkers = L * jax.random.uniform(
        kw, (n_walkers, n_elec, dim),
    )
    step_size = 0.5
    for _ in range(n_equil):
        rng, sk = jax.random.split(rng)
        walkers, acc = step(sk, walkers, step_size)
        # Adapt step
        step_size = step_size * (
            1.1 if float(acc) > 0.6 else (0.9 if float(acc) < 0.4 else 1.0)
        )
    for _ in range(n_steps):
        rng, sk = jax.random.split(rng)
        walkers, _ = step(sk, walkers, step_size)
    return walkers


@pytest.mark.parametrize('rs,N', [(2.0, 10), (5.0, 10)])
def test_2d_psiformer_HF_init_total_matches_finite_N_hf(rs, N):
    """Test 1 (Phase 0): HF baseline.  Run a short MCMC on the 2D
    PsiFormer at HF init (envelope-only) and confirm the total local
    energy <T + V_ee>_{|psi_HF|^2} matches the analytical finite-N HF
    energy to within MC statistical error.

    Sampling matters here: |psi_HF|^2 has Pauli exchange correlations
    (same-spin electrons repel) so uniform walker samples
    over-estimate V_ee.  We MCMC-sample to get the true HF
    expectation."""
    wf, sys_info, params, graphdef, other = _build_hf_only_psiformer(
        rs, N,
    )
    L = sys_info['L']
    n_elec = sys_info['N_elec']

    hf_analytic = hf_energy_2d_finite(sys_info)
    E_analytic = hf_analytic['total']

    ewald_tables = build_ewald_2d_tables(L)

    walkers = _mcmc_sample_hf(
        graphdef, params, other,
        L=L, n_elec=n_elec, dim=2,
        n_walkers=128, n_equil=200, n_steps=50,
        seed=0,
    )

    # Kinetic (constant at HF init: sum |k|^2/2).
    kin_fn = jax.jit(jax.vmap(
        lambda w: _local_kinetic_per_walker(
            graphdef, params, other, w, dim=2, n_elec=n_elec,
        ),
    ))
    T_per_elec = float(jnp.mean(kin_fn(walkers))) / n_elec

    pot_fn = jax.jit(jax.vmap(
        lambda w: ewald_2d_pair_energy(w, ewald_tables),
    ))
    V_per_walker = pot_fn(walkers)
    V_per_elec = float(jnp.mean(V_per_walker)) / n_elec
    serr_V = (
        float(jnp.std(V_per_walker)) / n_elec / np.sqrt(walkers.shape[0])
    )

    E_psi = T_per_elec + V_per_elec
    diff = E_psi - E_analytic
    print(
        f"\nrs={rs} N={N}: T={T_per_elec:.5f} (analytic "
        f"{hf_analytic['kinetic']:.5f}), "
        f"V={V_per_elec:.5f} +- {serr_V:.5f}, E_HF psi={E_psi:.5f} "
        f"vs analytic={E_analytic:.5f}, diff={diff*1000:.2f} mHa",
    )
    # Allow for a 5*serr margin on the potential, plus a small finite
    # equilibration error (~few mHa at 200 equil steps).
    tol = max(5.0 * serr_V, 0.005)
    assert abs(diff) < tol, (
        f"rs={rs}, N={N}: HF E/N psi={E_psi:.5f} vs analytic="
        f"{E_analytic:.5f}; diff={diff*1000:.2f} mHa, tol="
        f"{tol*1000:.2f} mHa"
    )


def test_2d_psiformer_HF_init_kinetic_matches_TD_at_large_N():
    """At large N the finite-N kinetic should approach the TD value
    1/(2 rs^2) — sanity check at N=58, rs=2."""
    wf, sys_info, params, graphdef, other = _build_hf_only_psiformer(
        rs=2.0, N=58,
    )
    L = sys_info['L']
    n_elec = sys_info['N_elec']

    rng = np.random.default_rng(0)
    walkers = jnp.asarray(
        rng.uniform(0, L, size=(2, n_elec, 2)),
    )
    kin_fn = jax.jit(jax.vmap(
        lambda w: _local_kinetic_per_walker(
            graphdef, params, other, w, dim=2, n_elec=n_elec,
        ),
    ))
    T_per_elec = float(jnp.mean(kin_fn(walkers))) / n_elec
    T_td = hf_energy_2d_td(2.0, 'unpolarized') - (
        - 4 * np.sqrt(2) / (3 * np.pi * 2.0)
    )  # subtract the exchange to leave only the kinetic
    # Should be within ~5% of TD limit at N=58
    assert abs(T_per_elec - T_td) / abs(T_td) < 0.05, (
        f"T/N at N=58 = {T_per_elec:.5f} vs TD {T_td:.5f}"
    )
