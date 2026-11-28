"""Hartree-Fock reference of the 2D electron gas.

Fermi sea, system construction and Stern's closed forms, finite-N to
thermodynamic limit, and the 2D PsiFormer at its HF initialization
reproducing the finite-N HF kinetic and total energies (the Laplacian
plus 2D Ewald path end to end).
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from OmegaQMC.psi.heg_2d import (  # noqa: E402
    CLOSED_SHELL_PER_SPIN_2D,
    build_2deg_system,
    closed_shell_n_per_spin,
    generate_2d_fermi_sea,
    hf_energy_2d_finite,
    hf_energy_2d_td,
    hf_exchange_energy_2d_td,
    hf_kinetic_energy_2d_td,
    is_closed_shell_n_per_spin,
)
from OmegaQMC.observables.ewald_2d import (  # noqa: E402
    build_ewald_2d_tables,
    ewald_2d_pair_energy,
)
from OmegaQMC.psi.nn.heg_wf_module import build_heg_psiformer_wf  # noqa: E402
from OmegaQMC.psi.nn.heg_wf import HEGPsiFormerConfig  # noqa: E402


# ---------------------------------------------------------------------------
# From test_heg_2d.py
#
# Unit tests for OmegaQMC.psi.heg_2d (2D HEG system + analytical HF).
#
# Verifies:
# * Closed-shell Fermi-sea generation for the standard 2D HEG sizes.
# * :func:`build_2deg_system` cell geometry: ``L = sqrt(pi N) * r_s``.
# * Hartree-Fock thermodynamic-limit closed forms (Stern 1973):
#     - unpolarized: ``E_HF/N = 1/(2 r_s^2) - 4 sqrt(2) / (3 pi r_s)``
#     - polarized  : ``E_HF/N = 1/r_s^2 - 8 / (3 pi r_s)``
# * Finite-N HF approaches the TD limit at small finite-size error.
# * Cross-check: HF total + correlation energy lies above QMC reference
#   (Attaccalite 2002), as it must by the variational principle.
# ---------------------------------------------------------------------------


# Attaccalite 2002 PRL 88, 256601 — FN-DMC backflow per-electron energies.
ATTACCALITE_2002_BACKFLOW = {
    # rs : {'unpolarized': E_per_elec [Ha], 'polarized': ...}
    1:  {'unpolarized': -0.20372,    'polarized': +0.13109},
    2:  {'unpolarized': -0.25721,    'polarized': -0.19359},
    5:  {'unpolarized': -0.149518,   'polarized': -0.143610},
    10: {'unpolarized': -0.085427,   'polarized': -0.084584},
    20: {'unpolarized': -0.046385,   'polarized': -0.0462488},
    30: {'unpolarized': -0.031941,   'polarized': -0.031938},
}


# ---------------------------------------------------------------------
# Closed-shell counts
# ---------------------------------------------------------------------
def test_closed_shell_per_spin_first_seven():
    """Documented closed-shell counts: 1, 5, 9, 13, 21, 25, 29 (= shells
    of |n|^2 = 0, 1, 2, 4, 5, 8, 9 cumulatively)."""
    assert CLOSED_SHELL_PER_SPIN_2D[:7] == (1, 5, 9, 13, 21, 25, 29)


def test_is_closed_shell():
    assert is_closed_shell_n_per_spin(29)
    assert not is_closed_shell_n_per_spin(30)
    assert is_closed_shell_n_per_spin(57)


def test_closed_shell_n_per_spin_round_up():
    assert closed_shell_n_per_spin(28) == 29
    assert closed_shell_n_per_spin(29) == 29
    assert closed_shell_n_per_spin(30) == 37


# ---------------------------------------------------------------------
# Fermi-sea generation
# ---------------------------------------------------------------------
@pytest.mark.parametrize('n_per_spin', [1, 5, 9, 13, 21, 25, 29, 37, 45, 57])
def test_fermi_sea_size_matches_closed_shell(n_per_spin):
    grid = generate_2d_fermi_sea(n_per_spin)
    assert grid.shape == (n_per_spin, 2)


def test_fermi_sea_n29_shells():
    """nup=29 fills shells |n|^2 = 0, 1, 2, 4, 5, 8, 9
    with multiplicities (1, 4, 4, 4, 8, 4, 4) summing to 29."""
    grid = generate_2d_fermi_sea(29)
    n_sq = np.sum(grid ** 2, axis=1)
    shells = sorted(set(n_sq.tolist()))
    assert shells == [0, 1, 2, 4, 5, 8, 9]
    counts = [int(np.sum(n_sq == s)) for s in shells]
    assert counts == [1, 4, 4, 4, 8, 4, 4]


def test_fermi_sea_no_duplicate_kpoints():
    grid = generate_2d_fermi_sea(57)
    s = {tuple(int(x) for x in v) for v in grid}
    assert len(s) == 57


def test_fermi_sea_invalid_n_raises():
    with pytest.raises(ValueError):
        generate_2d_fermi_sea(30)


# ---------------------------------------------------------------------
# System builder
# ---------------------------------------------------------------------
def test_build_2deg_system_unpolarized_geometry():
    """L = sqrt(pi N) * rs, A = pi N rs^2."""
    rs = 2.0
    N = 58
    sys = build_2deg_system(rs, N_elec=N, polarization='unpolarized')
    assert sys['nup'] == 29
    assert sys['ndown'] == 29
    np.testing.assert_allclose(sys['L'], rs * np.sqrt(np.pi * N))
    np.testing.assert_allclose(sys['area'], np.pi * N * rs ** 2)
    np.testing.assert_allclose(sys['dk'], 2 * np.pi / sys['L'])
    assert sys['kvecs'].shape == (29, 2)
    assert sys['dim'] == 2


def test_build_2deg_system_polarized_geometry():
    sys = build_2deg_system(rs=2.0, N_elec=57, polarization='polarized')
    assert sys['nup'] == 57
    assert sys['ndown'] == 0
    assert sys['kvecs'].shape == (57, 2)


def test_build_2deg_system_odd_N_unpolarized_raises():
    with pytest.raises(ValueError):
        build_2deg_system(rs=2.0, N_elec=57, polarization='unpolarized')


def test_build_2deg_system_open_shell_raises():
    """N=4 unpolarized would mean nup=2, which is not a closed shell."""
    with pytest.raises(ValueError):
        build_2deg_system(rs=2.0, N_elec=4, polarization='unpolarized')


def test_build_2deg_system_kappa_shifts_kvecs():
    sys0 = build_2deg_system(rs=2.0, N_elec=58, polarization='unpolarized')
    kappa = np.array([0.1, -0.2])
    sys_k = build_2deg_system(
        rs=2.0, N_elec=58, polarization='unpolarized', kappa=kappa,
    )
    expected_shift = kappa * sys0['dk']
    diff = sys_k['kvecs'] - sys0['kvecs']
    np.testing.assert_allclose(diff, np.tile(expected_shift, (29, 1)))


# ---------------------------------------------------------------------
# Hartree-Fock thermodynamic limit: closed-form
# ---------------------------------------------------------------------
@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0, 30.0])
def test_hf_kinetic_td_unpolarized_matches_formula(rs):
    """T/N (unpol, TD) = 1/(2 r_s^2) — half the Fermi energy."""
    expected = 0.5 / rs ** 2
    np.testing.assert_allclose(
        hf_kinetic_energy_2d_td(rs, 'unpolarized'), expected,
    )


@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0, 30.0])
def test_hf_kinetic_td_polarized_matches_formula(rs):
    """T/N (pol, TD) = 1/r_s^2."""
    expected = 1.0 / rs ** 2
    np.testing.assert_allclose(
        hf_kinetic_energy_2d_td(rs, 'polarized'), expected,
    )


@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0])
def test_hf_exchange_td_unpolarized_matches_stern(rs):
    """E_x/N (unpol, TD) = -4 sqrt(2) / (3 pi r_s) — Stern 1973."""
    expected = -4.0 * np.sqrt(2.0) / (3.0 * np.pi * rs)
    np.testing.assert_allclose(
        hf_exchange_energy_2d_td(rs, 'unpolarized'), expected,
    )


@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0])
def test_hf_exchange_td_polarized_matches_stern(rs):
    """E_x/N (pol, TD) = -8 / (3 pi r_s)."""
    expected = -8.0 / (3.0 * np.pi * rs)
    np.testing.assert_allclose(
        hf_exchange_energy_2d_td(rs, 'polarized'), expected,
    )


def test_hf_total_td_known_values():
    """Spot-check TD HF totals at standard rs.  E_HF/N at rs=2 unpol
    should be ~-0.175 Ha (= 0.125 - 0.300)."""
    e_unpol_rs2 = hf_energy_2d_td(2.0, 'unpolarized')
    np.testing.assert_allclose(e_unpol_rs2, -0.17511, atol=1e-4)
    e_pol_rs1 = hf_energy_2d_td(1.0, 'polarized')
    np.testing.assert_allclose(e_pol_rs1, +0.15117, atol=1e-4)


# ---------------------------------------------------------------------
# Finite-N HF: must approach TD limit at large N
# ---------------------------------------------------------------------
@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0, 20.0])
def test_finite_N_HF_converges_to_TD_unpolarized(rs):
    """Finite-N HF at the larger N=114 unpolarized should be within
    ~1 mHa/electron of the TD limit at all benchmark rs."""
    sys = build_2deg_system(rs, N_elec=114, polarization='unpolarized')
    hf = hf_energy_2d_finite(sys)
    e_td = hf_energy_2d_td(rs, 'unpolarized')
    assert abs(hf['total'] - e_td) < 1e-3, (
        f"rs={rs}: finite-N HF {hf['total']:.5f} differs from TD "
        f"{e_td:.5f} by {(hf['total']-e_td)*1000:.2f} mHa"
    )


@pytest.mark.parametrize('rs', [1.0, 2.0, 5.0, 10.0])
def test_finite_N_HF_converges_to_TD_polarized(rs):
    """Same convergence test for polarized N=113."""
    sys = build_2deg_system(rs, N_elec=113, polarization='polarized')
    hf = hf_energy_2d_finite(sys)
    e_td = hf_energy_2d_td(rs, 'polarized')
    assert abs(hf['total'] - e_td) < 2e-3


def test_finite_N_HF_kinetic_consistency_N58_rs2():
    """Numerical sanity check: at N=58 rs=2 unpolarized, T/N is 0.127
    Ha (within 1.5% of the TD value 0.125 Ha)."""
    sys = build_2deg_system(rs=2.0, N_elec=58, polarization='unpolarized')
    hf = hf_energy_2d_finite(sys)
    np.testing.assert_allclose(hf['kinetic'], 0.127, atol=1e-3)


# ---------------------------------------------------------------------
# Variational sanity: HF must lie ABOVE the QMC reference
# ---------------------------------------------------------------------
@pytest.mark.parametrize('rs', [1, 2, 5, 10, 20, 30])
def test_HF_TD_above_attaccalite_QMC_unpolarized(rs):
    """Variational principle: TD HF energy must lie *above* the
    Attaccalite 2002 backflow-DMC value at every density.  The
    difference is the correlation energy (typically 10-100 mHa
    for the 2D HEG at metallic densities)."""
    e_hf = hf_energy_2d_td(float(rs), 'unpolarized')
    e_qmc = ATTACCALITE_2002_BACKFLOW[rs]['unpolarized']
    assert e_hf > e_qmc, (
        f"rs={rs}: HF {e_hf:.5f} should exceed QMC {e_qmc:.5f}"
    )
    e_corr = e_qmc - e_hf
    # Correlation energy should be negative and bounded in magnitude.
    assert -0.2 < e_corr < 0.0


@pytest.mark.parametrize('rs', [1, 2, 5, 10])
def test_HF_TD_above_attaccalite_QMC_polarized(rs):
    e_hf = hf_energy_2d_td(float(rs), 'polarized')
    e_qmc = ATTACCALITE_2002_BACKFLOW[rs]['polarized']
    assert e_hf > e_qmc
    e_corr = e_qmc - e_hf
    assert -0.05 < e_corr < 0.0


# ---------------------------------------------------------------------------
# From test_2dheg_pipeline.py
#
# End-to-end pipeline tests for the 2D HEG PsiFormer.
#
# These verify that the 2D-aware PsiFormer + Ewald + system-builder
# pipeline produces the analytical Hartree-Fock energy at initialisation
# when:
#
# * the Jastrow / backflow / cusp / deep-Jastrow / pair-Jastrow are all
#   disabled, leaving only the plane-wave envelope (the Slater
#   determinant of plane waves at the closed-shell Fermi sea), and
# * the local energy is computed by the same Laplacian + Ewald path the
#   production VMC drivers use.
#
# The thermodynamic-limit HF (Stern 1973) is
#
#     E_HF/N = 1/(2 r_s^2) - 4 sqrt(2) / (3 pi r_s)   (unpolarized)
#
# and the analytical finite-N HF closed-form lives in
# :func:`OmegaQMC.psi.heg_2d.hf_energy_2d_finite`.  The test checks the
# PsiFormer's E_HF computed via the production code path matches the
# analytical finite-N HF to ~1 mHa/electron (statistical noise from the
# walker average).
# ---------------------------------------------------------------------------


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
