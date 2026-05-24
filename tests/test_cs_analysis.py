"""
Synthetic-data tests that lock the semantics of the pre-registered
compressed-sensing scaling analysis.

The data-collection pipeline must produce records that pass these tests; the
analysis module must continue to produce the regimes documented in the
``classify_regime`` decision rule.
"""

import numpy as np
import pytest

from OmegaQMC.cs.analysis import (
    SCHEMA_VERSION,
    apply_convergence_gate,
    classify_regime,
    compute_K_s_star_table,
    fit_scaling,
    flat_sparsity_diagnostic,
    run_analysis,
    validate_schema,
)


def _synthetic_k_star_table(
    exponent_K: float,
    exponent_logN: float = 1.0,
    noise: float = 0.02,
    seed: int = 0,
):
    """Direct K_s* table for the regression test, no pipeline involvement."""
    rng = np.random.default_rng(seed)
    K_eff_vals = [4, 8, 16, 32, 64, 128]
    N_det_vals = [1_000, 100_000, 10_000_000]
    table = []
    for K_eff in K_eff_vals:
        for N_det in N_det_vals:
            log_Ks = (
                exponent_K * np.log(K_eff)
                + exponent_logN * np.log(np.log(N_det))
                + noise * rng.standard_normal()
            )
            table.append(dict(
                R=float(K_eff), basis=str(N_det), eta=1e-3,
                K_eff=K_eff, N_det=N_det,
                K_s_star=int(round(np.exp(log_Ks))),
            ))
    return table


def _synthetic_pipeline(exponent_K: float, seed: int = 0):
    """Cells + aux for end-to-end pipeline tests. K_s sweep is sized to bracket
    K_s_true for the requested exponent across all (R, basis, eta) cells."""
    rng = np.random.default_rng(seed)
    R_values = [1.0, 1.4, 1.8, 2.2, 2.6]
    bases = [("cc-pVDZ", 36_100), ("cc-pVTZ", 2_400_000)]
    etas = [1e-2, 1e-3, 1e-4]
    K_base = {1.0: 2, 1.4: 3, 1.8: 4, 2.2: 6, 2.6: 8}
    eta_mult = {1e-2: 1.0, 1e-3: 1.5, 1e-4: 2.0}
    K_s_sweep = [2 ** k for k in range(2, 24)]  # 4 to ~8M

    cells = []
    aux = []
    for R in R_values:
        for basis, N_det in bases:
            aux.append(dict(
                R=R, basis=basis, N_det=N_det,
                psi_nn_energy_error=rng.uniform(-0.2, 0.2),
                max_c_corr=1.0 / np.sqrt(K_base[R]),
                K_eff_for_diag=K_base[R],
            ))
            for eta in etas:
                K_eff = int(round(K_base[R] * eta_mult[eta]))
                log_Ks_true = (
                    exponent_K * np.log(K_eff)
                    + np.log(np.log(N_det))
                )
                jitter = 0.05 * rng.standard_normal()
                Ks_true = np.exp(log_Ks_true + jitter)
                for K_s in K_s_sweep:
                    for seed_k in range(20):
                        crossed = K_s >= Ks_true * np.exp(
                            0.02 * rng.standard_normal()
                        )
                        err = eta * 0.5 if crossed else eta * 5.0
                        cells.append(dict(
                            R=R, basis=basis, eta=eta,
                            K_eff=K_eff, K_s=K_s, seed=seed_k,
                            L_inf_err=float(err),
                            L_2_err=float(err * 3.0),
                            support_err=0.0 if crossed else 0.5,
                            N_det=N_det,
                            lambda_used=eta,
                        ))
    return cells, aux


def test_schema_version_stable():
    assert SCHEMA_VERSION == "1.0.0", (
        "SCHEMA_VERSION changed; this invalidates pre-registration. "
        "If intentional, freeze a new version and re-run data collection."
    )


def test_validate_schema_catches_missing_fields():
    cells = [dict(R=1.0, basis="cc-pVDZ")]
    aux = [dict(R=1.0, basis="cc-pVDZ", N_det=100,
                psi_nn_energy_error=0.1, max_c_corr=0.3, K_eff_for_diag=10)]
    with pytest.raises(ValueError, match="missing fields"):
        validate_schema(cells, aux)


def test_validate_schema_empty():
    with pytest.raises(ValueError, match="empty"):
        validate_schema([], [])


def test_convergence_gate_drops_bad_geometries():
    cells, aux = _synthetic_pipeline(exponent_K=1.0)
    aux[0]["psi_nn_energy_error"] = 5.0
    kept, gated = apply_convergence_gate(cells, aux)
    bad = (aux[0]["R"], aux[0]["basis"])
    assert bad in gated
    assert all((c["R"], c["basis"]) != bad for c in kept)


def test_K_s_star_returns_None_when_unmet():
    cells = [dict(R=1.0, basis="cc-pVDZ", eta=1e-3, K_eff=10, K_s=ks, seed=s,
                  L_inf_err=1.0, L_2_err=1.0, support_err=1.0,
                  N_det=100, lambda_used=1e-3)
             for ks in (10, 100, 1000) for s in range(5)]
    table = compute_K_s_star_table(cells)
    assert table[0]["K_s_star"] is None


def test_K_s_star_finds_smallest_passing():
    rows = []
    for ks in (10, 100, 1000):
        err = 1e-2 if ks < 100 else 1e-4
        for s in range(5):
            rows.append(dict(R=1.0, basis="cc-pVDZ", eta=1e-3,
                             K_eff=10, K_s=ks, seed=s,
                             L_inf_err=err, L_2_err=err, support_err=0.0,
                             N_det=100, lambda_used=1e-3))
    table = compute_K_s_star_table(rows)
    assert table[0]["K_s_star"] == 100


def test_fit_recovers_RIP_exponent_direct():
    table = _synthetic_k_star_table(exponent_K=1.0, noise=0.02, seed=1)
    fit = fit_scaling(table, n_bootstrap=400, seed=1)
    assert 0.85 < fit["alpha"] < 1.15, f"alpha={fit['alpha']}"
    assert fit["alpha_ci"][0] < 1.0 < fit["alpha_ci"][1]


def test_fit_recovers_coherence_exponent_direct():
    table = _synthetic_k_star_table(exponent_K=2.0, noise=0.02, seed=2)
    fit = fit_scaling(table, n_bootstrap=400, seed=2)
    assert 1.85 < fit["alpha"] < 2.15, f"alpha={fit['alpha']}"


def test_fit_flags_broken_scaling_direct():
    table = _synthetic_k_star_table(exponent_K=3.5, noise=0.02, seed=3)
    fit = fit_scaling(table, n_bootstrap=400, seed=3)
    assert fit["alpha"] > 2.5


def test_flat_sparsity_diagnostic_healthy():
    aux = [dict(R=r, basis="b", N_det=1000, psi_nn_energy_error=0.0,
                max_c_corr=1.0 / np.sqrt(K), K_eff_for_diag=K)
           for r, K in zip([1.0, 1.4, 1.8, 2.2, 2.6], [8, 16, 32, 64, 128])]
    slope = flat_sparsity_diagnostic(aux)
    assert -0.55 < slope < -0.45


def test_flat_sparsity_diagnostic_violated():
    aux = [dict(R=r, basis="b", N_det=1000, psi_nn_energy_error=0.0,
                max_c_corr=0.9, K_eff_for_diag=K)
           for r, K in zip([1.0, 1.4, 1.8, 2.2, 2.6], [8, 16, 32, 64, 128])]
    slope = flat_sparsity_diagnostic(aux)
    assert slope > -0.1
    assert classify_regime(1.0, 1.0, slope) == "broken"


def test_classify_regime_decision_rule():
    assert classify_regime(1.0, 1.0, -0.5) == "RIP"
    assert classify_regime(2.0, 1.0, -0.5) == "coherence"
    assert classify_regime(3.0, 1.0, -0.5) == "broken"
    assert classify_regime(1.0, 3.0, -0.5) == "broken"
    assert classify_regime(1.0, 1.0, -0.1) == "broken"
    assert classify_regime(1.5, 1.0, -0.5) == "ambiguous"
    assert classify_regime(float("nan"), 1.0, -0.5) == "broken"


def test_pipeline_RIP_end_to_end():
    cells, aux = _synthetic_pipeline(exponent_K=1.0, seed=4)
    result = run_analysis(cells, aux, n_bootstrap=200, seed=4)
    assert len(result.k_star_table) == 5 * 2 * 3
    assert result.n_cells > 0
    assert 0.7 < result.alpha < 1.3, f"alpha={result.alpha}"
    assert result.regime in ("RIP", "ambiguous")


def test_pipeline_coherence_end_to_end():
    cells, aux = _synthetic_pipeline(exponent_K=2.0, seed=5)
    result = run_analysis(cells, aux, n_bootstrap=200, seed=5)
    assert 1.7 < result.alpha < 2.3, f"alpha={result.alpha}"
