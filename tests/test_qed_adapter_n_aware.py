"""Tests for the Tang-style n-aware QED-NN ansatz adapter (Phase 2f-1).

Verifies the Fock-head architecture and key invariants:
  1. Zero-initialised head means initial log Ψ(r, n) = log Ψ_e(r) for all n
     (n-aware ansatz reduces to standard NN at iter 0).
  2. Once the head's parameters are perturbed, the Fock-ladder ratio
     log Ψ(r, n+1) − log Ψ(r, n) becomes r-dependent — this is the
     architectural fix from Phase 2e (the factorized ansatz had ratios
     independent of r, which zeroed the bilinear-coupling contribution
     for symmetric systems).
  3. log Ψ flows through pauli_fierz_local_energy without errors.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom
from OmegaQMC.psi.nn.qed_adapter import (
    make_qed_nn_log_psi_n_aware,
    FockHead,
)
from OmegaQMC.psi.nn.qed_physics import pauli_fierz_local_energy


def _build_h2():
    L = 1.4010
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[[0.0, 0.0, -L / 2], [0.0, 0.0, L / 2]],
        n_up=1, n_down=1,
    )


@pytest.fixture
def h2_n_aware():
    mol = _build_h2()
    log_psi, params, graphdef = make_qed_nn_log_psi_n_aware(
        config="psiformer",
        mol_info=mol,
        rng_key=jax.random.key(99),
        omega=0.5,
        coupling_vec=jnp.array([0.0, 0.0, 0.05]),
        nph_max=4,
        fock_hidden_dim=32,
    )
    return log_psi, params, graphdef, mol


def test_n_aware_returns_combined_params(h2_n_aware):
    """Params dict should expose both 'nn' and 'fock_head'."""
    _, params, _, _ = h2_n_aware
    assert "nn" in params
    assert "fock_head" in params


def test_zero_init_head_means_no_n_dependence(h2_n_aware):
    """At iter 0 (zero-init output), log Ψ(r, n) must equal log Ψ_e(r)
    for *all* n. The Fock head adds exactly 0 at initialization."""
    log_psi, params, _, _ = h2_n_aware

    r = jnp.array([[0.1, 0.0, -0.5], [-0.05, 0.1, 0.6]])
    R = jnp.array([[0.0, 0.0, -0.7005], [0.0, 0.0, 0.7005]])

    val_n0 = float(log_psi(r, R, jnp.array(0), params))
    for n_val in [1, 2, 3, 4]:
        val_n = float(log_psi(r, R, jnp.array(n_val), params))
        np.testing.assert_allclose(
            val_n, val_n0, atol=1e-12,
            err_msg=f"n={n_val}: head should add 0 at zero-init",
        )


def test_perturbed_head_breaks_n_independence(h2_n_aware):
    """After perturbing the Fock head's output layer, log Ψ(r, n+1) −
    log Ψ(r, n) becomes r-dependent — the key architectural fix."""
    log_psi, params, _, _ = h2_n_aware

    # Perturb the output layer of the head with a small random kernel.
    head_params = params["fock_head"]
    rng = jax.random.key(7)
    flat, treedef = jax.tree.flatten(head_params)
    new_flat = []
    for leaf in flat:
        rng, sub = jax.random.split(rng)
        new_flat.append(leaf + 0.1 * jax.random.normal(sub, leaf.shape))
    head_params_pert = jax.tree.unflatten(treedef, new_flat)
    params_pert = {**params, "fock_head": head_params_pert}

    R = jnp.array([[0.0, 0.0, -0.7005], [0.0, 0.0, 0.7005]])

    # Check ratio depends on r (the architectural fix from Phase 2e).
    r1 = jnp.array([[0.1, 0.0, -0.5], [-0.05, 0.1, 0.6]])
    r2 = jnp.array([[-0.3, 0.2, 0.4], [0.5, -0.1, -0.6]])

    diff_n01_r1 = (
        float(log_psi(r1, R, jnp.array(1), params_pert))
        - float(log_psi(r1, R, jnp.array(0), params_pert))
    )
    diff_n01_r2 = (
        float(log_psi(r2, R, jnp.array(1), params_pert))
        - float(log_psi(r2, R, jnp.array(0), params_pert))
    )
    assert abs(diff_n01_r1 - diff_n01_r2) > 1e-6, (
        f"Fock-ladder ratio still r-independent after head perturbation: "
        f"r1 diff={diff_n01_r1:.6f}, r2 diff={diff_n01_r2:.6f}"
    )


def test_n_aware_integrates_with_pauli_fierz_local_energy(h2_n_aware):
    """Smoke test: n-aware log_psi flows through Pauli-Fierz local energy."""
    log_psi, params, _, _ = h2_n_aware

    nuc = jnp.array([[0.0, 0.0, -0.7005], [0.0, 0.0, 0.7005]])
    charges = jnp.array([1.0, 1.0])
    elec = jnp.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])

    e_loc = pauli_fierz_local_energy(
        log_psi, params, elec, jnp.array(0), nuc, charges,
        omega=0.5,
        coupling_vec=jnp.array([0.0, 0.0, 0.05]),
        nph_max=4,
    )
    # Untrained PsiFormer at fixed coords can produce very large kinetic
    # energies; only assert finiteness here. Physical reasonableness is
    # checked at the full-pipeline (MH-sampled) level in
    # test_qed_vmc_nn_n_aware_smoke.
    assert jnp.isfinite(e_loc)


def test_fock_head_zero_init_alone():
    """FockHead module on its own returns 0 at initialisation
    (independent of any inputs), since the output layer is zeros_init."""
    rngs = nnx.Rngs(jax.random.key(123))
    head = FockHead(
        nph_max=5, n_features=1, hidden_dim=16, rngs=rngs,
    )
    for n_val in [0, 1, 3, 5]:
        for f in [-2.0, 0.0, 0.5, 5.0]:
            out = float(head(jnp.array(n_val), jnp.array([f])))
            np.testing.assert_allclose(out, 0.0, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
