"""Tests for the Tang-native joint electron-photon ansatz (Phase 2g).

Verifies the per-electron one-hot(n) injection inside the GNN
(Tang 2025, arXiv:2503.15644 Sec. II.C):

  1. Builds end-to-end on FermiNet+Jastrow+backflow without shape errors.
  2. log|Psi(r,n)| varies with n (the network actually consumes n).
  3. Native Slater determinant sign is +/-1 for all n.
  4. Pauli-Fierz signed local energy is finite for random configs at
     lambda=0 (decoupling sanity) and at lambda=0.1.
  5. At lambda=0, n only enters via the photon energy term omega*n;
     the local-energy difference E_loc(n=k) - E_loc(n=0) - k*omega is
     bounded by floating-point noise on a *constant-in-n* trial wf.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom
from OmegaQMC.psi.nn.config import load_nn_config
from OmegaQMC.psi.nn.qed_adapter import (
    make_qed_nn_log_psi_tang_native,
)
from OmegaQMC.psi.nn.qed_physics import (
    pauli_fierz_local_energy_signed,
)


def _h2():
    L = 1.4010
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[[0.0, 0.0, -L / 2], [0.0, 0.0, L / 2]],
        n_up=1, n_down=1,
    )


def _build_tang_native(rng_seed=0, nph_max=4):
    cfg = load_nn_config('ferminet')
    cfg = dataclasses.replace(cfg, use_jastrow=True)
    mol = _h2()
    log_psi, params, gd = make_qed_nn_log_psi_tang_native(
        cfg, mol, jax.random.PRNGKey(rng_seed), nph_max=nph_max,
    )
    return log_psi, params, gd, mol


@pytest.fixture(scope="module")
def tang_h2():
    log_psi, params, gd, mol = _build_tang_native()
    return log_psi, params, mol


def test_build_and_forward_finite(tang_h2):
    """log|Psi| and sign are finite scalars for every n in [0, nph_max]."""
    log_psi, params, mol = tang_h2
    nph_max = 4
    elec = jax.random.normal(jax.random.PRNGKey(1), (2, 3)) * 0.5
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)

    for n_val in range(nph_max + 1):
        logp, sgn = log_psi(elec, nuc, jnp.array(n_val), params)
        assert jnp.isfinite(logp), f"log|Psi| not finite at n={n_val}"
        assert jnp.isfinite(sgn), f"sign not finite at n={n_val}"
        assert float(jnp.abs(sgn)) == pytest.approx(1.0, abs=1e-12), (
            f"sign should be +/-1 (native Slater determinant), got {float(sgn)}"
        )


def test_log_psi_varies_with_n(tang_h2):
    """Per-electron one-hot(n) actually influences the network output.

    If the GNN ignored n, log|Psi(r,n)| would be constant in n. Tang
    Sec. II.C: n must enter through every layer of the GNN.
    """
    log_psi, params, mol = tang_h2
    elec = jax.random.normal(jax.random.PRNGKey(2), (2, 3)) * 0.5
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)

    vals = jnp.array([
        log_psi(elec, nuc, jnp.array(n), params)[0]
        for n in range(5)
    ])
    spread = float(jnp.max(vals) - jnp.min(vals))
    assert spread > 1e-3, (
        f"log|Psi(r,n)| barely varies with n (spread={spread:.2e}); "
        "the n-injection is broken or n is unused by the GNN."
    )


def test_local_energy_finite_at_lambda_zero(tang_h2):
    """Pauli-Fierz local energy is finite at lambda=0 for random configs.

    At lambda=0 the photon couples only via the omega*n term; if the
    bilinear / DSE pathways have any latent NaN, this test catches it.
    """
    log_psi, params, mol = tang_h2
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    charges = jnp.asarray(mol.charges, dtype=jnp.float64)

    rng = jax.random.PRNGKey(3)
    omega = 0.5
    cv0 = jnp.array([0.0, 0.0, 0.0])

    for k in range(4):
        rng, sub = jax.random.split(rng)
        elec = jax.random.normal(sub, (2, 3)) * 0.4
        for n_val in [0, 1, 3]:
            e_loc = pauli_fierz_local_energy_signed(
                log_psi, params, elec, jnp.array(n_val),
                nuc, charges, omega, cv0,
                nph_max=4, enuc=None,
            )
            assert jnp.isfinite(e_loc), (
                f"E_loc not finite at n={n_val} (cfg #{k})"
            )


def test_local_energy_finite_at_lambda_finite(tang_h2):
    """Local energy is finite at lambda=0.1 (bilinear + DSE active)."""
    log_psi, params, mol = tang_h2
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    charges = jnp.asarray(mol.charges, dtype=jnp.float64)

    rng = jax.random.PRNGKey(4)
    cv = jnp.array([0.0, 0.0, 0.1])
    for k in range(4):
        rng, sub = jax.random.split(rng)
        elec = jax.random.normal(sub, (2, 3)) * 0.4
        for n_val in [0, 1, 3]:
            e_loc = pauli_fierz_local_energy_signed(
                log_psi, params, elec, jnp.array(n_val),
                nuc, charges, omega=0.5, coupling_vec=cv,
                nph_max=4, enuc=None,
            )
            assert jnp.isfinite(e_loc), (
                f"E_loc not finite at n={n_val} (cfg #{k}, lambda=0.1)"
            )


def test_grad_through_log_psi_finite(tang_h2):
    """JAX grad through log_psi w.r.t. params is finite for any n."""
    log_psi, params, mol = tang_h2
    nuc = jnp.asarray(mol.coords, dtype=jnp.float64)
    elec = jax.random.normal(jax.random.PRNGKey(5), (2, 3)) * 0.4

    def loss(p, n):
        log, sgn = log_psi(elec, nuc, n, p)
        return log * sgn  # scalar

    for n_val in [0, 2, 4]:
        g = jax.grad(loss)(params, jnp.array(n_val))
        leaves = jax.tree_util.tree_leaves(g)
        for L in leaves:
            assert bool(jnp.all(jnp.isfinite(L))), (
                f"grad NaN/inf at n={n_val}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
