"""Unit tests for the NN wavefunction modules.

Tests:
1. Forward pass produces finite Psi(sign, log).
2. jax.grad w.r.t. electron coords is finite.
3. Laplacian (kinetic energy) is finite.
4. Config loading returns correct NNAnsatzConfig.
"""

import pytest
import jax
import jax.numpy as jnp

from OmegaQMC.utils import Mole_custom
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.psi.nn.config import (
    NNAnsatzConfig,
    load_nn_config,
)
from OmegaQMC.psi.nn.physics import laplacian


@pytest.fixture
def h2_mol():
    """H2 molecule at 1.4 Bohr."""
    L = 1.4010
    return Mole_custom.from_arrays(
        charges=[1, 1],
        coords=[
            [0.0, 0.0, -L / 2],
            [0.0, 0.0, L / 2],
        ],
        n_up=1,
        n_down=1,
    )


@pytest.fixture
def psiformer_trial(h2_mol):
    """Build a PsiFormer trial on H2."""
    rng_key = jax.random.key(123)
    log_psi, params, graphdef, _lap_grad = make_nn_log_psi(
        'psiformer', h2_mol, rng_key,
    )
    return log_psi, params, h2_mol


class TestForwardPass:
    """Forward pass produces finite output."""

    def test_log_psi_finite(self, psiformer_trial):
        log_psi, params, mol = psiformer_trial
        nuc = mol.coords
        elec = jnp.array([
            [0.1, 0.0, -0.7],
            [0.0, 0.1, 0.7],
        ])
        val = log_psi(elec, nuc, params)
        assert jnp.isfinite(val), (
            f"log_psi not finite: {val}"
        )

    def test_log_psi_scalar(self, psiformer_trial):
        log_psi, params, mol = psiformer_trial
        nuc = mol.coords
        elec = jnp.array([
            [0.1, 0.0, -0.7],
            [0.0, 0.1, 0.7],
        ])
        val = log_psi(elec, nuc, params)
        assert val.shape == (), (
            f"Expected scalar, got shape {val.shape}"
        )


class TestGradient:
    """Gradient w.r.t. electron coordinates is finite."""

    def test_grad_wrt_elec(self, psiformer_trial):
        log_psi, params, mol = psiformer_trial
        nuc = mol.coords
        elec = jnp.array([
            [0.1, 0.0, -0.7],
            [0.0, 0.1, 0.7],
        ])
        grad_fn = jax.grad(log_psi, argnums=0)
        g = grad_fn(elec, nuc, params)
        assert g.shape == elec.shape
        assert jnp.all(jnp.isfinite(g)), (
            "Gradient contains non-finite values"
        )


class TestLaplacian:
    """Laplacian (kinetic energy) is finite."""

    def test_laplacian_finite(self, psiformer_trial):
        log_psi, params, mol = psiformer_trial
        nuc = mol.coords
        nelec = mol.n_up + mol.n_down
        elec = jnp.array([
            [0.1, 0.0, -0.7],
            [0.0, 0.1, 0.7],
        ])

        def f_flat(r_flat):
            r = r_flat.reshape(nelec, 3)
            return log_psi(r, nuc, params)

        r_flat = elec.reshape(-1)
        lap_fn = laplacian(f_flat)
        lap_val, grad_val = lap_fn(r_flat)

        assert jnp.isfinite(lap_val), (
            f"Laplacian not finite: {lap_val}"
        )
        assert jnp.all(jnp.isfinite(grad_val)), (
            "Laplacian gradient not finite"
        )

    def test_kinetic_energy_finite(
        self, psiformer_trial,
    ):
        log_psi, params, mol = psiformer_trial
        nuc = mol.coords
        nelec = mol.n_up + mol.n_down
        elec = jnp.array([
            [0.1, 0.0, -0.7],
            [0.0, 0.1, 0.7],
        ])

        def f_flat(r_flat):
            r = r_flat.reshape(nelec, 3)
            return log_psi(r, nuc, params)

        r_flat = elec.reshape(-1)
        lap_fn = laplacian(f_flat)
        lap_val, grad_val = lap_fn(r_flat)
        ke = -0.5 * (
            lap_val + jnp.dot(grad_val, grad_val)
        )
        assert jnp.isfinite(ke), (
            f"Kinetic energy not finite: {ke}"
        )


class TestConfig:
    """Config loading returns correct types."""

    def test_load_builtin(self):
        for name in [
            'paulinet', 'ferminet',
            'deeperwin', 'psiformer',
        ]:
            cfg = load_nn_config(name)
            assert isinstance(cfg, NNAnsatzConfig)

    def test_psiformer_fields(self):
        cfg = load_nn_config('psiformer')
        assert cfg.n_determinants == 16
        assert cfg.full_determinant is True
        assert cfg.use_jastrow is False
        assert cfg.use_backflow is True
        assert cfg.cusp_electrons_type == 'psiformer'

    def test_paulinet_fields(self):
        cfg = load_nn_config('paulinet')
        assert cfg.use_jastrow is True
        assert cfg.deep_features == 'shared'
        assert cfg.update_rule == 'concatenate'


class TestSignedLogPsi:
    """log_psi exposes a .signed callable returning (sign, log_amp)."""

    def test_signed_attribute_exists(self, psiformer_trial):
        log_psi, _params, _mol = psiformer_trial
        assert callable(log_psi.signed)

    def test_signed_shapes_and_finite(self, psiformer_trial):
        log_psi, params, mol = psiformer_trial
        nuc = mol.coords
        elec = jnp.array([
            [0.1, 0.0, -0.7],
            [0.0, 0.1, 0.7],
        ])
        sign, log_amp = log_psi.signed(elec, nuc, params)
        assert sign.shape == ()
        assert log_amp.shape == ()
        assert jnp.isfinite(log_amp)
        assert jnp.isfinite(sign)

    def test_signed_log_amp_matches_log_psi(self, psiformer_trial):
        log_psi, params, mol = psiformer_trial
        nuc = mol.coords
        elec = jnp.array([
            [0.2, 0.1, -0.6],
            [-0.1, 0.3, 0.8],
        ])
        log_check = log_psi(elec, nuc, params)
        _sign, log_amp = log_psi.signed(elec, nuc, params)
        jnp.allclose(log_amp, log_check)
        assert abs(float(log_amp) - float(log_check)) < 1e-6

    def test_signed_sign_is_unit_magnitude(self, psiformer_trial):
        """For real-valued wavefunctions ``|sign| == 1`` everywhere except
        on the nodal surface (probability-zero set)."""
        log_psi, params, mol = psiformer_trial
        nuc = mol.coords
        elec = jnp.array([
            [0.4, 0.0, -0.5],
            [-0.2, 0.1, 0.9],
        ])
        sign, _ = log_psi.signed(elec, nuc, params)
        assert abs(abs(float(sign)) - 1.0) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
