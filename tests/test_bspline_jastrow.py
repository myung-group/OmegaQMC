"""Tests for B-spline Jastrow factors."""

import pytest
import jax
import jax.numpy as jnp
from pyscf import gto, scf

from OmegaQMC.psi_gto import (
    _build_bspline_coefs,
    _bspline_eval,
    get_psi_fun,
)


jax.config.update("jax_enable_x64", True)


# ---- unit tests for helpers ----

class TestBuildBsplineCoefs:
    """Verify _build_bspline_coefs maps params correctly."""

    def test_cusp_constraint(self):
        """coefs[0] = coefs[2] - 2*dr*cusp."""
        params = jnp.array([0.1, 0.2, 0.3, 0.4])
        delta_r = 1.5
        cusp_val = -0.5
        coefs = _build_bspline_coefs(
            params, delta_r, cusp_val
        )
        # coefs[2] = params[1] = 0.2
        expected_c0 = 0.2 - 2.0 * 1.5 * (-0.5)
        assert coefs[0] == pytest.approx(expected_c0)

    def test_zero_boundary(self):
        """Last 3 coefs must be zero."""
        params = jnp.array([1.0, 2.0, 3.0])
        coefs = _build_bspline_coefs(params, 1.0, 0.0)
        assert coefs[-1] == 0.0
        assert coefs[-2] == 0.0
        assert coefs[-3] == 0.0

    def test_length(self):
        """Total length = N + 4."""
        for n in [2, 5, 10]:
            params = jnp.zeros(n)
            coefs = _build_bspline_coefs(
                params, 1.0, 0.0
            )
            assert coefs.shape[0] == n + 4


class TestBsplineEval:
    """Verify _bspline_eval against known values."""

    def test_at_cutoff_zero(self):
        """u(r_cut) = 0 when boundary coefs are zero."""
        params = jnp.array([0.5, 0.3, 0.1, 0.05])
        n = params.shape[0]
        r_cut = 5.0
        delta_r = r_cut / (n + 1)
        coefs = _build_bspline_coefs(
            params, delta_r, 0.0
        )
        r = jnp.array([r_cut])
        val = _bspline_eval(
            r, coefs, 1.0 / delta_r, n
        )
        assert abs(float(val[0])) < 1e-12

    def test_beyond_cutoff_masked(self):
        """r > r_cut zeroed by external cutoff mask."""
        params = jnp.array([0.5, 0.3, 0.1, 0.05])
        n = params.shape[0]
        r_cut = 5.0
        delta_r = r_cut / (n + 1)
        coefs = _build_bspline_coefs(
            params, delta_r, 0.0
        )
        r = jnp.array([r_cut - 0.01, r_cut + 1.0])
        vals = _bspline_eval(
            r, coefs, 1.0 / delta_r, n
        )
        mask = (r < r_cut).astype(r.dtype)
        masked = vals * mask
        # Beyond cutoff is zeroed by the mask
        assert masked[1] == 0.0
        # Within cutoff passes through
        assert jnp.isfinite(masked[0])

    def test_differentiable(self):
        """jax.grad through _bspline_eval works."""
        params = jnp.array([0.5, 0.3])
        n = params.shape[0]
        r_cut = 3.0
        delta_r = r_cut / (n + 1)

        def f(p):
            c = _build_bspline_coefs(p, delta_r, -0.5)
            r = jnp.array([0.5])
            return _bspline_eval(
                r, c, 1.0 / delta_r, n
            )[0]

        grad = jax.grad(f)(params)
        assert grad.shape == params.shape
        assert jnp.all(jnp.isfinite(grad))


class TestCuspDerivative:
    """Verify du/dr|_{r=0} equals cusp_val."""

    @pytest.mark.parametrize(
        "cusp_val", [-0.25, -0.5, -1.0]
    )
    def test_cusp_via_grad(self, cusp_val):
        params = jnp.array([0.0, 0.0, 0.0, 0.0])
        n = params.shape[0]
        r_cut = 5.0
        delta_r = r_cut / (n + 1)

        def u_scalar(r_scalar):
            c = _build_bspline_coefs(
                params, delta_r, cusp_val
            )
            r = jnp.array([r_scalar])
            return _bspline_eval(
                r, c, 1.0 / delta_r, n
            )[0]

        du_dr_at_0 = jax.grad(u_scalar)(0.0)
        assert du_dr_at_0 == pytest.approx(
            cusp_val, abs=1e-10
        )


# ---- integration test: H2 with J2_bspline ----

class TestH2BsplineIntegration:
    """Run a short VMC-like evaluation on H2."""

    @pytest.fixture(autouse=True)
    def setup_h2(self):
        mol = gto.M(
            atom='H 0 0 0; H 0 0 1.4',
            basis='sto-6g',
            unit='Bohr',
            verbose=0,
        )
        self.mf = scf.RHF(mol)
        self.mf.kernel()
        self.nelec = mol.tot_electrons()
        self.nuc_crds = jnp.array(
            mol.atom_coords(unit='Bohr')
        )

    def test_bspline_j2_evaluates(self):
        """Wavefunction evaluates with J2_bspline."""
        bspline_config = {"J2": {"r_cut": 5.0}}
        log_psi, _, _, _ = get_psi_fun(
            self.mf,
            bspline_config=bspline_config,
        )

        params = {
            "J2_bspline": {
                "like": jnp.zeros(8),
                "unlike": jnp.zeros(8),
            }
        }

        rng = jax.random.key(42)
        elec_crds = self.nuc_crds[
            jnp.array([0, 0])
        ] + 0.1 * jax.random.normal(
            rng, (self.nelec, 3)
        )

        val = log_psi(elec_crds, self.nuc_crds, params)
        assert jnp.isfinite(val)

    def test_bspline_j2_grad(self):
        """Gradient w.r.t. B-spline params is finite."""
        bspline_config = {"J2": {"r_cut": 5.0}}
        log_psi, _, _, _ = get_psi_fun(
            self.mf,
            bspline_config=bspline_config,
        )

        params = {
            "J2_bspline": {
                "like": jnp.zeros(4),
                "unlike": jnp.zeros(4),
            }
        }

        rng = jax.random.key(99)
        elec_crds = self.nuc_crds[
            jnp.array([0, 0])
        ] + 0.1 * jax.random.normal(
            rng, (self.nelec, 3)
        )

        def f(p):
            return log_psi(
                elec_crds, self.nuc_crds, p
            )

        grads = jax.grad(f)(params)
        for key in grads["J2_bspline"]:
            assert jnp.all(
                jnp.isfinite(grads["J2_bspline"][key])
            )


# ---- coexistence test: pade + bspline ----

class TestCoexistence:
    """Both pade and bspline present."""

    @pytest.fixture(autouse=True)
    def setup_h2(self):
        mol = gto.M(
            atom='H 0 0 0; H 0 0 1.4',
            basis='sto-6g',
            unit='Bohr',
            verbose=0,
        )
        self.mf = scf.RHF(mol)
        self.mf.kernel()
        self.nelec = mol.tot_electrons()
        self.nuc_crds = jnp.array(
            mol.atom_coords(unit='Bohr')
        )

    def test_pade_plus_bspline(self):
        """Both J2_pade and J2_bspline evaluate."""
        bspline_config = {"J2": {"r_cut": 5.0}}
        log_psi, _, _, _ = get_psi_fun(
            self.mf,
            bspline_config=bspline_config,
        )

        params = {
            "J2_pade": {
                "like": jnp.array([0.25, 1.0]),
                "unlike": jnp.array([0.5, 1.0]),
            },
            "J2_bspline": {
                "like": jnp.zeros(4),
                "unlike": jnp.zeros(4),
            },
        }

        rng = jax.random.key(7)
        elec_crds = self.nuc_crds[
            jnp.array([0, 0])
        ] + 0.1 * jax.random.normal(
            rng, (self.nelec, 3)
        )

        val = log_psi(elec_crds, self.nuc_crds, params)
        assert jnp.isfinite(val)

    def test_coexistence_warning(self):
        """Warning emitted when both are present."""
        import warnings
        from OmegaQMC.vmc_gto import (
            _validate_params_corr,
        )

        params = {
            "J2_pade": {
                "like": jnp.array([0.25, 1.0]),
                "unlike": jnp.array([0.5, 1.0]),
            },
            "J2_bspline": {
                "like": jnp.zeros(4),
                "unlike": jnp.zeros(4),
            },
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_params_corr(params, self.mf)
            msgs = [str(x.message) for x in w]
            assert any(
                "J2_pade" in m and "J2_bspline" in m
                for m in msgs
            )
