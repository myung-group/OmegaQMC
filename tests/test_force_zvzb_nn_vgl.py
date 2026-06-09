"""Regression: VGL ZV kinetic correction vs frozen v5
``vmc_nn_gradients_zvzb``.

The plan-mandated wiring of analytic forward-Laplacian
``Δlog|ψ|`` into the ``_ke_dpsi_component`` ``fori_loop``
in :func:`OmegaQMC.observables.force.vmc_nn_gradients_zvzb`
replaces the per-``(ia, k)`` ``laplacian`` call with a
single ``jax.jacfwd`` over nuclear coordinates on the
``(grad_x_lp, lap_x_lp)`` pair.  The mathematical
identity used (with ``h = ∇_R log|ψ|``,
``q = log|ψ| + log|h_{ia,k}|``) is

    (ke_dpsi[ia,k] - ke_psi) · h[ia,k]
    = -0.5 · Δ_x h[ia,k] - ∇_x log|ψ| · ∇_x h[ia,k]

— the ``1/h_{ia,k}`` factor cancels exactly against the
outer ``h[ia,k]`` from
``(KE_dψ - KE_ψ)·∇_R log|ψ|``.

This test pins the new bundled ``grd_ee_en, grd_ke,
grd_logpsi`` output against the frozen v5 baseline at
1e-9 in float64 across a batch of random walker
configurations on H₂ / PsiFormer.
"""
import sys
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from OmegaQMC.utils import Mole_custom
from OmegaQMC.psi.nn.adapter import make_nn_log_psi
from OmegaQMC.observables.force import (
    vmc_nn_gradients_zvzb as vmc_nn_gradients_zvzb_new,
)


TOL = 1e-9


def _load_v5_frozen():
    """Load the frozen v5 force kernel under its own name.

    The frozen module uses ``from ..psi.nn.physics import
    laplacian`` (deferred, inside ``vmc_nn_gradients_zvzb``).
    For that relative import to resolve when the module is
    loaded out-of-tree, we pin ``__package__`` to
    ``OmegaQMC.observables`` before executing.
    """
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "_force_v5_frozen", here / "force_v5_frozen.py",
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "OmegaQMC.observables"
    sys.modules["_force_v5_frozen"] = mod
    spec.loader.exec_module(mod)
    return mod.vmc_nn_gradients_zvzb


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


def _build_pieces(seed=900):
    mol = _h2_mol()
    rng_key = jax.random.key(seed)
    log_psi, params, _gd, lap_grad = make_nn_log_psi(
        'psiformer', mol, rng_key,
    )
    nuc = mol.coords
    charges = jnp.asarray([1.0, 1.0])
    nelec = mol.n_up + mol.n_down
    return dict(
        log_psi=log_psi, params=params,
        lap_grad=lap_grad, nuc=nuc,
        charges=charges, nelec=nelec,
    )


def test_h2_psiformer_zvzb_matches_v5_frozen():
    p = _build_pieces(seed=900)
    assert p['lap_grad'].use_vgl, (
        "PsiFormer adapter should expose the VGL path"
    )

    v5 = _load_v5_frozen()
    fn_v5 = v5(
        p['log_psi'], p['nuc'], p['charges'],
        p['nelec'], p['params'],
    )
    fn_new = vmc_nn_gradients_zvzb_new(
        p['log_psi'], p['nuc'], p['charges'],
        p['nelec'], p['params'],
        lap_grad=p['lap_grad'],
    )

    rng = np.random.default_rng(901)
    batch = jnp.asarray(
        rng.normal(size=(4, p['nelec'], 3)) * 0.7,
    )
    out_v5 = fn_v5(batch)
    out_new = fn_new(batch)
    assert len(out_v5) == len(out_new) == 3
    for label, a, b in zip(
        ('grd_ee_en', 'grd_ke', 'grd_logpsi'),
        out_v5, out_new,
    ):
        err = float(jnp.max(jnp.abs(a - b)))
        print(f"  {label:<10s} |Δ|={err:.3e}")
        assert err < TOL, (label, err)


def test_h2_psiformer_zvzb_no_vgl_falls_back():
    """Without ``lap_grad``, the new wrapper must reproduce
    the frozen v5 behaviour bit-for-bit (same code path)."""
    p = _build_pieces(seed=910)

    v5 = _load_v5_frozen()
    fn_v5 = v5(
        p['log_psi'], p['nuc'], p['charges'],
        p['nelec'], p['params'],
    )
    fn_new_fb = vmc_nn_gradients_zvzb_new(
        p['log_psi'], p['nuc'], p['charges'],
        p['nelec'], p['params'],
    )

    rng = np.random.default_rng(911)
    batch = jnp.asarray(
        rng.normal(size=(2, p['nelec'], 3)) * 0.7,
    )
    out_v5 = fn_v5(batch)
    out_fb = fn_new_fb(batch)
    for label, a, b in zip(
        ('grd_ee_en', 'grd_ke', 'grd_logpsi'),
        out_v5, out_fb,
    ):
        err = float(jnp.max(jnp.abs(a - b)))
        print(f"  [fallback] {label:<10s} |Δ|={err:.3e}")
        # Same code path → exact equality.
        assert err == 0.0, (label, err)


if __name__ == '__main__':
    test_h2_psiformer_zvzb_matches_v5_frozen()
    test_h2_psiformer_zvzb_no_vgl_falls_back()
    print('OK')
