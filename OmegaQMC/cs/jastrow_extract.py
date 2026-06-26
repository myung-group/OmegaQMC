"""
Extract the multiplicative log-Jastrow ``J(R)`` per walker for the TC decode.

The TC two-sided decode (:mod:`OmegaQMC.cs.transcorrelated`) needs ``J(R_k)``
where the trial factorises as ``Psi_NN = exp(ln_slater + J)`` -- i.e. ``J`` is
the additive ``jastrow_term`` in both the GTO Slater-Jastrow ansatz
(``OmegaQMC.psi.gto``, ``return ln_slater + jastrow_term``) and the NN ansatz
(``OmegaQMC.psi.nn``, additive deep-Jastrow head ``model.omni.jastrow``).

Two routes:

* **New banks (preferred).** Stream ``jastrow_term`` alongside ``log_psi`` at
  sample time (see :class:`OmegaQMC.cs.walkers.WalkerDumper`, schema >= 1.1.0).
  Then ``J`` is read directly from the dump -- no re-evaluation.

* **Existing banks (retrofit).** Re-evaluate ``J = log|Psi_full| - log|Phi_slater|``
  on the stored walkers from two log-amplitude callables. For the GTO ansatz
  ``log_slater`` is the bare determinant log-amplitude; for the NN adapter it is
  a second ``log_psi`` built from a model whose ``omni.jastrow`` is disabled
  (additive head -> setting it to ``None`` removes exactly ``J``). Both share
  the same trained ``params``.

The amplitude-difference is exact and ansatz-agnostic: any additive-log
correlation factor cancels the determinant part, leaving ``J`` regardless of
how the determinant part is parameterised (backflow included).
"""

from typing import Callable, Optional

import numpy as np


def jastrow_on_walkers(
    log_full: Callable,
    log_slater: Callable,
    walkers: np.ndarray,
    nuc_crds: np.ndarray,
    params,
    batch_size: int = 4096,
    jit: bool = True,
) -> np.ndarray:
    """``J(R_k) = log|Psi_full(R_k)| - log|Phi_slater(R_k)|`` per walker.

    ``log_full`` and ``log_slater`` are single-walker callables
    ``(elec_crds, nuc_crds, params) -> log_amplitude`` (the ``log_psi`` and
    its Jastrow-disabled sibling). ``walkers`` is ``(K_s, nelec, 3)`` in the
    ansatz's own electron layout. Returns ``(K_s,)``.

    Batched + ``jax.vmap``-traced, mirroring
    ``examples/validate_cs_h2_stretched.evaluate_signed_psi``.
    """
    import jax
    import jax.numpy as jnp

    full_v = jax.vmap(log_full, in_axes=(0, None, None))
    slat_v = jax.vmap(log_slater, in_axes=(0, None, None))
    if jit:
        full_v = jax.jit(full_v)
        slat_v = jax.jit(slat_v)

    w = jnp.asarray(walkers)
    nuc = jnp.asarray(nuc_crds)
    out = np.empty(w.shape[0], dtype=float)
    for start in range(0, w.shape[0], batch_size):
        batch = w[start:start + batch_size]
        lf = np.asarray(full_v(batch, nuc, params))
        ls = np.asarray(slat_v(batch, nuc, params))
        out[start:start + batch.shape[0]] = lf - ls
    return out


def kato_cusp_jastrow(
    walkers: np.ndarray,
    n_alpha: int,
    n_beta: int,
    b: float = 1.0,
    layout: str = "interleaved",
) -> np.ndarray:
    """Analytic short-range Kato e-e cusp factor ``J = sum_{i<j} u(r_ij)``.

    ``u(r) = a * r / (1 + b r)`` with ``a = 1/2`` for unlike-spin pairs and
    ``a = 1/4`` for like-spin pairs (the Kato cusp slopes ``u'(0) = a``).
    ``u(0) = 0`` and ``u`` saturates to ``a/b`` beyond ``~1/b``, so the
    factor carries *only* the non-analytic kink at coalescence and leaves
    the smooth long-range correlation untouched -- the Strategy-B cusp-only
    TC factor. Decode with ``tau = +1`` (``e^{-J}``) to remove the cusp.

    ``layout='interleaved'`` (NN convention: up = even indices, down = odd)
    or ``'grouped'`` (up first, then down). Returns ``(K_s,)``.
    """
    w = np.asarray(walkers, dtype=float)
    K, N, _ = w.shape
    if layout == "interleaved":
        spin = np.array([i % 2 for i in range(N)])  # 0=up(even), 1=down(odd)
    elif layout == "grouped":
        spin = np.array([0] * n_alpha + [1] * n_beta)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    J = np.zeros(K)
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(w[:, i, :] - w[:, j, :], axis=1)
            a = 0.5 if spin[i] != spin[j] else 0.25
            J += a * r / (1.0 + b * r)
    return J


def jastrow_from_logamps(
    log_psi_full: np.ndarray,
    log_slater: np.ndarray,
) -> np.ndarray:
    """``J = log|Psi_full| - log|Phi_slater|`` from two precomputed arrays.

    Use when both log-amplitudes were already streamed to the bank (the
    cheapest path: no model reload).
    """
    a = np.asarray(log_psi_full, dtype=float)
    b = np.asarray(log_slater, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return a - b


def nn_jastrow_on_walkers(
    config,
    mol_info,
    rng_key,
    trained_params,
    walkers: np.ndarray,
    nuc_crds: np.ndarray,
    batch_size: int = 2048,
    jit: bool = True,
) -> np.ndarray:
    """Retrofit ``J(R_k)`` for an NN bank with no streamed ``/jastrow``.

    Computes ``J = log|Psi_full| - log|Phi_slater|`` where ``Phi_slater`` is
    the same trained network with its additive deep-Jastrow head removed
    (``omni.jastrow = None``). The trained ``params`` carry a jastrow
    subtree that the jastrow-disabled graph does not expect, so the subtree
    is filtered out *once* (pure pytree restructuring) before merging --
    this is the only subtlety. Returns ``(K_s,)``.

    Prefer the streamed ``/jastrow`` dataset (schema >= 1.1.0) when
    available; this path exists for banks dumped before that schema.
    """
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from OmegaQMC.psi.nn.build import build_nn_wf
    from OmegaQMC.psi.nn.config import load_nn_config
    from OmegaQMC.psi.nn.types import PhysicalConfiguration

    if isinstance(config, str):
        config = load_nn_config(config)

    # Full model graph (for log|Psi_full|) and Jastrow-disabled graph.
    m_full = build_nn_wf(config, mol_info, nnx.Rngs(rng_key))
    gd_full, _pf, other_full = nnx.split(m_full, nnx.Param, ...)
    m_slat = build_nn_wf(config, mol_info, nnx.Rngs(rng_key))
    m_slat.omni.jastrow = None
    gd_slat, params_slat_tmpl, other_slat = nnx.split(m_slat, nnx.Param, ...)

    # Drop the jastrow subtree from the trained params (filter once).
    keys_slat = list(dict(params_slat_tmpl.flat_state()).keys())
    flat_full = dict(trained_params.flat_state())
    missing = [k for k in keys_slat if k not in flat_full]
    if missing:
        raise ValueError(
            f"trained params missing {len(missing)} slater leaves; "
            f"architecture/checkpoint mismatch (e.g. {missing[0]})"
        )
    params_slat = nnx.State.from_flat_path({k: flat_full[k]
                                            for k in keys_slat})

    def _logamp(graphdef, params, other):
        def fn(elec_crds, nuc):
            r_up = elec_crds[::2]
            r_dn = elec_crds[1::2]
            r_grouped = jnp.concatenate([r_up, r_dn], axis=0)
            pc = PhysicalConfiguration(R=nuc, r=r_grouped,
                                       mol_idx=jnp.array(0))
            return nnx.merge(graphdef, params, other)(pc).log
        return fn

    full_fn = jax.vmap(_logamp(gd_full, trained_params, other_full),
                       in_axes=(0, None))
    slat_fn = jax.vmap(_logamp(gd_slat, params_slat, other_slat),
                       in_axes=(0, None))
    if jit:
        full_fn = jax.jit(full_fn)
        slat_fn = jax.jit(slat_fn)

    w = jnp.asarray(walkers)
    nuc = jnp.asarray(nuc_crds)
    out = np.empty(w.shape[0], dtype=float)
    for s in range(0, w.shape[0], batch_size):
        b = w[s:s + batch_size]
        out[s:s + b.shape[0]] = (np.asarray(full_fn(b, nuc))
                                 - np.asarray(slat_fn(b, nuc)))
    return out
