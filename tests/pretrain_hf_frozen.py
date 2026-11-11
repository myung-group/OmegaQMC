"""Frozen copy of :func:`OmegaQMC.pretrain_nn.pretrain_hf` at 5fdb949.

The pretraining loop as it was before the optional HF-density sampling
(``hf_fraction``) was added.  :mod:`tests.test_pretrain_nn` checks that
``pretrain_hf(..., hf_fraction=0.0)`` reproduces it bit for bit (loss
history, parameters and walkers), because existing benchmark results
were produced with it.  Do not edit; the helpers it uses
(:class:`HartreeFockTarget`, :func:`init_electrons`,
:func:`_target_matrix`) are imported from the live module.
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from OmegaQMC.spin_layout import to_grouped
from OmegaQMC.pretrain_nn import (
    INIT_WIDTH,
    PRETRAIN_BASIS,
    PRETRAIN_ITERATIONS,
    PRETRAIN_LR,
    PRETRAIN_MCMC_WIDTH,
    HartreeFockTarget,
    _target_matrix,
    init_electrons,
)


def pretrain_hf(
    log_psi,
    mol_info,
    params,
    rng_key,
    *,
    iterations: int = PRETRAIN_ITERATIONS,
    num_walkers: int = 4096,
    basis: str = PRETRAIN_BASIS,
    lr: float = PRETRAIN_LR,
    mcmc_width: float = PRETRAIN_MCMC_WIDTH,
    init_width: float = INIT_WIDTH,
    walkers=None,
    log_every: int = 100,
    fout=None,
    verbose: int = 1,
):
    """Pretrain an NN wavefunction's orbitals against UHF.

    Args:
        log_psi: ``log|psi|`` callable from
            :func:`~OmegaQMC.psi.nn.adapter.make_nn_log_psi`; must
            carry the ``.orbitals`` and ``.full_determinant``
            attributes set there.
        mol_info: :class:`~OmegaQMC.utils.Mole_custom` (coordinates
            in bohr).
        params: Initial parameters.
        rng_key: JAX PRNG key.
        iterations: Adam steps (FermiNet: 1000).
        num_walkers: Walkers (FermiNet: the training batch, 4096).
        basis: Hartree-Fock basis (FermiNet: ``'sto-6g'``).
        lr: Adam learning rate (FermiNet: 3e-4).
        mcmc_width: Gaussian Metropolis step per iteration (bohr).
        init_width: Width of :func:`init_electrons`.
        walkers: Optional starting walkers; default
            :func:`init_electrons`.
        log_every: Iterations between log lines.
        fout: Text stream for the log (default stdout).
        verbose: 0 silences the log.

    Returns:
        ``(params, walkers, info)``: the pretrained parameters (same
        dtypes as *params*), the final walkers ``(W, n_e, 3)``, and a
        dict with ``'loss'`` (per-iteration array), ``'acceptance'``
        (mean acceptance), ``'e_hf'`` and ``'hf_converged'``.
    """
    import optax

    if not hasattr(log_psi, 'orbitals'):
        raise ValueError(
            "log_psi has no .orbitals; build it with make_nn_log_psi"
        )
    if fout is None:
        fout = sys.stdout
    if isinstance(rng_key, int):
        rng_key = jax.random.key(rng_key)
    full_det = bool(log_psi.full_determinant)
    orbitals = log_psi.orbitals
    nuc = jnp.asarray(mol_info.coords, dtype=jnp.float64)

    t0 = time.perf_counter()
    target = HartreeFockTarget(mol_info, basis=basis)
    if verbose:
        print(f"# HF pretraining: UHF/{basis} E = {target.energy:.8f} Ha"
              f" (converged: {target.converged}), {iterations}"
              f" iterations, {num_walkers} walkers, Adam lr {lr:g},"
              f" MCMC width {mcmc_width:g} bohr,"
              f" full_det={full_det}", file=fout)

    rng_key, k_init = jax.random.split(rng_key)
    if walkers is None:
        walkers = init_electrons(
            mol_info, num_walkers, k_init, init_width=init_width,
        )
    walkers = jnp.asarray(walkers, dtype=jnp.float64)

    batch_orbitals = jax.vmap(orbitals, in_axes=(0, None, None))
    batch_lp = jax.vmap(log_psi, in_axes=(0, None, None))

    def loss_fn(p, x, t_up, t_dn):
        o_up, o_dn, _ = batch_orbitals(x, nuc, p)
        if full_det:
            tgt = _target_matrix(t_up, t_dn, True)
            orb = jnp.concatenate([o_up, o_dn], axis=-2)
            return jnp.mean((tgt[:, None] - orb) ** 2)
        loss = 0.0
        for t, o in ((t_up, o_up), (t_dn, o_dn)):
            if t.size:
                loss = loss + jnp.mean((t[:, None] - o) ** 2)
        return loss

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p, state, x, t_up, t_dn, key):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, t_up, t_dn)
        updates, state = optimizer.update(grads, state, p)
        p = jax.tree.map(
            lambda a, u: (a + u).astype(a.dtype), p, updates,
        )
        # One all-electron Metropolis move under the updated |psi|^2.
        k_prop, k_acc = jax.random.split(key)
        prop = x + mcmc_width * jax.random.normal(
            k_prop, x.shape, dtype=x.dtype,
        )
        lp_old = batch_lp(x, nuc, p)
        lp_new = batch_lp(prop, nuc, p)
        acc = jnp.log(jax.random.uniform(
            k_acc, (x.shape[0],), dtype=x.dtype,
        )) < 2.0 * (lp_new - lp_old)
        x = jnp.where(acc[:, None, None], prop, x)
        return p, state, x, loss, jnp.mean(acc)

    n_up, n_down = int(mol_info.n_up), int(mol_info.n_down)

    def grouped(x):
        return np.asarray(to_grouped(x, n_up, n_down))

    losses = np.empty(iterations)
    accs = np.empty(iterations)
    for it in range(iterations):
        t_up, t_dn = target(grouped(walkers))
        rng_key, k = jax.random.split(rng_key)
        params, opt_state, walkers, loss, acc = step(
            params, opt_state, walkers,
            jnp.asarray(t_up), jnp.asarray(t_dn), k,
        )
        losses[it] = float(loss)
        accs[it] = float(acc)
        if verbose and (it % log_every == 0 or it == iterations - 1):
            print(f"# pretrain {it:6d}  loss {losses[it]:.6e}"
                  f"  acc {np.mean(accs[max(0, it - log_every + 1):it + 1]):.3f}"
                  f"  {time.perf_counter() - t0:8.1f} s", file=fout)

    if not np.all(np.isfinite(losses)):
        raise FloatingPointError("non-finite pretraining loss")
    return params, walkers, {
        'loss': losses,
        'acceptance': float(np.mean(accs)) if iterations else float('nan'),
        'e_hf': target.energy,
        'hf_converged': target.converged,
    }
