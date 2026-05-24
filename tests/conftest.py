"""
Shared test fixtures and helpers.
"""

import numpy as np


def metropolis_sample(
    psi_fn,
    n_walkers: int,
    n_electrons: int,
    n_steps: int,
    burnin: int = 100,
    step_size: float = 0.8,
    seed: int = 0,
):
    """Single-electron Metropolis sampler from ``|psi_fn|^2``.

    Returns ``(walkers, psi_at_walkers, acceptance_rate)``.
    """
    rng = np.random.default_rng(seed)
    R = rng.normal(scale=1.0, size=(n_walkers, n_electrons, 3))
    psi = np.asarray(psi_fn(R))
    accept_count = 0
    total = 0
    for _ in range(burnin + n_steps):
        for e in range(n_electrons):
            prop = R.copy()
            prop[:, e, :] = prop[:, e, :] + rng.normal(
                scale=step_size, size=(n_walkers, 3)
            )
            psi_new = np.asarray(psi_fn(prop))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(
                    np.abs(psi) > 1e-30,
                    (psi_new / psi) ** 2,
                    1.0,
                )
            accept = rng.uniform(size=n_walkers) < ratio
            R = np.where(accept[:, None, None], prop, R)
            psi = np.where(accept, psi_new, psi)
            accept_count += int(accept.sum())
            total += n_walkers
    return R, psi, accept_count / total
