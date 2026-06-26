"""Shared pairwise-Jastrow basis for ANOVA distillation and TC use.

u(r) = c0 * r/(1+r)  +  sum_k c_k * r^2 exp(-alpha_k r)
Only the first basis function (r/(1+r)) has nonzero slope at r=0, so c0 IS
u'(0). For TC we clamp c0 to the Kato cusp (1/2 opposite, 1/4 same) and keep
the fitted tail c_k (the NN's learned medium-range correlation).
"""

import numpy as np

ALPHAS = (0.3, 0.7, 1.5, 3.0)


def u_basis(r):
    r = np.asarray(r)
    cols = [r / (1.0 + r)] + [r * r * np.exp(-a * r) for a in ALPHAS]
    return np.stack(cols, axis=-1)


def u_basis_prime(r):
    r = np.asarray(r)
    cols = [1.0 / (1.0 + r) ** 2]
    cols += [(2.0 * r - a * r * r) * np.exp(-a * r) for a in ALPHAS]
    return np.stack(cols, axis=-1)


def make_uprime(coeffs, cusp, lam=1.0):
    """u'(r) callable from fitted coeffs: c0 clamped to the Kato cusp, the
    NN-distilled tail (c1..) scaled by lam (the TC strength knob). lam=1 is
    the raw distilled tail; lam=0 is the bare Kato cusp; lam in (0,1) damps
    the over-broad tail to the TC-optimal strength."""
    c = np.array(coeffs, dtype=float).copy()
    c[0] = float(cusp)
    c[1:] *= float(lam)
    def uprime(r):
        return np.asarray(u_basis_prime(r)) @ c
    return uprime
