"""
ANOVA distillation of the NN Jastrow J_NN -> pairwise u(r), then TC-FCI.

Fit (least squares over the bank, measure = |Psi_NN|^2):
    J_NN(R) ~ c + sum_{i,A} f(|r_i - R_A|)  +  u(r_12)
with a cusp-isolating 2-body basis: only one basis function has nonzero slope
at r=0, so its fitted coefficient IS u'(0) -- a direct read of whether the NN
carries the Kato e-e cusp (1/2 for the opposite-spin H2 pair). Reports the
variance hierarchy (how much of J_NN is constant / 1-body / 2-body), the cusp
slope, then runs TC-FCI with the distilled u vs the analytic cusp vs CBS.

  python examples/distill_jnn.py --npz h2_jnn.npz
"""

import argparse
import numpy as np
from pyscf import gto, scf, fci, ao2mo

from tc_fci_2e import (build_2e_hamiltonian, ground_state, u_profiles,
                       compute_tc_corrections, _mo_on_grid)


# ---- 2-body basis in r (one cusp fn + flat-at-0 smooth fns) ----
def u_basis(r):
    r = np.asarray(r)
    cols = [r / (1.0 + r)]                      # cusp: slope 1 at r=0
    for al in (0.3, 0.7, 1.5, 3.0):
        cols.append(r * r * np.exp(-al * r))    # slope 0 at r=0
    return np.stack(cols, axis=-1)              # (..., 5)

def u_basis_prime(r):
    r = np.asarray(r)
    cols = [1.0 / (1.0 + r) ** 2]
    for al in (0.3, 0.7, 1.5, 3.0):
        cols.append((2.0 * r - al * r * r) * np.exp(-al * r))
    return np.stack(cols, axis=-1)


def en_basis(d):
    d = np.asarray(d)
    cols = [d * np.exp(-d)]
    for be in (0.5, 1.0, 2.0):
        cols.append(np.exp(-be * d * d))
    return np.stack(cols, axis=-1)              # (..., 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="h2_jnn.npz")
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--grid-level", type=int, default=1)
    args = p.parse_args()

    d = np.load(args.npz)
    W = d["walkers"].astype(float)              # (K, 2, 3) Bohr
    J = d["jnn"].astype(float)                  # (K,)
    nuc = d["nuc"]                              # (2, 3)
    K = W.shape[0]

    r12 = np.linalg.norm(W[:, 0] - W[:, 1], axis=1)
    # electron-nucleus distances summed-basis 1-body feature
    en_feat = np.zeros((K, en_basis(np.array([1.0])).shape[-1]))
    for i in range(2):
        for A in range(nuc.shape[0]):
            dA = np.linalg.norm(W[:, i] - nuc[A][None, :], axis=1)
            en_feat += en_basis(dA)
    u_feat = u_basis(r12)                        # (K, 5)

    ones = np.ones((K, 1))
    X0 = ones
    X1 = np.hstack([ones, en_feat])
    X2 = np.hstack([ones, en_feat, u_feat])

    def r2(X):
        beta, *_ = np.linalg.lstsq(X, J, rcond=None)
        resid = J - X @ beta
        return 1.0 - np.var(resid) / np.var(J), beta

    r2_0, _ = r2(X0)
    r2_1, _ = r2(X1)
    r2_2, beta2 = r2(X2)
    # 2-body coefficients are the last 5 of beta2
    c_u = beta2[-u_feat.shape[1]:]
    cusp_slope = float(c_u[0])                   # coeff of r/(1+r), slope-1 fn

    print(f"=== ANOVA distillation of J_NN (H2, K={K}) ===")
    print(f"  variance explained:")
    print(f"    constant (mean)          : {r2_0*100:6.2f} %")
    print(f"    + 1-body (e-n)           : {r2_1*100:6.2f} %  "
          f"(1-body share {(r2_1-r2_0)*100:.2f}%)")
    print(f"    + 2-body u(r12)          : {r2_2*100:6.2f} %  "
          f"(2-body share {(r2_2-r2_1)*100:.2f}%)")
    print(f"    residual (unmodeled)     : {(1-r2_2)*100:6.2f} %  "
          f"(for H2 = position-dep 2-body; for N>2 would incl. 3-body)")
    print(f"  fitted cusp slope u'(0)    = {cusp_slope:+.4f}  "
          f"(Kato opposite-spin = +0.5)")

    # Distilled uprime(r) callable from the fitted 2-body coefficients.
    def uprime_distilled(r):
        return u_basis_prime(r) @ c_u

    # Cusp-constrained variant: force the slope-1 coefficient to 0.5.
    c_u_cc = c_u.copy(); c_u_cc[0] = 0.5
    def uprime_cc(r):
        return u_basis_prime(r) @ c_u_cc

    # ---- TC-FCI with distilled u vs analytic cusp vs CBS ----
    mol = gto.M(atom=[("H", nuc[0]), ("H", nuc[1])], unit="Bohr",
                basis=args.basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    C = np.asarray(mf.mo_coeff); n = C.shape[1]
    e_nuc = mol.energy_nuc()
    h1 = C.T @ mf.get_hcore() @ C
    g = np.transpose(ao2mo.restore(1, ao2mo.kernel(mol, C), n), (0, 2, 1, 3))
    e_fci = fci.FCI(mol, mf.mo_coeff).kernel()[0]
    grid = _mo_on_grid(mol, C, args.grid_level)

    def tc(uprime):
        dV = compute_tc_corrections(mol, C, uprime, args.grid_level, grid=grid)
        e, imag = ground_state(build_2e_hamiltonian(h1, g + dV, n), e_nuc)
        return e, imag

    # CBS from TZ/QZ
    def fcibasis(b):
        m = gto.M(atom=[("H", nuc[0]), ("H", nuc[1])], unit="Bohr", basis=b,
                  verbose=0)
        mfb = scf.RHF(m).run(verbose=0)
        return fci.FCI(m, mfb.mo_coeff).kernel()[0]
    e_tz, e_qz = fcibasis("cc-pvtz"), fcibasis("cc-pvqz")
    e_cbs = (64 * e_qz - 27 * e_tz) / 37

    e_d, _ = tc(uprime_distilled)
    e_cc, _ = tc(uprime_cc)
    e_an, _ = tc(u_profiles("cusp", 0.5, 0.2924))   # the step-2 NN-matched cusp

    print(f"\n=== TC-FCI ({args.basis}) ===")
    print(f"  FCI(basis) ceiling        = {e_fci:.6f}")
    print(f"  CBS (TZ/QZ extrap)        = {e_cbs:.6f}")
    print(f"  TC-FCI (distilled u, free)= {e_d:.6f}  "
          f"({(e_d-e_cbs)*1e3:+.2f} mE_h vs CBS)")
    print(f"  TC-FCI (distilled, cusp=.5)= {e_cc:.6f}  "
          f"({(e_cc-e_cbs)*1e3:+.2f} mE_h vs CBS)")
    print(f"  TC-FCI (analytic a=.5,b=.29)={e_an:.6f}  "
          f"({(e_an-e_cbs)*1e3:+.2f} mE_h vs CBS)")


if __name__ == "__main__":
    main()
