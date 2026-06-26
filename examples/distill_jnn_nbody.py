"""
ANOVA distillation of J_NN for N>2 electrons -> the genuine many-body
(3-body) Sobol variance fraction, plus spin-resolved pairwise cusps.

Fit (least squares over the bank):
  J_NN ~ c + sum_i f_el(|r_i-R_A|)
            + sum_{opp pairs} u_opp(r) + sum_{same pairs} u_same(r)
            + sum_{triples} 3body(d_ab,d_ac,d_bc)
and report the incremental variance explained at each order. The 3-body
increment is the real "how pairwise is J_NN" number (zero by construction for
2 electrons; nonzero only for N>=3). Cusp slopes u'_opp(0), u'_same(0) read
off the cusp-isolating basis (Kato: 1/2 opposite, 1/4 same).

  python examples/distill_jnn_nbody.py --npz beh2_jnn.npz
"""

import argparse
from itertools import combinations

import numpy as np

from distilled_u import u_basis


def en_basis(d):
    d = np.asarray(d)
    cols = [d * np.exp(-d)]
    for be in (0.5, 1.0, 2.0):
        cols.append(np.exp(-be * d * d))
    return np.stack(cols, axis=-1)


def tb_feature(d_ab, d_ac, d_bc):
    """Symmetric 3-body features: sum over the 3 vertex choices of the
    product of the two incident edges, for several ranges (TC-3-body form)."""
    out = []
    for al in (0.5, 1.0, 2.0):
        pa, pb, pc = (np.exp(-al * d_ab), np.exp(-al * d_ac),
                      np.exp(-al * d_bc))
        out.append(pa * pb + pa * pc + pb * pc)  # vertex a,b,c
    return np.stack(out, axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    args = p.parse_args()
    d = np.load(args.npz)
    W = d["walkers"].astype(float)               # (K, N, 3)
    J = d["jnn"].astype(float)
    nuc = d["nuc"]
    na, nb = int(d["na"]), int(d["nb"])
    K, N, _ = W.shape
    spin = np.array([i % 2 for i in range(N)])   # interleaved: even=up

    # pairwise distances
    pairs = list(combinations(range(N), 2))
    rij = {pr: np.linalg.norm(W[:, pr[0]] - W[:, pr[1]], axis=1)
           for pr in pairs}
    opp = [pr for pr in pairs if spin[pr[0]] != spin[pr[1]]]
    same = [pr for pr in pairs if spin[pr[0]] == spin[pr[1]]]

    # 1-body (e-n), separated by element: nucleus 0 = Be (heavy), rest = H
    f_heavy = np.zeros((K, en_basis(np.array([1.0])).shape[-1]))
    f_light = np.zeros_like(f_heavy)
    for i in range(N):
        f_heavy += en_basis(np.linalg.norm(W[:, i] - nuc[0][None], axis=1))
        for A in range(1, nuc.shape[0]):
            f_light += en_basis(np.linalg.norm(W[:, i] - nuc[A][None], axis=1))

    nuB = u_basis(np.array([1.0])).shape[-1]
    u_opp = (sum(u_basis(rij[pr]) for pr in opp) if opp
             else np.zeros((K, nuB)))            # (K, 5)
    u_same = (sum(u_basis(rij[pr]) for pr in same) if same
              else np.zeros((K, nuB)))

    tb = np.zeros((K, 3))
    for (a, b, c) in combinations(range(N), 3):
        tb += tb_feature(rij[(a, b)] if (a, b) in rij else rij[(b, a)],
                         rij[(a, c)] if (a, c) in rij else rij[(c, a)],
                         rij[(b, c)] if (b, c) in rij else rij[(c, b)])

    ones = np.ones((K, 1))
    blocks = {
        "const": ones,
        "1body": np.hstack([f_heavy, f_light]),
        "2body": np.hstack([u_opp, u_same]),
        "3body": tb,
    }

    def r2(X):
        beta, *_ = np.linalg.lstsq(X, J, rcond=None)
        return 1.0 - np.var(J - X @ beta) / np.var(J), beta

    order = ["const", "1body", "2body", "3body"]
    X = np.zeros((K, 0)); prev = 0.0
    print(f"=== ANOVA distillation J_NN ({args.npz}, K={K}, N={N}e-, "
          f"{len(opp)} opp / {len(same)} same pairs, "
          f"{len(list(combinations(range(N),3)))} triples) ===")
    beta_full = None
    for name in order:
        X = np.hstack([X, blocks[name]])
        cur, beta = r2(X)
        print(f"  + {name:6s}: cumulative {cur*100:6.2f} %   "
              f"(this order +{(cur-prev)*100:5.2f}%)")
        prev = cur
        beta_full = beta
    print(f"    residual (unmodeled): {(1-prev)*100:6.2f} %")

    # cusp slopes: coefficients of the slope-1 basis fn in u_opp, u_same
    n_en = blocks["1body"].shape[1]
    off = 1 + n_en
    nuB = u_basis(np.array([1.0])).shape[-1]
    c_opp_vec = beta_full[off:off + nuB]
    c_same_vec = beta_full[off + nuB:off + 2 * nuB]
    print(f"  free-fit cusp slopes: u'_opp(0) = {c_opp_vec[0]:+.3f} (Kato +0.50), "
          f"u'_same(0) = {c_same_vec[0]:+.3f} (Kato +0.25) [unreliable]")

    # Cusp-CONSTRAINED refit: clamp cusp to Kato, fit the tail consistently
    # (the free tail compensates the free cusp, so it can't be reused).
    J_resid = J - 0.5 * u_opp[:, 0] - 0.25 * u_same[:, 0]
    X_cc = np.hstack([ones, blocks["1body"], u_opp[:, 1:], u_same[:, 1:], tb])
    beta_cc, *_ = np.linalg.lstsq(X_cc, J_resid, rcond=None)
    o = 1 + blocks["1body"].shape[1]
    tail_opp = beta_cc[o:o + (nuB - 1)]
    tail_same = beta_cc[o + (nuB - 1):o + 2 * (nuB - 1)]
    c_opp = np.concatenate([[0.5], tail_opp])
    c_same = np.concatenate([[0.25], tail_same])
    out = args.npz.replace(".npz", "_u.npz")
    np.savez(out, c_opp=c_opp, c_same=c_same)
    print(f"  saved cusp-constrained distilled u -> {out} "
          f"(c0=Kato, tail refit consistently)")


if __name__ == "__main__":
    main()
