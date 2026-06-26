"""
Three-body transcorrelation term for general-N TC-FCI (step 3b).

The N>2 piece: -1/2 sum_i (grad_i J)^2 has, beyond the pairwise term,
the cross contribution

    O_3 = - sum_i sum_{j<k != i} u'(r_ij) u'(r_ik) (rhat_ij . rhat_ik),

a Hermitian multiplicative 3-body operator. Its MO integral factorises at the
common vertex through exactly the vector field A used for the 2-body term:

    <pqr|g(1;2,3)|stu> = sum_g w_g phi_p phi_s(g) [A^a12_qt(g) . A^a13_ru(g)],

so no 9-D quadrature. Cusp coefficients are spin-resolved (a=1/2 opposite,
1/4 same). We build the symmetric W_sym = sum over the 3 vertex choices and
add it to the FCI matrix by second-quantised operator action in a unified
spin-orbital representation.

Verification: (G2) u=0 -> H3=0; H3 Hermitian (multiplicative real operator);
(G3) total spectrum real.
"""

import numpy as np
from itertools import permutations

from tc_fci_2e import _mo_on_grid, _build_A, u_profiles
from tc_fci_general import apply_op


def _A_field(mol, C, a, b, grid):
    coords, w, P, Gmo = grid
    n = C.shape[1]
    rho = (P[:, None, :] * P[None, :, :]).reshape(n * n, -1)
    return _build_A(rho, coords, w, u_profiles("cusp", a, b))  # (n^2, G, 3)


def vertex_tensor(P, w, A1, A2, n):
    """V[p,s,q,t,r,u] = sum_g w_g P_p P_s(g) (A1_qt(g).A2_ru(g)),
    with A1 the vertex->2 edge (cusp a1) and A2 the vertex->3 edge (a2)."""
    AA = np.einsum("Qgc,Rgc->QRg", A1, A2)               # (n^2, n^2, G)
    PS = (P[:, None, :] * P[None, :, :])                 # (n, n, G)
    V = np.einsum("g,psg,QRg->psQR", w, PS, AA)          # (n,n,n^2,n^2)
    return V.reshape(n, n, n, n, n, n)                   # [p,s,q,t,r,u]


def build_3body_fci(mol, C, b, dets, grid):
    """Return the 3-body contribution H3 (ndet x ndet) to the TC-FCI matrix."""
    n = C.shape[1]
    coords, w, P, Gmo = grid
    A_half = _A_field(mol, C, 0.5, b, grid)      # opposite-spin edge
    A_quar = _A_field(mol, C, 0.25, b, grid)     # same-spin edge

    # Vertex tensors for the 4 (edge1, edge2) cusp combinations.
    Av = {0.5: A_half, 0.25: A_quar}
    V = {(a1, a2): vertex_tensor(P, w, Av[a1], Av[a2], n)
         for a1 in (0.5, 0.25) for a2 in (0.5, 0.25)}

    def cusp(s1, s2):
        return 0.5 if s1 != s2 else 0.25

    # Unified spin-orbital indexing: alpha p -> p, beta p -> n+p.
    def to_so(d):
        oa, ob = d
        return tuple(sorted(list(oa) + [n + x for x in ob]))

    def spatial_spin(I):
        return (I, 0) if I < n else (I - n, 1)

    def so_to_det(so):
        oa = tuple(sorted(x for x in so if x < n))
        ob = tuple(sorted(x - n for x in so if x >= n))
        return (oa, ob)

    idx = {d: I for I, d in enumerate(dets)}
    so_dets = [to_so(d) for d in dets]
    ndet = len(dets)
    H3 = np.zeros((ndet, ndet))

    def wsym(P_, Q_, R_, S_, T_, U_):
        # spatial + spins of the three electrons (1:p/s, 2:q/t, 3:r/u)
        p, s1 = spatial_spin(P_); s, _ = spatial_spin(S_)
        q, s2 = spatial_spin(Q_); t, _ = spatial_spin(T_)
        r, s3 = spatial_spin(R_); u, _ = spatial_spin(U_)
        val = (V[(cusp(s1, s2), cusp(s1, s3))][p, s, q, t, r, u]
               + V[(cusp(s2, s1), cusp(s2, s3))][q, t, p, s, r, u]
               + V[(cusp(s3, s1), cusp(s3, s2))][r, u, p, s, q, t])
        return val

    for J, soJ in enumerate(so_dets):
        occ = soJ
        # annihilate an ordered distinct triple (S,T,U) from occupied,
        # create an ordered distinct triple (P,Q,R) with matching spins.
        for S_, T_, U_ in permutations(occ, 3):
            sS = S_ < n; sT = T_ < n; sU = U_ < n   # spin bools (alpha=True)
            # creations must match electron spins: P~S, Q~T, R~U
            P_choices = [x for x in range(2 * n) if (x < n) == sS]
            Q_choices = [x for x in range(2 * n) if (x < n) == sT]
            R_choices = [x for x in range(2 * n) if (x < n) == sU]
            for P_ in P_choices:
                for Q_ in Q_choices:
                    for R_ in R_choices:
                        sgn, new = apply_op(occ, [S_, T_, U_], [R_, Q_, P_])
                        if sgn == 0:
                            continue
                        I = idx.get(so_to_det(new))
                        if I is None:
                            continue
                        H3[I, J] += -(1.0 / 6.0) * sgn * wsym(
                            P_, Q_, R_, S_, T_, U_)
    return H3
