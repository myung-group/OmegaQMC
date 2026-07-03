"""
Spin-blocked (UHF) QED-CCSD with density-fitted integrals and precomputed
integral-block intermediates — for open-shell references (radicals,
cations/anions with unpaired electrons).

Alpha/beta-blocked counterpart of :mod:`OmegaQMC.addons.qed_ccsd_rhf`,
obtained by exactly spin-tracing the same Wick-derived spin-orbital
equations (tools/qed_ccsd_df_derivation/) while keeping the two spins
independent: amplitudes are stored as blocks

    t1_10_a[a,i], t1_10_b            (and likewise t2_11_*, t2_12_*)
    t2_20_aa, t2_20_bb               antisymmetric same-spin doubles
    t2_20_ab[a,b,i,j]                mixed block (a,i alpha; b,j beta)
    (analogously t2_21_*, t2_22_*; t1_01, t2_02 scalars)

Each kernel evaluates the literal spin block of the spin-orbital residual,
so denominators, DIIS and convergence carry over per block. Cost is ~3-4x
a closed-shell calculation of the same size instead of the ~16x of the
spin-orbital code.

Intermediate reuse: all two-electron contractions except the all-virtual
ladders use spin-resolved 4-index chemist blocks (pq|rs)_{s1 s2} built
once at setup; the (vv|vv) blocks are never formed — ladder terms go
through the two-factor batched helper :func:`_vvvv_ladder2` (peak
intermediate nvir^3).

The reference is a QED-UHF dict from :mod:`OmegaQMC.addons.qed_uhf`
(run_qed_uhf); a closed-shell QED-HF dict is rejected — use qed_ccsd_rhf
for those (this module reproduces it exactly in the closed-shell limit).
Equations are complete (they include the quartic t2_21/t2_22 BCH terms
missing from the published reference implementation); each kernel was
validated to machine precision against the Wick-derived spin-orbital
equations (tools/qed_ccsd_df_derivation/validate_uhf_kernels.py).
"""

import math
import time

import numpy as np
from opt_einsum import contract

from pyscf import gto

from .qed_ccsd_utils import _DiisHistory, _ao_df_factor

_ALL_AMPS = frozenset(
    ("t1_01", "t2_02", "t2_11", "t2_21", "t2_12", "t2_22"))


def _vvvv_ladder2(B1, B2, W, out, alpha=1.0):
    """``out[a,b,i,j] += alpha * sum_{x,c,d} B1[x,a,c] B2[x,b,d]
    W[c,d,i,j]`` batched over ``b`` (peak intermediate ``nv1^2 * nv2``).
    ``B1``/``B2`` are the all-virtual DF slices of the two spin cases
    (identical objects for same-spin ladders). ``W`` must be C-contiguous.
    """
    naux, nv1 = B1.shape[0], B1.shape[1]
    nv2 = B2.shape[1]
    no12 = W.shape[2] * W.shape[3]
    Wm = W.reshape(nv1 * nv2, no12)
    Bm1 = np.ascontiguousarray(B1.transpose(1, 2, 0)).reshape(
        nv1 * nv1, naux)
    for b in range(nv2):
        g_b = Bm1 @ B2[:, b, :]                    # ((a, c), d)
        out[:, b] += alpha * (
            g_b.reshape(nv1, nv1 * nv2) @ Wm).reshape(
                nv1, W.shape[2], W.shape[3])
    return out


_NEEDED_V = ['v_oooo_aa', 'v_oooo_ab', 'v_oooo_bb', 'v_ooov_aa', 'v_ooov_ab', 'v_ooov_ba', 'v_ooov_bb', 'v_oovv_aa', 'v_oovv_ab', 'v_oovv_ba', 'v_oovv_bb', 'v_ovov_aa', 'v_ovov_ab', 'v_ovov_bb', 'v_ovvv_aa', 'v_ovvv_ab', 'v_ovvv_ba', 'v_ovvv_bb']


def ccsd_energy(ints, w, amps, active=_ALL_AMPS):
    res = 0.0

    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ck->', ints.d_a_vo, amps.t1_10_a)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ck->', ints.d_a_vo, amps.t2_11_a)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ck->', ints.d_b_vo, amps.t1_10_b)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ck->', ints.d_b_vo, amps.t2_11_b)
    res += 1.0 * contract('ck,ck->', ints.f_a_vo, amps.t1_10_a)
    res += 1.0 * contract('ck,ck->', ints.f_b_vo, amps.t1_10_b)
    res += 0.5 * contract('ck,dl,kcld->', amps.t1_10_a, amps.t1_10_a, ints.v_ovov_aa)
    res += -0.5 * contract('ck,dl,kdlc->', amps.t1_10_a, amps.t1_10_a, ints.v_ovov_aa)
    res += 1.0 * contract('ck,dl,kcld->', amps.t1_10_a, amps.t1_10_b, ints.v_ovov_ab)
    res += 0.5 * contract('ck,dl,kcld->', amps.t1_10_b, amps.t1_10_b, ints.v_ovov_bb)
    res += -0.5 * contract('ck,dl,kdlc->', amps.t1_10_b, amps.t1_10_b, ints.v_ovov_bb)
    res += 0.5 * contract('cdkl,kcld->', amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('cdkl,kcld->', amps.t2_20_ab, ints.v_ovov_ab)
    res += 0.5 * contract('cdkl,kcld->', amps.t2_20_bb, ints.v_ovov_bb)

    return float(res)

def ccsd_t1_10_a(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_a.size, ints.eps_occ_a.size))

    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ac,ci->ai', ints.d_a_vv, amps.t1_10_a)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci->ai', ints.d_a_vv, amps.t2_11_a)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ai->ai', ints.d_a_vo)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ck,acik->ai', ints.d_a_vo, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ck,ai->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't2_21' in active:
        res += 1.0 * contract('ck,acik->ai', ints.d_a_vo, amps.t2_21_aa)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ik,ak->ai', ints.d_a_oo, amps.t1_10_a)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak->ai', ints.d_a_oo, amps.t2_11_a)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ck,ai->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_11_a)
    if 't2_21' in active:
        res += 1.0 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_21_ab)
    res += 1.0 * contract('ac,ci->ai', ints.f_a_vv, amps.t1_10_a)
    res += 1.0 * contract('ai->ai', ints.f_a_vo)
    res += -1.0 * contract('ck,ak,ci->ai', ints.f_a_vo, amps.t1_10_a, amps.t1_10_a)
    res += 1.0 * contract('ck,acik->ai', ints.f_a_vo, amps.t2_20_aa)
    res += -1.0 * contract('ik,ak->ai', ints.f_a_oo, amps.t1_10_a)
    res += 1.0 * contract('ck,acik->ai', ints.f_b_vo, amps.t2_20_ab)
    res += -1.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ovov_aa)
    res += -1.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t1_10_b, ints.v_ovov_ab)
    res += 1.0 * contract('ak,cl,di,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ovov_aa)
    res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_a, amps.t1_10_a, ints.v_ooov_aa)
    res += 1.0 * contract('ak,cl,ilkc->ai', amps.t1_10_a, amps.t1_10_a, ints.v_ooov_aa)
    res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_a, amps.t1_10_b, ints.v_ooov_ab)
    res += -1.0 * contract('ak,cdil,kcld->ai', amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('ak,cdil,kcld->ai', amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ci,dk,kcad->ai', amps.t1_10_a, amps.t1_10_a, ints.v_ovvv_aa)
    res += 1.0 * contract('ci,dk,kdac->ai', amps.t1_10_a, amps.t1_10_b, ints.v_ovvv_ba)
    res += -1.0 * contract('ci,adkl,kcld->ai', amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('ci,adkl,kcld->ai', amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_a, amps.t1_10_a, ints.v_ovvv_aa)
    res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('ck,adil,kdlc->ai', amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ck,ikac->ai', amps.t1_10_a, ints.v_oovv_aa)
    res += 1.0 * contract('ck,iakc->ai', amps.t1_10_a, ints.v_ovov_aa)
    res += 1.0 * contract('ck,adil,ldkc->ai', amps.t1_10_b, amps.t2_20_aa, ints.v_ovov_ab)
    res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_bb)
    res += -1.0 * contract('ck,adil,kdlc->ai', amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_bb)
    res += 1.0 * contract('ck,iakc->ai', amps.t1_10_b, ints.v_ovov_ab)
    res += -1.0 * contract('ackl,iklc->ai', amps.t2_20_aa, ints.v_ooov_aa)
    res += -1.0 * contract('cdik,kcad->ai', amps.t2_20_aa, ints.v_ovvv_aa)
    res += -1.0 * contract('ackl,iklc->ai', amps.t2_20_ab, ints.v_ooov_ab)
    res += 1.0 * contract('cdik,kdac->ai', amps.t2_20_ab, ints.v_ovvv_ba)

    e_denom = 1.0 / (ints.eps_occ_a[None, :] - ints.eps_vir_a[:, None] - (0.0))
    return amps.t1_10_a + res * e_denom

def ccsd_t1_10_b(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_b.size, ints.eps_occ_b.size))

    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ck,caki->ai', ints.d_a_vo, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ck,ai->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_11_b)
    if 't2_21' in active:
        res += 1.0 * contract('ck,caki->ai', ints.d_a_vo, amps.t2_21_ab)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ac,ci->ai', ints.d_b_vv, amps.t1_10_b)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci->ai', ints.d_b_vv, amps.t2_11_b)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ai->ai', ints.d_b_vo)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ck,ai->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't2_21' in active:
        res += 1.0 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_21_bb)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ik,ak->ai', ints.d_b_oo, amps.t1_10_b)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak->ai', ints.d_b_oo, amps.t2_11_b)
    res += 1.0 * contract('ck,caki->ai', ints.f_a_vo, amps.t2_20_ab)
    res += 1.0 * contract('ac,ci->ai', ints.f_b_vv, amps.t1_10_b)
    res += 1.0 * contract('ai->ai', ints.f_b_vo)
    res += -1.0 * contract('ck,ak,ci->ai', ints.f_b_vo, amps.t1_10_b, amps.t1_10_b)
    res += 1.0 * contract('ck,acik->ai', ints.f_b_vo, amps.t2_20_bb)
    res += -1.0 * contract('ik,ak->ai', ints.f_b_oo, amps.t1_10_b)
    res += -1.0 * contract('ck,al,di,kcld->ai', amps.t1_10_a, amps.t1_10_b, amps.t1_10_b, ints.v_ovov_ab)
    res += -1.0 * contract('ck,al,ilkc->ai', amps.t1_10_a, amps.t1_10_b, ints.v_ooov_ba)
    res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_a, amps.t1_10_b, ints.v_ovvv_ab)
    res += 1.0 * contract('ck,dali,kcld->ai', amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_aa)
    res += -1.0 * contract('ck,dali,kdlc->ai', amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_aa)
    res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_a, amps.t2_20_bb, ints.v_ovov_ab)
    res += 1.0 * contract('ck,kcia->ai', amps.t1_10_a, ints.v_ovov_ab)
    res += -1.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ovov_bb)
    res += 1.0 * contract('ak,cl,di,kcld->ai', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ovov_bb)
    res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_b, amps.t1_10_b, ints.v_ooov_bb)
    res += 1.0 * contract('ak,cl,ilkc->ai', amps.t1_10_b, amps.t1_10_b, ints.v_ooov_bb)
    res += -1.0 * contract('ak,cdli,lckd->ai', amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ak,cdil,kcld->ai', amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('ci,dk,kcad->ai', amps.t1_10_b, amps.t1_10_b, ints.v_ovvv_bb)
    res += -1.0 * contract('ci,dakl,kdlc->ai', amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ci,adkl,kcld->ai', amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_b, amps.t1_10_b, ints.v_ovvv_bb)
    res += 1.0 * contract('ck,dali,ldkc->ai', amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('ck,adil,kdlc->ai', amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('ck,ikac->ai', amps.t1_10_b, ints.v_oovv_bb)
    res += 1.0 * contract('ck,iakc->ai', amps.t1_10_b, ints.v_ovov_bb)
    res += -1.0 * contract('cakl,ilkc->ai', amps.t2_20_ab, ints.v_ooov_ba)
    res += 1.0 * contract('cdki,kcad->ai', amps.t2_20_ab, ints.v_ovvv_ab)
    res += -1.0 * contract('ackl,iklc->ai', amps.t2_20_bb, ints.v_ooov_bb)
    res += -1.0 * contract('cdik,kcad->ai', amps.t2_20_bb, ints.v_ovvv_bb)

    e_denom = 1.0 / (ints.eps_occ_b[None, :] - ints.eps_vir_b[:, None] - (0.0))
    return amps.t1_10_b + res * e_denom

def ccsd_t1_01(ints, w, amps, active=_ALL_AMPS):
    res = 0.0

    if 't1_01' in active and 't2_11' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ck->', ints.d_a_vo, amps.t2_11_a)
    res += 1.0 * contract('ck,ck->', ints.d_a_vo, amps.t1_10_a)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ck,ck->', ints.d_a_vo, amps.t1_10_a)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ck->', ints.d_a_vo, amps.t2_12_a)
    if 't1_01' in active and 't2_11' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ck->', ints.d_b_vo, amps.t2_11_b)
    res += 1.0 * contract('ck,ck->', ints.d_b_vo, amps.t1_10_b)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ck,ck->', ints.d_b_vo, amps.t1_10_b)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ck->', ints.d_b_vo, amps.t2_12_b)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ck->', ints.f_a_vo, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ck->', ints.f_b_vo, amps.t2_11_b)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * w
    if 't2_11' in active:
        res += 1.0 * contract('ck,dl,kcld->', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dl,kdlc->', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dl,kcld->', amps.t1_10_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dl,ldkc->', amps.t1_10_b, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dl,kcld->', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dl,kdlc->', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 0.5 * contract('cdkl,kcld->', amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cdkl,kcld->', amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 0.5 * contract('cdkl,kcld->', amps.t2_21_bb, ints.v_ovov_bb)

    if w == 0:
        return 0.0
    return amps.t1_01 - res / w

def ccsd_t2_20_aa(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_a.size, ints.eps_vir_a.size,
                    ints.eps_occ_a.size, ints.eps_occ_a.size))

    res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_a, ints.B_vv_a, amps.t1_10_a, amps.t1_10_a)
    res += -1.0 * contract('xac,xbd,cj,di->abij', ints.B_vv_a, ints.B_vv_a, amps.t1_10_a, amps.t1_10_a)
    _vvvv_ladder2(ints.B_vv_a, ints.B_vv_a, amps.t2_20_aa, out=res, alpha=1.0)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ac,bcij->abij', ints.d_a_vv, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci,bj->abij', ints.d_a_vv, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('ac,cj,bi->abij', ints.d_a_vv, amps.t1_10_a, amps.t2_11_a)
    if 't2_21' in active:
        res += -1.0 * contract('ac,bcij->abij', ints.d_a_vv, amps.t2_21_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ai,bj->abij', ints.d_a_vo, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('aj,bi->abij', ints.d_a_vo, amps.t2_11_a)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('bc,acij->abij', ints.d_a_vv, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bc,ci,aj->abij', ints.d_a_vv, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('bc,cj,ai->abij', ints.d_a_vv, amps.t1_10_a, amps.t2_11_a)
    if 't2_21' in active:
        res += 1.0 * contract('bc,acij->abij', ints.d_a_vv, amps.t2_21_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bi,aj->abij', ints.d_a_vo, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('bj,ai->abij', ints.d_a_vo, amps.t2_11_a)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ak,cj,bi->abij', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bk,ci,aj->abij', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,cj,ai->abij', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ik,abjk->abij', ints.d_a_oo, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_a_oo, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('ik,bk,aj->abij', ints.d_a_oo, amps.t1_10_a, amps.t2_11_a)
    if 't2_21' in active:
        res += 1.0 * contract('ik,abjk->abij', ints.d_a_oo, amps.t2_21_aa)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('jk,abik->abij', ints.d_a_oo, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('jk,ak,bi->abij', ints.d_a_oo, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('jk,bk,ai->abij', ints.d_a_oo, amps.t1_10_a, amps.t2_11_a)
    if 't2_21' in active:
        res += -1.0 * contract('jk,abik->abij', ints.d_a_oo, amps.t2_21_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_20_ab)
    res += -1.0 * contract('ac,bcij->abij', ints.f_a_vv, amps.t2_20_aa)
    res += 1.0 * contract('bc,acij->abij', ints.f_a_vv, amps.t2_20_aa)
    res += 1.0 * contract('ck,ak,bcij->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_20_aa)
    res += -1.0 * contract('ck,bk,acij->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_20_aa)
    res += 1.0 * contract('ck,ci,abjk->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_20_aa)
    res += -1.0 * contract('ck,cj,abik->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_20_aa)
    res += 1.0 * contract('ik,abjk->abij', ints.f_a_oo, amps.t2_20_aa)
    res += -1.0 * contract('jk,abik->abij', ints.f_a_oo, amps.t2_20_aa)
    res += -1.0 * contract('ak,bl,ci,dj,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ovov_aa)
    res += -1.0 * contract('ak,bl,ci,jklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ooov_aa)
    res += 1.0 * contract('ak,bl,ci,jlkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ooov_aa)
    res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ovov_aa)
    res += 1.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ooov_aa)
    res += -1.0 * contract('ak,bl,cj,ilkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ooov_aa)
    res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('ak,bl,ikjl->abij', amps.t1_10_a, amps.t1_10_a, ints.v_oooo_aa)
    res += -1.0 * contract('ak,bl,iljk->abij', amps.t1_10_a, amps.t1_10_a, ints.v_oooo_aa)
    res += -1.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ovvv_aa)
    res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('ak,ci,jkbc->abij', amps.t1_10_a, amps.t1_10_a, ints.v_oovv_aa)
    res += -1.0 * contract('ak,ci,jbkc->abij', amps.t1_10_a, amps.t1_10_a, ints.v_ovov_aa)
    res += 1.0 * contract('ak,cj,di,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ovvv_aa)
    res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ak,cj,ikbc->abij', amps.t1_10_a, amps.t1_10_a, ints.v_oovv_aa)
    res += 1.0 * contract('ak,cj,ibkc->abij', amps.t1_10_a, amps.t1_10_a, ints.v_ovov_aa)
    res += -1.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_aa, ints.v_ovov_ab)
    res += 1.0 * contract('ak,bcil,jklc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += -1.0 * contract('ak,bcil,jlkc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += 1.0 * contract('ak,bcjl,ilkc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += 1.0 * contract('ak,bcil,jklc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_ab)
    res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_ab)
    res += -1.0 * contract('ak,ikjb->abij', amps.t1_10_a, ints.v_ooov_aa)
    res += 1.0 * contract('ak,jkib->abij', amps.t1_10_a, ints.v_ooov_aa)
    res += 1.0 * contract('bk,ci,dj,kcad->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ovvv_aa)
    res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('bk,ci,jkac->abij', amps.t1_10_a, amps.t1_10_a, ints.v_oovv_aa)
    res += 1.0 * contract('bk,ci,jakc->abij', amps.t1_10_a, amps.t1_10_a, ints.v_ovov_aa)
    res += -1.0 * contract('bk,cj,di,kcad->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, ints.v_ovvv_aa)
    res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('bk,cj,ikac->abij', amps.t1_10_a, amps.t1_10_a, ints.v_oovv_aa)
    res += -1.0 * contract('bk,cj,iakc->abij', amps.t1_10_a, amps.t1_10_a, ints.v_ovov_aa)
    res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_aa, ints.v_ovov_ab)
    res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += 1.0 * contract('bk,acil,jlkc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += 1.0 * contract('bk,acjl,iklc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += -1.0 * contract('bk,acjl,ilkc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += 1.0 * contract('bk,cdij,kcad->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_ab)
    res += 1.0 * contract('bk,acjl,iklc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_ab)
    res += 1.0 * contract('bk,ikja->abij', amps.t1_10_a, ints.v_ooov_aa)
    res += -1.0 * contract('bk,jkia->abij', amps.t1_10_a, ints.v_ooov_aa)
    res += -1.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('ci,dk,abjl,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_aa, ints.v_ovov_ab)
    res += -1.0 * contract('ci,abkl,jklc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += 1.0 * contract('ci,adjk,kcbd->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += -1.0 * contract('ci,bdjk,kcad->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_ba)
    res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_ba)
    res += -1.0 * contract('ci,jabc->abij', amps.t1_10_a, ints.v_ovvv_aa)
    res += 1.0 * contract('ci,jbac->abij', amps.t1_10_a, ints.v_ovvv_aa)
    res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('cj,dk,abil,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_aa, ints.v_ovov_ab)
    res += 1.0 * contract('cj,abkl,iklc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += -1.0 * contract('cj,adik,kcbd->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += 1.0 * contract('cj,bdik,kcad->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += -1.0 * contract('cj,bdik,kdac->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_ba)
    res += -1.0 * contract('cj,bdik,kdac->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_ba)
    res += 1.0 * contract('cj,iabc->abij', amps.t1_10_a, ints.v_ovvv_aa)
    res += -1.0 * contract('cj,ibac->abij', amps.t1_10_a, ints.v_ovvv_aa)
    res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('ck,abil,jklc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += -1.0 * contract('ck,abjl,iklc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ooov_aa)
    res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += -1.0 * contract('ck,adij,kdbc->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += 1.0 * contract('ck,bdij,kdac->abij', amps.t1_10_a, amps.t2_20_aa, ints.v_ovvv_aa)
    res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_b, amps.t2_20_aa, ints.v_ooov_ab)
    res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_b, amps.t2_20_aa, ints.v_ooov_ab)
    res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_b, amps.t2_20_aa, ints.v_ovvv_ba)
    res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_b, amps.t2_20_aa, ints.v_ovvv_ba)
    res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_aa, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('abjk,cdil,kcld->abij', amps.t2_20_aa, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('abjk,cdil,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_ab)
    res += 0.5 * contract('abkl,cdij,kcld->abij', amps.t2_20_aa, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('abkl,ikjl->abij', amps.t2_20_aa, ints.v_oooo_aa)
    res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_aa, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_aa, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_aa, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('acik,jkbc->abij', amps.t2_20_aa, ints.v_oovv_aa)
    res += 1.0 * contract('acik,jbkc->abij', amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_aa, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('acjk,bdil,kdlc->abij', amps.t2_20_aa, amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('acjk,ikbc->abij', amps.t2_20_aa, ints.v_oovv_aa)
    res += -1.0 * contract('acjk,ibkc->abij', amps.t2_20_aa, ints.v_ovov_aa)
    res += -1.0 * contract('ackl,bdij,kcld->abij', amps.t2_20_aa, amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('bcij,adkl,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('bcik,adjl,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('bcik,jkac->abij', amps.t2_20_aa, ints.v_oovv_aa)
    res += -1.0 * contract('bcik,jakc->abij', amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('bcjk,ikac->abij', amps.t2_20_aa, ints.v_oovv_aa)
    res += 1.0 * contract('bcjk,iakc->abij', amps.t2_20_aa, ints.v_ovov_aa)
    res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_bb)
    res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_bb)
    res += 1.0 * contract('acik,jbkc->abij', amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_bb)
    res += 1.0 * contract('acjk,bdil,kdlc->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_bb)
    res += -1.0 * contract('acjk,ibkc->abij', amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('bcik,jakc->abij', amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('bcjk,iakc->abij', amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('iajb->abij', ints.v_ovov_aa)
    res += -1.0 * contract('ibja->abij', ints.v_ovov_aa)

    # exact antisymmetry of the same-spin residual; enforce against rounding drift
    res = 0.5 * (res - res.transpose(1, 0, 2, 3))
    res = 0.5 * (res - res.transpose(0, 1, 3, 2))
    nocc_j = ints.eps_occ_a.size
    for j in range(nocc_j):
        res[:, :, :, j] *= 1.0 / (ints.eps_occ_a[None, None, :] + ints.eps_occ_a[j]
                                  - ints.eps_vir_a[:, None, None]
                                  - ints.eps_vir_a[None, :, None] - (0.0))
    res += amps.t2_20_aa
    return res

def ccsd_t2_20_ab(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_a.size, ints.eps_vir_b.size,
                    ints.eps_occ_a.size, ints.eps_occ_b.size))

    res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_a, ints.B_vv_b, amps.t1_10_a, amps.t1_10_b)
    _vvvv_ladder2(ints.B_vv_a, ints.B_vv_b, amps.t2_20_ab, out=res, alpha=1.0)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ac,cbij->abij', ints.d_a_vv, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci,bj->abij', ints.d_a_vv, amps.t1_10_a, amps.t2_11_b)
    if 't2_21' in active:
        res += 1.0 * contract('ac,cbij->abij', ints.d_a_vv, amps.t2_21_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ai,bj->abij', ints.d_a_vo, amps.t2_11_b)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_ab)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a, amps.t2_11_b)
    if 't2_21' in active:
        res += -1.0 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ai,cbkj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_20_aa)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ik,abkj->abij', ints.d_a_oo, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_a_oo, amps.t1_10_a, amps.t2_11_b)
    if 't2_21' in active:
        res += -1.0 * contract('ik,abkj->abij', ints.d_a_oo, amps.t2_21_ab)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bc,cj,ai->abij', ints.d_b_vv, amps.t1_10_b, amps.t2_11_a)
    if 't2_21' in active:
        res += 1.0 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_21_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bj,ai->abij', ints.d_b_vo, amps.t2_11_a)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_ab)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,cj,ai->abij', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b, amps.t2_11_a)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('jk,bk,ai->abij', ints.d_b_oo, amps.t1_10_b, amps.t2_11_a)
    if 't2_21' in active:
        res += -1.0 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_21_ab)
    res += 1.0 * contract('ac,cbij->abij', ints.f_a_vv, amps.t2_20_ab)
    res += -1.0 * contract('ck,ak,cbij->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_20_ab)
    res += -1.0 * contract('ck,ci,abkj->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_20_ab)
    res += -1.0 * contract('ik,abkj->abij', ints.f_a_oo, amps.t2_20_ab)
    res += 1.0 * contract('bc,acij->abij', ints.f_b_vv, amps.t2_20_ab)
    res += -1.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_20_ab)
    res += -1.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_20_ab)
    res += -1.0 * contract('jk,abik->abij', ints.f_b_oo, amps.t2_20_ab)
    res += 1.0 * contract('ak,ci,bl,dj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_b, amps.t1_10_b, ints.v_ovov_ab)
    res += 1.0 * contract('ak,ci,bl,jlkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_b, ints.v_ooov_ba)
    res += -1.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_b, ints.v_ovvv_ab)
    res += -1.0 * contract('ak,ci,dblj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_aa)
    res += 1.0 * contract('ak,ci,dblj,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_aa)
    res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_bb, ints.v_ovov_ab)
    res += -1.0 * contract('ak,ci,kcjb->abij', amps.t1_10_a, amps.t1_10_a, ints.v_ovov_ab)
    res += 1.0 * contract('ak,cl,dbij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_aa)
    res += -1.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_aa)
    res += 1.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_a, amps.t1_10_b, amps.t1_10_b, ints.v_ooov_ab)
    res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('ak,bl,ikjl->abij', amps.t1_10_a, amps.t1_10_b, ints.v_oooo_ab)
    res += 1.0 * contract('ak,cj,dbil,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ak,cj,ikbc->abij', amps.t1_10_a, amps.t1_10_b, ints.v_oovv_ab)
    res += -1.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('ak,cbil,jlkc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_ba)
    res += -1.0 * contract('ak,cblj,iklc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_aa)
    res += 1.0 * contract('ak,cblj,ilkc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_aa)
    res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_ab)
    res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_a, amps.t2_20_bb, ints.v_ooov_ab)
    res += -1.0 * contract('ak,ikjb->abij', amps.t1_10_a, ints.v_ooov_ab)
    res += 1.0 * contract('ci,dk,ablj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_aa)
    res += -1.0 * contract('ci,bk,dj,kdac->abij', amps.t1_10_a, amps.t1_10_b, amps.t1_10_b, ints.v_ovvv_ba)
    res += 1.0 * contract('ci,bk,adlj,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ci,bk,jkac->abij', amps.t1_10_a, amps.t1_10_b, ints.v_oovv_ba)
    res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ci,dk,ablj,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('ci,abkl,jlkc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_ba)
    res += -1.0 * contract('ci,adkj,kcbd->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_ab)
    res += -1.0 * contract('ci,dbkj,kcad->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_aa)
    res += 1.0 * contract('ci,dbkj,kdac->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_aa)
    res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_a, amps.t2_20_bb, ints.v_ovvv_ba)
    res += 1.0 * contract('ci,jbac->abij', amps.t1_10_a, ints.v_ovvv_ba)
    res += -1.0 * contract('ck,di,ablj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_20_ab, ints.v_ovov_aa)
    res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_ba)
    res += 1.0 * contract('ck,ablj,iklc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_aa)
    res += -1.0 * contract('ck,ablj,ilkc->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ooov_aa)
    res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_ab)
    res += 1.0 * contract('ck,dbij,kcad->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_aa)
    res += -1.0 * contract('ck,dbij,kdac->abij', amps.t1_10_a, amps.t2_20_ab, ints.v_ovvv_aa)
    res += -1.0 * contract('bk,cj,adil,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_aa, ints.v_ovov_ab)
    res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_bb)
    res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_bb)
    res += -1.0 * contract('bk,cj,iakc->abij', amps.t1_10_b, amps.t1_10_b, ints.v_ovov_ab)
    res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_bb)
    res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_bb)
    res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_b, amps.t2_20_aa, ints.v_ooov_ba)
    res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_bb)
    res += 1.0 * contract('bk,acil,jlkc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_bb)
    res += 1.0 * contract('bk,aclj,ilkc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_ab)
    res += -1.0 * contract('bk,cdij,kdac->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_ba)
    res += -1.0 * contract('bk,jkia->abij', amps.t1_10_b, ints.v_ooov_ba)
    res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_bb)
    res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_b, amps.t2_20_aa, ints.v_ovvv_ab)
    res += 1.0 * contract('cj,abkl,iklc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_ab)
    res += -1.0 * contract('cj,adik,kcbd->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_bb)
    res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_bb)
    res += -1.0 * contract('cj,dbik,kcad->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_ba)
    res += 1.0 * contract('cj,iabc->abij', amps.t1_10_b, ints.v_ovvv_ab)
    res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_bb)
    res += 1.0 * contract('ck,abil,jklc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_bb)
    res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_bb)
    res += -1.0 * contract('ck,ablj,ilkc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_ab)
    res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_bb)
    res += -1.0 * contract('ck,adij,kdbc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_bb)
    res += 1.0 * contract('ck,dbij,kcad->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_ba)
    res += 1.0 * contract('acik,dblj,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_aa)
    res += -1.0 * contract('acik,dblj,kdlc->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_aa)
    res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_aa, amps.t2_20_bb, ints.v_ovov_ab)
    res += 1.0 * contract('acik,kcjb->abij', amps.t2_20_aa, ints.v_ovov_ab)
    res += 1.0 * contract('ackl,dbij,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_aa)
    res += 1.0 * contract('cdik,ablj,kcld->abij', amps.t2_20_aa, amps.t2_20_ab, ints.v_ovov_aa)
    res += -1.0 * contract('abik,cdlj,lckd->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('abkj,cdil,kcld->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('abkl,cdij,kcld->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('abkl,ikjl->abij', amps.t2_20_ab, ints.v_oooo_ab)
    res += -1.0 * contract('acij,dbkl,kdlc->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('acik,dblj,ldkc->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('acik,jkbc->abij', amps.t2_20_ab, ints.v_oovv_bb)
    res += 1.0 * contract('acik,jbkc->abij', amps.t2_20_ab, ints.v_ovov_bb)
    res += 1.0 * contract('ackj,dbil,kdlc->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ackj,ikbc->abij', amps.t2_20_ab, ints.v_oovv_ab)
    res += -1.0 * contract('ackl,dbij,kdlc->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('cbik,jkac->abij', amps.t2_20_ab, ints.v_oovv_ba)
    res += -1.0 * contract('cbkj,ikac->abij', amps.t2_20_ab, ints.v_oovv_aa)
    res += 1.0 * contract('cbkj,iakc->abij', amps.t2_20_ab, ints.v_ovov_aa)
    res += 1.0 * contract('bcjk,iakc->abij', amps.t2_20_bb, ints.v_ovov_ab)
    res += 1.0 * contract('iajb->abij', ints.v_ovov_ab)

    nocc_j = ints.eps_occ_b.size
    for j in range(nocc_j):
        res[:, :, :, j] *= 1.0 / (ints.eps_occ_a[None, None, :] + ints.eps_occ_b[j]
                                  - ints.eps_vir_a[:, None, None]
                                  - ints.eps_vir_b[None, :, None] - (0.0))
    res += amps.t2_20_ab
    return res

def ccsd_t2_20_bb(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_b.size, ints.eps_vir_b.size,
                    ints.eps_occ_b.size, ints.eps_occ_b.size))

    res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_b, ints.B_vv_b, amps.t1_10_b, amps.t1_10_b)
    res += -1.0 * contract('xac,xbd,cj,di->abij', ints.B_vv_b, ints.B_vv_b, amps.t1_10_b, amps.t1_10_b)
    _vvvv_ladder2(ints.B_vv_b, ints.B_vv_b, amps.t2_20_bb, out=res, alpha=1.0)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ai,cbkj->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,aj,cbki->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bi,cakj->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bj,caki->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ac,bcij->abij', ints.d_b_vv, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci,bj->abij', ints.d_b_vv, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('ac,cj,bi->abij', ints.d_b_vv, amps.t1_10_b, amps.t2_11_b)
    if 't2_21' in active:
        res += -1.0 * contract('ac,bcij->abij', ints.d_b_vv, amps.t2_21_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ai,bj->abij', ints.d_b_vo, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('aj,bi->abij', ints.d_b_vo, amps.t2_11_b)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bc,ci,aj->abij', ints.d_b_vv, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('bc,cj,ai->abij', ints.d_b_vv, amps.t1_10_b, amps.t2_11_b)
    if 't2_21' in active:
        res += 1.0 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_21_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bi,aj->abij', ints.d_b_vo, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('bj,ai->abij', ints.d_b_vo, amps.t2_11_b)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci,bj->abij', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ak,cj,bi->abij', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bk,ci,aj->abij', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,cj,ai->abij', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't1_01' in active:
        res += 1.0 * amps.t1_01 * contract('ik,abjk->abij', ints.d_b_oo, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_b_oo, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('ik,bk,aj->abij', ints.d_b_oo, amps.t1_10_b, amps.t2_11_b)
    if 't2_21' in active:
        res += 1.0 * contract('ik,abjk->abij', ints.d_b_oo, amps.t2_21_bb)
    if 't1_01' in active:
        res += -1.0 * amps.t1_01 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('jk,ak,bi->abij', ints.d_b_oo, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('jk,bk,ai->abij', ints.d_b_oo, amps.t1_10_b, amps.t2_11_b)
    if 't2_21' in active:
        res += -1.0 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_21_bb)
    res += -1.0 * contract('ac,bcij->abij', ints.f_b_vv, amps.t2_20_bb)
    res += 1.0 * contract('bc,acij->abij', ints.f_b_vv, amps.t2_20_bb)
    res += 1.0 * contract('ck,ak,bcij->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_20_bb)
    res += -1.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_20_bb)
    res += 1.0 * contract('ck,ci,abjk->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_20_bb)
    res += -1.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_20_bb)
    res += 1.0 * contract('ik,abjk->abij', ints.f_b_oo, amps.t2_20_bb)
    res += -1.0 * contract('jk,abik->abij', ints.f_b_oo, amps.t2_20_bb)
    res += 1.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_ab)
    res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_ab)
    res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_ab)
    res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_ab)
    res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_a, amps.t2_20_bb, ints.v_ooov_ba)
    res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_a, amps.t2_20_bb, ints.v_ooov_ba)
    res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_a, amps.t2_20_bb, ints.v_ovvv_ab)
    res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_a, amps.t2_20_bb, ints.v_ovvv_ab)
    res += -1.0 * contract('ak,bl,ci,dj,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ovov_bb)
    res += -1.0 * contract('ak,bl,ci,jklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ooov_bb)
    res += 1.0 * contract('ak,bl,ci,jlkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ooov_bb)
    res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ovov_bb)
    res += 1.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ooov_bb)
    res += -1.0 * contract('ak,bl,cj,ilkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ooov_bb)
    res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('ak,bl,ikjl->abij', amps.t1_10_b, amps.t1_10_b, ints.v_oooo_bb)
    res += -1.0 * contract('ak,bl,iljk->abij', amps.t1_10_b, amps.t1_10_b, ints.v_oooo_bb)
    res += -1.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ovvv_bb)
    res += -1.0 * contract('ak,ci,dblj,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('ak,ci,jkbc->abij', amps.t1_10_b, amps.t1_10_b, ints.v_oovv_bb)
    res += -1.0 * contract('ak,ci,jbkc->abij', amps.t1_10_b, amps.t1_10_b, ints.v_ovov_bb)
    res += 1.0 * contract('ak,cj,di,kcbd->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ovvv_bb)
    res += 1.0 * contract('ak,cj,dbli,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('ak,cj,ikbc->abij', amps.t1_10_b, amps.t1_10_b, ints.v_oovv_bb)
    res += 1.0 * contract('ak,cj,ibkc->abij', amps.t1_10_b, amps.t1_10_b, ints.v_ovov_bb)
    res += -1.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('ak,cbli,jklc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_ba)
    res += -1.0 * contract('ak,cblj,iklc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_ba)
    res += 1.0 * contract('ak,bcil,jklc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += -1.0 * contract('ak,bcil,jlkc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += 1.0 * contract('ak,bcjl,ilkc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += -1.0 * contract('ak,ikjb->abij', amps.t1_10_b, ints.v_ooov_bb)
    res += 1.0 * contract('ak,jkib->abij', amps.t1_10_b, ints.v_ooov_bb)
    res += 1.0 * contract('bk,ci,dj,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ovvv_bb)
    res += 1.0 * contract('bk,ci,dalj,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('bk,ci,jkac->abij', amps.t1_10_b, amps.t1_10_b, ints.v_oovv_bb)
    res += 1.0 * contract('bk,ci,jakc->abij', amps.t1_10_b, amps.t1_10_b, ints.v_ovov_bb)
    res += -1.0 * contract('bk,cj,di,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, ints.v_ovvv_bb)
    res += -1.0 * contract('bk,cj,dali,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('bk,cj,ikac->abij', amps.t1_10_b, amps.t1_10_b, ints.v_oovv_bb)
    res += -1.0 * contract('bk,cj,iakc->abij', amps.t1_10_b, amps.t1_10_b, ints.v_ovov_bb)
    res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('bk,cali,jklc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_ba)
    res += 1.0 * contract('bk,calj,iklc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ooov_ba)
    res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += 1.0 * contract('bk,acil,jlkc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += 1.0 * contract('bk,acjl,iklc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += -1.0 * contract('bk,acjl,ilkc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += 1.0 * contract('bk,cdij,kcad->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += 1.0 * contract('bk,ikja->abij', amps.t1_10_b, ints.v_ooov_bb)
    res += -1.0 * contract('bk,jkia->abij', amps.t1_10_b, ints.v_ooov_bb)
    res += -1.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('ci,dakj,kdbc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_ab)
    res += 1.0 * contract('ci,dbkj,kdac->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_ab)
    res += -1.0 * contract('ci,abkl,jklc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += 1.0 * contract('ci,adjk,kcbd->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += -1.0 * contract('ci,bdjk,kcad->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += -1.0 * contract('ci,jabc->abij', amps.t1_10_b, ints.v_ovvv_bb)
    res += 1.0 * contract('ci,jbac->abij', amps.t1_10_b, ints.v_ovvv_bb)
    res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('cj,daki,kdbc->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_ab)
    res += -1.0 * contract('cj,dbki,kdac->abij', amps.t1_10_b, amps.t2_20_ab, ints.v_ovvv_ab)
    res += 1.0 * contract('cj,abkl,iklc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += -1.0 * contract('cj,adik,kcbd->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += 1.0 * contract('cj,bdik,kcad->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += -1.0 * contract('cj,bdik,kdac->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += 1.0 * contract('cj,iabc->abij', amps.t1_10_b, ints.v_ovvv_bb)
    res += -1.0 * contract('cj,ibac->abij', amps.t1_10_b, ints.v_ovvv_bb)
    res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('ck,abil,jklc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += -1.0 * contract('ck,abjl,iklc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ooov_bb)
    res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += -1.0 * contract('ck,adij,kdbc->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += 1.0 * contract('ck,bdij,kdac->abij', amps.t1_10_b, amps.t2_20_bb, ints.v_ovvv_bb)
    res += 1.0 * contract('caki,dblj,kcld->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_aa)
    res += 1.0 * contract('caki,bdjl,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_ab)
    res += 1.0 * contract('caki,kcjb->abij', amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('cakj,dbli,kdlc->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_aa)
    res += -1.0 * contract('cakj,bdil,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_ab)
    res += -1.0 * contract('cakj,kcib->abij', amps.t2_20_ab, ints.v_ovov_ab)
    res += 1.0 * contract('cakl,bdij,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_ab)
    res += -1.0 * contract('cbki,dalj,kcld->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_aa)
    res += -1.0 * contract('cbki,adjl,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_ab)
    res += -1.0 * contract('cbki,kcja->abij', amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('cbkj,dali,kdlc->abij', amps.t2_20_ab, amps.t2_20_ab, ints.v_ovov_aa)
    res += 1.0 * contract('cbkj,adil,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_ab)
    res += 1.0 * contract('cbkj,kcia->abij', amps.t2_20_ab, ints.v_ovov_ab)
    res += -1.0 * contract('cbkl,adij,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_ab)
    res += 1.0 * contract('cdki,abjl,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_ab)
    res += -1.0 * contract('cdkj,abil,kcld->abij', amps.t2_20_ab, amps.t2_20_bb, ints.v_ovov_ab)
    res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_bb, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('abjk,cdil,kcld->abij', amps.t2_20_bb, amps.t2_20_bb, ints.v_ovov_bb)
    res += 0.5 * contract('abkl,cdij,kcld->abij', amps.t2_20_bb, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('abkl,ikjl->abij', amps.t2_20_bb, ints.v_oooo_bb)
    res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_bb, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_bb, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_bb, amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('acik,jkbc->abij', amps.t2_20_bb, ints.v_oovv_bb)
    res += 1.0 * contract('acik,jbkc->abij', amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_bb, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('acjk,bdil,kdlc->abij', amps.t2_20_bb, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('acjk,ikbc->abij', amps.t2_20_bb, ints.v_oovv_bb)
    res += -1.0 * contract('acjk,ibkc->abij', amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('ackl,bdij,kcld->abij', amps.t2_20_bb, amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('bcik,jkac->abij', amps.t2_20_bb, ints.v_oovv_bb)
    res += -1.0 * contract('bcik,jakc->abij', amps.t2_20_bb, ints.v_ovov_bb)
    res += -1.0 * contract('bcjk,ikac->abij', amps.t2_20_bb, ints.v_oovv_bb)
    res += 1.0 * contract('bcjk,iakc->abij', amps.t2_20_bb, ints.v_ovov_bb)
    res += 1.0 * contract('iajb->abij', ints.v_ovov_bb)
    res += -1.0 * contract('ibja->abij', ints.v_ovov_bb)

    # exact antisymmetry of the same-spin residual; enforce against rounding drift
    res = 0.5 * (res - res.transpose(1, 0, 2, 3))
    res = 0.5 * (res - res.transpose(0, 1, 3, 2))
    nocc_j = ints.eps_occ_b.size
    for j in range(nocc_j):
        res[:, :, :, j] *= 1.0 / (ints.eps_occ_b[None, None, :] + ints.eps_occ_b[j]
                                  - ints.eps_vir_b[:, None, None]
                                  - ints.eps_vir_b[None, :, None] - (0.0))
    res += amps.t2_20_bb
    return res

def ccsd_t2_02(ints, w, amps, active=_ALL_AMPS):
    res = 0.0

    if 't1_01' in active and 't2_12' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ck->', ints.d_a_vo, amps.t2_12_a)
    if 't2_02' in active and 't2_11' in active:
        res += 2.0 * amps.t2_02 * contract('ck,ck->', ints.d_a_vo, amps.t2_11_a)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ck->', ints.d_a_vo, amps.t2_11_a)
    if 't1_01' in active and 't2_12' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ck->', ints.d_b_vo, amps.t2_12_b)
    if 't2_02' in active and 't2_11' in active:
        res += 2.0 * amps.t2_02 * contract('ck,ck->', ints.d_b_vo, amps.t2_11_b)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ck->', ints.d_b_vo, amps.t2_11_b)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ck->', ints.f_a_vo, amps.t2_12_a)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ck->', ints.f_b_vo, amps.t2_12_b)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dl,kcld->', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,dl,kdlc->', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dl,kcld->', amps.t1_10_a, amps.t2_12_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dl,ldkc->', amps.t1_10_b, amps.t2_12_a, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dl,kcld->', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,dl,kdlc->', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_02' in active:
        res += 2.0 * amps.t2_02 * w
    if 't2_11' in active:
        res += 1.0 * contract('ck,dl,kcld->', amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dl,kdlc->', amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ck,dl,kcld->', amps.t2_11_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dl,kcld->', amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dl,kdlc->', amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 0.5 * contract('cdkl,kcld->', amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cdkl,kcld->', amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 0.5 * contract('cdkl,kcld->', amps.t2_22_bb, ints.v_ovov_bb)

    if w == 0:
        return 0.0
    return amps.t2_02 - res / (2.0 * w)

def ccsd_t2_11_a(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_a.size, ints.eps_occ_a.size))

    if 't1_01' in active and 't2_11' in active:
        res += 1.0 * amps.t1_01 * contract('ac,ci->ai', ints.d_a_vv, amps.t2_11_a)
    res += 1.0 * contract('ac,ci->ai', ints.d_a_vv, amps.t1_10_a)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ac,ci->ai', ints.d_a_vv, amps.t1_10_a)
    if 't2_12' in active:
        res += 1.0 * contract('ac,ci->ai', ints.d_a_vv, amps.t2_12_a)
    res += 1.0 * contract('ai->ai', ints.d_a_vo)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ai->ai', ints.d_a_vo)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ci,ak->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ck,acik->ai', ints.d_a_vo, amps.t2_21_aa)
    res += -1.0 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ck,ai->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_12_a)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ck,acik->ai', ints.d_a_vo, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ai,ck->ai', ints.d_a_vo, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t2_11_a, amps.t2_11_a)
    res += 1.0 * contract('ck,acik->ai', ints.d_a_vo, amps.t2_20_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,acik->ai', ints.d_a_vo, amps.t2_22_aa)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ik,ak->ai', ints.d_a_oo, amps.t2_11_a)
    res += -1.0 * contract('ik,ak->ai', ints.d_a_oo, amps.t1_10_a)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ik,ak->ai', ints.d_a_oo, amps.t1_10_a)
    if 't2_12' in active:
        res += -1.0 * contract('ik,ak->ai', ints.d_a_oo, amps.t2_12_a)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_21_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ck,ai->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_12_a)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ai,ck->ai', ints.d_b_vo, amps.t2_11_a, amps.t2_11_b)
    res += 1.0 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_20_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_22_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci->ai', ints.f_a_vv, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.f_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.f_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't2_21' in active:
        res += 1.0 * contract('ck,acik->ai', ints.f_a_vo, amps.t2_21_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak->ai', ints.f_a_oo, amps.t2_11_a)
    if 't2_21' in active:
        res += 1.0 * contract('ck,acik->ai', ints.f_b_vo, amps.t2_21_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,dl,kdlc->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,di,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,di,kdlc->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,di,kdlc->ai', amps.t1_10_a, amps.t1_10_b, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,ilkc->ai', amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_a, amps.t2_11_b, ints.v_ooov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,al,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dk,al,lckd->ai', amps.t1_10_a, amps.t1_10_b, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dk,kcad->ai', amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,kdac->ai', amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,kdac->ai', amps.t1_10_a, amps.t2_11_b, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,di,al,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,al,iklc->ai', amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,al,ilkc->ai', amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,di,kdac->ai', amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,al,ilkc->ai', amps.t1_10_b, amps.t2_11_a, ints.v_ooov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_b, amps.t2_11_a, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adil,ldkc->ai', amps.t1_10_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * w * contract('ai->ai', amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ikac->ai', amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,iakc->ai', amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adil,ldkc->ai', amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,iakc->ai', amps.t2_11_b, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ackl,iklc->ai', amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cdik,kcad->ai', amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ackl,iklc->ai', amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cdik,kdac->ai', amps.t2_21_ab, ints.v_ovvv_ba)

    e_denom = 1.0 / (ints.eps_occ_a[None, :] - ints.eps_vir_a[:, None] - (w))
    return amps.t2_11_a + res * e_denom

def ccsd_t2_11_b(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_b.size, ints.eps_occ_b.size))

    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ck,caki->ai', ints.d_a_vo, amps.t2_21_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ck,ai->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_12_b)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ck,caki->ai', ints.d_a_vo, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ck,ai->ai', ints.d_a_vo, amps.t2_11_a, amps.t2_11_b)
    res += 1.0 * contract('ck,caki->ai', ints.d_a_vo, amps.t2_20_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,caki->ai', ints.d_a_vo, amps.t2_22_ab)
    if 't1_01' in active and 't2_11' in active:
        res += 1.0 * amps.t1_01 * contract('ac,ci->ai', ints.d_b_vv, amps.t2_11_b)
    res += 1.0 * contract('ac,ci->ai', ints.d_b_vv, amps.t1_10_b)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ac,ci->ai', ints.d_b_vv, amps.t1_10_b)
    if 't2_12' in active:
        res += 1.0 * contract('ac,ci->ai', ints.d_b_vv, amps.t2_12_b)
    res += 1.0 * contract('ai->ai', ints.d_b_vo)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ai->ai', ints.d_b_vo)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ci,ak->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_21_bb)
    res += -1.0 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ck,ai->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_12_b)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ai,ck->ai', ints.d_b_vo, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t2_11_b, amps.t2_11_b)
    res += 1.0 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_20_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_22_bb)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ik,ak->ai', ints.d_b_oo, amps.t2_11_b)
    res += -1.0 * contract('ik,ak->ai', ints.d_b_oo, amps.t1_10_b)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ik,ak->ai', ints.d_b_oo, amps.t1_10_b)
    if 't2_12' in active:
        res += -1.0 * contract('ik,ak->ai', ints.d_b_oo, amps.t2_12_b)
    if 't2_21' in active:
        res += 1.0 * contract('ck,caki->ai', ints.f_a_vo, amps.t2_21_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci->ai', ints.f_b_vv, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.f_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.f_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't2_21' in active:
        res += 1.0 * contract('ck,acik->ai', ints.f_b_vo, amps.t2_21_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak->ai', ints.f_b_oo, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('ck,al,di,kcld->ai', amps.t1_10_a, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,di,al,kcld->ai', amps.t1_10_a, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,al,ilkc->ai', amps.t1_10_a, amps.t2_11_b, ints.v_ooov_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_a, amps.t2_11_b, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ck,dali,kcld->ai', amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,dali,kdlc->ai', amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_a, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,dl,ldkc->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,dl,kdlc->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,di,kcld->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,di,kdlc->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_b, amps.t2_11_a, ints.v_ooov_ba)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,ilkc->ai', amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cdli,lckd->ai', amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,al,kcld->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,kdac->ai', amps.t1_10_b, amps.t2_11_a, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dk,kcad->ai', amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,kdac->ai', amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ci,dakl,kdlc->ai', amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,di,al,kcld->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,al,iklc->ai', amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,al,ilkc->ai', amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,di,kdac->ai', amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,dali,ldkc->ai', amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dali,kcld->ai', amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dali,kdlc->ai', amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t2_11_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,kcia->ai', amps.t2_11_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * w * contract('ai->ai', amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cdli,lckd->ai', amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dakl,kdlc->ai', amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dali,ldkc->ai', amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ikac->ai', amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,iakc->ai', amps.t2_11_b, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('cakl,ilkc->ai', amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_21' in active:
        res += 1.0 * contract('cdki,kcad->ai', amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ackl,iklc->ai', amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('cdik,kcad->ai', amps.t2_21_bb, ints.v_ovvv_bb)

    e_denom = 1.0 / (ints.eps_occ_b[None, :] - ints.eps_vir_b[:, None] - (w))
    return amps.t2_11_b + res * e_denom

def ccsd_t2_21_aa(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_a.size, ints.eps_vir_a.size,
                    ints.eps_occ_a.size, ints.eps_occ_a.size))

    if 't2_11' in active:
        res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_a, ints.B_vv_a, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('xac,xbd,cj,di->abij', ints.B_vv_a, ints.B_vv_a, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('xac,xbd,di,cj->abij', ints.B_vv_a, ints.B_vv_a, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('xac,xbd,dj,ci->abij', ints.B_vv_a, ints.B_vv_a, amps.t1_10_a, amps.t2_11_a)
    if 't2_21' in active:
        _vvvv_ladder2(ints.B_vv_a, ints.B_vv_a, amps.t2_21_aa, out=res, alpha=1.0)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ac,bcij->abij', ints.d_a_vv, amps.t2_21_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ac,ci,bj->abij', ints.d_a_vv, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += -1.0 * contract('ac,cj,bi->abij', ints.d_a_vv, amps.t1_10_a, amps.t2_12_a)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ac,bcij->abij', ints.d_a_vv, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ac,bi,cj->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('ac,bj,ci->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_11_a)
    res += -1.0 * contract('ac,bcij->abij', ints.d_a_vv, amps.t2_20_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ac,bcij->abij', ints.d_a_vv, amps.t2_22_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ai,bj->abij', ints.d_a_vo, amps.t2_12_a)
    if 't2_12' in active:
        res += -1.0 * contract('aj,bi->abij', ints.d_a_vo, amps.t2_12_a)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('bc,acij->abij', ints.d_a_vv, amps.t2_21_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bc,ci,aj->abij', ints.d_a_vv, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += 1.0 * contract('bc,cj,ai->abij', ints.d_a_vv, amps.t1_10_a, amps.t2_12_a)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('bc,acij->abij', ints.d_a_vv, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bc,ai,cj->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('bc,aj,ci->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_11_a)
    res += 1.0 * contract('bc,acij->abij', ints.d_a_vv, amps.t2_20_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bc,acij->abij', ints.d_a_vv, amps.t2_22_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bi,aj->abij', ints.d_a_vo, amps.t2_12_a)
    if 't2_12' in active:
        res += 1.0 * contract('bj,ai->abij', ints.d_a_vo, amps.t2_12_a)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't1_01' in active and 't2_11' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't1_01' in active and 't2_11' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,ci,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ak,cj,bi->abij', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ak,bi,cj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,bj,ci->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a)
    res += 1.0 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bk,ci,aj->abij', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,cj,ai->abij', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,ai,cj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bk,aj,ci->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a)
    res += -1.0 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ci,aj,bk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,ak,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a)
    res += 1.0 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,ai,bk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('ck,cj,ak,bi->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a)
    res += -1.0 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ik,abjk->abij', ints.d_a_oo, amps.t2_21_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_a_oo, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += 1.0 * contract('ik,bk,aj->abij', ints.d_a_oo, amps.t1_10_a, amps.t2_12_a)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ik,abjk->abij', ints.d_a_oo, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ik,aj,bk->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_11_a)
    res += 1.0 * contract('ik,abjk->abij', ints.d_a_oo, amps.t2_20_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ik,abjk->abij', ints.d_a_oo, amps.t2_22_aa)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('jk,abik->abij', ints.d_a_oo, amps.t2_21_aa)
    if 't2_12' in active:
        res += 1.0 * contract('jk,ak,bi->abij', ints.d_a_oo, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += -1.0 * contract('jk,bk,ai->abij', ints.d_a_oo, amps.t1_10_a, amps.t2_12_a)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('jk,abik->abij', ints.d_a_oo, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('jk,ai,bk->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += 1.0 * contract('jk,ak,bi->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_11_a)
    res += -1.0 * contract('jk,abik->abij', ints.d_a_oo, amps.t2_20_aa)
    if 't2_22' in active:
        res += -1.0 * contract('jk,abik->abij', ints.d_a_oo, amps.t2_22_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_b_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_b_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ac,bcij->abij', ints.f_a_vv, amps.t2_21_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bc,acij->abij', ints.f_a_vv, amps.t2_21_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ik,abjk->abij', ints.f_a_oo, amps.t2_21_aa)
    if 't2_21' in active:
        res += -1.0 * contract('jk,abik->abij', ints.f_a_oo, amps.t2_21_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ci,dj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bl,ci,dj,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bl,cj,di,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bl,ci,jklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ci,jlkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bl,cj,ilkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,dj,bl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,bl,jklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,bl,jlkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,dj,kdbc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,di,bl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,bl,iklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,bl,ilkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,di,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,di,kdbc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ikjl->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oooo_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bl,iljk->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oooo_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,jkbc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,jbkc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,ikbc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,ibkc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ak,bcil,jlkc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,dj,al,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,al,jklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,al,jlkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,dj,kcad->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,dj,kdac->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,di,al,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,al,iklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,al,ilkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,di,kcad->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,di,kdac->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('bk,al,cdij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,al,ikjl->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oooo_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,al,iljk->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oooo_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,jkac->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,jakc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,ikac->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,iakc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bk,acjl,ilkc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cdij,kcad->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dj,ak,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,bk,kcad->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ci,dk,abjl,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,ak,bdjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ci,ak,jkbc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,ak,jbkc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bk,adjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bk,adjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bk,adjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bk,jkac->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bk,jakc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,abjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,abjl,lckd->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ci,abkl,jklc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ci,adjk,kcbd->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += 1.0 * contract('cj,di,ak,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,di,bk,kcad->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cj,dk,abil,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('cj,ak,bdil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,ak,bdil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('cj,ak,bdil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('cj,ak,ikbc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('cj,ak,ibkc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('cj,bk,adil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('cj,bk,ikac->abij', amps.t1_10_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bk,iakc->abij', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,dk,abil,lckd->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cj,bdik,kcad->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,al,bdij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bl,adij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,di,abjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dj,abil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,abjl,iklc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,bdij,kdac->abij', amps.t1_10_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,al,bdij,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bl,adij,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,di,abjl,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dj,abil,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_b, amps.t2_21_aa, ints.v_ooov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_b, amps.t2_21_aa, ints.v_ooov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_b, amps.t2_21_aa, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_b, amps.t2_21_aa, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bcil,jlkc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ikjb->abij', amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,jkib->abij', amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,acjl,ilkc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cdij,kcad->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ikja->abij', amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,jkia->abij', amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,abkl,jklc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,adjk,kcbd->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += -1.0 * contract('ci,jabc->abij', amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,jbac->abij', amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('cj,bdik,kcad->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += 1.0 * contract('cj,iabc->abij', amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('cj,ibac->abij', amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,abjl,iklc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bdij,kdac->abij', amps.t2_11_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_11_b, amps.t2_20_aa, ints.v_ooov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t2_11_b, amps.t2_20_aa, ints.v_ooov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_11_b, amps.t2_20_aa, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t2_11_b, amps.t2_20_aa, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('abjk,cdil,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('abjk,cdil,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 0.5 * contract('abkl,cdij,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('acjk,bdil,kdlc->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ackl,bdij,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bcij,adkl,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bcij,adkl,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('bcik,adjl,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bcik,adjl,kdlc->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bcik,adjl,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bcjk,adil,kdlc->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bckl,adij,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 0.5 * contract('cdij,abkl,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cdik,abjl,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cdjk,abil,kcld->abij', amps.t2_20_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('acik,bdjl,ldkc->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('acjk,bdil,ldkc->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('acjk,bdil,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ackl,bdij,kdlc->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('bcik,adjl,ldkc->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('bcik,adjl,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bcik,adjl,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,adil,ldkc->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bcjk,adil,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bckl,adij,kdlc->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cdik,abjl,lckd->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cdjk,abil,lckd->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * w * contract('abij->abij', amps.t2_21_aa)
    if 't2_21' in active:
        res += 1.0 * contract('abkl,ikjl->abij', amps.t2_21_aa, ints.v_oooo_aa)
    if 't2_21' in active:
        res += -1.0 * contract('acik,jkbc->abij', amps.t2_21_aa, ints.v_oovv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('acik,jbkc->abij', amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('acjk,ikbc->abij', amps.t2_21_aa, ints.v_oovv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('acjk,ibkc->abij', amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bcik,jkac->abij', amps.t2_21_aa, ints.v_oovv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bcik,jakc->abij', amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('bcjk,ikac->abij', amps.t2_21_aa, ints.v_oovv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,iakc->abij', amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('acik,jbkc->abij', amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('acjk,ibkc->abij', amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('bcik,jakc->abij', amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,iakc->abij', amps.t2_21_ab, ints.v_ovov_ab)

    # exact antisymmetry of the same-spin residual; enforce against rounding drift
    res = 0.5 * (res - res.transpose(1, 0, 2, 3))
    res = 0.5 * (res - res.transpose(0, 1, 3, 2))
    nocc_j = ints.eps_occ_a.size
    for j in range(nocc_j):
        res[:, :, :, j] *= 1.0 / (ints.eps_occ_a[None, None, :] + ints.eps_occ_a[j]
                                  - ints.eps_vir_a[:, None, None]
                                  - ints.eps_vir_a[None, :, None] - (w))
    res += amps.t2_21_aa
    return res

def ccsd_t2_21_ab(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_a.size, ints.eps_vir_b.size,
                    ints.eps_occ_a.size, ints.eps_occ_b.size))

    if 't2_11' in active:
        res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_a, ints.B_vv_b, amps.t1_10_a, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('xac,xbd,dj,ci->abij', ints.B_vv_a, ints.B_vv_b, amps.t1_10_b, amps.t2_11_a)
    if 't2_21' in active:
        _vvvv_ladder2(ints.B_vv_a, ints.B_vv_b, amps.t2_21_ab, out=res, alpha=1.0)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ac,cbij->abij', ints.d_a_vv, amps.t2_21_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ac,ci,bj->abij', ints.d_a_vv, amps.t1_10_a, amps.t2_12_b)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ac,cbij->abij', ints.d_a_vv, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci,bj->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_11_b)
    res += 1.0 * contract('ac,cbij->abij', ints.d_a_vv, amps.t2_20_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ac,cbij->abij', ints.d_a_vv, amps.t2_22_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ai,bj->abij', ints.d_a_vo, amps.t2_12_b)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,ci,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t1_10_a, amps.t2_12_b)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_11_b)
    res += -1.0 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_ab)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,ak,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_11_b)
    res += -1.0 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_20_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ai,cbkj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_21_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ai,cbkj->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_a_vo, amps.t2_12_b, amps.t2_20_aa)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ik,abkj->abij', ints.d_a_oo, amps.t2_21_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_a_oo, amps.t1_10_a, amps.t2_12_b)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ik,abkj->abij', ints.d_a_oo, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_11_b)
    res += -1.0 * contract('ik,abkj->abij', ints.d_a_oo, amps.t2_20_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ik,abkj->abij', ints.d_a_oo, amps.t2_22_ab)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_21_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bc,cj,ai->abij', ints.d_b_vv, amps.t1_10_b, amps.t2_12_a)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_20_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bc,ai,cj->abij', ints.d_b_vv, amps.t2_11_a, amps.t2_11_b)
    res += 1.0 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_20_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_22_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bj,ai->abij', ints.d_b_vo, amps.t2_12_a)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,cj,ai->abij', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b, amps.t2_12_a)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,ai,cj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_a, amps.t2_11_b)
    res += -1.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_ab)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,ai,bk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_a, amps.t2_11_b)
    res += -1.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_12_a, amps.t2_20_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_21_ab)
    if 't2_12' in active:
        res += -1.0 * contract('jk,bk,ai->abij', ints.d_b_oo, amps.t1_10_b, amps.t2_12_a)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('jk,ai,bk->abij', ints.d_b_oo, amps.t2_11_a, amps.t2_11_b)
    res += -1.0 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_20_ab)
    if 't2_22' in active:
        res += -1.0 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_22_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ac,cbij->abij', ints.f_a_vv, amps.t2_21_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,ak,cbij->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,cbij->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ik,abkj->abij', ints.f_a_oo, amps.t2_21_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bc,acij->abij', ints.f_b_vv, amps.t2_21_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_21' in active:
        res += -1.0 * contract('jk,abik->abij', ints.f_b_oo, amps.t2_21_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,bl,dj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,dj,bl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,bl,jlkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_b, ints.v_ooov_ba)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_b, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ak,ci,dblj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ak,ci,dblj,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cl,dbij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t1_10_b, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ci,jlkc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_a, ints.v_ooov_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,di,kdbc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_a, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,bl,iklc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cj,dbil,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,dblj,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,dblj,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,kcjb->abij', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,dbij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ikjl->abij', amps.t1_10_a, amps.t2_11_b, ints.v_oooo_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,dbil,kdlc->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,ikbc->abij', amps.t1_10_a, amps.t2_11_b, ints.v_oovv_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cbil,jlkc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cblj,iklc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cblj,ilkc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_a, amps.t2_21_bb, ints.v_ooov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ci,dk,ablj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bk,dj,al,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t1_10_b, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bk,al,jklc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_a, ints.v_ooov_ba)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bk,dj,kdac->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += 1.0 * contract('ci,bk,adlj,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dj,ak,kcbd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_a, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dj,bk,kdac->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ci,dk,ablj,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,ak,dblj,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,ak,dblj,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,ak,kcjb->abij', amps.t1_10_a, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,ablj,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dk,ablj,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bk,adlj,lckd->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bk,jkac->abij', amps.t1_10_a, amps.t2_11_b, ints.v_oovv_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dk,ablj,lckd->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ci,abkl,jlkc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_21' in active:
        res += -1.0 * contract('ci,adkj,kcbd->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ci,dbkj,kcad->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ci,dbkj,kdac->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_a, amps.t2_21_bb, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += -1.0 * contract('ck,di,ablj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,al,dbij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,al,dbij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,di,ablj,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,di,ablj,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ablj,iklc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,ablj,ilkc->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ooov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ck,dbij,kcad->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_aa)
    if 't2_21' in active:
        res += -1.0 * contract('ck,dbij,kdac->abij', amps.t1_10_a, amps.t2_21_ab, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,al,ilkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_a, ints.v_ooov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,di,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_a, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cj,adil,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,al,cdij,lckd->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bk,al,iljk->abij', amps.t1_10_b, amps.t2_11_a, ints.v_oooo_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,adlj,lckd->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,jkac->abij', amps.t1_10_b, amps.t2_11_a, ints.v_oovv_ba)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cl,adij,lckd->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,adil,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,iakc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_b, amps.t2_21_aa, ints.v_ooov_ba)
    if 't2_21' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bk,aclj,ilkc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cdij,kdac->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,ak,dbil,kdlc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('cj,ak,ikbc->abij', amps.t1_10_b, amps.t2_11_a, ints.v_oovv_ab)
    if 't2_11' in active:
        res += 1.0 * contract('cj,di,abkl,kdlc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bk,adil,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,bk,adil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bk,iakc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_b, amps.t2_21_aa, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('cj,dbik,kcad->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,al,dbij,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,di,ablj,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bl,adij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dj,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,ablj,ilkc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,dbij,kcad->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cbil,jlkc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cblj,iklc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cblj,ilkc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t2_11_a, amps.t2_20_bb, ints.v_ooov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ikjb->abij', amps.t2_11_a, ints.v_ooov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ci,abkl,jlkc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_11' in active:
        res += -1.0 * contract('ci,adkj,kcbd->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dbkj,kcad->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dbkj,kdac->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t2_11_a, amps.t2_20_bb, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ci,jbac->abij', amps.t2_11_a, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ablj,iklc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ablj,ilkc->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dbij,kcad->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dbij,kdac->abij', amps.t2_11_a, amps.t2_20_ab, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t2_11_b, amps.t2_20_aa, ints.v_ooov_ba)
    if 't2_11' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,aclj,ilkc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cdij,kdac->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += -1.0 * contract('bk,jkia->abij', amps.t2_11_b, ints.v_ooov_ba)
    if 't2_11' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t2_11_b, amps.t2_20_aa, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,dbik,kcad->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += 1.0 * contract('cj,iabc->abij', amps.t2_11_b, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ablj,ilkc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dbij,kcad->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_21' in active:
        res += 1.0 * contract('acik,dblj,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('acik,dblj,kdlc->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_aa, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ackl,dbij,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cdik,ablj,kcld->abij', amps.t2_20_aa, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('abik,cdlj,lckd->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('abkj,cdil,kcld->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('abkj,cdil,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('abkl,cdij,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('acij,dbkl,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('acik,dblj,ldkc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ackj,dbil,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ackl,dbij,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cbij,adkl,kcld->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cbij,adkl,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cbik,adlj,lckd->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cbkj,adil,kcld->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cbkj,adil,kdlc->abij', amps.t2_20_ab, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cbkj,adil,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cbkl,adij,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cdij,abkl,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cdik,ablj,lckd->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cdkj,abil,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,adil,ldkc->abij', amps.t2_20_bb, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bcjk,adil,kdlc->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bckl,adij,kcld->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('cdjk,abil,kcld->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('acik,kcjb->abij', amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * w * contract('abij->abij', amps.t2_21_ab)
    if 't2_21' in active:
        res += 1.0 * contract('abkl,ikjl->abij', amps.t2_21_ab, ints.v_oooo_ab)
    if 't2_21' in active:
        res += -1.0 * contract('acik,jkbc->abij', amps.t2_21_ab, ints.v_oovv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('acik,jbkc->abij', amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ackj,ikbc->abij', amps.t2_21_ab, ints.v_oovv_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cbik,jkac->abij', amps.t2_21_ab, ints.v_oovv_ba)
    if 't2_21' in active:
        res += -1.0 * contract('cbkj,ikac->abij', amps.t2_21_ab, ints.v_oovv_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cbkj,iakc->abij', amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,iakc->abij', amps.t2_21_bb, ints.v_ovov_ab)

    nocc_j = ints.eps_occ_b.size
    for j in range(nocc_j):
        res[:, :, :, j] *= 1.0 / (ints.eps_occ_a[None, None, :] + ints.eps_occ_b[j]
                                  - ints.eps_vir_a[:, None, None]
                                  - ints.eps_vir_b[None, :, None] - (w))
    res += amps.t2_21_ab
    return res

def ccsd_t2_21_bb(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_b.size, ints.eps_vir_b.size,
                    ints.eps_occ_b.size, ints.eps_occ_b.size))

    if 't2_11' in active:
        res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_b, ints.B_vv_b, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('xac,xbd,cj,di->abij', ints.B_vv_b, ints.B_vv_b, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('xac,xbd,di,cj->abij', ints.B_vv_b, ints.B_vv_b, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('xac,xbd,dj,ci->abij', ints.B_vv_b, ints.B_vv_b, amps.t1_10_b, amps.t2_11_b)
    if 't2_21' in active:
        _vvvv_ladder2(ints.B_vv_b, ints.B_vv_b, amps.t2_21_bb, out=res, alpha=1.0)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ai,cbkj->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -1.0 * contract('ck,aj,cbki->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -1.0 * contract('ck,bi,cakj->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,bj,caki->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ai,cbkj->abij', ints.d_a_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,aj,cbki->abij', ints.d_a_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bi,cakj->abij', ints.d_a_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bj,caki->abij', ints.d_a_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ac,bcij->abij', ints.d_b_vv, amps.t2_21_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ac,ci,bj->abij', ints.d_b_vv, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('ac,cj,bi->abij', ints.d_b_vv, amps.t1_10_b, amps.t2_12_b)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ac,bcij->abij', ints.d_b_vv, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ac,bi,cj->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('ac,bj,ci->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_11_b)
    res += -1.0 * contract('ac,bcij->abij', ints.d_b_vv, amps.t2_20_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ac,bcij->abij', ints.d_b_vv, amps.t2_22_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ai,bj->abij', ints.d_b_vo, amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('aj,bi->abij', ints.d_b_vo, amps.t2_12_b)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_21_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bc,ci,aj->abij', ints.d_b_vv, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += 1.0 * contract('bc,cj,ai->abij', ints.d_b_vv, amps.t1_10_b, amps.t2_12_b)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bc,ai,cj->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('bc,aj,ci->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_11_b)
    res += 1.0 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_20_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_22_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bi,aj->abij', ints.d_b_vo, amps.t2_12_b)
    if 't2_12' in active:
        res += 1.0 * contract('bj,ai->abij', ints.d_b_vo, amps.t2_12_b)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't1_01' in active and 't2_11' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't1_01' in active and 't2_11' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,ci,bj->abij', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ak,cj,bi->abij', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ak,bi,cj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,bj,ci->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b)
    res += 1.0 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bk,ci,aj->abij', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,cj,ai->abij', ints.d_b_vo, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,ai,cj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bk,aj,ci->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b)
    res += -1.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ci,aj,bk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,ak,bj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b)
    res += 1.0 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,ai,bk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('ck,cj,ak,bi->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b)
    res += -1.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_20_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * amps.t1_01 * contract('ik,abjk->abij', ints.d_b_oo, amps.t2_21_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_b_oo, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += 1.0 * contract('ik,bk,aj->abij', ints.d_b_oo, amps.t1_10_b, amps.t2_12_b)
    if 't2_02' in active:
        res += 1.0 * amps.t2_02 * contract('ik,abjk->abij', ints.d_b_oo, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ik,aj,bk->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_11_b)
    res += 1.0 * contract('ik,abjk->abij', ints.d_b_oo, amps.t2_20_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ik,abjk->abij', ints.d_b_oo, amps.t2_22_bb)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * amps.t1_01 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_21_bb)
    if 't2_12' in active:
        res += 1.0 * contract('jk,ak,bi->abij', ints.d_b_oo, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('jk,bk,ai->abij', ints.d_b_oo, amps.t1_10_b, amps.t2_12_b)
    if 't2_02' in active:
        res += -1.0 * amps.t2_02 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('jk,ai,bk->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += 1.0 * contract('jk,ak,bi->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_11_b)
    res += -1.0 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_20_bb)
    if 't2_22' in active:
        res += -1.0 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_22_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ac,bcij->abij', ints.f_b_vv, amps.t2_21_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bc,acij->abij', ints.f_b_vv, amps.t2_21_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ik,abjk->abij', ints.f_b_oo, amps.t2_21_bb)
    if 't2_21' in active:
        res += -1.0 * contract('jk,abik->abij', ints.f_b_oo, amps.t2_21_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_a, amps.t2_21_bb, ints.v_ooov_ba)
    if 't2_21' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_a, amps.t2_21_bb, ints.v_ooov_ba)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_a, amps.t2_21_bb, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_a, amps.t2_21_bb, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ci,dj,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bl,ci,dj,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bl,cj,di,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bl,ci,jklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ci,jlkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bl,cj,ilkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,dj,bl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,bl,jklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,bl,jlkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,dj,kdbc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ak,ci,dblj,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,di,bl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,bl,iklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,bl,ilkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,di,kcbd->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,di,kdbc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cj,dbli,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,bdij,lckd->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ikjl->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oooo_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bl,iljk->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oooo_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,dblj,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,jkbc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,jbkc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,dbli,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,ikbc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,ibkc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cbli,jklc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cblj,iklc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ak,bcil,jlkc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,dj,al,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,al,jklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,al,jlkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,dj,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,dj,kdac->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bk,ci,dalj,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,di,al,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,al,iklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,al,ilkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,di,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,di,kdac->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cj,dali,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cl,adij,lckd->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('bk,al,cdij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,al,ikjl->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oooo_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,al,iljk->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oooo_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,dalj,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,jkac->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,jakc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,dali,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,ikac->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,iakc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cali,jklc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_21' in active:
        res += 1.0 * contract('bk,calj,iklc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_21' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bk,acjl,ilkc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cdij,kcad->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dj,ak,kcbd->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,bk,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,abjl,kdlc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,ak,dblj,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,ak,bdjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,ak,jkbc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,ak,jbkc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bk,dalj,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bk,adjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bk,adjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bk,jkac->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bk,jakc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,abjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ci,dakj,kdbc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += 1.0 * contract('ci,dbkj,kdac->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += -1.0 * contract('ci,abkl,jklc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ci,adjk,kcbd->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,di,ak,kcbd->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,di,bk,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('cj,ak,dbli,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('cj,ak,bdil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,ak,bdil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,ak,ikbc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,ak,ibkc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bk,dali,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,bk,adil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,bk,ikac->abij', amps.t1_10_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bk,iakc->abij', amps.t1_10_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('cj,daki,kdbc->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cj,dbki,kdac->abij', amps.t1_10_b, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('cj,bdik,kcad->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,al,bdij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bl,adij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,di,abjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dj,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,abjl,iklc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('ck,bdij,kdac->abij', amps.t1_10_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_11_a, amps.t2_20_bb, ints.v_ooov_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t2_11_a, amps.t2_20_bb, ints.v_ooov_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_11_a, amps.t2_20_bb, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t2_11_a, amps.t2_20_bb, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cbli,jklc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cblj,iklc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bcil,jlkc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ikjb->abij', amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ak,jkib->abij', amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cali,jklc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_11' in active:
        res += 1.0 * contract('bk,calj,iklc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_11' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,acjl,ilkc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cdij,kcad->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ikja->abij', amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('bk,jkia->abij', amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dakj,kdbc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dbkj,kdac->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += -1.0 * contract('ci,abkl,jklc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,adjk,kcbd->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,jabc->abij', amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,jbac->abij', amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,daki,kdbc->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += -1.0 * contract('cj,dbki,kdac->abij', amps.t2_11_b, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,bdik,kcad->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,iabc->abij', amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,ibac->abij', amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,abjl,iklc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bdij,kdac->abij', amps.t2_11_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('caki,dblj,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('caki,dblj,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('caki,bdjl,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cakj,dbli,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cakj,dbli,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cakj,bdil,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cakl,bdij,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cbki,dalj,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cbki,dalj,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cbki,adjl,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cbkj,dali,kcld->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cbkj,dali,kdlc->abij', amps.t2_20_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cbkj,adil,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cbkl,adij,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cdki,abjl,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cdkj,abil,kcld->abij', amps.t2_20_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('abik,cdlj,lckd->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('abjk,cdli,lckd->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('abjk,cdil,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 0.5 * contract('abkl,cdij,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('acij,dbkl,kdlc->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('acik,dblj,ldkc->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('acjk,dbli,ldkc->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('acjk,bdil,kdlc->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('ackl,bdij,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bcij,dakl,kdlc->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bcij,adkl,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bcik,dalj,ldkc->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('bcik,adjl,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bcik,adjl,kdlc->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,dali,ldkc->abij', amps.t2_20_bb, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bcjk,adil,kdlc->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bckl,adij,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 0.5 * contract('cdij,abkl,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('cdik,abjl,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('cdjk,abil,kcld->abij', amps.t2_20_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('caki,kcjb->abij', amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cakj,kcib->abij', amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cbki,kcja->abij', amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cbkj,kcia->abij', amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * w * contract('abij->abij', amps.t2_21_bb)
    if 't2_21' in active:
        res += 1.0 * contract('abkl,ikjl->abij', amps.t2_21_bb, ints.v_oooo_bb)
    if 't2_21' in active:
        res += -1.0 * contract('acik,jkbc->abij', amps.t2_21_bb, ints.v_oovv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('acik,jbkc->abij', amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('acjk,ikbc->abij', amps.t2_21_bb, ints.v_oovv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('acjk,ibkc->abij', amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bcik,jkac->abij', amps.t2_21_bb, ints.v_oovv_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bcik,jakc->abij', amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -1.0 * contract('bcjk,ikac->abij', amps.t2_21_bb, ints.v_oovv_bb)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,iakc->abij', amps.t2_21_bb, ints.v_ovov_bb)

    # exact antisymmetry of the same-spin residual; enforce against rounding drift
    res = 0.5 * (res - res.transpose(1, 0, 2, 3))
    res = 0.5 * (res - res.transpose(0, 1, 3, 2))
    nocc_j = ints.eps_occ_b.size
    for j in range(nocc_j):
        res[:, :, :, j] *= 1.0 / (ints.eps_occ_b[None, None, :] + ints.eps_occ_b[j]
                                  - ints.eps_vir_b[:, None, None]
                                  - ints.eps_vir_b[None, :, None] - (w))
    res += amps.t2_21_bb
    return res

def ccsd_t2_12_a(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_a.size, ints.eps_occ_a.size))

    if 't1_01' in active and 't2_12' in active:
        res += 1.0 * amps.t1_01 * contract('ac,ci->ai', ints.d_a_vv, amps.t2_12_a)
    if 't2_02' in active and 't2_11' in active:
        res += 2.0 * amps.t2_02 * contract('ac,ci->ai', ints.d_a_vv, amps.t2_11_a)
    if 't2_11' in active:
        res += 2.0 * contract('ac,ci->ai', ints.d_a_vv, amps.t2_11_a)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_12_a)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ci,ak->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_12_a)
    if 't1_01' in active and 't2_11' in active:
        res += -2.0 * amps.t1_01 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t2_11_a, amps.t2_11_a)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ck,acik->ai', ints.d_a_vo, amps.t2_22_aa)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,ci,ak->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ci,ak->ai', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ck,acik->ai', ints.d_a_vo, amps.t2_21_aa)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,ai,ck->ai', ints.d_a_vo, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -3.0 * contract('ck,ak,ci->ai', ints.d_a_vo, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -3.0 * contract('ck,ci,ak->ai', ints.d_a_vo, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,ck,ai->ai', ints.d_a_vo, amps.t2_11_a, amps.t2_12_a)
    if 't2_21' in active:
        res += 2.0 * contract('ck,acik->ai', ints.d_a_vo, amps.t2_21_aa)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ik,ak->ai', ints.d_a_oo, amps.t2_12_a)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ik,ak->ai', ints.d_a_oo, amps.t2_11_a)
    if 't2_11' in active:
        res += -2.0 * contract('ik,ak->ai', ints.d_a_oo, amps.t2_11_a)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_22_ab)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_21_ab)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,ai,ck->ai', ints.d_b_vo, amps.t2_11_a, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,ck,ai->ai', ints.d_b_vo, amps.t2_11_b, amps.t2_12_a)
    if 't2_21' in active:
        res += 2.0 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_21_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ac,ci->ai', ints.f_a_vv, amps.t2_12_a)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.f_a_vo, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.f_a_vo, amps.t1_10_a, amps.t2_12_a)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,ci->ai', ints.f_a_vo, amps.t2_11_a, amps.t2_11_a)
    if 't2_22' in active:
        res += 1.0 * contract('ck,acik->ai', ints.f_a_vo, amps.t2_22_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ik,ak->ai', ints.f_a_oo, amps.t2_12_a)
    if 't2_22' in active:
        res += 1.0 * contract('ck,acik->ai', ints.f_b_vo, amps.t2_22_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,dl,kdlc->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_12_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,di,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,di,kdlc->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,di,kdlc->ai', amps.t1_10_a, amps.t1_10_b, amps.t2_12_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_a, amps.t2_11_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,di,kcld->ai', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,ilkc->ai', amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_a, amps.t2_12_b, ints.v_ooov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,al,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dk,al,lckd->ai', amps.t1_10_a, amps.t1_10_b, amps.t2_12_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ci,ak,dl,kcld->ai', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ci,ak,dl,kdlc->ai', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ci,ak,dl,kcld->ai', amps.t1_10_a, amps.t2_11_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dk,kcad->ai', amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,kdac->ai', amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,kdac->ai', amps.t1_10_a, amps.t2_12_b, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,di,al,kcld->ai', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ck,al,di,kcld->ai', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ck,al,di,kdlc->ai', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,al,iklc->ai', amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,al,ilkc->ai', amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,di,kdac->ai', amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ck,al,di,ldkc->ai', amps.t1_10_b, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,al,ilkc->ai', amps.t1_10_b, amps.t2_12_a, ints.v_ooov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_b, amps.t2_12_a, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adil,ldkc->ai', amps.t1_10_b, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,iklc->ai', amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,ilkc->ai', amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,iklc->ai', amps.t2_11_a, amps.t2_11_b, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cdil,kcld->ai', amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cdil,kcld->ai', amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ci,dk,kcad->ai', amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ci,dk,kdac->ai', amps.t2_11_a, amps.t2_11_b, ints.v_ovvv_ba)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,adkl,kcld->ai', amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,adkl,kcld->ai', amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ck,di,kcad->ai', amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adil,kcld->ai', amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,adil,kdlc->ai', amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adil,kcld->ai', amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adil,ldkc->ai', amps.t2_11_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adil,kcld->ai', amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,adil,kdlc->ai', amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 2.0 * w * contract('ai->ai', amps.t2_12_a)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ikac->ai', amps.t2_12_a, ints.v_oovv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,iakc->ai', amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adil,ldkc->ai', amps.t2_12_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,iakc->ai', amps.t2_12_b, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ackl,iklc->ai', amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cdik,kcad->ai', amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ackl,iklc->ai', amps.t2_22_ab, ints.v_ooov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cdik,kdac->ai', amps.t2_22_ab, ints.v_ovvv_ba)

    e_denom = 1.0 / (ints.eps_occ_a[None, :] - ints.eps_vir_a[:, None] - (2.0 * w))
    return amps.t2_12_a + res * e_denom

def ccsd_t2_12_b(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_b.size, ints.eps_occ_b.size))

    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ck,caki->ai', ints.d_a_vo, amps.t2_22_ab)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ck,caki->ai', ints.d_a_vo, amps.t2_21_ab)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,ck,ai->ai', ints.d_a_vo, amps.t2_11_a, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,ai,ck->ai', ints.d_a_vo, amps.t2_11_b, amps.t2_12_a)
    if 't2_21' in active:
        res += 2.0 * contract('ck,caki->ai', ints.d_a_vo, amps.t2_21_ab)
    if 't1_01' in active and 't2_12' in active:
        res += 1.0 * amps.t1_01 * contract('ac,ci->ai', ints.d_b_vv, amps.t2_12_b)
    if 't2_02' in active and 't2_11' in active:
        res += 2.0 * amps.t2_02 * contract('ac,ci->ai', ints.d_b_vv, amps.t2_11_b)
    if 't2_11' in active:
        res += 2.0 * contract('ac,ci->ai', ints.d_b_vv, amps.t2_11_b)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_12_b)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ci,ak->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_12_b)
    if 't1_01' in active and 't2_11' in active:
        res += -2.0 * amps.t1_01 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t2_11_b, amps.t2_11_b)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_22_bb)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,ci,ak->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ci,ak->ai', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_21_bb)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,ai,ck->ai', ints.d_b_vo, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -3.0 * contract('ck,ak,ci->ai', ints.d_b_vo, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -3.0 * contract('ck,ci,ak->ai', ints.d_b_vo, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,ck,ai->ai', ints.d_b_vo, amps.t2_11_b, amps.t2_12_b)
    if 't2_21' in active:
        res += 2.0 * contract('ck,acik->ai', ints.d_b_vo, amps.t2_21_bb)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ik,ak->ai', ints.d_b_oo, amps.t2_12_b)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ik,ak->ai', ints.d_b_oo, amps.t2_11_b)
    if 't2_11' in active:
        res += -2.0 * contract('ik,ak->ai', ints.d_b_oo, amps.t2_11_b)
    if 't2_22' in active:
        res += 1.0 * contract('ck,caki->ai', ints.f_a_vo, amps.t2_22_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ac,ci->ai', ints.f_b_vv, amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.f_b_vo, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.f_b_vo, amps.t1_10_b, amps.t2_12_b)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,ci->ai', ints.f_b_vo, amps.t2_11_b, amps.t2_11_b)
    if 't2_22' in active:
        res += 1.0 * contract('ck,acik->ai', ints.f_b_vo, amps.t2_22_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ik,ak->ai', ints.f_b_oo, amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('ck,al,di,kcld->ai', amps.t1_10_a, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,di,al,kcld->ai', amps.t1_10_a, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ck,al,di,kcld->ai', amps.t1_10_a, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,al,ilkc->ai', amps.t1_10_a, amps.t2_12_b, ints.v_ooov_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_a, amps.t2_12_b, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,dali,kcld->ai', amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,dali,kdlc->ai', amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_a, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,dl,ldkc->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_12_a, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,dl,kdlc->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,di,kcld->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,di,kdlc->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,di,lckd->ai', amps.t1_10_b, amps.t2_11_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,dl,kcld->ai', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,di,kcld->ai', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_b, amps.t2_12_a, ints.v_ooov_ba)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,iklc->ai', amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,ilkc->ai', amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cdli,lckd->ai', amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,al,kcld->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ci,dk,al,kdlc->ai', amps.t1_10_b, amps.t2_11_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ci,ak,dl,kcld->ai', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ci,ak,dl,kdlc->ai', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,kdac->ai', amps.t1_10_b, amps.t2_12_a, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dk,kcad->ai', amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,kdac->ai', amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ci,dakl,kdlc->ai', amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,di,al,kcld->ai', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ck,al,di,kcld->ai', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ck,al,di,kdlc->ai', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,al,iklc->ai', amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,al,ilkc->ai', amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,di,kcad->ai', amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,di,kdac->ai', amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,dali,ldkc->ai', amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ck,al,ilkc->ai', amps.t2_11_a, amps.t2_11_b, ints.v_ooov_ba)
    if 't2_11' in active:
        res += 2.0 * contract('ck,di,kcad->ai', amps.t2_11_a, amps.t2_11_b, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,dali,kcld->ai', amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,dali,kdlc->ai', amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adil,kcld->ai', amps.t2_11_a, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,iklc->ai', amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,ilkc->ai', amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cdli,lckd->ai', amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cdil,kcld->ai', amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ci,dk,kcad->ai', amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,dakl,kdlc->ai', amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,adkl,kcld->ai', amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ck,di,kcad->ai', amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,dali,ldkc->ai', amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adil,kcld->ai', amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,adil,kdlc->ai', amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dali,kcld->ai', amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,dali,kdlc->ai', amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t2_12_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,kcia->ai', amps.t2_12_a, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 2.0 * w * contract('ai->ai', amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cdli,lckd->ai', amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cdil,kcld->ai', amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dakl,kdlc->ai', amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,adkl,kcld->ai', amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dali,ldkc->ai', amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adil,kcld->ai', amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,adil,kdlc->ai', amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ikac->ai', amps.t2_12_b, ints.v_oovv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,iakc->ai', amps.t2_12_b, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('cakl,ilkc->ai', amps.t2_22_ab, ints.v_ooov_ba)
    if 't2_22' in active:
        res += 1.0 * contract('cdki,kcad->ai', amps.t2_22_ab, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ackl,iklc->ai', amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('cdik,kcad->ai', amps.t2_22_bb, ints.v_ovvv_bb)

    e_denom = 1.0 / (ints.eps_occ_b[None, :] - ints.eps_vir_b[:, None] - (2.0 * w))
    return amps.t2_12_b + res * e_denom

def ccsd_t2_22_aa(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_a.size, ints.eps_vir_a.size,
                    ints.eps_occ_a.size, ints.eps_occ_a.size))

    if 't2_12' in active:
        res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_a, ints.B_vv_a, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += -1.0 * contract('xac,xbd,cj,di->abij', ints.B_vv_a, ints.B_vv_a, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += -1.0 * contract('xac,xbd,di,cj->abij', ints.B_vv_a, ints.B_vv_a, amps.t1_10_a, amps.t2_12_a)
    if 't2_12' in active:
        res += 1.0 * contract('xac,xbd,dj,ci->abij', ints.B_vv_a, ints.B_vv_a, amps.t1_10_a, amps.t2_12_a)
    if 't2_11' in active:
        res += 2.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_a, ints.B_vv_a, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -2.0 * contract('xac,xbd,cj,di->abij', ints.B_vv_a, ints.B_vv_a, amps.t2_11_a, amps.t2_11_a)
    if 't2_22' in active:
        _vvvv_ladder2(ints.B_vv_a, ints.B_vv_a, amps.t2_22_aa, out=res, alpha=1.0)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ac,bcij->abij', ints.d_a_vv, amps.t2_22_aa)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ac,bcij->abij', ints.d_a_vv, amps.t2_21_aa)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ac,bi,cj->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ac,bj,ci->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ac,ci,bj->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ac,cj,bi->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_12_a)
    if 't2_21' in active:
        res += -2.0 * contract('ac,bcij->abij', ints.d_a_vv, amps.t2_21_aa)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('bc,acij->abij', ints.d_a_vv, amps.t2_22_aa)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('bc,acij->abij', ints.d_a_vv, amps.t2_21_aa)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('bc,ai,cj->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('bc,aj,ci->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('bc,ci,aj->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('bc,cj,ai->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_12_a)
    if 't2_21' in active:
        res += 2.0 * contract('bc,acij->abij', ints.d_a_vv, amps.t2_21_aa)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += 2.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += 2.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't1_01' in active and 't2_12' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't1_01' in active and 't2_12' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,ak,bi,cj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,ak,bj,ci->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,ak,ci,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,ak,cj,bi->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_21' in active:
        res += 2.0 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,bk,ai,cj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,bk,aj,ci->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,bk,ci,aj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,bk,cj,ai->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,ci,aj,bk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,ci,ak,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,ci,bj,ak->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,ci,bk,aj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_21' in active:
        res += 2.0 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,cj,ai,bk->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,cj,ak,bi->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,cj,bi,ak->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,cj,bk,ai->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_a)
    if 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_aa)
    if 't2_02' in active and 't2_11' in active:
        res += 2.0 * amps.t2_02 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_02' in active and 't2_11' in active:
        res += 2.0 * amps.t2_02 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ai,bk,cj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ck,aj,bk,ci->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active and 't2_22' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ak,bi,cj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,bj,ci->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_11_a, amps.t2_11_a)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active and 't2_22' in active:
        res += 3.0 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_aa)
    if 't2_11' in active and 't2_22' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_aa)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active and 't2_22' in active:
        res += 3.0 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_aa)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_aa)
    if 't2_11' in active and 't2_22' in active:
        res += 2.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_aa)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ai,bcjk->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_aa)
    if 't2_12' in active and 't2_21' in active:
        res += -2.0 * contract('ck,aj,bcik->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_aa)
    if 't2_12' in active and 't2_21' in active:
        res += 3.0 * contract('ck,ak,bcij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_aa)
    if 't2_12' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bi,acjk->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_aa)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bj,acik->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_aa)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,bk,acij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_aa)
    if 't2_12' in active and 't2_21' in active:
        res += 3.0 * contract('ck,ci,abjk->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_aa)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,cj,abik->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_aa)
    if 't2_12' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_aa)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ik,abjk->abij', ints.d_a_oo, amps.t2_22_aa)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ik,abjk->abij', ints.d_a_oo, amps.t2_21_aa)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ik,aj,bk->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ik,ak,bj->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ik,bj,ak->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ik,bk,aj->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_12_a)
    if 't2_21' in active:
        res += 2.0 * contract('ik,abjk->abij', ints.d_a_oo, amps.t2_21_aa)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('jk,abik->abij', ints.d_a_oo, amps.t2_22_aa)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('jk,abik->abij', ints.d_a_oo, amps.t2_21_aa)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('jk,ai,bk->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('jk,ak,bi->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('jk,bi,ak->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_12_a)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('jk,bk,ai->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_12_a)
    if 't2_21' in active:
        res += -2.0 * contract('jk,abik->abij', ints.d_a_oo, amps.t2_21_aa)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_22_ab)
    if 't2_11' in active and 't2_22' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_22_ab)
    if 't2_11' in active and 't2_22' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_22_ab)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_22_ab)
    if 't2_11' in active and 't2_22' in active:
        res += 2.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_aa)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_12_a, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += -2.0 * contract('ck,aj,bcik->abij', ints.d_b_vo, amps.t2_12_a, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bi,acjk->abij', ints.d_b_vo, amps.t2_12_a, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_12_a, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ac,bcij->abij', ints.f_a_vv, amps.t2_22_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bc,acij->abij', ints.f_a_vv, amps.t2_22_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_22_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ak,bcij->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ci,abjk->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_21_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.f_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.f_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_a_vo, amps.t2_12_a, amps.t2_20_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ik,abjk->abij', ints.f_a_oo, amps.t2_22_aa)
    if 't2_22' in active:
        res += -1.0 * contract('jk,abik->abij', ints.f_a_oo, amps.t2_22_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ci,dj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bl,ci,dj,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bl,cj,di,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,bl,ci,dj,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cj,di,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bl,ci,jklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ci,jlkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bl,cj,ilkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,dj,bl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,bl,dj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,bl,dj,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,bl,jklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,bl,jlkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,dj,kdbc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,di,bl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cj,bl,di,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,bl,di,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,bl,iklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,bl,ilkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,di,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,di,kdbc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ak,bl,ci,jklc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,ci,jlkc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,bl,cj,ilkc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,di,kcbd->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ikjl->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oooo_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bl,iljk->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oooo_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,jkbc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oovv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,jbkc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,ikbc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oovv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,ibkc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ak,bcil,jlkc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,dj,al,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,ci,al,dj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,al,dj,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,al,jklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,al,jlkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,dj,kcad->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,dj,kdac->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,di,al,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,al,di,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,al,di,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,al,iklc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,al,ilkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,di,kcad->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,di,kdac->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('bk,al,ci,jklc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,al,ci,jlkc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,al,cj,iklc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('bk,al,cj,ilkc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,al,cdij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,dj,kcad->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,di,kcad->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('bk,al,cdij,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,al,ikjl->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oooo_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,al,iljk->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oooo_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,jkac->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oovv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,jakc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,ikac->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oovv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,iakc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bk,acjl,ilkc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cdij,kcad->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,ak,bl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dj,ak,bl,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dj,ak,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dj,bk,kcad->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ci,dk,abjl,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ci,ak,bl,jklc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ci,ak,bl,jlkc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ci,ak,dj,kcbd->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ci,ak,dj,kdbc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,ak,bdjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ci,bk,dj,kcad->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ci,bk,dj,kdac->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,bk,adjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,bk,adjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,bk,adjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dk,abjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dk,abjl,lckd->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,ak,bdjl,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ci,ak,jkbc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oovv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,ak,jbkc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bk,adjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bk,adjl,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bk,adjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bk,jkac->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oovv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bk,jakc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,abjl,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,abjl,lckd->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ci,abkl,jklc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ci,adjk,kcbd->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += -1.0 * contract('cj,di,ak,bl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 1.0 * contract('cj,di,ak,bl,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('cj,di,ak,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,di,bk,kcad->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cj,dk,abil,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('cj,ak,bl,iklc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('cj,ak,bl,ilkc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('cj,ak,di,kcbd->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -2.0 * contract('cj,ak,di,kdbc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,ak,bdil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,ak,bdil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,ak,bdil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('cj,bk,di,kcad->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 2.0 * contract('cj,bk,di,kdac->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,bk,adil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,dk,abil,lckd->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('cj,ak,bdil,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,ak,bdil,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('cj,ak,bdil,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('cj,ak,ikbc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oovv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('cj,ak,ibkc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('cj,bk,adil,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('cj,bk,ikac->abij', amps.t1_10_a, amps.t2_12_a, ints.v_oovv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bk,iakc->abij', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,dk,abil,lckd->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cj,bdik,kcad->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,al,bdij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bl,adij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,di,abjl,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,dj,abil,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,al,bdij,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bl,adij,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,di,abjl,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dj,abil,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,abjl,iklc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ooov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,bdij,kdac->abij', amps.t1_10_a, amps.t2_22_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,al,bdij,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bl,adij,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,di,abjl,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,dj,abil,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,al,bdij,ldkc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bl,adij,ldkc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,di,abjl,ldkc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,dj,abil,ldkc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_b, amps.t2_22_aa, ints.v_ooov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_b, amps.t2_22_aa, ints.v_ooov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_b, amps.t2_22_aa, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_b, amps.t2_22_aa, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cdij,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,ikjl->abij', amps.t2_11_a, amps.t2_11_a, ints.v_oooo_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,bl,iljk->abij', amps.t2_11_a, amps.t2_11_a, ints.v_oooo_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,jkbc->abij', amps.t2_11_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,jbkc->abij', amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,bdil,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cj,bdil,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,bdil,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cj,ikbc->abij', amps.t2_11_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,ibkc->abij', amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,bdij,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,bdij,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,bdij,kdlc->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bcil,jklc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,bcil,jlkc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,bcjl,iklc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bcjl,ilkc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cdij,kcbd->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bcil,jklc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,bcjl,iklc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,adjl,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,ci,adjl,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,adjl,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('bk,ci,jkac->abij', amps.t2_11_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,jakc->abij', amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,adil,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,ikac->abij', amps.t2_11_a, amps.t2_11_a, ints.v_oovv_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,iakc->abij', amps.t2_11_a, amps.t2_11_a, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cl,adij,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,acil,jklc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,acil,jlkc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,acjl,iklc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,acjl,ilkc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cdij,kcad->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,acil,jklc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,acjl,iklc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ci,dk,abjl,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ci,dk,abjl,lckd->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,abkl,jklc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,adjk,kcbd->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,adjk,kdbc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,bdjk,kcad->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,bdjk,kdac->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,adjk,kdbc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,bdjk,kdac->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('cj,dk,abil,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('cj,dk,abil,lckd->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,abkl,iklc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,adik,kcbd->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,adik,kdbc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,bdik,kcad->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,bdik,kdac->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,adik,kdbc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,bdik,kdac->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += -2.0 * contract('ck,di,abjl,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ck,dj,abil,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_aa, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,abil,jklc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,abil,jlkc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,abjl,iklc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,abjl,ilkc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adij,kcbd->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,adij,kdbc->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bdij,kcad->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bdij,kdac->abij', amps.t2_11_a, amps.t2_21_aa, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,abil,jlkc->abij', amps.t2_11_b, amps.t2_21_aa, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,abjl,ilkc->abij', amps.t2_11_b, amps.t2_21_aa, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adij,kcbd->abij', amps.t2_11_b, amps.t2_21_aa, ints.v_ovvv_ba)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bdij,kcad->abij', amps.t2_11_b, amps.t2_21_aa, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bcil,jlkc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ikjb->abij', amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,jkib->abij', amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,acjl,ilkc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cdij,kcad->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ikja->abij', amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,jkia->abij', amps.t2_12_a, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,abkl,jklc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,adjk,kcbd->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += -1.0 * contract('ci,jabc->abij', amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,jbac->abij', amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('cj,bdik,kcad->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += 1.0 * contract('cj,iabc->abij', amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('cj,ibac->abij', amps.t2_12_a, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,abjl,iklc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bdij,kdac->abij', amps.t2_12_a, amps.t2_20_aa, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_12_b, amps.t2_20_aa, ints.v_ooov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t2_12_b, amps.t2_20_aa, ints.v_ooov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_12_b, amps.t2_20_aa, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t2_12_b, amps.t2_20_aa, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('abjk,cdil,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('abjk,cdil,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 0.5 * contract('abkl,cdij,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('acjk,bdil,kdlc->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ackl,bdij,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bcij,adkl,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bcij,adkl,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('bcik,adjl,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bcik,adjl,kdlc->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bcik,adjl,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bcjk,adil,kdlc->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bckl,adij,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 0.5 * contract('cdij,abkl,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cdik,abjl,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cdjk,abil,kcld->abij', amps.t2_20_aa, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('acik,bdjl,ldkc->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('acjk,bdil,ldkc->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('acjk,bdil,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ackl,bdij,kdlc->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('bcik,adjl,ldkc->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('bcik,adjl,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bcik,adjl,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,adil,ldkc->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bcjk,adil,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bckl,adij,kdlc->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cdik,abjl,lckd->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cdjk,abil,lckd->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('abik,cdjl,kcld->abij', amps.t2_21_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -2.0 * contract('abik,cdjl,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 2.0 * contract('abjk,cdil,kcld->abij', amps.t2_21_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 2.0 * contract('abjk,cdil,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('abkl,cdij,kcld->abij', amps.t2_21_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -2.0 * contract('acij,bdkl,kcld->abij', amps.t2_21_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -2.0 * contract('acij,bdkl,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 2.0 * contract('acik,bdjl,kcld->abij', amps.t2_21_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -2.0 * contract('acik,bdjl,kdlc->abij', amps.t2_21_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 2.0 * contract('acik,bdjl,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('acjk,bdil,kcld->abij', amps.t2_21_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 2.0 * contract('acjk,bdil,kdlc->abij', amps.t2_21_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -2.0 * contract('acjk,bdil,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('ackl,bdij,kcld->abij', amps.t2_21_aa, amps.t2_21_aa, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 2.0 * contract('bcij,adkl,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('bcik,adjl,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 2.0 * contract('bcjk,adil,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 2.0 * contract('acik,bdjl,kcld->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -2.0 * contract('acik,bdjl,kdlc->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -2.0 * contract('acjk,bdil,kcld->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 2.0 * contract('acjk,bdil,kdlc->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 2.0 * w * contract('abij->abij', amps.t2_22_aa)
    if 't2_22' in active:
        res += 1.0 * contract('abkl,ikjl->abij', amps.t2_22_aa, ints.v_oooo_aa)
    if 't2_22' in active:
        res += -1.0 * contract('acik,jkbc->abij', amps.t2_22_aa, ints.v_oovv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('acik,jbkc->abij', amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('acjk,ikbc->abij', amps.t2_22_aa, ints.v_oovv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('acjk,ibkc->abij', amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bcik,jkac->abij', amps.t2_22_aa, ints.v_oovv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bcik,jakc->abij', amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('bcjk,ikac->abij', amps.t2_22_aa, ints.v_oovv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,iakc->abij', amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('acik,jbkc->abij', amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('acjk,ibkc->abij', amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('bcik,jakc->abij', amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,iakc->abij', amps.t2_22_ab, ints.v_ovov_ab)

    # exact antisymmetry of the same-spin residual; enforce against rounding drift
    res = 0.5 * (res - res.transpose(1, 0, 2, 3))
    res = 0.5 * (res - res.transpose(0, 1, 3, 2))
    nocc_j = ints.eps_occ_a.size
    for j in range(nocc_j):
        res[:, :, :, j] *= 1.0 / (ints.eps_occ_a[None, None, :] + ints.eps_occ_a[j]
                                  - ints.eps_vir_a[:, None, None]
                                  - ints.eps_vir_a[None, :, None] - (2.0 * w))
    res += amps.t2_22_aa
    return res

def ccsd_t2_22_ab(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_a.size, ints.eps_vir_b.size,
                    ints.eps_occ_a.size, ints.eps_occ_b.size))

    if 't2_12' in active:
        res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_a, ints.B_vv_b, amps.t1_10_a, amps.t2_12_b)
    if 't2_12' in active:
        res += 1.0 * contract('xac,xbd,dj,ci->abij', ints.B_vv_a, ints.B_vv_b, amps.t1_10_b, amps.t2_12_a)
    if 't2_11' in active:
        res += 2.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_a, ints.B_vv_b, amps.t2_11_a, amps.t2_11_b)
    if 't2_22' in active:
        _vvvv_ladder2(ints.B_vv_a, ints.B_vv_b, amps.t2_22_ab, out=res, alpha=1.0)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ac,cbij->abij', ints.d_a_vv, amps.t2_22_ab)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ac,cbij->abij', ints.d_a_vv, amps.t2_21_ab)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ac,ci,bj->abij', ints.d_a_vv, amps.t2_11_a, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ac,bj,ci->abij', ints.d_a_vv, amps.t2_11_b, amps.t2_12_a)
    if 't2_21' in active:
        res += 2.0 * contract('ac,cbij->abij', ints.d_a_vv, amps.t2_21_ab)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_ab)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_22_ab)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * amps.t1_01 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * amps.t1_01 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,ak,ci,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,ak,bj,ci->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_b, amps.t2_12_a)
    if 't2_21' in active:
        res += -2.0 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,ci,ak,bj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_a, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,ci,bj,ak->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_11_b, amps.t2_12_a)
    if 't2_21' in active:
        res += -2.0 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t1_10_a, amps.t2_21_ab)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,ai,cbkj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,ci,bj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_11_a, amps.t2_11_b)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_20_ab)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_ab)
    if 't2_11' in active and 't2_22' in active:
        res += 2.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_ab)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_22_aa)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ai,cbkj->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,ak,cbij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,ci,abkj->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bj,acik->abij', ints.d_a_vo, amps.t2_12_b, amps.t2_21_aa)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ik,abkj->abij', ints.d_a_oo, amps.t2_22_ab)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ik,abkj->abij', ints.d_a_oo, amps.t2_21_ab)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ik,ak,bj->abij', ints.d_a_oo, amps.t2_11_a, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ik,bj,ak->abij', ints.d_a_oo, amps.t2_11_b, amps.t2_12_a)
    if 't2_21' in active:
        res += -2.0 * contract('ik,abkj->abij', ints.d_a_oo, amps.t2_21_ab)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_22_ab)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_21_ab)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('bc,ai,cj->abij', ints.d_b_vv, amps.t2_11_a, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('bc,cj,ai->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_12_a)
    if 't2_21' in active:
        res += 2.0 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_21_ab)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_ab)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_ab)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,bk,ai,cj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_a, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,bk,cj,ai->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_a)
    if 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,cj,ai,bk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_a, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,cj,bk,ai->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_a)
    if 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_ab)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ai,bk,cj->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_11_a, amps.t2_22_bb)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_ab)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_ab)
    if 't2_11' in active and 't2_22' in active:
        res += 2.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_ab)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_12_a, amps.t2_21_bb)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_ab)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_22_ab)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_21_ab)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('jk,ai,bk->abij', ints.d_b_oo, amps.t2_11_a, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('jk,bk,ai->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_12_a)
    if 't2_21' in active:
        res += -2.0 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_21_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ac,cbij->abij', ints.f_a_vv, amps.t2_22_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,ak,cbij->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_22_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.f_a_vo, amps.t1_10_a, amps.t2_22_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ak,cbij->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ci,abkj->abij', ints.f_a_vo, amps.t2_11_a, amps.t2_21_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,cbij->abij', ints.f_a_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.f_a_vo, amps.t2_12_a, amps.t2_20_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ik,abkj->abij', ints.f_a_oo, amps.t2_22_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bc,acij->abij', ints.f_b_vv, amps.t2_22_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_22_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_22_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_21_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t2_12_b, amps.t2_20_ab)
    if 't2_22' in active:
        res += -1.0 * contract('jk,abik->abij', ints.f_b_oo, amps.t2_22_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,bl,dj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,dj,bl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,bl,dj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,bl,jlkc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_b, ints.v_ooov_ba)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_12_b, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ak,ci,dblj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ak,ci,dblj,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cl,dbij,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t1_10_b, amps.t2_12_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,ci,dj,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ci,jlkc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_12_a, ints.v_ooov_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,di,bl,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,di,kdbc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_12_a, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,bl,iklc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cj,dbil,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,bl,jlkc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_b, ints.v_ooov_ba)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_b, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,ci,dblj,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,ci,dblj,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cl,dbij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cj,dbil,kdlc->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,dblj,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,dblj,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,kcjb->abij', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,dbij,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ikjl->abij', amps.t1_10_a, amps.t2_12_b, ints.v_oooo_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,dbil,kdlc->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,ikbc->abij', amps.t1_10_a, amps.t2_12_b, ints.v_oovv_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,dbij,kdlc->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cbil,jlkc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_ba)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cblj,iklc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cblj,ilkc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_a, amps.t2_22_bb, ints.v_ooov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ci,dk,ablj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bk,dj,al,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t1_10_b, amps.t2_12_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ci,bk,al,dj,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bk,al,jklc->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_12_a, ints.v_ooov_ba)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bk,dj,kdac->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += 1.0 * contract('ci,bk,adlj,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ci,dj,ak,bl,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_11_a, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dj,ak,kcbd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_12_a, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dj,bk,kdac->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ci,dk,ablj,lckd->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ci,ak,bl,jlkc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_b, ints.v_ooov_ba)
    if 't2_11' in active:
        res += -2.0 * contract('ci,ak,dj,kcbd->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_11_b, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,ak,dblj,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,ak,dblj,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dk,ablj,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,dk,ablj,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ci,bk,dj,kdac->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_ba)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,bk,adlj,lckd->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,dk,ablj,lckd->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,ak,dblj,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,ak,dblj,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,ak,kcjb->abij', amps.t1_10_a, amps.t2_12_a, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,ablj,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dk,ablj,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bk,adlj,lckd->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bk,jkac->abij', amps.t1_10_a, amps.t2_12_b, ints.v_oovv_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dk,ablj,lckd->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ci,abkl,jlkc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_ba)
    if 't2_22' in active:
        res += -1.0 * contract('ci,adkj,kcbd->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ci,dbkj,kcad->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ci,dbkj,kdac->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_a, amps.t2_22_bb, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += -1.0 * contract('ck,di,ablj,kcld->abij', amps.t1_10_a, amps.t1_10_a, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,al,dbij,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,al,dbij,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,di,ablj,kcld->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,di,ablj,kdlc->abij', amps.t1_10_a, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,al,dbij,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,al,dbij,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,di,ablj,kcld->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,di,ablj,kdlc->abij', amps.t1_10_a, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_ba)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ablj,iklc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,ablj,ilkc->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ooov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,dbij,kcad->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_aa)
    if 't2_22' in active:
        res += -1.0 * contract('ck,dbij,kdac->abij', amps.t1_10_a, amps.t2_22_ab, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,al,di,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_a, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,al,ilkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_a, ints.v_ooov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,di,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_a, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cj,adil,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,al,ci,jklc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_11_a, ints.v_ooov_ba)
    if 't2_11' in active:
        res += 2.0 * contract('bk,al,cj,ilkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_11_b, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,al,cdij,lckd->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('bk,ci,dj,kdac->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_11_b, ints.v_ovvv_ba)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,ci,adlj,lckd->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cl,adij,lckd->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cj,adil,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,al,cdij,lckd->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bk,al,iljk->abij', amps.t1_10_b, amps.t2_12_a, ints.v_oooo_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,adlj,lckd->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,jkac->abij', amps.t1_10_b, amps.t2_12_a, ints.v_oovv_ba)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cl,adij,lckd->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,adil,ldkc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,iakc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_b, amps.t2_22_aa, ints.v_ooov_ba)
    if 't2_22' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bk,aclj,ilkc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cdij,kdac->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('cj,ak,di,kdbc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_11_a, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += 2.0 * contract('cj,ak,bl,iklc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_11_b, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,ak,dbil,kdlc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('cj,di,bk,kcad->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_11_b, ints.v_ovvv_ba)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,di,abkl,kdlc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,bk,adil,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_aa, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,bk,adil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,ak,dbil,kdlc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('cj,ak,ikbc->abij', amps.t1_10_b, amps.t2_12_a, ints.v_oovv_ab)
    if 't2_12' in active:
        res += 1.0 * contract('cj,di,abkl,kdlc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bk,adil,ldkc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,bk,adil,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bk,iakc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_b, amps.t2_22_aa, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('cj,dbik,kcad->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,al,dbij,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,di,ablj,ldkc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bl,adij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,dj,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,al,dbij,ldkc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,di,ablj,ldkc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bl,adij,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dj,abil,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,ablj,ilkc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,dbij,kcad->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,dblj,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,dblj,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,kcjb->abij', amps.t2_11_a, amps.t2_11_a, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,dbij,kcld->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,dbij,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cdij,kcld->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,ikjl->abij', amps.t2_11_a, amps.t2_11_b, ints.v_oooo_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,dbil,kdlc->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cj,ikbc->abij', amps.t2_11_a, amps.t2_11_b, ints.v_oovv_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,dbij,kdlc->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cbil,jlkc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cblj,iklc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cblj,ilkc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cdij,kcbd->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,bcjl,iklc->abij', amps.t2_11_a, amps.t2_21_bb, ints.v_ooov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ci,dk,ablj,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += 2.0 * contract('ci,bk,adlj,lckd->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ci,bk,jkac->abij', amps.t2_11_a, amps.t2_11_b, ints.v_oovv_ba)
    if 't2_11' in active:
        res += 2.0 * contract('ci,dj,abkl,kcld->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ci,dk,ablj,lckd->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,abkl,jlkc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,adkj,kcbd->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,dbkj,kcad->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dbkj,kdac->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,bdjk,kdac->abij', amps.t2_11_a, amps.t2_21_bb, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += 2.0 * contract('ck,di,ablj,kdlc->abij', amps.t2_11_a, amps.t2_11_a, amps.t2_20_ab, ints.v_ovov_aa)
    if 't2_11' in active:
        res += -2.0 * contract('ck,bl,adij,kcld->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,abil,jlkc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ablj,iklc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ablj,ilkc->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ooov_aa)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adij,kcbd->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,dbij,kcad->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_aa)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,dbij,kdac->abij', amps.t2_11_a, amps.t2_21_ab, ints.v_ovvv_aa)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,adil,ldkc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_aa, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,adil,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,iakc->abij', amps.t2_11_b, amps.t2_11_b, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cl,adij,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,acil,jklc->abij', amps.t2_11_b, amps.t2_21_aa, ints.v_ooov_ba)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,acil,jklc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,acil,jlkc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,aclj,ilkc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cdij,kdac->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += -2.0 * contract('cj,dk,abil,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,adik,kdbc->abij', amps.t2_11_b, amps.t2_21_aa, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,abkl,iklc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,adik,kcbd->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,adik,kdbc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,dbik,kcad->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_11' in active:
        res += 2.0 * contract('ck,dj,abil,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,abil,jklc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,abil,jlkc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ablj,ilkc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adij,kcbd->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,adij,kdbc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,dbij,kcad->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cbil,jlkc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cblj,iklc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cblj,ilkc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t2_12_a, amps.t2_20_bb, ints.v_ooov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ikjb->abij', amps.t2_12_a, ints.v_ooov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ci,abkl,jlkc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_12' in active:
        res += -1.0 * contract('ci,adkj,kcbd->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dbkj,kcad->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dbkj,kdac->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t2_12_a, amps.t2_20_bb, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ci,jbac->abij', amps.t2_12_a, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ablj,iklc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ablj,ilkc->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ooov_aa)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dbij,kcad->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('ck,dbij,kdac->abij', amps.t2_12_a, amps.t2_20_ab, ints.v_ovvv_aa)
    if 't2_12' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t2_12_b, amps.t2_20_aa, ints.v_ooov_ba)
    if 't2_12' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,aclj,ilkc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cdij,kdac->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += -1.0 * contract('bk,jkia->abij', amps.t2_12_b, ints.v_ooov_ba)
    if 't2_12' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t2_12_b, amps.t2_20_aa, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,dbik,kcad->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_12' in active:
        res += 1.0 * contract('cj,iabc->abij', amps.t2_12_b, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ablj,ilkc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dbij,kcad->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_ba)
    if 't2_22' in active:
        res += 1.0 * contract('acik,dblj,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('acik,dblj,kdlc->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_aa, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ackl,dbij,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cdik,ablj,kcld->abij', amps.t2_20_aa, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('abik,cdlj,lckd->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('abkj,cdil,kcld->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('abkj,cdil,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('abkl,cdij,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('acij,dbkl,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('acik,dblj,ldkc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ackj,dbil,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ackl,dbij,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cbij,adkl,kcld->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cbij,adkl,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cbik,adlj,lckd->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cbkj,adil,kcld->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cbkj,adil,kdlc->abij', amps.t2_20_ab, amps.t2_22_aa, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cbkj,adil,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cbkl,adij,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cdij,abkl,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cdik,ablj,lckd->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cdkj,abil,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,adil,ldkc->abij', amps.t2_20_bb, amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bcjk,adil,kdlc->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bckl,adij,kcld->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('cdjk,abil,kcld->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 2.0 * contract('acik,dblj,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -2.0 * contract('acik,dblj,kdlc->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 2.0 * contract('acik,bdjl,kcld->abij', amps.t2_21_aa, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 2.0 * contract('ackl,dbij,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 2.0 * contract('cdik,ablj,kcld->abij', amps.t2_21_aa, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -2.0 * contract('abik,cdlj,lckd->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('abik,cdjl,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -2.0 * contract('abkj,cdil,kcld->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 2.0 * contract('abkl,cdij,kcld->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('acij,dbkl,kdlc->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('acij,bdkl,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 2.0 * contract('acik,dblj,ldkc->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 2.0 * contract('acik,bdjl,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -2.0 * contract('acik,bdjl,kdlc->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 2.0 * contract('ackj,dbil,kdlc->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('ackl,dbij,kdlc->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('acik,kcjb->abij', amps.t2_22_aa, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 2.0 * w * contract('abij->abij', amps.t2_22_ab)
    if 't2_22' in active:
        res += 1.0 * contract('abkl,ikjl->abij', amps.t2_22_ab, ints.v_oooo_ab)
    if 't2_22' in active:
        res += -1.0 * contract('acik,jkbc->abij', amps.t2_22_ab, ints.v_oovv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('acik,jbkc->abij', amps.t2_22_ab, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ackj,ikbc->abij', amps.t2_22_ab, ints.v_oovv_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cbik,jkac->abij', amps.t2_22_ab, ints.v_oovv_ba)
    if 't2_22' in active:
        res += -1.0 * contract('cbkj,ikac->abij', amps.t2_22_ab, ints.v_oovv_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cbkj,iakc->abij', amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,iakc->abij', amps.t2_22_bb, ints.v_ovov_ab)

    nocc_j = ints.eps_occ_b.size
    for j in range(nocc_j):
        res[:, :, :, j] *= 1.0 / (ints.eps_occ_a[None, None, :] + ints.eps_occ_b[j]
                                  - ints.eps_vir_a[:, None, None]
                                  - ints.eps_vir_b[None, :, None] - (2.0 * w))
    res += amps.t2_22_ab
    return res

def ccsd_t2_22_bb(ints, w, amps, active=_ALL_AMPS):
    res = np.zeros((ints.eps_vir_b.size, ints.eps_vir_b.size,
                    ints.eps_occ_b.size, ints.eps_occ_b.size))

    if 't2_12' in active:
        res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_b, ints.B_vv_b, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('xac,xbd,cj,di->abij', ints.B_vv_b, ints.B_vv_b, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += -1.0 * contract('xac,xbd,di,cj->abij', ints.B_vv_b, ints.B_vv_b, amps.t1_10_b, amps.t2_12_b)
    if 't2_12' in active:
        res += 1.0 * contract('xac,xbd,dj,ci->abij', ints.B_vv_b, ints.B_vv_b, amps.t1_10_b, amps.t2_12_b)
    if 't2_11' in active:
        res += 2.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv_b, ints.B_vv_b, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -2.0 * contract('xac,xbd,cj,di->abij', ints.B_vv_b, ints.B_vv_b, amps.t2_11_b, amps.t2_11_b)
    if 't2_22' in active:
        _vvvv_ladder2(ints.B_vv_b, ints.B_vv_b, amps.t2_22_bb, out=res, alpha=1.0)
    if 't2_11' in active and 't2_22' in active:
        res += 2.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t2_11_a, amps.t2_22_bb)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,ai,cbkj->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_22_ab)
    if 't2_11' in active and 't2_22' in active:
        res += -1.0 * contract('ck,aj,cbki->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_22_ab)
    if 't2_11' in active and 't2_22' in active:
        res += -1.0 * contract('ck,bi,cakj->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_22_ab)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,bj,caki->abij', ints.d_a_vo, amps.t2_11_b, amps.t2_22_ab)
    if 't2_12' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_a_vo, amps.t2_12_a, amps.t2_21_bb)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ai,cbkj->abij', ints.d_a_vo, amps.t2_12_b, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += -2.0 * contract('ck,aj,cbki->abij', ints.d_a_vo, amps.t2_12_b, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bi,cakj->abij', ints.d_a_vo, amps.t2_12_b, amps.t2_21_ab)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bj,caki->abij', ints.d_a_vo, amps.t2_12_b, amps.t2_21_ab)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ac,bcij->abij', ints.d_b_vv, amps.t2_22_bb)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ac,bcij->abij', ints.d_b_vv, amps.t2_21_bb)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ac,bi,cj->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ac,bj,ci->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ac,ci,bj->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ac,cj,bi->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_12_b)
    if 't2_21' in active:
        res += -2.0 * contract('ac,bcij->abij', ints.d_b_vv, amps.t2_21_bb)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_22_bb)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_21_bb)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('bc,ai,cj->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('bc,aj,ci->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('bc,ci,aj->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('bc,cj,ai->abij', ints.d_b_vv, amps.t2_11_b, amps.t2_12_b)
    if 't2_21' in active:
        res += 2.0 * contract('bc,acij->abij', ints.d_b_vv, amps.t2_21_bb)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += 2.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += 2.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't1_01' in active and 't2_12' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't1_01' in active and 't2_12' in active:
        res += 1.0 * amps.t1_01 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * amps.t1_01 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,ak,bi,cj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,ak,bj,ci->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,ak,ci,bj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,ak,cj,bi->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_21' in active:
        res += 2.0 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,bk,ai,cj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,bk,aj,ci->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,bk,ci,aj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,bk,cj,ai->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,ci,aj,bk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,ci,ak,bj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,ci,bj,ak->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,ci,bk,aj->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_21' in active:
        res += 2.0 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,cj,ai,bk->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,cj,ak,bi->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ck,cj,bi,ak->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,cj,bk,ai->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_11_b, amps.t2_12_b)
    if 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t1_10_b, amps.t2_21_bb)
    if 't2_02' in active and 't2_11' in active:
        res += 2.0 * amps.t2_02 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_02' in active and 't2_11' in active:
        res += 2.0 * amps.t2_02 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * amps.t2_02 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ai,bk,cj->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ck,aj,bk,ci->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active and 't2_22' in active:
        res += -1.0 * contract('ck,aj,bcik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ak,bi,cj->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,bj,ci->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_11_b, amps.t2_11_b)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active and 't2_22' in active:
        res += 3.0 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_bb)
    if 't2_11' in active and 't2_22' in active:
        res += -1.0 * contract('ck,bi,acjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_bb)
    if 't2_11' in active and 't2_22' in active:
        res += 1.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active and 't2_22' in active:
        res += 3.0 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_20_bb)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_bb)
    if 't2_11' in active and 't2_22' in active:
        res += 2.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t2_11_b, amps.t2_22_bb)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ai,bcjk->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_bb)
    if 't2_12' in active and 't2_21' in active:
        res += -2.0 * contract('ck,aj,bcik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_bb)
    if 't2_12' in active and 't2_21' in active:
        res += 3.0 * contract('ck,ak,bcij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_bb)
    if 't2_12' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bi,acjk->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_bb)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bj,acik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_bb)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,bk,acij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_bb)
    if 't2_12' in active and 't2_21' in active:
        res += 3.0 * contract('ck,ci,abjk->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_bb)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,cj,abik->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_bb)
    if 't2_12' in active and 't2_21' in active:
        res += 1.0 * contract('ck,ck,abij->abij', ints.d_b_vo, amps.t2_12_b, amps.t2_21_bb)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * amps.t1_01 * contract('ik,abjk->abij', ints.d_b_oo, amps.t2_22_bb)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * amps.t2_02 * contract('ik,abjk->abij', ints.d_b_oo, amps.t2_21_bb)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ik,aj,bk->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ik,ak,bj->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ik,bj,ak->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ik,bk,aj->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_12_b)
    if 't2_21' in active:
        res += 2.0 * contract('ik,abjk->abij', ints.d_b_oo, amps.t2_21_bb)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * amps.t1_01 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_22_bb)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * amps.t2_02 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_21_bb)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('jk,ai,bk->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('jk,ak,bi->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('jk,bi,ak->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_12_b)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('jk,bk,ai->abij', ints.d_b_oo, amps.t2_11_b, amps.t2_12_b)
    if 't2_21' in active:
        res += -2.0 * contract('jk,abik->abij', ints.d_b_oo, amps.t2_21_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ac,bcij->abij', ints.f_b_vv, amps.t2_22_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bc,acij->abij', ints.f_b_vv, amps.t2_22_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t1_10_b, amps.t2_22_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ak,bcij->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ci,abjk->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t2_11_b, amps.t2_21_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ak,bcij->abij', ints.f_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ci,abjk->abij', ints.f_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_b_vo, amps.t2_12_b, amps.t2_20_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ik,abjk->abij', ints.f_b_oo, amps.t2_22_bb)
    if 't2_22' in active:
        res += -1.0 * contract('jk,abik->abij', ints.f_b_oo, amps.t2_22_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_a, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_a, amps.t2_22_bb, ints.v_ooov_ba)
    if 't2_22' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_a, amps.t2_22_bb, ints.v_ooov_ba)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_a, amps.t2_22_bb, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_a, amps.t2_22_bb, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ci,dj,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bl,ci,dj,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bl,cj,di,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,bl,ci,dj,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cj,di,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bl,ci,jklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ci,jlkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bl,cj,ilkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,dj,bl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,bl,dj,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,bl,dj,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,bl,jklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,bl,jlkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,dj,kdbc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ak,ci,dblj,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,di,bl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cj,bl,di,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,bl,di,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,bl,iklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,bl,ilkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,di,kcbd->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,di,kdbc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cj,dbli,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cl,bdij,lckd->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ak,bl,ci,jklc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,ci,jlkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cj,iklc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,bl,cj,ilkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,dj,kcbd->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,ci,dblj,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,di,kcbd->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cj,dbli,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,bdij,lckd->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ikjl->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oooo_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bl,iljk->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oooo_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,dblj,ldkc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,bdjl,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,jkbc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oovv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,jbkc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,dbli,ldkc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,bdil,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,bdil,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,ikbc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oovv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,ibkc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cl,bdij,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,bdij,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cbli,jklc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_ba)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cblj,iklc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_ba)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ak,bcil,jlkc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,dj,al,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,ci,al,dj,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,al,dj,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,al,jklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,al,jlkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,dj,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,dj,kdac->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bk,ci,dalj,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,di,al,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,al,di,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,al,di,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,al,iklc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,al,ilkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,di,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,di,kdac->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cj,dali,ldkc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cl,adij,lckd->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('bk,al,ci,jklc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,al,ci,jlkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,al,cj,iklc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,al,cj,ilkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,al,cdij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,dj,kcad->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,ci,dalj,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,di,kcad->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cj,dali,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cl,adij,lckd->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('bk,al,cdij,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,al,ikjl->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oooo_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,al,iljk->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oooo_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,dalj,ldkc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,adjl,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,adjl,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,jkac->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oovv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,jakc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,dali,ldkc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,adil,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,ikac->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oovv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,iakc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cl,adij,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cali,jklc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_ba)
    if 't2_22' in active:
        res += 1.0 * contract('bk,calj,iklc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ooov_ba)
    if 't2_22' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bk,acjl,ilkc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cdij,kcad->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,ak,bl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dj,ak,bl,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dj,ak,kcbd->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dj,bk,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dk,abjl,kdlc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ci,ak,bl,jklc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ci,ak,bl,jlkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ci,ak,dj,kcbd->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ci,ak,dj,kdbc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,ak,dblj,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,ak,bdjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ci,bk,dj,kcad->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ci,bk,dj,kdac->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,bk,dalj,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,bk,adjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,bk,adjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dk,abjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,abjl,kdlc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,ak,dblj,ldkc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,ak,bdjl,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,ak,bdjl,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,ak,jkbc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oovv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ci,ak,jbkc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bk,dalj,ldkc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bk,adjl,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bk,adjl,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bk,jkac->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oovv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bk,jakc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dk,abjl,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,abjl,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ci,dakj,kdbc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += 1.0 * contract('ci,dbkj,kdac->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += -1.0 * contract('ci,abkl,jklc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ci,adjk,kcbd->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,di,ak,bl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 1.0 * contract('cj,di,ak,bl,kdlc->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,di,ak,kcbd->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,di,bk,kcad->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_a, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('cj,ak,bl,iklc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('cj,ak,bl,ilkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ooov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('cj,ak,di,kcbd->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -2.0 * contract('cj,ak,di,kdbc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,ak,dbli,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,ak,bdil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,ak,bdil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('cj,bk,di,kcad->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 2.0 * contract('cj,bk,di,kdac->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_11_b, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,bk,dali,ldkc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_ab, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,bk,adil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_12_a, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('cj,ak,dbli,ldkc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += 1.0 * contract('cj,ak,bdil,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,ak,bdil,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,ak,ikbc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oovv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,ak,ibkc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bk,dali,ldkc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bk,adil,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,bk,adil,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,bk,ikac->abij', amps.t1_10_b, amps.t2_12_b, ints.v_oovv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bk,iakc->abij', amps.t1_10_b, amps.t2_12_b, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,dk,abil,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('cj,daki,kdbc->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cj,dbki,kdac->abij', amps.t1_10_b, amps.t2_22_ab, ints.v_ovvv_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('cj,bdik,kcad->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t1_10_b, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,al,bdij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bl,adij,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,di,abjl,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,dj,abil,kdlc->abij', amps.t1_10_b, amps.t2_11_b, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,al,bdij,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,al,bdij,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bl,adij,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bl,adij,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,di,abjl,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,di,abjl,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,dj,abil,kcld->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dj,abil,kdlc->abij', amps.t1_10_b, amps.t2_12_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,abjl,iklc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ooov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('ck,bdij,kdac->abij', amps.t1_10_b, amps.t2_22_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ck,al,bdij,kcld->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ck,bl,adij,kcld->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ck,di,abjl,kcld->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', amps.t2_11_a, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,abil,jlkc->abij', amps.t2_11_a, amps.t2_21_bb, ints.v_ooov_ba)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,abjl,ilkc->abij', amps.t2_11_a, amps.t2_21_bb, ints.v_ooov_ba)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adij,kcbd->abij', amps.t2_11_a, amps.t2_21_bb, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bdij,kcad->abij', amps.t2_11_a, amps.t2_21_bb, ints.v_ovvv_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cdij,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,ikjl->abij', amps.t2_11_b, amps.t2_11_b, ints.v_oooo_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,bl,iljk->abij', amps.t2_11_b, amps.t2_11_b, ints.v_oooo_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,dblj,ldkc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,bdjl,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,jkbc->abij', amps.t2_11_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,jbkc->abij', amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,dbli,ldkc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,bdil,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cj,bdil,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cj,ikbc->abij', amps.t2_11_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,ibkc->abij', amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,bdij,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,bdij,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cbli,jklc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cblj,iklc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bcil,jklc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,bcil,jlkc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,bcjl,iklc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bcjl,ilkc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cdij,kcbd->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,dalj,ldkc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,adjl,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,ci,adjl,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,ci,jkac->abij', amps.t2_11_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,jakc->abij', amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,dali,ldkc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_ab, ints.v_ovov_ab)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,adil,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,ikac->abij', amps.t2_11_b, amps.t2_11_b, ints.v_oovv_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,iakc->abij', amps.t2_11_b, amps.t2_11_b, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cl,adij,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cali,jklc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,calj,iklc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ooov_ba)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,acil,jklc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,acil,jlkc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,acjl,iklc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,acjl,ilkc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cdij,kcad->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ci,dk,abjl,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,dakj,kdbc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dbkj,kdac->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,abkl,jklc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,adjk,kcbd->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,adjk,kdbc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,bdjk,kcad->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,bdjk,kdac->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -1.0 * contract('cj,di,abkl,kcld->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += -2.0 * contract('cj,dk,abil,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,daki,kdbc->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,dbki,kdac->abij', amps.t2_11_b, amps.t2_21_ab, ints.v_ovvv_ab)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,abkl,iklc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,adik,kcbd->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,adik,kdbc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,bdik,kcad->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,bdik,kdac->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active:
        res += -2.0 * contract('ck,di,abjl,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active:
        res += 2.0 * contract('ck,dj,abil,kdlc->abij', amps.t2_11_b, amps.t2_11_b, amps.t2_20_bb, ints.v_ovov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,abil,jklc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,abil,jlkc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,abjl,iklc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,abjl,ilkc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ooov_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adij,kcbd->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,adij,kdbc->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bdij,kcad->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bdij,kdac->abij', amps.t2_11_b, amps.t2_21_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_12_a, amps.t2_20_bb, ints.v_ooov_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t2_12_a, amps.t2_20_bb, ints.v_ooov_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_12_a, amps.t2_20_bb, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t2_12_a, amps.t2_20_bb, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cbli,jklc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cblj,iklc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bcil,jklc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bcil,jlkc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,bcjl,iklc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ikjb->abij', amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ak,jkib->abij', amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cali,jklc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_12' in active:
        res += 1.0 * contract('bk,calj,iklc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ooov_ba)
    if 't2_12' in active:
        res += -1.0 * contract('bk,acil,jklc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,acjl,iklc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,acjl,ilkc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cdij,kcad->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ikja->abij', amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('bk,jkia->abij', amps.t2_12_b, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dakj,kdbc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dbkj,kdac->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += -1.0 * contract('ci,abkl,jklc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,adjk,kcbd->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ci,adjk,kdbc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bdjk,kdac->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ci,jabc->abij', amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ci,jbac->abij', amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,daki,kdbc->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += -1.0 * contract('cj,dbki,kdac->abij', amps.t2_12_b, amps.t2_20_ab, ints.v_ovvv_ab)
    if 't2_12' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,adik,kdbc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,bdik,kcad->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bdik,kdac->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('cj,iabc->abij', amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('cj,ibac->abij', amps.t2_12_b, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,abil,jlkc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,abjl,iklc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,abjl,ilkc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ooov_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adij,kcbd->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bdij,kcad->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bdij,kdac->abij', amps.t2_12_b, amps.t2_20_bb, ints.v_ovvv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('caki,dblj,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('caki,dblj,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('caki,bdjl,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cakj,dbli,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cakj,dbli,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cakj,bdil,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cakl,bdij,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cbki,dalj,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cbki,dalj,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cbki,adjl,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cbkj,dali,kcld->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += -1.0 * contract('cbkj,dali,kdlc->abij', amps.t2_20_ab, amps.t2_22_ab, ints.v_ovov_aa)
    if 't2_22' in active:
        res += 1.0 * contract('cbkj,adil,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cbkl,adij,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cdki,abjl,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cdkj,abil,kcld->abij', amps.t2_20_ab, amps.t2_22_bb, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('abik,cdlj,lckd->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('abik,cdjl,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('abjk,cdli,lckd->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('abjk,cdil,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 0.5 * contract('abkl,cdij,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('acij,dbkl,kdlc->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('acij,bdkl,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('acik,dblj,ldkc->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('acik,bdjl,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('acik,bdjl,kdlc->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('acjk,dbli,ldkc->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('acjk,bdil,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('acjk,bdil,kdlc->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('ackl,bdij,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bcij,dakl,kdlc->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bcij,adkl,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bcik,dalj,ldkc->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('bcik,adjl,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bcik,adjl,kdlc->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,dali,ldkc->abij', amps.t2_20_bb, amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,adil,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bcjk,adil,kdlc->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bckl,adij,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 0.5 * contract('cdij,abkl,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('cdik,abjl,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('cdjk,abil,kcld->abij', amps.t2_20_bb, amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('caki,dblj,kcld->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('caki,dblj,kdlc->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 2.0 * contract('caki,bdjl,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cakj,dbli,kcld->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cakj,dbli,kdlc->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -2.0 * contract('cakj,bdil,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 2.0 * contract('cakl,bdij,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -1.0 * contract('cbki,dalj,kcld->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 1.0 * contract('cbki,dalj,kdlc->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -2.0 * contract('cbki,adjl,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 1.0 * contract('cbkj,dali,kcld->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += -1.0 * contract('cbkj,dali,kdlc->abij', amps.t2_21_ab, amps.t2_21_ab, ints.v_ovov_aa)
    if 't2_21' in active:
        res += 2.0 * contract('cbkj,adil,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('cbkl,adij,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += 2.0 * contract('cdki,abjl,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('cdkj,abil,kcld->abij', amps.t2_21_ab, amps.t2_21_bb, ints.v_ovov_ab)
    if 't2_21' in active:
        res += -2.0 * contract('abik,cdjl,kcld->abij', amps.t2_21_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 2.0 * contract('abjk,cdil,kcld->abij', amps.t2_21_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 1.0 * contract('abkl,cdij,kcld->abij', amps.t2_21_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -2.0 * contract('acij,bdkl,kcld->abij', amps.t2_21_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 2.0 * contract('acik,bdjl,kcld->abij', amps.t2_21_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -2.0 * contract('acik,bdjl,kdlc->abij', amps.t2_21_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -2.0 * contract('acjk,bdil,kcld->abij', amps.t2_21_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += 2.0 * contract('acjk,bdil,kdlc->abij', amps.t2_21_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_21' in active:
        res += -2.0 * contract('ackl,bdij,kcld->abij', amps.t2_21_bb, amps.t2_21_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('caki,kcjb->abij', amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cakj,kcib->abij', amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += -1.0 * contract('cbki,kcja->abij', amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 1.0 * contract('cbkj,kcia->abij', amps.t2_22_ab, ints.v_ovov_ab)
    if 't2_22' in active:
        res += 2.0 * w * contract('abij->abij', amps.t2_22_bb)
    if 't2_22' in active:
        res += 1.0 * contract('abkl,ikjl->abij', amps.t2_22_bb, ints.v_oooo_bb)
    if 't2_22' in active:
        res += -1.0 * contract('acik,jkbc->abij', amps.t2_22_bb, ints.v_oovv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('acik,jbkc->abij', amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('acjk,ikbc->abij', amps.t2_22_bb, ints.v_oovv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('acjk,ibkc->abij', amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bcik,jkac->abij', amps.t2_22_bb, ints.v_oovv_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bcik,jakc->abij', amps.t2_22_bb, ints.v_ovov_bb)
    if 't2_22' in active:
        res += -1.0 * contract('bcjk,ikac->abij', amps.t2_22_bb, ints.v_oovv_bb)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,iakc->abij', amps.t2_22_bb, ints.v_ovov_bb)

    # exact antisymmetry of the same-spin residual; enforce against rounding drift
    res = 0.5 * (res - res.transpose(1, 0, 2, 3))
    res = 0.5 * (res - res.transpose(0, 1, 3, 2))
    nocc_j = ints.eps_occ_b.size
    for j in range(nocc_j):
        res[:, :, :, j] *= 1.0 / (ints.eps_occ_b[None, None, :] + ints.eps_occ_b[j]
                                  - ints.eps_vir_b[:, None, None]
                                  - ints.eps_vir_b[None, :, None] - (2.0 * w))
    res += amps.t2_22_bb
    return res

# ---------------------------------------------------------------------------
# Spin-resolved integral setup (built once per run)
# ---------------------------------------------------------------------------
class _Ints:
    """Namespace for spin-blocked Fock/dipole slices, all-virtual DF
    slices, orbital energies and 4-index chemist blocks."""


class _Amps:
    """Namespace for the blocked amplitudes."""


def _build_ints(qedhf, frozen=0):
    if 'Ca' not in qedhf:
        raise NotImplementedError(
            "qed_ccsd_uhf requires a QED-UHF reference dict "
            "(OmegaQMC.addons.qed_uhf.run_qed_uhf); use qed_ccsd_rhf for "
            "closed-shell references")
    omega = qedhf['omega']
    lambda_x, lambda_y, lambda_z = qedhf['lambda_cav']
    mu = (qedhf['mu_x_ao'], qedhf['mu_y_ao'], qedhf['mu_z_ao'])
    cf = (lambda_x * math.sqrt(omega / 2.0),
          lambda_y * math.sqrt(omega / 2.0),
          lambda_z * math.sqrt(omega / 2.0))
    B_ao = _ao_df_factor(qedhf)

    ints = _Ints()
    nocc = {}
    nvir = {}
    B_mo = {}
    for s, (C, F) in (('a', (qedhf['Ca'], qedhf['Fa'])),
                      ('b', (qedhf['Cb'], qedhf['Fb']))):
        C = np.asarray(C)
        f = C.T @ np.asarray(F) @ C
        dip = sum(cf[k] * (C.T @ mu[k] @ C) for k in range(3))
        B = contract('pi,Ppq,qj->Pij', C, B_ao, C)
        no = qedhf[f'nocc_{s}']
        if frozen:
            f = f[frozen:, frozen:]
            dip = dip[frozen:, frozen:]
            B = B[:, frozen:, frozen:]
            no -= frozen
        nmo = f.shape[0]
        o = slice(None, no)
        v = slice(no, None)
        nocc[s] = no
        nvir[s] = nmo - no
        eps = f.diagonal()
        setattr(ints, f'eps_occ_{s}', np.ascontiguousarray(eps[o]))
        setattr(ints, f'eps_vir_{s}', np.ascontiguousarray(eps[v]))
        for c1, sl1 in (('o', o), ('v', v)):
            for c2, sl2 in (('o', o), ('v', v)):
                setattr(ints, f'f_{s}_{c1}{c2}',
                        np.ascontiguousarray(f[sl1, sl2]))
                setattr(ints, f'd_{s}_{c1}{c2}',
                        -np.ascontiguousarray(dip[sl1, sl2]))
        setattr(ints, f'B_vv_{s}', np.ascontiguousarray(B[:, v, v]))
        B_mo[s] = (B, o, v)

    # 4-index chemist blocks (pq|rs)_{s1 s2}; (vv|vv) intentionally absent
    for name in _NEEDED_V:
        c12, sp = name[2:].split('_')
        c1, c2 = c12[:2], c12[2:]
        B1, o1, v1 = B_mo[sp[0]]
        B2, o2, v2 = B_mo[sp[1]]
        sl = {'o': {0: o1, 1: o2}, 'v': {0: v1, 1: v2}}
        setattr(ints, name, contract(
            'xpq,xrs->pqrs',
            B1[:, sl[c1[0]][0], sl[c1[1]][0]],
            B2[:, sl[c2[0]][1], sl[c2[1]][1]]))

    return ints, nocc, nvir


_BLOCKS_2 = ('a', 'b')
_BLOCKS_4 = ('aa', 'ab', 'bb')


# ---------------------------------------------------------------------------
# QED-CCSD driver (spin-blocked, open-shell references)
# ---------------------------------------------------------------------------
def run_qed_ccsd(qedhf, do_t1_01=True, do_t2_11=True, do_t2_21=True,
                 do_t2_02=False, do_t2_12=False, do_t2_22=False,
                 frozen=0, max_iter=50, tol=1e-8, tol_amp=1e-7,
                 max_diis=8, diis_on_disk=False, verbose=True):
    """Spin-blocked QED-CCSD on a QED-UHF reference (open shells).

    Same flavour flags, convergence criteria and return-dict layout as
    qed_ccsd_rhf.run_qed_ccsd; amplitudes are returned as spin blocks
    (keys like 't2_20_ab'). ``frozen`` drops the lowest ``frozen``
    spatial orbitals of both spins (doubly-occupied cores).
    """
    omega = qedhf['omega']
    lambda_x, lambda_y, lambda_z = qedhf['lambda_cav']
    E_qed_hf_ref = qedhf['E_qed_uhf']
    E_hf_ref = qedhf.get('E_uhf')

    t_setup = time.time()
    ints, nocc, nvir = _build_ints(qedhf, frozen=frozen)
    t_setup = time.time() - t_setup

    active = set()
    for flag, name in ((do_t1_01, 't1_01'), (do_t2_02, 't2_02'),
                       (do_t2_11, 't2_11'), (do_t2_21, 't2_21'),
                       (do_t2_12, 't2_12'), (do_t2_22, 't2_22')):
        if flag:
            active.add(name)
    active = frozenset(active)

    if verbose:
        flavour = "QED-CCSD"
        if not active:
            flavour = "conventional CCSD"
        elif active == {'t1_01', 't2_11', 't2_21'}:
            flavour = "QED-CCSD-1 (QED-CCSD-21, Deprince)"
        elif active == {'t1_01', 't2_11', 't2_02', 't2_12'}:
            flavour = "QED-CCSD-White (QED-CCSD-12)"
        elif active == _ALL_AMPS:
            flavour = "QED-CCSD-Full (QED-CCSD-22)"
        naux = ints.B_vv_a.shape[0]
        print(f"\nQED-CCSD (spin-blocked UHF, DF): flavour = {flavour}")
        print(f"  nocc = ({nocc['a']},{nocc['b']}),"
              f" nvir = ({nvir['a']},{nvir['b']}) (spatial),"
              f" naux = {naux}, frozen = {frozen},"
              f" setup {t_setup:.1f} s")
        print(f"  omega = {omega:.6f} Ha,"
              f"  lambda = ({lambda_x},{lambda_y},{lambda_z})")

    amps = _Amps()
    amps.t1_01 = 0.0
    amps.t2_02 = 0.0
    active_arrays = []
    for base, blocks, always in (('t1_10', _BLOCKS_2, True),
                                 ('t2_20', _BLOCKS_4, True),
                                 ('t2_11', _BLOCKS_2, do_t2_11),
                                 ('t2_21', _BLOCKS_4, do_t2_21),
                                 ('t2_12', _BLOCKS_2, do_t2_12),
                                 ('t2_22', _BLOCKS_4, do_t2_22)):
        for blk in blocks:
            if not always:
                shape = (1, 1) if len(blk) == 1 else (1, 1, 1, 1)
            elif len(blk) == 1:
                shape = (nvir[blk], nocc[blk])
            else:
                shape = (nvir[blk[0]], nvir[blk[1]],
                         nocc[blk[0]], nocc[blk[1]])
            setattr(amps, f'{base}_{blk}', np.zeros(shape))
            if always:
                active_arrays.append(f'{base}_{blk}')

    hist = _DiisHistory(active_arrays, max_diis, on_disk=diis_on_disk)
    hist.vals.append({k: hist._store(getattr(amps, k), f'val0_{k}')
                      for k in active_arrays})

    kernels = []
    for blk in _BLOCKS_2:
        kernels.append((f't1_10_{blk}', globals()[f'ccsd_t1_10_{blk}'],
                        True))
    kernels.append(('t1_01', ccsd_t1_01, do_t1_01))
    for blk in _BLOCKS_4:
        kernels.append((f't2_20_{blk}', globals()[f'ccsd_t2_20_{blk}'],
                        True))
    kernels.append(('t2_02', ccsd_t2_02, do_t2_02))
    for blk in _BLOCKS_2:
        kernels.append((f't2_11_{blk}', globals()[f'ccsd_t2_11_{blk}'],
                        do_t2_11))
    for blk in _BLOCKS_4:
        kernels.append((f't2_21_{blk}', globals()[f'ccsd_t2_21_{blk}'],
                        do_t2_21))
    for blk in _BLOCKS_2:
        kernels.append((f't2_12_{blk}', globals()[f'ccsd_t2_12_{blk}'],
                        do_t2_12))
    for blk in _BLOCKS_4:
        kernels.append((f't2_22_{blk}', globals()[f'ccsd_t2_22_{blk}'],
                        do_t2_22))

    E_CCSD_old = 0.0
    E_CCSD_new = 0.0
    time_total = 0.0
    converged = False
    ccsd_iter = 0

    if verbose:
        print('\nStarting QED-CCSD iteration:')
        print('Iter   E(QED-CCSD corr)        |dE|         |dT|'
              '        time (s)')

    try:
        for ccsd_iter in range(1, max_iter + 1):
            t_start = time.time()

            old_t1_01 = float(amps.t1_01)
            old_t2_02 = float(amps.t2_02)
            err = {}
            step_sq = 0.0

            for name, fn, enabled in kernels:
                if not enabled:
                    continue
                new = fn(ints, omega, amps, active=active)
                if name in ('t1_01', 't2_02'):
                    setattr(amps, name, new)
                    continue
                e = new - getattr(amps, name)
                err[name] = e
                step_sq += float(np.dot(e.ravel(), e.ravel()))
                setattr(amps, name, new)

            E_CCSD_new = ccsd_energy(ints, omega, amps, active=active)

            amp_norm = float(np.sqrt(
                step_sq
                + (float(amps.t1_01) - old_t1_01) ** 2
                + (float(amps.t2_02) - old_t2_02) ** 2))

            t_total = time.time() - t_start
            time_total += t_total
            if verbose:
                print('%3d:  %20.12f  %1.5E  %1.5E   %.3f'
                      % (ccsd_iter, E_CCSD_new,
                         abs(E_CCSD_new - E_CCSD_old), amp_norm, t_total))

            if abs(E_CCSD_new - E_CCSD_old) < tol and amp_norm < tol_amp:
                converged = True
                break

            hist.append({k: getattr(amps, k) for k in active_arrays}, err)
            del err
            E_CCSD_old = E_CCSD_new

            new_amps = hist.extrapolate()
            if new_amps is not None:
                for k in active_arrays:
                    setattr(amps, k, new_amps[k])
        else:
            if verbose:
                print('Warning: QED-CCSD did not converge in %d iterations'
                      % max_iter)
    finally:
        hist.cleanup()

    out = {
        'converged': bool(converged),
        'E_qed_ccsd_corr': float(E_CCSD_new),
        'E_qed_ccsd_total': float(E_CCSD_new) + E_qed_hf_ref,
        'E_qed_uhf': E_qed_hf_ref,
        'E_uhf': E_hf_ref,
        'frozen': frozen,
        't1_01': amps.t1_01,
        't2_02': amps.t2_02,
        'iterations': ccsd_iter,
        'time_total': time_total,
    }
    for base, blocks in (('t1_10', _BLOCKS_2), ('t2_20', _BLOCKS_4),
                         ('t2_11', _BLOCKS_2), ('t2_21', _BLOCKS_4),
                         ('t2_12', _BLOCKS_2), ('t2_22', _BLOCKS_4)):
        for blk in blocks:
            out[f'{base}_{blk}'] = getattr(amps, f'{base}_{blk}')
    return out


# ---------------------------------------------------------------------------
# Demo: OH radical / STO-3G QED-CCSD-21
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from .qed_uhf import run_qed_uhf
    mol = gto.M(atom="O 0 0 0; H 0 0 0.9697", basis='sto-3g',
                spin=1, verbose=0)
    omega = 3.0 / 27.211386245988
    qeduhf = run_qed_uhf(mol, omega, (0.0, 0.0, 0.05), verbose=False)
    print(f"E_QED_UHF = {qeduhf['E_qed_uhf']:.12f}")
    result = run_qed_ccsd(qeduhf, do_t1_01=True, do_t2_11=True,
                          do_t2_21=True, verbose=True)
    print(f"\nE_QED_CCSD total = {result['E_qed_ccsd_total']:.12f}")
