"""
Closed-shell (spin-adapted) QED-CCSD with density-fitted integrals and
precomputed integral-block intermediates.

Closed-shell backend of :mod:`OmegaQMC.addons.qed_ccsd` (whose
run_qed_ccsd dispatches here for restricted QED-HF references). The
residual equations were obtained by exactly spin-tracing the Wick-derived
spin-orbital QED-CCSD equations with a density-fitted two-electron
operator (see tools/qed_ccsd_df_derivation/): the doubles-type
spatial amplitudes are the mixed-spin blocks

    T[a,b,i,j] := t2_so[a-alpha, b-beta, i-alpha, j-beta],

singles-type amplitudes are the alpha blocks, and each residual computed
here is the corresponding block of the spin-orbital residual — so
denominators, DIIS and convergence behaviour carry over unchanged, while
every tensor index runs over spatial orbitals (16x fewer flops, 16x less
amplitude memory than the spin-orbital code).

Intermediate reuse: all two-electron contractions except the all-virtual
ladder use 4-index chemist blocks

    (pq|rs) = sum_x B[x,p,q] B[x,r,s]   (DSE folded into B)

that are built ONCE at setup — (oo|oo), (oo|ov), (oo|vv), (ov|ov),
(ov|vv) — instead of re-contracting the 3-index factor for every term of
every iteration (the dominant cost of the term-by-term spin-orbital
code). The (vv|vv) block is never formed: ladder terms are grouped per
kernel and evaluated by the batched :func:`_vvvv_ladder` with peak
intermediate ``nvir**3``.

The equations are complete: they include the quartic (4-amplitude) BCH
terms of the t2_21/t2_22 residuals that the published reference
implementation lacks, so converged QED-CCSD-21 energies differ from it
at the ~1e-7 Ha level (glycolaldehyde/STO-3G, lambda=0.1:
-262.416985787 here vs the published -262.416986187). Every kernel was
validated to machine precision against the spin-orbital equations
(tools/qed_ccsd_df_derivation/validate_rhf_kernels.py).

For naphthalene/cc-pVDZ (frozen core) the entire working set is
~1.5 GB and an iteration takes minutes, versus ~8 GB and ~an hour for
a spin-orbital implementation on the same machine.
"""

import math
import time

import numpy as np
from opt_einsum import contract

from pyscf import gto

from .qed_hf import run_qed_hf
from .qed_ccsd_utils import _DiisHistory, _vvvv_ladder, _ao_df_factor

_ALL_AMPS = frozenset(
    ("t1_01", "t2_02", "t2_11", "t2_21", "t2_12", "t2_22"))


_NEEDED_V = ['v_oooo', 'v_ooov', 'v_oovv', 'v_ovov', 'v_ovvv']
_NEEDED_SLICES = ['B_vv', 'd_oo', 'd_vo', 'd_vv', 'f_oo', 'f_vo', 'f_vv']


def ccsd_energy(ints, w, t1_10, t1_01, t2_20, t2_02,
                t2_11, t2_21, t2_12, t2_22, active=_ALL_AMPS):
    nocc = t2_20.shape[2]
    nvir = t2_20.shape[0]
    res = 0.0

    if 't1_01' in active:
        res += 2.0 * t1_01 * contract('ck,ck->', ints.d_vo, t1_10)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ck->', ints.d_vo, t2_11)
    res += 2.0 * contract('ck,ck->', ints.f_vo, t1_10)
    res += 2.0 * contract('ck,dl,kcld->', t1_10, t1_10, ints.v_ovov)
    res += -1.0 * contract('ck,dl,kdlc->', t1_10, t1_10, ints.v_ovov)
    res += 2.0 * contract('cdkl,kcld->', t2_20, ints.v_ovov)
    res += -1.0 * contract('cdkl,kdlc->', t2_20, ints.v_ovov)

    return float(res)

def ccsd_t1_10(ints, w, t1_10, t1_01, t2_20, t2_02,
               t2_11, t2_21, t2_12, t2_22, active=_ALL_AMPS):
    nocc = t2_20.shape[2]
    nvir = t2_20.shape[0]
    res = np.zeros((nvir, nocc))

    if 't1_01' in active:
        res += 1.0 * t1_01 * contract('ac,ci->ai', ints.d_vv, t1_10)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci->ai', ints.d_vv, t2_11)
    if 't1_01' in active:
        res += 1.0 * t1_01 * contract('ai->ai', ints.d_vo)
    if 't1_01' in active:
        res += -1.0 * t1_01 * contract('ck,ak,ci->ai', ints.d_vo, t1_10, t1_10)
    if 't1_01' in active:
        res += 2.0 * t1_01 * contract('ck,acik->ai', ints.d_vo, t2_20)
    if 't1_01' in active:
        res += -1.0 * t1_01 * contract('ck,acki->ai', ints.d_vo, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.d_vo, t1_10, t2_11)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.d_vo, t1_10, t2_11)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ck,ai->ai', ints.d_vo, t1_10, t2_11)
    if 't2_21' in active:
        res += 2.0 * contract('ck,acik->ai', ints.d_vo, t2_21)
    if 't2_21' in active:
        res += -1.0 * contract('ck,acki->ai', ints.d_vo, t2_21)
    if 't1_01' in active:
        res += -1.0 * t1_01 * contract('ik,ak->ai', ints.d_oo, t1_10)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak->ai', ints.d_oo, t2_11)
    res += 1.0 * contract('ac,ci->ai', ints.f_vv, t1_10)
    res += 1.0 * contract('ai->ai', ints.f_vo)
    res += -1.0 * contract('ck,ak,ci->ai', ints.f_vo, t1_10, t1_10)
    res += 2.0 * contract('ck,acik->ai', ints.f_vo, t2_20)
    res += -1.0 * contract('ck,acki->ai', ints.f_vo, t2_20)
    res += -1.0 * contract('ik,ak->ai', ints.f_oo, t1_10)
    res += -2.0 * contract('ak,ci,dl,kcld->ai', t1_10, t1_10, t1_10, ints.v_ovov)
    res += 1.0 * contract('ak,cl,di,kcld->ai', t1_10, t1_10, t1_10, ints.v_ovov)
    res += -2.0 * contract('ak,cl,iklc->ai', t1_10, t1_10, ints.v_ooov)
    res += 1.0 * contract('ak,cl,ilkc->ai', t1_10, t1_10, ints.v_ooov)
    res += -2.0 * contract('ak,cdil,kcld->ai', t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('ak,cdil,kdlc->ai', t1_10, t2_20, ints.v_ovov)
    res += -1.0 * contract('ci,dk,kcad->ai', t1_10, t1_10, ints.v_ovvv)
    res += -2.0 * contract('ci,adkl,kcld->ai', t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('ci,adkl,kdlc->ai', t1_10, t2_20, ints.v_ovov)
    res += 2.0 * contract('ck,di,kcad->ai', t1_10, t1_10, ints.v_ovvv)
    res += 4.0 * contract('ck,adil,kcld->ai', t1_10, t2_20, ints.v_ovov)
    res += -2.0 * contract('ck,adil,kdlc->ai', t1_10, t2_20, ints.v_ovov)
    res += -2.0 * contract('ck,adli,kcld->ai', t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('ck,adli,kdlc->ai', t1_10, t2_20, ints.v_ovov)
    res += -1.0 * contract('ck,ikac->ai', t1_10, ints.v_oovv)
    res += 2.0 * contract('ck,iakc->ai', t1_10, ints.v_ovov)
    res += -2.0 * contract('ackl,iklc->ai', t2_20, ints.v_ooov)
    res += 1.0 * contract('ackl,ilkc->ai', t2_20, ints.v_ooov)
    res += -1.0 * contract('cdik,kcad->ai', t2_20, ints.v_ovvv)
    res += 2.0 * contract('cdik,kdac->ai', t2_20, ints.v_ovvv)

    e_denom = 1.0 / (ints.eps_occ[None, :] - ints.eps_vir[:, None])
    return t1_10 + res * e_denom

def ccsd_t1_01(ints, w, t1_10, t1_01, t2_20, t2_02,
               t2_11, t2_21, t2_12, t2_22, active=_ALL_AMPS):
    nocc = t2_20.shape[2]
    nvir = t2_20.shape[0]
    res = 0.0

    if 't1_01' in active and 't2_11' in active:
        res += 2.0 * t1_01 * contract('ck,ck->', ints.d_vo, t2_11)
    res += 2.0 * contract('ck,ck->', ints.d_vo, t1_10)
    if 't2_02' in active:
        res += 2.0 * t2_02 * contract('ck,ck->', ints.d_vo, t1_10)
    if 't2_12' in active:
        res += 2.0 * contract('ck,ck->', ints.d_vo, t2_12)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ck->', ints.f_vo, t2_11)
    if 't1_01' in active:
        res += 1.0 * t1_01 * w
    if 't2_11' in active:
        res += 4.0 * contract('ck,dl,kcld->', t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,dl,kdlc->', t1_10, t2_11, ints.v_ovov)
    if 't2_21' in active:
        res += 2.0 * contract('cdkl,kcld->', t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -1.0 * contract('cdkl,kdlc->', t2_21, ints.v_ovov)

    if w == 0:
        return 0.0
    return t1_01 - res / w

def ccsd_t2_20(ints, w, t1_10, t1_01, t2_20, t2_02,
               t2_11, t2_21, t2_12, t2_22, active=_ALL_AMPS):
    nocc = t2_20.shape[2]
    nvir = t2_20.shape[0]
    res = np.zeros((nvir, nvir, nocc, nocc))

    res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv, ints.B_vv, t1_10, t1_10)
    if 't1_01' in active:
        res += 1.0 * t1_01 * contract('ac,bcji->abij', ints.d_vv, t2_20)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci,bj->abij', ints.d_vv, t1_10, t2_11)
    if 't2_21' in active:
        res += 1.0 * contract('ac,bcji->abij', ints.d_vv, t2_21)
    if 't2_11' in active:
        res += 1.0 * contract('ai,bj->abij', ints.d_vo, t2_11)
    if 't1_01' in active:
        res += 1.0 * t1_01 * contract('bc,acij->abij', ints.d_vv, t2_20)
    if 't2_11' in active:
        res += 1.0 * contract('bc,cj,ai->abij', ints.d_vv, t1_10, t2_11)
    if 't2_21' in active:
        res += 1.0 * contract('bc,acij->abij', ints.d_vv, t2_21)
    if 't2_11' in active:
        res += 1.0 * contract('bj,ai->abij', ints.d_vo, t2_11)
    if 't1_01' in active:
        res += -1.0 * t1_01 * contract('ck,ak,bcji->abij', ints.d_vo, t1_10, t2_20)
    if 't1_01' in active:
        res += -1.0 * t1_01 * contract('ck,bk,acij->abij', ints.d_vo, t1_10, t2_20)
    if 't1_01' in active:
        res += -1.0 * t1_01 * contract('ck,ci,abkj->abij', ints.d_vo, t1_10, t2_20)
    if 't1_01' in active:
        res += -1.0 * t1_01 * contract('ck,cj,abik->abij', ints.d_vo, t1_10, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci,bj->abij', ints.d_vo, t1_10, t1_10, t2_11)
    if 't2_21' in active:
        res += -1.0 * contract('ck,ak,bcji->abij', ints.d_vo, t1_10, t2_21)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,cj,ai->abij', ints.d_vo, t1_10, t1_10, t2_11)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_vo, t1_10, t2_21)
    if 't2_21' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.d_vo, t1_10, t2_21)
    if 't2_21' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_vo, t1_10, t2_21)
    if 't2_21' in active:
        res += 2.0 * contract('ck,ck,abij->abij', ints.d_vo, t1_10, t2_21)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ai,bcjk->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ai,bckj->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,bcji->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += 2.0 * contract('ck,bj,acik->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bj,acki->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_vo, t2_11, t2_20)
    if 't1_01' in active:
        res += -1.0 * t1_01 * contract('ik,abkj->abij', ints.d_oo, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_oo, t1_10, t2_11)
    if 't2_21' in active:
        res += -1.0 * contract('ik,abkj->abij', ints.d_oo, t2_21)
    if 't1_01' in active:
        res += -1.0 * t1_01 * contract('jk,abik->abij', ints.d_oo, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('jk,bk,ai->abij', ints.d_oo, t1_10, t2_11)
    if 't2_21' in active:
        res += -1.0 * contract('jk,abik->abij', ints.d_oo, t2_21)
    res += 1.0 * contract('ac,bcji->abij', ints.f_vv, t2_20)
    res += 1.0 * contract('bc,acij->abij', ints.f_vv, t2_20)
    res += -1.0 * contract('ck,ak,bcji->abij', ints.f_vo, t1_10, t2_20)
    res += -1.0 * contract('ck,bk,acij->abij', ints.f_vo, t1_10, t2_20)
    res += -1.0 * contract('ck,ci,abkj->abij', ints.f_vo, t1_10, t2_20)
    res += -1.0 * contract('ck,cj,abik->abij', ints.f_vo, t1_10, t2_20)
    res += -1.0 * contract('ik,abkj->abij', ints.f_oo, t2_20)
    res += -1.0 * contract('jk,abik->abij', ints.f_oo, t2_20)
    res += 1.0 * contract('ak,bl,ci,jlkc->abij', t1_10, t1_10, t1_10, ints.v_ooov)
    res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', t1_10, t1_10, t1_10, t1_10, ints.v_ovov)
    res += 1.0 * contract('ak,bl,cj,iklc->abij', t1_10, t1_10, t1_10, ints.v_ooov)
    res += 1.0 * contract('ak,bl,cdij,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('ak,bl,ikjl->abij', t1_10, t1_10, ints.v_oooo)
    res += -1.0 * contract('ak,ci,dj,kcbd->abij', t1_10, t1_10, t1_10, ints.v_ovvv)
    res += -2.0 * contract('ak,ci,bdjl,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('ak,ci,bdlj,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += -1.0 * contract('ak,ci,jbkc->abij', t1_10, t1_10, ints.v_ovov)
    res += 1.0 * contract('ak,cj,bdli,kdlc->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += -1.0 * contract('ak,cj,ikbc->abij', t1_10, t1_10, ints.v_oovv)
    res += 1.0 * contract('ak,cl,bdji,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += -2.0 * contract('ak,cl,bdji,kdlc->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += -2.0 * contract('ak,bcjl,iklc->abij', t1_10, t2_20, ints.v_ooov)
    res += 1.0 * contract('ak,bcjl,ilkc->abij', t1_10, t2_20, ints.v_ooov)
    res += 1.0 * contract('ak,bcli,jlkc->abij', t1_10, t2_20, ints.v_ooov)
    res += 1.0 * contract('ak,bclj,iklc->abij', t1_10, t2_20, ints.v_ooov)
    res += -1.0 * contract('ak,cdij,kcbd->abij', t1_10, t2_20, ints.v_ovvv)
    res += -1.0 * contract('ak,ikjb->abij', t1_10, ints.v_ooov)
    res += 1.0 * contract('bk,ci,adlj,kdlc->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += -1.0 * contract('bk,ci,jkac->abij', t1_10, t1_10, ints.v_oovv)
    res += -1.0 * contract('bk,cj,di,kcad->abij', t1_10, t1_10, t1_10, ints.v_ovvv)
    res += -2.0 * contract('bk,cj,adil,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('bk,cj,adil,kdlc->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('bk,cj,adli,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += -1.0 * contract('bk,cj,iakc->abij', t1_10, t1_10, ints.v_ovov)
    res += 1.0 * contract('bk,cl,adij,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += -2.0 * contract('bk,cl,adij,kdlc->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += -2.0 * contract('bk,acil,jklc->abij', t1_10, t2_20, ints.v_ooov)
    res += 1.0 * contract('bk,acil,jlkc->abij', t1_10, t2_20, ints.v_ooov)
    res += 1.0 * contract('bk,acli,jklc->abij', t1_10, t2_20, ints.v_ooov)
    res += 1.0 * contract('bk,aclj,ilkc->abij', t1_10, t2_20, ints.v_ooov)
    res += -1.0 * contract('bk,cdij,kdac->abij', t1_10, t2_20, ints.v_ovvv)
    res += -1.0 * contract('bk,jkia->abij', t1_10, ints.v_ooov)
    res += 1.0 * contract('ci,dk,ablj,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('ci,abkl,jlkc->abij', t1_10, t2_20, ints.v_ooov)
    res += -1.0 * contract('ci,adkj,kcbd->abij', t1_10, t2_20, ints.v_ovvv)
    res += -1.0 * contract('ci,bdjk,kcad->abij', t1_10, t2_20, ints.v_ovvv)
    res += 2.0 * contract('ci,bdjk,kdac->abij', t1_10, t2_20, ints.v_ovvv)
    res += -1.0 * contract('ci,bdkj,kdac->abij', t1_10, t2_20, ints.v_ovvv)
    res += 1.0 * contract('ci,jbac->abij', t1_10, ints.v_ovvv)
    res += 1.0 * contract('cj,di,abkl,kdlc->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('cj,dk,abil,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('cj,abkl,iklc->abij', t1_10, t2_20, ints.v_ooov)
    res += -1.0 * contract('cj,adik,kcbd->abij', t1_10, t2_20, ints.v_ovvv)
    res += 2.0 * contract('cj,adik,kdbc->abij', t1_10, t2_20, ints.v_ovvv)
    res += -1.0 * contract('cj,adki,kdbc->abij', t1_10, t2_20, ints.v_ovvv)
    res += -1.0 * contract('cj,bdki,kcad->abij', t1_10, t2_20, ints.v_ovvv)
    res += 1.0 * contract('cj,iabc->abij', t1_10, ints.v_ovvv)
    res += -2.0 * contract('ck,di,ablj,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += -2.0 * contract('ck,dj,abil,kcld->abij', t1_10, t1_10, t2_20, ints.v_ovov)
    res += 1.0 * contract('ck,abil,jklc->abij', t1_10, t2_20, ints.v_ooov)
    res += -2.0 * contract('ck,abil,jlkc->abij', t1_10, t2_20, ints.v_ooov)
    res += 1.0 * contract('ck,ablj,iklc->abij', t1_10, t2_20, ints.v_ooov)
    res += -2.0 * contract('ck,ablj,ilkc->abij', t1_10, t2_20, ints.v_ooov)
    res += 2.0 * contract('ck,adij,kcbd->abij', t1_10, t2_20, ints.v_ovvv)
    res += -1.0 * contract('ck,adij,kdbc->abij', t1_10, t2_20, ints.v_ovvv)
    res += 2.0 * contract('ck,bdji,kcad->abij', t1_10, t2_20, ints.v_ovvv)
    res += -1.0 * contract('ck,bdji,kdac->abij', t1_10, t2_20, ints.v_ovvv)
    res += -2.0 * contract('abik,cdjl,kcld->abij', t2_20, t2_20, ints.v_ovov)
    res += 1.0 * contract('abik,cdjl,kdlc->abij', t2_20, t2_20, ints.v_ovov)
    res += -2.0 * contract('abkj,cdil,kcld->abij', t2_20, t2_20, ints.v_ovov)
    res += 1.0 * contract('abkj,cdil,kdlc->abij', t2_20, t2_20, ints.v_ovov)
    res += 1.0 * contract('abkl,cdij,kcld->abij', t2_20, t2_20, ints.v_ovov)
    res += 1.0 * contract('abkl,ikjl->abij', t2_20, ints.v_oooo)
    res += -2.0 * contract('acij,bdkl,kcld->abij', t2_20, t2_20, ints.v_ovov)
    res += 1.0 * contract('acij,bdkl,kdlc->abij', t2_20, t2_20, ints.v_ovov)
    res += 4.0 * contract('acik,bdjl,kcld->abij', t2_20, t2_20, ints.v_ovov)
    res += -2.0 * contract('acik,bdjl,kdlc->abij', t2_20, t2_20, ints.v_ovov)
    res += -2.0 * contract('acik,bdlj,kcld->abij', t2_20, t2_20, ints.v_ovov)
    res += 1.0 * contract('acik,bdlj,kdlc->abij', t2_20, t2_20, ints.v_ovov)
    res += -1.0 * contract('acik,jkbc->abij', t2_20, ints.v_oovv)
    res += 2.0 * contract('acik,jbkc->abij', t2_20, ints.v_ovov)
    res += -2.0 * contract('acki,bdjl,kcld->abij', t2_20, t2_20, ints.v_ovov)
    res += 1.0 * contract('acki,bdjl,kdlc->abij', t2_20, t2_20, ints.v_ovov)
    res += 1.0 * contract('acki,bdlj,kcld->abij', t2_20, t2_20, ints.v_ovov)
    res += -1.0 * contract('acki,jbkc->abij', t2_20, ints.v_ovov)
    res += 1.0 * contract('ackj,bdli,kdlc->abij', t2_20, t2_20, ints.v_ovov)
    res += -1.0 * contract('ackj,ikbc->abij', t2_20, ints.v_oovv)
    res += 1.0 * contract('ackl,bdji,kcld->abij', t2_20, t2_20, ints.v_ovov)
    res += -2.0 * contract('ackl,bdji,kdlc->abij', t2_20, t2_20, ints.v_ovov)
    res += -1.0 * contract('bcjk,ikac->abij', t2_20, ints.v_oovv)
    res += 2.0 * contract('bcjk,iakc->abij', t2_20, ints.v_ovov)
    res += -1.0 * contract('bcki,jkac->abij', t2_20, ints.v_oovv)
    res += -1.0 * contract('bckj,iakc->abij', t2_20, ints.v_ovov)
    res += 1.0 * contract('iajb->abij', ints.v_ovov)
    _W = 1.0 * t2_20
    _vvvv_ladder(ints.B_vv, np.ascontiguousarray(_W), out=res, alpha=1.0)

    # exact (ab)(ij) pair symmetry of the mixed-spin block; enforce against rounding drift
    res = 0.5 * (res + res.transpose(1, 0, 3, 2))
    for i in range(nocc):
        res[:, :, i, :] *= 1.0 / (ints.eps_occ[i] + ints.eps_occ[None, None, :]
                                  - ints.eps_vir[:, None, None]
                                  - ints.eps_vir[None, :, None] - (0.0))
    res += t2_20
    return res

def ccsd_t2_02(ints, w, t1_10, t1_01, t2_20, t2_02,
               t2_11, t2_21, t2_12, t2_22, active=_ALL_AMPS):
    nocc = t2_20.shape[2]
    nvir = t2_20.shape[0]
    res = 0.0

    if 't1_01' in active and 't2_12' in active:
        res += 2.0 * t1_01 * contract('ck,ck->', ints.d_vo, t2_12)
    if 't2_02' in active and 't2_11' in active:
        res += 4.0 * t2_02 * contract('ck,ck->', ints.d_vo, t2_11)
    if 't2_11' in active:
        res += 4.0 * contract('ck,ck->', ints.d_vo, t2_11)
    if 't2_12' in active:
        res += 2.0 * contract('ck,ck->', ints.f_vo, t2_12)
    if 't2_12' in active:
        res += 4.0 * contract('ck,dl,kcld->', t1_10, t2_12, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,dl,kdlc->', t1_10, t2_12, ints.v_ovov)
    if 't2_02' in active:
        res += 2.0 * t2_02 * w
    if 't2_11' in active:
        res += 4.0 * contract('ck,dl,kcld->', t2_11, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,dl,kdlc->', t2_11, t2_11, ints.v_ovov)
    if 't2_22' in active:
        res += 2.0 * contract('cdkl,kcld->', t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -1.0 * contract('cdkl,kdlc->', t2_22, ints.v_ovov)

    if w == 0:
        return 0.0
    return t2_02 - res / (2.0 * w)

def ccsd_t2_11(ints, w, t1_10, t1_01, t2_20, t2_02,
               t2_11, t2_21, t2_12, t2_22, active=_ALL_AMPS):
    nocc = t2_20.shape[2]
    nvir = t2_20.shape[0]
    res = np.zeros((nvir, nocc))

    if 't1_01' in active and 't2_11' in active:
        res += 1.0 * t1_01 * contract('ac,ci->ai', ints.d_vv, t2_11)
    res += 1.0 * contract('ac,ci->ai', ints.d_vv, t1_10)
    if 't2_02' in active:
        res += 1.0 * t2_02 * contract('ac,ci->ai', ints.d_vv, t1_10)
    if 't2_12' in active:
        res += 1.0 * contract('ac,ci->ai', ints.d_vv, t2_12)
    res += 1.0 * contract('ai->ai', ints.d_vo)
    if 't2_02' in active:
        res += 1.0 * t2_02 * contract('ai->ai', ints.d_vo)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * t1_01 * contract('ck,ak,ci->ai', ints.d_vo, t1_10, t2_11)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * t1_01 * contract('ck,ci,ak->ai', ints.d_vo, t1_10, t2_11)
    if 't1_01' in active and 't2_21' in active:
        res += 2.0 * t1_01 * contract('ck,acik->ai', ints.d_vo, t2_21)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * t1_01 * contract('ck,acki->ai', ints.d_vo, t2_21)
    res += -1.0 * contract('ck,ak,ci->ai', ints.d_vo, t1_10, t1_10)
    if 't2_02' in active:
        res += -1.0 * t2_02 * contract('ck,ak,ci->ai', ints.d_vo, t1_10, t1_10)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.d_vo, t1_10, t2_12)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.d_vo, t1_10, t2_12)
    if 't2_12' in active:
        res += 2.0 * contract('ck,ck,ai->ai', ints.d_vo, t1_10, t2_12)
    if 't2_02' in active:
        res += 2.0 * t2_02 * contract('ck,acik->ai', ints.d_vo, t2_20)
    if 't2_02' in active:
        res += -1.0 * t2_02 * contract('ck,acki->ai', ints.d_vo, t2_20)
    if 't2_11' in active:
        res += 2.0 * contract('ck,ai,ck->ai', ints.d_vo, t2_11, t2_11)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,ci->ai', ints.d_vo, t2_11, t2_11)
    res += 2.0 * contract('ck,acik->ai', ints.d_vo, t2_20)
    res += -1.0 * contract('ck,acki->ai', ints.d_vo, t2_20)
    if 't2_22' in active:
        res += 2.0 * contract('ck,acik->ai', ints.d_vo, t2_22)
    if 't2_22' in active:
        res += -1.0 * contract('ck,acki->ai', ints.d_vo, t2_22)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * t1_01 * contract('ik,ak->ai', ints.d_oo, t2_11)
    res += -1.0 * contract('ik,ak->ai', ints.d_oo, t1_10)
    if 't2_02' in active:
        res += -1.0 * t2_02 * contract('ik,ak->ai', ints.d_oo, t1_10)
    if 't2_12' in active:
        res += -1.0 * contract('ik,ak->ai', ints.d_oo, t2_12)
    if 't2_11' in active:
        res += 1.0 * contract('ac,ci->ai', ints.f_vv, t2_11)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.f_vo, t1_10, t2_11)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.f_vo, t1_10, t2_11)
    if 't2_21' in active:
        res += 2.0 * contract('ck,acik->ai', ints.f_vo, t2_21)
    if 't2_21' in active:
        res += -1.0 * contract('ck,acki->ai', ints.f_vo, t2_21)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak->ai', ints.f_oo, t2_11)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,dl,kcld->ai', t1_10, t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,dl,kdlc->ai', t1_10, t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,di,kcld->ai', t1_10, t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,di,kdlc->ai', t1_10, t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,iklc->ai', t1_10, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,ilkc->ai', t1_10, t2_11, ints.v_ooov)
    if 't2_21' in active:
        res += -2.0 * contract('ak,cdil,kcld->ai', t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cdil,kdlc->ai', t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,al,kcld->ai', t1_10, t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dk,kcad->ai', t1_10, t2_11, ints.v_ovvv)
    if 't2_11' in active:
        res += 2.0 * contract('ci,dk,kdac->ai', t1_10, t2_11, ints.v_ovvv)
    if 't2_21' in active:
        res += -2.0 * contract('ci,adkl,kcld->ai', t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('ci,adkl,kdlc->ai', t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,di,al,kcld->ai', t1_10, t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ck,al,iklc->ai', t1_10, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,al,ilkc->ai', t1_10, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += 2.0 * contract('ck,di,kcad->ai', t1_10, t2_11, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('ck,di,kdac->ai', t1_10, t2_11, ints.v_ovvv)
    if 't2_21' in active:
        res += 4.0 * contract('ck,adil,kcld->ai', t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('ck,adil,kdlc->ai', t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('ck,adli,kcld->ai', t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('ck,adli,kdlc->ai', t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * w * contract('ai->ai', t2_11)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cdil,kcld->ai', t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cdil,kdlc->ai', t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ci,adkl,kcld->ai', t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ci,adkl,kdlc->ai', t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 4.0 * contract('ck,adil,kcld->ai', t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,adil,kdlc->ai', t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,adli,kcld->ai', t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ck,adli,kdlc->ai', t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ikac->ai', t2_11, ints.v_oovv)
    if 't2_11' in active:
        res += 2.0 * contract('ck,iakc->ai', t2_11, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('ackl,iklc->ai', t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += 1.0 * contract('ackl,ilkc->ai', t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += -1.0 * contract('cdik,kcad->ai', t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += 2.0 * contract('cdik,kdac->ai', t2_21, ints.v_ovvv)

    e_denom = 1.0 / (ints.eps_occ[None, :] - ints.eps_vir[:, None] - w)
    return t2_11 + res * e_denom

def ccsd_t2_21(ints, w, t1_10, t1_01, t2_20, t2_02,
               t2_11, t2_21, t2_12, t2_22, active=_ALL_AMPS):
    nocc = t2_20.shape[2]
    nvir = t2_20.shape[0]
    res = np.zeros((nvir, nvir, nocc, nocc))

    if 't2_11' in active:
        res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv, ints.B_vv, t1_10, t2_11)
    if 't2_11' in active:
        res += 1.0 * contract('xac,xbd,dj,ci->abij', ints.B_vv, ints.B_vv, t1_10, t2_11)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * t1_01 * contract('ac,bcji->abij', ints.d_vv, t2_21)
    if 't2_12' in active:
        res += 1.0 * contract('ac,ci,bj->abij', ints.d_vv, t1_10, t2_12)
    if 't2_02' in active:
        res += 1.0 * t2_02 * contract('ac,bcji->abij', ints.d_vv, t2_20)
    if 't2_11' in active:
        res += 1.0 * contract('ac,bj,ci->abij', ints.d_vv, t2_11, t2_11)
    res += 1.0 * contract('ac,bcji->abij', ints.d_vv, t2_20)
    if 't2_22' in active:
        res += 1.0 * contract('ac,bcji->abij', ints.d_vv, t2_22)
    if 't2_12' in active:
        res += 1.0 * contract('ai,bj->abij', ints.d_vo, t2_12)
    if 't1_01' in active and 't2_21' in active:
        res += 1.0 * t1_01 * contract('bc,acij->abij', ints.d_vv, t2_21)
    if 't2_12' in active:
        res += 1.0 * contract('bc,cj,ai->abij', ints.d_vv, t1_10, t2_12)
    if 't2_02' in active:
        res += 1.0 * t2_02 * contract('bc,acij->abij', ints.d_vv, t2_20)
    if 't2_11' in active:
        res += 1.0 * contract('bc,ai,cj->abij', ints.d_vv, t2_11, t2_11)
    res += 1.0 * contract('bc,acij->abij', ints.d_vv, t2_20)
    if 't2_22' in active:
        res += 1.0 * contract('bc,acij->abij', ints.d_vv, t2_22)
    if 't2_12' in active:
        res += 1.0 * contract('bj,ai->abij', ints.d_vo, t2_12)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * t1_01 * contract('ck,ak,bcji->abij', ints.d_vo, t1_10, t2_21)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * t1_01 * contract('ck,bk,acij->abij', ints.d_vo, t1_10, t2_21)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * t1_01 * contract('ck,ci,abkj->abij', ints.d_vo, t1_10, t2_21)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * t1_01 * contract('ck,cj,abik->abij', ints.d_vo, t1_10, t2_21)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * t1_01 * contract('ck,ak,bcji->abij', ints.d_vo, t2_11, t2_20)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * t1_01 * contract('ck,bk,acij->abij', ints.d_vo, t2_11, t2_20)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * t1_01 * contract('ck,ci,abkj->abij', ints.d_vo, t2_11, t2_20)
    if 't1_01' in active and 't2_11' in active:
        res += -1.0 * t1_01 * contract('ck,cj,abik->abij', ints.d_vo, t2_11, t2_20)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,ci,bj->abij', ints.d_vo, t1_10, t1_10, t2_12)
    if 't2_02' in active:
        res += -1.0 * t2_02 * contract('ck,ak,bcji->abij', ints.d_vo, t1_10, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,bj,ci->abij', ints.d_vo, t1_10, t2_11, t2_11)
    res += -1.0 * contract('ck,ak,bcji->abij', ints.d_vo, t1_10, t2_20)
    if 't2_22' in active:
        res += -1.0 * contract('ck,ak,bcji->abij', ints.d_vo, t1_10, t2_22)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,cj,ai->abij', ints.d_vo, t1_10, t1_10, t2_12)
    if 't2_02' in active:
        res += -1.0 * t2_02 * contract('ck,bk,acij->abij', ints.d_vo, t1_10, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,ai,cj->abij', ints.d_vo, t1_10, t2_11, t2_11)
    res += -1.0 * contract('ck,bk,acij->abij', ints.d_vo, t1_10, t2_20)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_vo, t1_10, t2_22)
    if 't2_02' in active:
        res += -1.0 * t2_02 * contract('ck,ci,abkj->abij', ints.d_vo, t1_10, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,ak,bj->abij', ints.d_vo, t1_10, t2_11, t2_11)
    res += -1.0 * contract('ck,ci,abkj->abij', ints.d_vo, t1_10, t2_20)
    if 't2_22' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.d_vo, t1_10, t2_22)
    if 't2_02' in active:
        res += -1.0 * t2_02 * contract('ck,cj,abik->abij', ints.d_vo, t1_10, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,ai,bk->abij', ints.d_vo, t1_10, t2_11, t2_11)
    res += -1.0 * contract('ck,cj,abik->abij', ints.d_vo, t1_10, t2_20)
    if 't2_22' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_vo, t1_10, t2_22)
    if 't2_22' in active:
        res += 2.0 * contract('ck,ck,abij->abij', ints.d_vo, t1_10, t2_22)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ai,bcjk->abij', ints.d_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += -1.0 * contract('ck,ai,bckj->abij', ints.d_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ak,bcji->abij', ints.d_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bj,acik->abij', ints.d_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += -1.0 * contract('ck,bj,acki->abij', ints.d_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ci,abkj->abij', ints.d_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ck,abij->abij', ints.d_vo, t2_11, t2_21)
    if 't2_12' in active:
        res += 2.0 * contract('ck,ai,bcjk->abij', ints.d_vo, t2_12, t2_20)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ai,bckj->abij', ints.d_vo, t2_12, t2_20)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,bcji->abij', ints.d_vo, t2_12, t2_20)
    if 't2_12' in active:
        res += 2.0 * contract('ck,bj,acik->abij', ints.d_vo, t2_12, t2_20)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bj,acki->abij', ints.d_vo, t2_12, t2_20)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.d_vo, t2_12, t2_20)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.d_vo, t2_12, t2_20)
    if 't2_12' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.d_vo, t2_12, t2_20)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * t1_01 * contract('ik,abkj->abij', ints.d_oo, t2_21)
    if 't2_12' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_oo, t1_10, t2_12)
    if 't2_02' in active:
        res += -1.0 * t2_02 * contract('ik,abkj->abij', ints.d_oo, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ik,ak,bj->abij', ints.d_oo, t2_11, t2_11)
    res += -1.0 * contract('ik,abkj->abij', ints.d_oo, t2_20)
    if 't2_22' in active:
        res += -1.0 * contract('ik,abkj->abij', ints.d_oo, t2_22)
    if 't1_01' in active and 't2_21' in active:
        res += -1.0 * t1_01 * contract('jk,abik->abij', ints.d_oo, t2_21)
    if 't2_12' in active:
        res += -1.0 * contract('jk,bk,ai->abij', ints.d_oo, t1_10, t2_12)
    if 't2_02' in active:
        res += -1.0 * t2_02 * contract('jk,abik->abij', ints.d_oo, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('jk,ai,bk->abij', ints.d_oo, t2_11, t2_11)
    res += -1.0 * contract('jk,abik->abij', ints.d_oo, t2_20)
    if 't2_22' in active:
        res += -1.0 * contract('jk,abik->abij', ints.d_oo, t2_22)
    if 't2_21' in active:
        res += 1.0 * contract('ac,bcji->abij', ints.f_vv, t2_21)
    if 't2_21' in active:
        res += 1.0 * contract('bc,acij->abij', ints.f_vv, t2_21)
    if 't2_21' in active:
        res += -1.0 * contract('ck,ak,bcji->abij', ints.f_vo, t1_10, t2_21)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_vo, t1_10, t2_21)
    if 't2_21' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.f_vo, t1_10, t2_21)
    if 't2_21' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_vo, t1_10, t2_21)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ak,bcji->abij', ints.f_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.f_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_vo, t2_11, t2_20)
    if 't2_21' in active:
        res += -1.0 * contract('ik,abkj->abij', ints.f_oo, t2_21)
    if 't2_21' in active:
        res += -1.0 * contract('jk,abik->abij', ints.f_oo, t2_21)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ci,dj,kcld->abij', t1_10, t1_10, t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', t1_10, t1_10, t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ci,jlkc->abij', t1_10, t1_10, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cj,iklc->abij', t1_10, t1_10, t2_11, ints.v_ooov)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,dj,bl,kcld->abij', t1_10, t1_10, t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,bl,jlkc->abij', t1_10, t1_10, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,dj,kcbd->abij', t1_10, t1_10, t2_11, ints.v_ovvv)
    if 't2_21' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('ak,ci,bdlj,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,bl,iklc->abij', t1_10, t1_10, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,di,kdbc->abij', t1_10, t1_10, t2_11, ints.v_ovvv)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cj,bdli,kdlc->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('ak,cl,bdji,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('ak,cl,bdji,kdlc->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bl,ikjl->abij', t1_10, t2_11, ints.v_oooo)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,ci,bdlj,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ci,jbkc->abij', t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cj,bdli,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cj,ikbc->abij', t1_10, t2_11, ints.v_oovv)
    if 't2_11' in active:
        res += 1.0 * contract('ak,cl,bdji,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cl,bdji,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('ak,bcjl,iklc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bcli,jlkc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += 1.0 * contract('ak,bclj,iklc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,al,jklc->abij', t1_10, t1_10, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,dj,kdac->abij', t1_10, t1_10, t2_11, ints.v_ovvv)
    if 't2_21' in active:
        res += 1.0 * contract('bk,ci,adlj,kdlc->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,di,al,kcld->abij', t1_10, t1_10, t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,al,ilkc->abij', t1_10, t1_10, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,di,kcad->abij', t1_10, t1_10, t2_11, ints.v_ovvv)
    if 't2_21' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cj,adli,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('bk,al,cdij,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('bk,al,iljk->abij', t1_10, t2_11, ints.v_oooo)
    if 't2_11' in active:
        res += 1.0 * contract('bk,ci,adlj,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -1.0 * contract('bk,ci,jkac->abij', t1_10, t2_11, ints.v_oovv)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cj,adli,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cj,iakc->abij', t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('bk,acil,jklc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += 1.0 * contract('bk,acli,jklc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += 1.0 * contract('bk,aclj,ilkc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += -1.0 * contract('bk,cdij,kdac->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('ci,dj,ak,kcbd->abij', t1_10, t1_10, t2_11, ints.v_ovvv)
    if 't2_21' in active:
        res += 1.0 * contract('ci,dk,ablj,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ci,ak,bdjl,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ci,ak,bdjl,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ci,ak,bdlj,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -1.0 * contract('ci,ak,jbkc->abij', t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ci,bk,adlj,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bk,jkac->abij', t1_10, t2_11, ints.v_oovv)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dk,ablj,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ci,dk,ablj,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('ci,abkl,jlkc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += -1.0 * contract('ci,adkj,kcbd->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += 2.0 * contract('ci,bdjk,kdac->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += -1.0 * contract('ci,bdkj,kdac->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('cj,di,bk,kcad->abij', t1_10, t1_10, t2_11, ints.v_ovvv)
    if 't2_21' in active:
        res += 1.0 * contract('cj,di,abkl,kdlc->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('cj,ak,bdli,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -1.0 * contract('cj,ak,ikbc->abij', t1_10, t2_11, ints.v_oovv)
    if 't2_11' in active:
        res += -2.0 * contract('cj,bk,adil,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('cj,bk,adil,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('cj,bk,adli,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bk,iakc->abij', t1_10, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('cj,di,abkl,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('cj,dk,abil,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += 2.0 * contract('cj,adik,kdbc->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += -1.0 * contract('cj,adki,kdbc->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += -1.0 * contract('cj,bdki,kcad->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += -2.0 * contract('ck,di,ablj,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', t1_10, t1_10, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,al,bdji,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ck,al,bdji,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,bl,adij,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ck,bl,adij,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,di,ablj,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ck,di,ablj,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 1.0 * contract('ck,dj,abil,kdlc->abij', t1_10, t2_11, t2_20, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += -2.0 * contract('ck,abil,jlkc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += 1.0 * contract('ck,ablj,iklc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += -2.0 * contract('ck,ablj,ilkc->abij', t1_10, t2_21, ints.v_ooov)
    if 't2_21' in active:
        res += 2.0 * contract('ck,adij,kcbd->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += 2.0 * contract('ck,bdji,kcad->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_21' in active:
        res += -1.0 * contract('ck,bdji,kdac->abij', t1_10, t2_21, ints.v_ovvv)
    if 't2_11' in active:
        res += -2.0 * contract('ak,bcjl,iklc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bcli,jlkc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += 1.0 * contract('ak,bclj,iklc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('ak,ikjb->abij', t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += -2.0 * contract('bk,acil,jklc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += 1.0 * contract('bk,acli,jklc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += 1.0 * contract('bk,aclj,ilkc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += -1.0 * contract('bk,cdij,kdac->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('bk,jkia->abij', t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += 1.0 * contract('ci,abkl,jlkc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += -1.0 * contract('ci,adkj,kcbd->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += 2.0 * contract('ci,bdjk,kdac->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('ci,bdkj,kdac->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += 1.0 * contract('ci,jbac->abij', t2_11, ints.v_ovvv)
    if 't2_11' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += 2.0 * contract('cj,adik,kdbc->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('cj,adki,kdbc->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('cj,bdki,kcad->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += 1.0 * contract('cj,iabc->abij', t2_11, ints.v_ovvv)
    if 't2_11' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,abil,jlkc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += 1.0 * contract('ck,ablj,iklc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ablj,ilkc->abij', t2_11, t2_20, ints.v_ooov)
    if 't2_11' in active:
        res += 2.0 * contract('ck,adij,kcbd->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += 2.0 * contract('ck,bdji,kcad->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_11' in active:
        res += -1.0 * contract('ck,bdji,kdac->abij', t2_11, t2_20, ints.v_ovvv)
    if 't2_21' in active:
        res += -2.0 * contract('abik,cdjl,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('abik,cdjl,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('abkj,cdil,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('abkj,cdil,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('abkl,cdij,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('acij,bdkl,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('acij,bdkl,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 4.0 * contract('acik,bdjl,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('acik,bdjl,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('acik,bdlj,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('acik,bdlj,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('acki,bdjl,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('acki,bdjl,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('acki,bdlj,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('ackj,bdli,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('ackl,bdji,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('ackl,bdji,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('bcji,adkl,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('bcji,adkl,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 4.0 * contract('bcjk,adil,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('bcjk,adil,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('bcjk,adli,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('bcjk,adli,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('bcki,adlj,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('bckj,adil,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('bckj,adil,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('bckj,adli,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('bckl,adij,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('bckl,adij,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('cdij,abkl,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('cdik,ablj,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('cdik,ablj,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * contract('cdjk,abil,kcld->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -2.0 * contract('cdjk,abil,kdlc->abij', t2_20, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 1.0 * w * contract('abij->abij', t2_21)
    if 't2_21' in active:
        res += 1.0 * contract('abkl,ikjl->abij', t2_21, ints.v_oooo)
    if 't2_21' in active:
        res += -1.0 * contract('acik,jkbc->abij', t2_21, ints.v_oovv)
    if 't2_21' in active:
        res += 2.0 * contract('acik,jbkc->abij', t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -1.0 * contract('acki,jbkc->abij', t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -1.0 * contract('ackj,ikbc->abij', t2_21, ints.v_oovv)
    if 't2_21' in active:
        res += -1.0 * contract('bcjk,ikac->abij', t2_21, ints.v_oovv)
    if 't2_21' in active:
        res += 2.0 * contract('bcjk,iakc->abij', t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -1.0 * contract('bcki,jkac->abij', t2_21, ints.v_oovv)
    if 't2_21' in active:
        res += -1.0 * contract('bckj,iakc->abij', t2_21, ints.v_ovov)
    if 't2_21' in active:
        _W = 1.0 * t2_21
        _vvvv_ladder(ints.B_vv, np.ascontiguousarray(_W), out=res, alpha=1.0)

    # exact (ab)(ij) pair symmetry of the mixed-spin block; enforce against rounding drift
    res = 0.5 * (res + res.transpose(1, 0, 3, 2))
    for i in range(nocc):
        res[:, :, i, :] *= 1.0 / (ints.eps_occ[i] + ints.eps_occ[None, None, :]
                                  - ints.eps_vir[:, None, None]
                                  - ints.eps_vir[None, :, None] - (w))
    res += t2_21
    return res

def ccsd_t2_12(ints, w, t1_10, t1_01, t2_20, t2_02,
               t2_11, t2_21, t2_12, t2_22, active=_ALL_AMPS):
    nocc = t2_20.shape[2]
    nvir = t2_20.shape[0]
    res = np.zeros((nvir, nocc))

    if 't1_01' in active and 't2_12' in active:
        res += 1.0 * t1_01 * contract('ac,ci->ai', ints.d_vv, t2_12)
    if 't2_02' in active and 't2_11' in active:
        res += 2.0 * t2_02 * contract('ac,ci->ai', ints.d_vv, t2_11)
    if 't2_11' in active:
        res += 2.0 * contract('ac,ci->ai', ints.d_vv, t2_11)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * t1_01 * contract('ck,ak,ci->ai', ints.d_vo, t1_10, t2_12)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * t1_01 * contract('ck,ci,ak->ai', ints.d_vo, t1_10, t2_12)
    if 't1_01' in active and 't2_11' in active:
        res += -2.0 * t1_01 * contract('ck,ak,ci->ai', ints.d_vo, t2_11, t2_11)
    if 't1_01' in active and 't2_22' in active:
        res += 2.0 * t1_01 * contract('ck,acik->ai', ints.d_vo, t2_22)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * t1_01 * contract('ck,acki->ai', ints.d_vo, t2_22)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * t2_02 * contract('ck,ak,ci->ai', ints.d_vo, t1_10, t2_11)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,ci->ai', ints.d_vo, t1_10, t2_11)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * t2_02 * contract('ck,ci,ak->ai', ints.d_vo, t1_10, t2_11)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ci,ak->ai', ints.d_vo, t1_10, t2_11)
    if 't2_02' in active and 't2_21' in active:
        res += 4.0 * t2_02 * contract('ck,acik->ai', ints.d_vo, t2_21)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * t2_02 * contract('ck,acki->ai', ints.d_vo, t2_21)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ck,ai,ck->ai', ints.d_vo, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += -3.0 * contract('ck,ak,ci->ai', ints.d_vo, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += -3.0 * contract('ck,ci,ak->ai', ints.d_vo, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += 4.0 * contract('ck,ck,ai->ai', ints.d_vo, t2_11, t2_12)
    if 't2_21' in active:
        res += 4.0 * contract('ck,acik->ai', ints.d_vo, t2_21)
    if 't2_21' in active:
        res += -2.0 * contract('ck,acki->ai', ints.d_vo, t2_21)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * t1_01 * contract('ik,ak->ai', ints.d_oo, t2_12)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * t2_02 * contract('ik,ak->ai', ints.d_oo, t2_11)
    if 't2_11' in active:
        res += -2.0 * contract('ik,ak->ai', ints.d_oo, t2_11)
    if 't2_12' in active:
        res += 1.0 * contract('ac,ci->ai', ints.f_vv, t2_12)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,ci->ai', ints.f_vo, t1_10, t2_12)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ci,ak->ai', ints.f_vo, t1_10, t2_12)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,ci->ai', ints.f_vo, t2_11, t2_11)
    if 't2_22' in active:
        res += 2.0 * contract('ck,acik->ai', ints.f_vo, t2_22)
    if 't2_22' in active:
        res += -1.0 * contract('ck,acki->ai', ints.f_vo, t2_22)
    if 't2_12' in active:
        res += -1.0 * contract('ik,ak->ai', ints.f_oo, t2_12)
    if 't2_12' in active:
        res += -2.0 * contract('ak,ci,dl,kcld->ai', t1_10, t1_10, t2_12, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,dl,kdlc->ai', t1_10, t1_10, t2_12, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,di,kcld->ai', t1_10, t1_10, t2_12, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ak,cl,di,kdlc->ai', t1_10, t1_10, t2_12, ints.v_ovov)
    if 't2_11' in active:
        res += -4.0 * contract('ak,ci,dl,kcld->ai', t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,di,kcld->ai', t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ak,cl,iklc->ai', t1_10, t2_12, ints.v_ooov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,ilkc->ai', t1_10, t2_12, ints.v_ooov)
    if 't2_22' in active:
        res += -2.0 * contract('ak,cdil,kcld->ai', t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cdil,kdlc->ai', t1_10, t2_22, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,al,kcld->ai', t1_10, t1_10, t2_12, ints.v_ovov)
    if 't2_11' in active:
        res += -4.0 * contract('ci,ak,dl,kcld->ai', t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ci,ak,dl,kdlc->ai', t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dk,kcad->ai', t1_10, t2_12, ints.v_ovvv)
    if 't2_12' in active:
        res += 2.0 * contract('ci,dk,kdac->ai', t1_10, t2_12, ints.v_ovvv)
    if 't2_22' in active:
        res += -2.0 * contract('ci,adkl,kcld->ai', t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('ci,adkl,kdlc->ai', t1_10, t2_22, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,di,al,kcld->ai', t1_10, t1_10, t2_12, ints.v_ovov)
    if 't2_11' in active:
        res += -4.0 * contract('ck,al,di,kcld->ai', t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ck,al,di,kdlc->ai', t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ck,al,iklc->ai', t1_10, t2_12, ints.v_ooov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,al,ilkc->ai', t1_10, t2_12, ints.v_ooov)
    if 't2_12' in active:
        res += 2.0 * contract('ck,di,kcad->ai', t1_10, t2_12, ints.v_ovvv)
    if 't2_12' in active:
        res += -1.0 * contract('ck,di,kdac->ai', t1_10, t2_12, ints.v_ovvv)
    if 't2_22' in active:
        res += 4.0 * contract('ck,adil,kcld->ai', t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('ck,adil,kdlc->ai', t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('ck,adli,kcld->ai', t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('ck,adli,kdlc->ai', t1_10, t2_22, ints.v_ovov)
    if 't2_11' in active:
        res += -4.0 * contract('ak,cl,iklc->ai', t2_11, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,ilkc->ai', t2_11, t2_11, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ak,cdil,kcld->ai', t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cdil,kdlc->ai', t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ci,dk,kcad->ai', t2_11, t2_11, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ci,adkl,kcld->ai', t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,adkl,kdlc->ai', t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += 4.0 * contract('ck,di,kcad->ai', t2_11, t2_11, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += 8.0 * contract('ck,adil,kcld->ai', t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ck,adil,kdlc->ai', t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ck,adli,kcld->ai', t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,adli,kdlc->ai', t2_11, t2_21, ints.v_ovov)
    if 't2_12' in active:
        res += 2.0 * w * contract('ai->ai', t2_12)
    if 't2_12' in active:
        res += -2.0 * contract('ak,cdil,kcld->ai', t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cdil,kdlc->ai', t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ci,adkl,kcld->ai', t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ci,adkl,kdlc->ai', t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 4.0 * contract('ck,adil,kcld->ai', t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,adil,kdlc->ai', t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,adli,kcld->ai', t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ck,adli,kdlc->ai', t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ikac->ai', t2_12, ints.v_oovv)
    if 't2_12' in active:
        res += 2.0 * contract('ck,iakc->ai', t2_12, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('ackl,iklc->ai', t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += 1.0 * contract('ackl,ilkc->ai', t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += -1.0 * contract('cdik,kcad->ai', t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += 2.0 * contract('cdik,kdac->ai', t2_22, ints.v_ovvv)

    e_denom = 1.0 / (ints.eps_occ[None, :] - ints.eps_vir[:, None] - 2.0 * w)
    return t2_12 + res * e_denom

def ccsd_t2_22(ints, w, t1_10, t1_01, t2_20, t2_02,
               t2_11, t2_21, t2_12, t2_22, active=_ALL_AMPS):
    nocc = t2_20.shape[2]
    nvir = t2_20.shape[0]
    res = np.zeros((nvir, nvir, nocc, nocc))

    if 't2_12' in active:
        res += 1.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv, ints.B_vv, t1_10, t2_12)
    if 't2_12' in active:
        res += 1.0 * contract('xac,xbd,dj,ci->abij', ints.B_vv, ints.B_vv, t1_10, t2_12)
    if 't2_11' in active:
        res += 2.0 * contract('xac,xbd,ci,dj->abij', ints.B_vv, ints.B_vv, t2_11, t2_11)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * t1_01 * contract('ac,bcji->abij', ints.d_vv, t2_22)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * t2_02 * contract('ac,bcji->abij', ints.d_vv, t2_21)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('ac,bj,ci->abij', ints.d_vv, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('ac,ci,bj->abij', ints.d_vv, t2_11, t2_12)
    if 't2_21' in active:
        res += 2.0 * contract('ac,bcji->abij', ints.d_vv, t2_21)
    if 't1_01' in active and 't2_22' in active:
        res += 1.0 * t1_01 * contract('bc,acij->abij', ints.d_vv, t2_22)
    if 't2_02' in active and 't2_21' in active:
        res += 2.0 * t2_02 * contract('bc,acij->abij', ints.d_vv, t2_21)
    if 't2_11' in active and 't2_12' in active:
        res += 1.0 * contract('bc,ai,cj->abij', ints.d_vv, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += 2.0 * contract('bc,cj,ai->abij', ints.d_vv, t2_11, t2_12)
    if 't2_21' in active:
        res += 2.0 * contract('bc,acij->abij', ints.d_vv, t2_21)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * t1_01 * contract('ck,ak,bcji->abij', ints.d_vo, t1_10, t2_22)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * t1_01 * contract('ck,bk,acij->abij', ints.d_vo, t1_10, t2_22)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * t1_01 * contract('ck,ci,abkj->abij', ints.d_vo, t1_10, t2_22)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * t1_01 * contract('ck,cj,abik->abij', ints.d_vo, t1_10, t2_22)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * t1_01 * contract('ck,ak,bcji->abij', ints.d_vo, t2_11, t2_21)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * t1_01 * contract('ck,bk,acij->abij', ints.d_vo, t2_11, t2_21)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * t1_01 * contract('ck,ci,abkj->abij', ints.d_vo, t2_11, t2_21)
    if 't1_01' in active and 't2_11' in active and 't2_21' in active:
        res += -2.0 * t1_01 * contract('ck,cj,abik->abij', ints.d_vo, t2_11, t2_21)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * t1_01 * contract('ck,ak,bcji->abij', ints.d_vo, t2_12, t2_20)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * t1_01 * contract('ck,bk,acij->abij', ints.d_vo, t2_12, t2_20)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * t1_01 * contract('ck,ci,abkj->abij', ints.d_vo, t2_12, t2_20)
    if 't1_01' in active and 't2_12' in active:
        res += -1.0 * t1_01 * contract('ck,cj,abik->abij', ints.d_vo, t2_12, t2_20)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * t2_02 * contract('ck,ak,bcji->abij', ints.d_vo, t1_10, t2_21)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,ak,bj,ci->abij', ints.d_vo, t1_10, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,ak,ci,bj->abij', ints.d_vo, t1_10, t2_11, t2_12)
    if 't2_21' in active:
        res += -2.0 * contract('ck,ak,bcji->abij', ints.d_vo, t1_10, t2_21)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * t2_02 * contract('ck,bk,acij->abij', ints.d_vo, t1_10, t2_21)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,bk,ai,cj->abij', ints.d_vo, t1_10, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,bk,cj,ai->abij', ints.d_vo, t1_10, t2_11, t2_12)
    if 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_vo, t1_10, t2_21)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * t2_02 * contract('ck,ci,abkj->abij', ints.d_vo, t1_10, t2_21)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,ci,ak,bj->abij', ints.d_vo, t1_10, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,ci,bj,ak->abij', ints.d_vo, t1_10, t2_11, t2_12)
    if 't2_21' in active:
        res += -2.0 * contract('ck,ci,abkj->abij', ints.d_vo, t1_10, t2_21)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * t2_02 * contract('ck,cj,abik->abij', ints.d_vo, t1_10, t2_21)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ck,cj,ai,bk->abij', ints.d_vo, t1_10, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ck,cj,bk,ai->abij', ints.d_vo, t1_10, t2_11, t2_12)
    if 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_vo, t1_10, t2_21)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * t2_02 * contract('ck,ak,bcji->abij', ints.d_vo, t2_11, t2_20)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * t2_02 * contract('ck,bk,acij->abij', ints.d_vo, t2_11, t2_20)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * t2_02 * contract('ck,ci,abkj->abij', ints.d_vo, t2_11, t2_20)
    if 't2_02' in active and 't2_11' in active:
        res += -2.0 * t2_02 * contract('ck,cj,abik->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ai,bk,cj->abij', ints.d_vo, t2_11, t2_11, t2_11)
    if 't2_11' in active and 't2_22' in active:
        res += 2.0 * contract('ck,ai,bcjk->abij', ints.d_vo, t2_11, t2_22)
    if 't2_11' in active and 't2_22' in active:
        res += -1.0 * contract('ck,ai,bckj->abij', ints.d_vo, t2_11, t2_22)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,bj,ci->abij', ints.d_vo, t2_11, t2_11, t2_11)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ak,bcji->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,ak,bcji->abij', ints.d_vo, t2_11, t2_22)
    if 't2_11' in active and 't2_22' in active:
        res += 2.0 * contract('ck,bj,acik->abij', ints.d_vo, t2_11, t2_22)
    if 't2_11' in active and 't2_22' in active:
        res += -1.0 * contract('ck,bj,acki->abij', ints.d_vo, t2_11, t2_22)
    if 't2_11' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,bk,acij->abij', ints.d_vo, t2_11, t2_22)
    if 't2_11' in active:
        res += -2.0 * contract('ck,ci,abkj->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,ci,abkj->abij', ints.d_vo, t2_11, t2_22)
    if 't2_11' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.d_vo, t2_11, t2_20)
    if 't2_11' in active and 't2_22' in active:
        res += -3.0 * contract('ck,cj,abik->abij', ints.d_vo, t2_11, t2_22)
    if 't2_11' in active and 't2_22' in active:
        res += 4.0 * contract('ck,ck,abij->abij', ints.d_vo, t2_11, t2_22)
    if 't2_12' in active and 't2_21' in active:
        res += 4.0 * contract('ck,ai,bcjk->abij', ints.d_vo, t2_12, t2_21)
    if 't2_12' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ai,bckj->abij', ints.d_vo, t2_12, t2_21)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,ak,bcji->abij', ints.d_vo, t2_12, t2_21)
    if 't2_12' in active and 't2_21' in active:
        res += 4.0 * contract('ck,bj,acik->abij', ints.d_vo, t2_12, t2_21)
    if 't2_12' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bj,acki->abij', ints.d_vo, t2_12, t2_21)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,bk,acij->abij', ints.d_vo, t2_12, t2_21)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,ci,abkj->abij', ints.d_vo, t2_12, t2_21)
    if 't2_12' in active and 't2_21' in active:
        res += -3.0 * contract('ck,cj,abik->abij', ints.d_vo, t2_12, t2_21)
    if 't2_12' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ck,abij->abij', ints.d_vo, t2_12, t2_21)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * t1_01 * contract('ik,abkj->abij', ints.d_oo, t2_22)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * t2_02 * contract('ik,abkj->abij', ints.d_oo, t2_21)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('ik,ak,bj->abij', ints.d_oo, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('ik,bj,ak->abij', ints.d_oo, t2_11, t2_12)
    if 't2_21' in active:
        res += -2.0 * contract('ik,abkj->abij', ints.d_oo, t2_21)
    if 't1_01' in active and 't2_22' in active:
        res += -1.0 * t1_01 * contract('jk,abik->abij', ints.d_oo, t2_22)
    if 't2_02' in active and 't2_21' in active:
        res += -2.0 * t2_02 * contract('jk,abik->abij', ints.d_oo, t2_21)
    if 't2_11' in active and 't2_12' in active:
        res += -1.0 * contract('jk,ai,bk->abij', ints.d_oo, t2_11, t2_12)
    if 't2_11' in active and 't2_12' in active:
        res += -2.0 * contract('jk,bk,ai->abij', ints.d_oo, t2_11, t2_12)
    if 't2_21' in active:
        res += -2.0 * contract('jk,abik->abij', ints.d_oo, t2_21)
    if 't2_22' in active:
        res += 1.0 * contract('ac,bcji->abij', ints.f_vv, t2_22)
    if 't2_22' in active:
        res += 1.0 * contract('bc,acij->abij', ints.f_vv, t2_22)
    if 't2_22' in active:
        res += -1.0 * contract('ck,ak,bcji->abij', ints.f_vo, t1_10, t2_22)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_vo, t1_10, t2_22)
    if 't2_22' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.f_vo, t1_10, t2_22)
    if 't2_22' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_vo, t1_10, t2_22)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ak,bcji->abij', ints.f_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bk,acij->abij', ints.f_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,ci,abkj->abij', ints.f_vo, t2_11, t2_21)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,cj,abik->abij', ints.f_vo, t2_11, t2_21)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ak,bcji->abij', ints.f_vo, t2_12, t2_20)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bk,acij->abij', ints.f_vo, t2_12, t2_20)
    if 't2_12' in active:
        res += -1.0 * contract('ck,ci,abkj->abij', ints.f_vo, t2_12, t2_20)
    if 't2_12' in active:
        res += -1.0 * contract('ck,cj,abik->abij', ints.f_vo, t2_12, t2_20)
    if 't2_22' in active:
        res += -1.0 * contract('ik,abkj->abij', ints.f_oo, t2_22)
    if 't2_22' in active:
        res += -1.0 * contract('jk,abik->abij', ints.f_oo, t2_22)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ci,dj,kcld->abij', t1_10, t1_10, t1_10, t2_12, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cj,di,kdlc->abij', t1_10, t1_10, t1_10, t2_12, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cj,di,kdlc->abij', t1_10, t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ci,jlkc->abij', t1_10, t1_10, t2_12, ints.v_ooov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cj,iklc->abij', t1_10, t1_10, t2_12, ints.v_ooov)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,dj,bl,kcld->abij', t1_10, t1_10, t1_10, t2_12, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,bl,dj,kcld->abij', t1_10, t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,bl,jlkc->abij', t1_10, t1_10, t2_12, ints.v_ooov)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,dj,kcbd->abij', t1_10, t1_10, t2_12, ints.v_ovvv)
    if 't2_22' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('ak,ci,bdlj,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,bl,di,kdlc->abij', t1_10, t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,bl,iklc->abij', t1_10, t1_10, t2_12, ints.v_ooov)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,di,kdbc->abij', t1_10, t1_10, t2_12, ints.v_ovvv)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cj,bdli,kdlc->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('ak,cl,bdji,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('ak,cl,bdji,kdlc->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,ci,jlkc->abij', t1_10, t2_11, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cj,iklc->abij', t1_10, t2_11, t2_11, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bl,cdij,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,dj,kcbd->abij', t1_10, t2_11, t2_11, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ak,ci,bdjl,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,ci,bdjl,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,ci,bdlj,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cj,bdli,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,cl,bdji,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ak,cl,bdji,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,cdij,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bl,ikjl->abij', t1_10, t2_12, ints.v_oooo)
    if 't2_12' in active:
        res += -2.0 * contract('ak,ci,bdjl,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,bdjl,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,ci,bdlj,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ci,jbkc->abij', t1_10, t2_12, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cj,bdli,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cj,ikbc->abij', t1_10, t2_12, ints.v_oovv)
    if 't2_12' in active:
        res += 1.0 * contract('ak,cl,bdji,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ak,cl,bdji,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('ak,bcjl,iklc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bcli,jlkc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += 1.0 * contract('ak,bclj,iklc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,al,dj,kdlc->abij', t1_10, t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,al,jklc->abij', t1_10, t1_10, t2_12, ints.v_ooov)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,dj,kdac->abij', t1_10, t1_10, t2_12, ints.v_ovvv)
    if 't2_22' in active:
        res += 1.0 * contract('bk,ci,adlj,kdlc->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,di,al,kcld->abij', t1_10, t1_10, t1_10, t2_12, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,al,di,kcld->abij', t1_10, t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,al,ilkc->abij', t1_10, t1_10, t2_12, ints.v_ooov)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,di,kcad->abij', t1_10, t1_10, t2_12, ints.v_ovvv)
    if 't2_22' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cj,adli,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('bk,al,ci,jklc->abij', t1_10, t2_11, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += 2.0 * contract('bk,al,cj,ilkc->abij', t1_10, t2_11, t2_11, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,al,cdij,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,ci,adlj,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,di,kcad->abij', t1_10, t2_11, t2_11, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('bk,cj,adil,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cj,adil,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cj,adli,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,cl,adij,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('bk,cl,adij,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,al,cdij,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,al,iljk->abij', t1_10, t2_12, ints.v_oooo)
    if 't2_12' in active:
        res += 1.0 * contract('bk,ci,adlj,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('bk,ci,jkac->abij', t1_10, t2_12, ints.v_oovv)
    if 't2_12' in active:
        res += -2.0 * contract('bk,cj,adil,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,adil,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cj,adli,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cj,iakc->abij', t1_10, t2_12, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,cl,adij,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('bk,cl,adij,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('bk,acil,jklc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += 1.0 * contract('bk,acli,jklc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += 1.0 * contract('bk,aclj,ilkc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += -1.0 * contract('bk,cdij,kdac->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,ak,bl,kcld->abij', t1_10, t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('ci,dj,ak,kcbd->abij', t1_10, t1_10, t2_12, ints.v_ovvv)
    if 't2_22' in active:
        res += 1.0 * contract('ci,dk,ablj,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ci,ak,bl,jlkc->abij', t1_10, t2_11, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += -2.0 * contract('ci,ak,dj,kcbd->abij', t1_10, t2_11, t2_11, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ci,ak,bdjl,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,ak,bdjl,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,ak,bdlj,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ci,bk,dj,kdac->abij', t1_10, t2_11, t2_11, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,bk,adlj,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dj,abkl,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,dk,ablj,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ci,dk,ablj,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ci,ak,bdjl,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ci,ak,bdjl,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ci,ak,bdlj,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('ci,ak,jbkc->abij', t1_10, t2_12, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ci,bk,adlj,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bk,jkac->abij', t1_10, t2_12, ints.v_oovv)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ci,dk,ablj,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ci,dk,ablj,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('ci,abkl,jlkc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += -1.0 * contract('ci,adkj,kcbd->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += 2.0 * contract('ci,bdjk,kdac->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += -1.0 * contract('ci,bdkj,kdac->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_11' in active:
        res += 1.0 * contract('cj,di,ak,bl,kdlc->abij', t1_10, t1_10, t2_11, t2_11, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('cj,di,bk,kcad->abij', t1_10, t1_10, t2_12, ints.v_ovvv)
    if 't2_22' in active:
        res += 1.0 * contract('cj,di,abkl,kdlc->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('cj,ak,bl,iklc->abij', t1_10, t2_11, t2_11, ints.v_ooov)
    if 't2_11' in active:
        res += -2.0 * contract('cj,ak,di,kdbc->abij', t1_10, t2_11, t2_11, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,ak,bdli,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('cj,bk,di,kcad->abij', t1_10, t2_11, t2_11, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('cj,bk,adil,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,bk,adil,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,bk,adli,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,di,abkl,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,dk,abil,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('cj,dk,abil,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('cj,ak,bdli,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('cj,ak,ikbc->abij', t1_10, t2_12, ints.v_oovv)
    if 't2_12' in active:
        res += -2.0 * contract('cj,bk,adil,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('cj,bk,adil,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('cj,bk,adli,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bk,iakc->abij', t1_10, t2_12, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('cj,di,abkl,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('cj,dk,abil,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('cj,dk,abil,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += 2.0 * contract('cj,adik,kdbc->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += -1.0 * contract('cj,adki,kdbc->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += -1.0 * contract('cj,bdki,kcad->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += -2.0 * contract('ck,di,ablj,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', t1_10, t1_10, t2_22, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ck,al,bdji,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,al,bdji,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ck,bl,adij,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,bl,adij,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ck,di,ablj,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,di,ablj,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ck,dj,abil,kcld->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,dj,abil,kdlc->abij', t1_10, t2_11, t2_21, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,al,bdji,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ck,al,bdji,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,bl,adij,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ck,bl,adij,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,di,ablj,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ck,di,ablj,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,dj,abil,kcld->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_12' in active:
        res += 1.0 * contract('ck,dj,abil,kdlc->abij', t1_10, t2_12, t2_20, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += -2.0 * contract('ck,abil,jlkc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += 1.0 * contract('ck,ablj,iklc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += -2.0 * contract('ck,ablj,ilkc->abij', t1_10, t2_22, ints.v_ooov)
    if 't2_22' in active:
        res += 2.0 * contract('ck,adij,kcbd->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += 2.0 * contract('ck,bdji,kcad->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_22' in active:
        res += -1.0 * contract('ck,bdji,kdac->abij', t1_10, t2_22, ints.v_ovvv)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,cdij,kcld->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,bl,ikjl->abij', t2_11, t2_11, ints.v_oooo)
    if 't2_11' in active:
        res += -4.0 * contract('ak,ci,bdjl,kcld->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,bdjl,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,ci,bdlj,kcld->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ak,ci,jbkc->abij', t2_11, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cj,bdli,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('ak,cj,ikbc->abij', t2_11, t2_11, ints.v_oovv)
    if 't2_11' in active:
        res += 2.0 * contract('ak,cl,bdji,kcld->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -4.0 * contract('ak,cl,bdji,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ak,bcjl,iklc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bcjl,ilkc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bcli,jlkc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ak,bclj,iklc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ak,cdij,kcbd->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active:
        res += 2.0 * contract('bk,ci,adlj,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('bk,ci,jkac->abij', t2_11, t2_11, ints.v_oovv)
    if 't2_11' in active:
        res += -4.0 * contract('bk,cj,adil,kcld->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,adil,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cj,adli,kcld->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -2.0 * contract('bk,cj,iakc->abij', t2_11, t2_11, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('bk,cl,adij,kcld->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -4.0 * contract('bk,cl,adij,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('bk,acil,jklc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,acil,jlkc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,acli,jklc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('bk,aclj,ilkc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('bk,cdij,kdac->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active:
        res += 1.0 * contract('ci,dj,abkl,kcld->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -4.0 * contract('ci,dk,ablj,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ci,abkl,jlkc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,adkj,kcbd->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,bdjk,kcad->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += 4.0 * contract('ci,bdjk,kdac->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ci,bdkj,kdac->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active:
        res += 1.0 * contract('cj,di,abkl,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += -4.0 * contract('cj,dk,abil,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('cj,abkl,iklc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,adik,kcbd->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += 4.0 * contract('cj,adik,kdbc->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,adki,kdbc->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('cj,bdki,kcad->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active:
        res += 2.0 * contract('ck,di,ablj,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active:
        res += 2.0 * contract('ck,dj,abil,kdlc->abij', t2_11, t2_11, t2_20, ints.v_ovov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,abil,jklc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ck,abil,jlkc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += 2.0 * contract('ck,ablj,iklc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += -4.0 * contract('ck,ablj,ilkc->abij', t2_11, t2_21, ints.v_ooov)
    if 't2_11' in active and 't2_21' in active:
        res += 4.0 * contract('ck,adij,kcbd->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,adij,kdbc->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += 4.0 * contract('ck,bdji,kcad->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_11' in active and 't2_21' in active:
        res += -2.0 * contract('ck,bdji,kdac->abij', t2_11, t2_21, ints.v_ovvv)
    if 't2_12' in active:
        res += -2.0 * contract('ak,bcjl,iklc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bcjl,ilkc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bcli,jlkc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += 1.0 * contract('ak,bclj,iklc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += -1.0 * contract('ak,cdij,kcbd->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += -1.0 * contract('ak,ikjb->abij', t2_12, ints.v_ooov)
    if 't2_12' in active:
        res += -2.0 * contract('bk,acil,jklc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,acil,jlkc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,acli,jklc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += 1.0 * contract('bk,aclj,ilkc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += -1.0 * contract('bk,cdij,kdac->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += -1.0 * contract('bk,jkia->abij', t2_12, ints.v_ooov)
    if 't2_12' in active:
        res += 1.0 * contract('ci,abkl,jlkc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += -1.0 * contract('ci,adkj,kcbd->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bdjk,kcad->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += 2.0 * contract('ci,bdjk,kdac->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += -1.0 * contract('ci,bdkj,kdac->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += 1.0 * contract('ci,jbac->abij', t2_12, ints.v_ovvv)
    if 't2_12' in active:
        res += 1.0 * contract('cj,abkl,iklc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += -1.0 * contract('cj,adik,kcbd->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += 2.0 * contract('cj,adik,kdbc->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += -1.0 * contract('cj,adki,kdbc->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += -1.0 * contract('cj,bdki,kcad->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += 1.0 * contract('cj,iabc->abij', t2_12, ints.v_ovvv)
    if 't2_12' in active:
        res += 1.0 * contract('ck,abil,jklc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,abil,jlkc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += 1.0 * contract('ck,ablj,iklc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += -2.0 * contract('ck,ablj,ilkc->abij', t2_12, t2_20, ints.v_ooov)
    if 't2_12' in active:
        res += 2.0 * contract('ck,adij,kcbd->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += -1.0 * contract('ck,adij,kdbc->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += 2.0 * contract('ck,bdji,kcad->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_12' in active:
        res += -1.0 * contract('ck,bdji,kdac->abij', t2_12, t2_20, ints.v_ovvv)
    if 't2_22' in active:
        res += -2.0 * contract('abik,cdjl,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('abik,cdjl,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('abkj,cdil,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('abkj,cdil,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('abkl,cdij,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('acij,bdkl,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('acij,bdkl,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 4.0 * contract('acik,bdjl,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('acik,bdjl,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('acik,bdlj,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('acik,bdlj,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('acki,bdjl,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('acki,bdjl,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('acki,bdlj,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('ackj,bdli,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('ackl,bdji,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('ackl,bdji,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('bcji,adkl,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('bcji,adkl,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 4.0 * contract('bcjk,adil,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('bcjk,adil,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('bcjk,adli,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('bcjk,adli,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('bcki,adlj,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('bckj,adil,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('bckj,adil,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('bckj,adli,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('bckl,adij,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('bckl,adij,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('cdij,abkl,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('cdik,ablj,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('cdik,ablj,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += 1.0 * contract('cdjk,abil,kcld->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -2.0 * contract('cdjk,abil,kdlc->abij', t2_20, t2_22, ints.v_ovov)
    if 't2_21' in active:
        res += -4.0 * contract('abik,cdjl,kcld->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 2.0 * contract('abik,cdjl,kdlc->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -4.0 * contract('abkj,cdil,kcld->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 2.0 * contract('abkj,cdil,kdlc->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 2.0 * contract('abkl,cdij,kcld->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -4.0 * contract('acij,bdkl,kcld->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 2.0 * contract('acij,bdkl,kdlc->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 8.0 * contract('acik,bdjl,kcld->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -4.0 * contract('acik,bdjl,kdlc->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -4.0 * contract('acik,bdlj,kcld->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 2.0 * contract('acik,bdlj,kdlc->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -4.0 * contract('acki,bdjl,kcld->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 2.0 * contract('acki,bdjl,kdlc->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 2.0 * contract('acki,bdlj,kcld->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 2.0 * contract('ackj,bdli,kdlc->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += 2.0 * contract('ackl,bdji,kcld->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_21' in active:
        res += -4.0 * contract('ackl,bdji,kdlc->abij', t2_21, t2_21, ints.v_ovov)
    if 't2_22' in active:
        res += 2.0 * w * contract('abij->abij', t2_22)
    if 't2_22' in active:
        res += 1.0 * contract('abkl,ikjl->abij', t2_22, ints.v_oooo)
    if 't2_22' in active:
        res += -1.0 * contract('acik,jkbc->abij', t2_22, ints.v_oovv)
    if 't2_22' in active:
        res += 2.0 * contract('acik,jbkc->abij', t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -1.0 * contract('acki,jbkc->abij', t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -1.0 * contract('ackj,ikbc->abij', t2_22, ints.v_oovv)
    if 't2_22' in active:
        res += -1.0 * contract('bcjk,ikac->abij', t2_22, ints.v_oovv)
    if 't2_22' in active:
        res += 2.0 * contract('bcjk,iakc->abij', t2_22, ints.v_ovov)
    if 't2_22' in active:
        res += -1.0 * contract('bcki,jkac->abij', t2_22, ints.v_oovv)
    if 't2_22' in active:
        res += -1.0 * contract('bckj,iakc->abij', t2_22, ints.v_ovov)
    if 't2_22' in active:
        _W = 1.0 * t2_22
        _vvvv_ladder(ints.B_vv, np.ascontiguousarray(_W), out=res, alpha=1.0)

    # exact (ab)(ij) pair symmetry of the mixed-spin block; enforce against rounding drift
    res = 0.5 * (res + res.transpose(1, 0, 3, 2))
    for i in range(nocc):
        res[:, :, i, :] *= 1.0 / (ints.eps_occ[i] + ints.eps_occ[None, None, :]
                                  - ints.eps_vir[:, None, None]
                                  - ints.eps_vir[None, :, None] - (2.0 * w))
    res += t2_22
    return res

# ---------------------------------------------------------------------------
# Spatial-orbital integral setup (built once per run)
# ---------------------------------------------------------------------------
class _Ints:
    """Namespace holding the spatial Fock/dipole blocks, the DF factor's
    all-virtual slice, and the precomputed 4-index chemist blocks."""


def _build_ints(qedhf, frozen=0):
    if 'Ca' in qedhf:
        raise NotImplementedError(
            "qed_ccsd_rhf requires a restricted (closed-shell) QED-HF "
            "reference; use qed_ccsd_uhf for QED-UHF")
    omega = qedhf['omega']
    lambda_x, lambda_y, lambda_z = qedhf['lambda_cav']
    C = np.asarray(qedhf['C'])
    F_ao = np.asarray(qedhf['F'])
    mu_x_ao = qedhf['mu_x_ao']
    mu_y_ao = qedhf['mu_y_ao']
    mu_z_ao = qedhf['mu_z_ao']
    nocc = qedhf['nocc_spatial']
    nmo = qedhf['nmo_spatial']

    B_ao = _ao_df_factor(qedhf)
    B = contract('pi,Ppq,qj->Pij', C, B_ao, C)          # (P, nmo, nmo)

    cf_x = lambda_x * math.sqrt(omega / 2.0)
    cf_y = lambda_y * math.sqrt(omega / 2.0)
    cf_z = lambda_z * math.sqrt(omega / 2.0)
    dip = (cf_x * (C.T @ mu_x_ao @ C) + cf_y * (C.T @ mu_y_ao @ C)
           + cf_z * (C.T @ mu_z_ao @ C))
    f = C.T @ F_ao @ C

    if frozen:
        f = f[frozen:, frozen:]
        dip = dip[frozen:, frozen:]
        B = B[:, frozen:, frozen:]
        nocc -= frozen
        nmo -= frozen
    nvir = nmo - nocc

    o = slice(None, nocc)
    v = slice(nocc, None)
    ints = _Ints()
    eps = f.diagonal()
    ints.eps_occ = np.ascontiguousarray(eps[o])
    ints.eps_vir = np.ascontiguousarray(eps[v])
    ints.f_oo = np.ascontiguousarray(f[o, o])
    ints.f_vo = np.ascontiguousarray(f[v, o])
    ints.f_vv = np.ascontiguousarray(f[v, v])
    ints.d_oo = -np.ascontiguousarray(dip[o, o])
    ints.d_vo = -np.ascontiguousarray(dip[v, o])
    ints.d_vv = -np.ascontiguousarray(dip[v, v])
    ints.B_vv = np.ascontiguousarray(B[:, v, v])

    # 4-index chemist blocks (pq|rs); (vv|vv) intentionally absent
    sl = {'o': o, 'v': v}
    for name in _NEEDED_V:
        c1, c2 = name[2:4], name[4:6]
        ints.__dict__[name] = contract(
            'xpq,xrs->pqrs',
            B[:, sl[c1[0]], sl[c1[1]]], B[:, sl[c2[0]], sl[c2[1]]])

    return ints, nocc, nvir


# ---------------------------------------------------------------------------
# QED-CCSD driver (closed-shell, spatial orbitals)
# ---------------------------------------------------------------------------
def run_qed_ccsd(qedhf, do_t1_01=True, do_t2_11=True, do_t2_21=True,
                 do_t2_02=False, do_t2_12=False, do_t2_22=False,
                 frozen=0, max_iter=50, tol=1e-8, tol_amp=1e-7,
                 max_diis=8, diis_on_disk=False, verbose=True):
    """Spin-adapted closed-shell QED-CCSD on a restricted QED-HF reference.

    Same flavour flags, convergence criteria and return dict as
    qed_ccsd.run_qed_ccsd; requires a closed-shell reference. The
    amplitude convention for the returned doubles is the mixed-spin block
    T[a,b,i,j] (spatial orbitals).
    """
    omega = qedhf['omega']
    lambda_x, lambda_y, lambda_z = qedhf['lambda_cav']
    E_qed_hf_ref = qedhf['E_qed_hf']
    E_hf_ref = qedhf.get('E_rhf')

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
        naux = ints.B_vv.shape[0]
        print(f"\nQED-CCSD (closed-shell, DF): flavour = {flavour}")
        print(f"  nocc = {nocc}, nvir = {nvir} (spatial), naux = {naux},"
              f" frozen = {frozen}, integral setup {t_setup:.1f} s")
        print(f"  omega = {omega:.6f} Ha,"
              f"  lambda = ({lambda_x},{lambda_y},{lambda_z})")

    t1_10 = np.zeros((nvir, nocc))
    t1_01 = 0.0
    t2_20 = np.zeros((nvir, nvir, nocc, nocc))
    t2_02 = 0.0
    t2_11 = np.zeros((nvir, nocc)) if do_t2_11 else np.zeros((1, 1))
    t2_21 = (np.zeros((nvir, nvir, nocc, nocc)) if do_t2_21
             else np.zeros((1, 1, 1, 1)))
    t2_12 = np.zeros((nvir, nocc)) if do_t2_12 else np.zeros((1, 1))
    t2_22 = (np.zeros((nvir, nvir, nocc, nocc)) if do_t2_22
             else np.zeros((1, 1, 1, 1)))

    diis_names = ['t1_10', 't2_20']
    for flag, name in ((do_t2_11, 't2_11'), (do_t2_21, 't2_21'),
                       (do_t2_12, 't2_12'), (do_t2_22, 't2_22')):
        if flag:
            diis_names.append(name)
    hist = _DiisHistory(diis_names, max_diis, on_disk=diis_on_disk)
    zero_init = {
        't1_10': t1_10, 't2_20': t2_20, 't2_11': t2_11,
        't2_21': t2_21, 't2_12': t2_12, 't2_22': t2_22}
    hist.vals.append({k: hist._store(zero_init[k], f'val0_{k}')
                      for k in diis_names})

    kernel_args = lambda: (ints, omega, t1_10, t1_01, t2_20, t2_02,
                           t2_11, t2_21, t2_12, t2_22)

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

            old_t1_01 = float(t1_01)
            old_t2_02 = float(t2_02)
            err = {}
            step_sq = 0.0

            def _step(name, new, old):
                nonlocal step_sq
                e = new - old
                if name in diis_names:
                    err[name] = e
                step_sq += float(np.dot(e.ravel(), e.ravel()))
                return new

            t1_10 = _step('t1_10',
                          ccsd_t1_10(*kernel_args(), active=active), t1_10)
            if do_t1_01:
                t1_01 = ccsd_t1_01(*kernel_args(), active=active)
            t2_20 = _step('t2_20',
                          ccsd_t2_20(*kernel_args(), active=active), t2_20)
            if do_t2_02:
                t2_02 = ccsd_t2_02(*kernel_args(), active=active)
            if do_t2_11:
                t2_11 = _step('t2_11',
                              ccsd_t2_11(*kernel_args(), active=active),
                              t2_11)
            if do_t2_21:
                t2_21 = _step('t2_21',
                              ccsd_t2_21(*kernel_args(), active=active),
                              t2_21)
            if do_t2_12:
                t2_12 = _step('t2_12',
                              ccsd_t2_12(*kernel_args(), active=active),
                              t2_12)
            if do_t2_22:
                t2_22 = _step('t2_22',
                              ccsd_t2_22(*kernel_args(), active=active),
                              t2_22)

            E_CCSD_new = ccsd_energy(*kernel_args(), active=active)

            amp_norm = float(np.sqrt(
                step_sq
                + (float(t1_01) - old_t1_01) ** 2
                + (float(t2_02) - old_t2_02) ** 2))

            t_total = time.time() - t_start
            time_total += t_total
            if verbose:
                print('%3d:  %20.12f  %1.5E  %1.5E   %.3f'
                      % (ccsd_iter, E_CCSD_new,
                         abs(E_CCSD_new - E_CCSD_old), amp_norm, t_total))

            if abs(E_CCSD_new - E_CCSD_old) < tol and amp_norm < tol_amp:
                converged = True
                break

            cur = {'t1_10': t1_10, 't2_20': t2_20, 't2_11': t2_11,
                   't2_21': t2_21, 't2_12': t2_12, 't2_22': t2_22}
            hist.append({k: cur[k] for k in diis_names}, err)
            del err
            E_CCSD_old = E_CCSD_new

            new_amps = hist.extrapolate()
            if new_amps is not None:
                t1_10 = new_amps['t1_10']
                t2_20 = new_amps['t2_20']
                if do_t2_11:
                    t2_11 = new_amps['t2_11']
                if do_t2_21:
                    t2_21 = new_amps['t2_21']
                if do_t2_12:
                    t2_12 = new_amps['t2_12']
                if do_t2_22:
                    t2_22 = new_amps['t2_22']
        else:
            if verbose:
                print('Warning: QED-CCSD did not converge in %d iterations'
                      % max_iter)
    finally:
        hist.cleanup()

    return {
        'converged': bool(converged),
        'E_qed_ccsd_corr': float(E_CCSD_new),
        'E_qed_ccsd_total': float(E_CCSD_new) + E_qed_hf_ref,
        'E_qed_hf': E_qed_hf_ref,
        'E_rhf': E_hf_ref,
        'frozen': frozen,
        't1_10': t1_10,
        't1_01': t1_01,
        't2_20': t2_20,
        't2_02': t2_02,
        't2_11': t2_11,
        't2_21': t2_21,
        't2_12': t2_12,
        't2_22': t2_22,
        'iterations': ccsd_iter,
        'time_total': time_total,
    }


# ---------------------------------------------------------------------------
# Demo: glycolaldehyde / STO-3G QED-CCSD-21 (complete-equations value:
# -262.416985787; the published -262.416986187 lacks the quartic terms)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    mol = gto.M(
        atom="""
            C   0.00000000   0.00000000   0.00000000
            O   0.00000000   1.23456800   0.00000000
            H   0.97075033  -0.54577032   0.00000000
            C  -1.21509881  -0.80991169   0.00000000
            H  -1.15288176  -1.89931439   0.00000000
            C  -2.43440063  -0.19144555   0.00000000
            H  -3.37262777  -0.75937214   0.00000000
            O  -2.62194056   1.12501165   0.00000000
            H  -1.71446384   1.51627790   0.00000000
        """,
        basis='STO-3G', unit='Angstrom', symmetry=False, verbose=0)
    omega = 3.0 / 27.211386245988
    qedhf = run_qed_hf(mol, omega, (0.0, 0.0, 0.1), verbose=False)
    print(f"E_QED_HF = {qedhf['E_qed_hf']:.12f}")
    result = run_qed_ccsd(qedhf, do_t1_01=True, do_t2_11=True,
                          do_t2_21=True, verbose=True)
    print(f"\nE_QED_CCSD total = {result['E_qed_ccsd_total']:.12f}")
    print("spin-orbital complete-equations reference:"
          " -262.416985787002")
