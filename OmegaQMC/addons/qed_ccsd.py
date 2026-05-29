"""
Reference: https://pubs.acs.org/doi/10.1021/jacs.1c13201

QED-CCSD with complete double excitations (PySCF + JAX port).

t = t1_10 + t1_01 + t2_20 + t2_02 + t2_11 + t2_21 + t2_12 + t2_22

This is a port of the psi4 + numba reference (qed_ccsd.py at the repo
root) to PySCF for SCF/integrals and JAX for tensor contractions. The
residual equations themselves are unchanged — they were derived with
Wick (https://github.com/awhite862/wick) and are reused verbatim. The
inner loop is driven by ``jax.numpy.einsum`` so contractions can run
on GPU/TPU through XLA, while DIIS and the SCF iteration use plain
NumPy for small linear-algebra solves.

The module exposes:

* :func:`run_qed_hf`   — self-consistent QED-Hartree-Fock (dipole gauge,
  Pauli-Fierz Hamiltonian) returning the dressed mean-field object.
* :func:`run_qed_ccsd` — DIIS-accelerated QED-CCSD iteration on top of a
  QED-HF reference, with selectable photonic excitation levels:

    - conventional CCSD       (all do_* flags False)
    - QED-CCSD-21 / Deprince  (do_t1_01, do_t2_11, do_t2_21)
    - QED-CCSD-12 / White     (do_t1_01, do_t2_11, do_t2_02, do_t2_12)
    - QED-CCSD-22 / full      (all do_* flags True)

Because the residual equations are written in spin orbitals, they are
agnostic to the reference: ``run_qed_ccsd`` accepts either a restricted
QED-HF dict (:mod:`OmegaQMC.addons.qed_hf`) or a spin-unrestricted
QED-UHF dict (:mod:`OmegaQMC.addons.qed_uhf`, for open-shell systems)
and builds the spin-orbital tensors accordingly — only the integral
build differs.

Running the module directly reproduces the glycolaldehyde / STO-3G
demo from the reference.
"""

import math
import time

import numpy as _np
import jax
import jax.numpy as np  # all residual-equation contractions go through jax.numpy
import scipy.linalg as la

from pyscf import gto

from .qed_hf import run_qed_hf

# JAX's default dtype is float32. QED-CCSD energy convergence (~1e-8)
# requires float64; enable it once at import.
jax.config.update("jax_enable_x64", True)

# Validation note: with the geometry and parameters from the reference
# psi4 implementation (glycolaldehyde, STO-3G, ω = 3 eV, λ = (0, 0, 0.1))
# this port reproduces the published QED-CCSD-21 total energy
#     E(QED-CCSD-21) = -262.416986187232396
# to within ~2×10⁻¹⁰ Ha. At λ = 0 it reproduces pyscf's plain CCSD energy.


# ---------------------------------------------------------------------------
# QED-CCSD residual equations (derived with Wick:
#   https://github.com/awhite862/wick)
# Bodies are byte-for-byte the same as the reference psi4 implementation;
# only the contraction backend changed (jax.numpy.einsum).
# ---------------------------------------------------------------------------
def ccsd_t2_20(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir)

    g_vvoo = g_so[nocc:, nocc:, :nocc, :nocc]
    g_vvvv = g_so[nocc:, nocc:, nocc:, nocc:]
    g_oooo = g_so[:nocc, :nocc, :nocc, :nocc]
    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovoo = g_so[:nocc, nocc:, :nocc, :nocc]
    g_vvov = g_so[nocc:, nocc:, :nocc, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]
    d_vo = -dip[nocc:, :nocc]

    res_t2_20 = np.zeros((nvir, nvir, nocc, nocc))

    res_t2_20 += 1.0 * np.einsum('baji->abij', g_vvoo, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ki,bakj->abij', f_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kj,baki->abij', f_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bc,acji->abij', f_vv, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ac,bcji->abij', f_vv, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbji,ak->abij', g_ovoo, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kaji,bk->abij', g_ovoo, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('baic,cj->abij', g_vvov, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bajc,ci->abij', g_vvov, t1_10, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klji,balk->abij', g_oooo, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kbic,ackj->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kaic,bckj->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbjc,acki->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kajc,bcki->abij', g_ovov, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('bacd,dcji->abij', g_vvvv, t2_20, optimize=True)
    #res_t2_20 += 1.0 * np.einsum('I,baji->abijI', G, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bi,aj->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ai,bj->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bj,ai->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('aj,bi->abij', d_vo, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ci,bakj->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,cj,baki->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bk,acji->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,ak,bcji->abij', f_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klji,bk,al->abij', g_oooo, t1_10, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * t1_01 * np.einsum('ki,bakj->abij', d_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('kj,baki->abij', d_oo, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('bc,acji->abij', d_vv, t2_20, optimize=True)
    res_t2_20 += 1.0 * t1_01 *  np.einsum('ac,bcji->abij', d_vv, t2_20, optimize=True)
    res_t2_20 += 0.5 * np.einsum('klic,cj,balk->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klic,ck,balj->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('kljc,ci,balk->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += 0.5 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 0.5 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 0.25 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_20, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t1_10, t2_21, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_20, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t1_10, t1_10, optimize=True)
    res_t2_20 += 1.0 * t1_01 *  np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * t1_01 *  np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * t1_01 *  np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -0.5 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += -1.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t1_10, t2_20, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t1_10, t2_11, optimize=True)
    res_t2_20 += 1.0 * np.einsum('klcd,di,cj,bk,al->abij', g_oovv, t1_10, t1_10, t1_10, t1_10, optimize=True)


    t2_20 += np.einsum('abij,iajb -> abij', res_t2_20, e_denom, optimize=True)

    return t2_20

def ccsd_t2_21(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    eps_vir_p_w = eps[nocc:] + w
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir_p_w)

    g_vvvv = g_so[nocc:, nocc:, nocc:, nocc:]
    g_oooo = g_so[:nocc, :nocc, :nocc, :nocc]
    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovoo = g_so[:nocc, nocc:, :nocc, :nocc]
    g_vvov = g_so[nocc:, nocc:, :nocc, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]
    d_vo = -dip[nocc:, :nocc]

    res_t2_21 = np.zeros((nvir, nvir, nocc, nocc))

    res_t2_21 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,bakj->abij', f_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,baki->abij', f_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,acji->abij', f_vv, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,bcji->abij', f_vv, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbji,ak->abij', g_ovoo, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kaji,bk->abij', g_ovoo, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('baic,cj->abij', g_vvov, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bajc,ci->abij', g_vvov, t2_11, optimize = True)
    res_t2_21 += 1.0 * w * np.einsum('baji->abij', t2_21, optimize = True)
    #res_t2_21 += 1.0 * G * np.einsum('J,baji->abij', t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bi,aj->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ai,bj->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bj,ai->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('aj,bi->abij', d_vo, t2_12, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klji,balk->abij', g_oooo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbic,ackj->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kaic,bckj->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbjc,acki->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kajc,bcki->abij', g_ovov, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('bacd,dcji->abij', g_vvvv, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_22, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_22, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('ki,bakj->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('kj,baki->abij', d_oo, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('bc,acji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('ac,bcji->abij', d_vv, t2_20, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bakj->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,baki->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,acji->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,bcji->abij', f_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bcji,ak->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,acji,bk->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,baki,cj->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bakj,ci->abij', f_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klji,bk,al->abij', g_oooo, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klji,ak,bl->abij', g_oooo, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbic,ak,cj->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kaic,bk,cj->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbjc,ak,ci->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kajc,bk,ci->abij', g_ovov, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bacd,dj,ci->abij', g_vvvv, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klic,cj,balk->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,ck,balj->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kljc,ci,balk->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,bakj,cl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,bckj,al->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,ackj,bl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klic,balk,cj->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,baki,cl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,bcki,al->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,acki,bl->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kljc,balk,ci->abij', g_ooov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,adji,ck->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('kbcd,dcji,ak->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,bdji,ck->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('kacd,dcji,bk->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,adki,cj->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,bdki,cj->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,adkj,ci->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,bdkj,ci->abij', g_ovvv, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += 1.0 * t2_02 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -1.0 * t2_02 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_20, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.25 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,bakj,dcli->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,bdkj,acli->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,adkj,bcli->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,dckj,bali->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.25 * np.einsum('klcd,balk,dcji->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bdlk,acji->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,adlk,bcji->abij', g_oovv, t2_20, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ki,bj,ak->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ki,aj,bk->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kj,bi,ak->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kj,ai,bk->abij', d_oo, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('bc,ai,cj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('ac,bi,cj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bi,ackj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ai,bckj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 2.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bj,acki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,aj,bcki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -2.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 2.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -2.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klic,cj,ak,bl->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klic,bk,al,cj->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kljc,ci,ak,bl->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kljc,bk,al,ci->abij', g_ooov, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kbcd,di,ak,cj->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kacd,di,bk,cj->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kbcd,dj,ak,ci->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kacd,dj,bk,ci->abij', g_ovvv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * t1_01 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * t1_01 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t1_10, t2_21, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,di,bakj,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,di,bckj,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,di,ackj,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,di,balk,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dj,baki,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dj,bcki,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dj,acki,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,bk,adji,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -0.5 * np.einsum('klcd,bk,dcji,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,ak,bdji,cl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dk,bcji,al->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,ak,dcji,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dk,acji,bl->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,bk,adli,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,ak,bdli,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,dk,bali,cj->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 0.5 * np.einsum('klcd,dj,balk,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,bk,adlj,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('klcd,ak,bdlj,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('klcd,dk,balj,ci->abij', g_oovv, t1_10, t2_20, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ci,bj,ak->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ci,aj,bk->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,cj,bi,ak->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,cj,ai,bk->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,bk,ai,cj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,bk,ci,aj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += 1.0 * np.einsum('kc,ak,bi,cj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_21 += -1.0 * np.einsum('kc,ak,ci,bj->abij', d_ov, t1_10, t2_11, t2_11, optimize = True)

    t2_21 += np.einsum('abij,iajb -> abij', res_t2_21, e_denom, optimize=True)

    return t2_21

def ccsd_t2_22(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    eps_vir_p_2w = eps[nocc:] + 2 * w
    e_denom = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir_p_2w)

    g_vvvv = g_so[nocc:, nocc:, nocc:, nocc:]
    g_oooo = g_so[:nocc, :nocc, :nocc, :nocc]
    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovoo = g_so[:nocc, nocc:, :nocc, :nocc]
    g_vvov = g_so[nocc:, nocc:, :nocc, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]

    res_t2_22 = np.zeros((nvir, nvir, nocc, nocc))

    res_t2_22 += 1.0 * np.einsum('ki,bakj->abij', f_oo, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,baki->abij', f_oo, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,acji->abij', f_vv, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bcji->abij', f_vv, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbji,ak->abij', g_ovoo, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kaji,bk->abij', g_ovoo, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('baic,cj->abij', g_vvov, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bajc,ci->abij', g_vvov, t2_12, optimize = True)
    res_t2_22 += 1.0 * w * np.einsum('baji->abij', t2_22, optimize = True)
    res_t2_22 += 1.0 * w * np.einsum('baji->abij', t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klji,balk->abij', g_oooo, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbic,ackj->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kaic,bckj->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbjc,acki->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kajc,bcki->abij', g_ovov, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('bacd,dcji->abij', g_vvvv, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', f_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', f_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klji,bk,al->abij', g_oooo, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klji,ak,bl->abij', g_oooo, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbic,ak,cj->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kaic,bk,cj->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbjc,ak,ci->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kajc,bk,ci->abij', g_ovov, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bacd,dj,ci->abij', g_vvvv, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('ki,bakj->abij', d_oo, t2_22, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kj,baki->abij', d_oo, t2_22, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ki,bakj->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kj,baki->abij', d_oo, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('bc,acji->abij', d_vv, t2_22, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('ac,bcji->abij', d_vv, t2_22, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('bc,acji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('ac,bcji->abij', d_vv, t2_21, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klic,cj,balk->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,ck,balj->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kljc,ci,balk->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,bakj,cl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,bckj,al->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,ackj,bl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klic,balk,cj->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,baki,cl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,bcki,al->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,acki,bl->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kljc,balk,ci->abij', g_ooov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,adji,ck->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('kbcd,dcji,ak->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,bdji,ck->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('kacd,dcji,bk->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,adki,cj->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,bdki,cj->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,adkj,ci->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,bdkj,ci->abij', g_ovvv, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.25 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,bakj,dcli->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bdkj,acli->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,adkj,bcli->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,dckj,bali->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.25 * np.einsum('klcd,balk,dcji->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bdlk,acji->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,adlk,bcji->abij', g_oovv, t2_20, t2_22, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,ci,bakj->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,cj,baki->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,bk,acji->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,ak,bcji->abij', f_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klji,bk,al->abij', g_oooo, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kbic,cj,ak->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kaic,cj,bk->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbjc,ci,ak->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kajc,ci,bk->abij', g_ovov, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('bacd,di,cj->abij', g_vvvv, t2_11, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,bk,aj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ki,ak,bj->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,bk,ai->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kj,ak,bi->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ki,bj,ak->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ki,aj,bk->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kj,bi,ak->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kj,ai,bk->abij', d_oo, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,ci,aj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,ci,bj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bc,cj,ai->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ac,cj,bi->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('bc,ai,cj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('ac,bi,cj->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('bc,aj,ci->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('ac,bj,ci->abij', d_vv, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,cj,balk->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,bk,aclj->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klic,ak,bclj->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,ck,balj->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,ci,balk->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,bk,acli->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kljc,ak,bcli->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,ck,bali->abij', g_ooov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,di,ackj->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,di,bckj->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kbcd,dj,acki->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kacd,dj,bcki->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,ak,dcji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,dk,acji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,bk,dcji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,dk,bcji->abij', g_ovvv, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ck,baji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bi,ackj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ai,bckj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bj,acki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,aj,bcki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bcki,aj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,acki,bj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bckj,ai->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ackj,bi->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,baji,ck->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bcji,ak->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,acji,bk->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,baki,cj->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bakj,ci->abij', d_ov, t2_21, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klic,cj,ak,bl->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klic,bk,al,cj->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kljc,ci,ak,bl->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kljc,bk,al,ci->abij', g_ooov, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kbcd,di,ak,cj->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kacd,di,bk,cj->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kbcd,dj,ak,ci->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kacd,dj,bk,ci->abij', g_ovvv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,ci,bakj->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,cj,baki->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bk,acji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,ak,bcji->abij', d_ov, t1_10, t2_21, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * t1_01 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * t1_01 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,bcji,ak->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,acji,bk->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * t2_02 * np.einsum('kc,baki,cj->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += 1.0 * t2_02 * np.einsum('kc,bakj,ci->abij', d_ov, t2_20, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bdji,aclk->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,adji,bclk->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,dcji,balk->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,baki,dclj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bdki,aclj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,adki,bclj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dcki,balj->abij', g_oovv, t2_21, t2_21, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t1_10, t2_22, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,bakj,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,di,bckj,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,ackj,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,di,balk,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,baki,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dj,bcki,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,acki,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bk,adji,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -0.5 * np.einsum('klcd,bk,dcji,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,ak,bdji,cl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dk,bcji,al->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,ak,dcji,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dk,acji,bl->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,bk,adli,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,ak,bdli,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dk,bali,cj->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 0.5 * np.einsum('klcd,dj,balk,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bk,adlj,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,ak,bdlj,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dk,balj,ci->abij', g_oovv, t1_10, t2_20, t2_12, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,cj,bk,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klic,bk,cj,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klic,ak,cj,bl->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,ci,bk,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kljc,bk,ci,al->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kljc,ak,ci,bl->abij', g_ooov, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,di,cj,ak->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,di,cj,bk->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kbcd,dj,ci,ak->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kacd,dj,ci,bk->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kbcd,ak,di,cj->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kacd,bk,di,cj->abij', g_ovvv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,ci,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,ci,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bk,cj,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ak,cj,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,bk,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ci,ak,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,ci,aj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,ci,bj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,bk,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,cj,ak,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bk,cj,ai->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ak,cj,bi->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ci,bj,ak->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ci,aj,bk->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,cj,bi,ak->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,cj,ai,bk->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,bk,ai,cj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,ak,bi,cj->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 1.0 * np.einsum('kc,bk,aj,ci->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += -1.0 * np.einsum('kc,ak,bj,ci->abij', d_ov, t1_10, t2_11, t2_12, optimize = True)
    res_t2_22 += 2.0 * t1_01 * np.einsum('kc,ci,bakj->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * t1_01 * np.einsum('kc,cj,baki->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * t1_01 * np.einsum('kc,bk,acji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * t1_01 * np.einsum('kc,ak,bcji->abij', d_ov, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,di,cj,balk->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,di,bk,aclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,di,ak,bclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,di,ck,balj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,dj,ci,balk->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bk,di,aclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,ak,di,bclj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dk,ci,balj->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dj,bk,acli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,dj,ak,bcli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dj,ck,bali->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,bk,dj,acli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,ak,dj,bcli->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,dk,cj,bali->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,bk,al,dcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bk,dl,acji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 1.0 * np.einsum('klcd,ak,bl,dcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,dk,bl,acji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,ak,dl,bcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,dk,al,bcji->abij', g_oovv, t1_10, t2_11, t2_21, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bdji,ak,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,adji,bk,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,dcji,bk,al->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,baki,dj,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,bdki,cj,al->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,adki,cj,bl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,bakj,di,cl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('klcd,bdkj,ci,al->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('klcd,adkj,ci,bl->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += -1.0 * np.einsum('klcd,balk,di,cj->abij', g_oovv, t2_20, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,bi,cj,ak->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,ci,bj,ak->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)
    res_t2_22 += -2.0 * np.einsum('kc,ai,cj,bk->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)
    res_t2_22 += 2.0 * np.einsum('kc,ci,aj,bk->abij', d_ov, t2_11, t2_11, t2_11, optimize = True)

    t2_22 += np.einsum('abij,iajb -> abij', res_t2_22, e_denom, optimize=True)

    return t2_22


def ccsd_t1_10(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]
    e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]
    f_vo = f_so[nocc:, :nocc]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]
    d_vo = -dip[nocc:, :nocc]

    res_t1_10 = np.zeros((nvir, nocc))

    res_t1_10 += 1.0 * np.einsum('ai->ai', f_vo, optimize=True)
    res_t1_10 += -1.0 * np.einsum('ji,aj->ai', f_oo, t1_10, optimize=True)
    res_t1_10 += 1.0 * np.einsum('ab,bi->ai', f_vv, t1_10, optimize=True)
    res_t1_10 += 1.0 * t1_01 * np.einsum('ai->ai', d_vo, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,abji->ai', f_ov, t2_20, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jaib,bj->ai', g_ovov, t1_10, optimize=True)
    res_t1_10 += 0.5 * np.einsum('jkib,abkj->ai', g_ooov, t2_20, optimize=True)
    res_t1_10 += -0.5 * np.einsum('jabc,cbji->ai', g_ovvv, t2_20, optimize=True)
    #res_t1_10 += 1.0 * np.einsum('ai->aiI', G, t2_11, optimize=True)
    res_t1_10 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize=True)
    res_t1_10 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,bi,aj->ai', f_ov, t1_10, t1_10, optimize=True)
    res_t1_10 += -1.0 * t1_01 * np.einsum('ji,aj->ai', d_oo, t1_10, optimize=True)
    res_t1_10 += 1.0 * t1_01 * np.einsum('ab,bi->ai', d_vv, t1_10, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t1_10, t1_10, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t1_10, t1_10, optimize=True)
    res_t1_10 += -1.0 * t1_01 * np.einsum('jb,abji->ai', d_ov, t2_20, optimize=True)
    res_t1_10 += -0.5 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t1_10, t2_20, optimize=True)
    res_t1_10 += -0.5 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t1_10, t2_20, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t1_10, t2_20, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize=True)
    res_t1_10 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t1_10, t2_11, optimize=True)
    res_t1_10 += -1.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t1_10, optimize=True)
    res_t1_10 += 1.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t1_10, t1_10, optimize=True)

    t1_10 += np.einsum('ai,ia -> ai', res_t1_10, e_denom, optimize=True)

    return t1_10

def ccsd_t1_01(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]
    d_ov = -dip[:nocc, nocc:]

    res_t1_01 = 0
    G = 0
    #res_t1_01 += 1.0 * G
    res_t1_01 += 1.0 * w * t1_01
    res_t1_01 += 1.0 * G * t2_02
    res_t1_01 += 1.0 * np.einsum('ia,ai->', d_ov, t1_10, optimize=True)
    res_t1_01 += 1.0 * np.einsum('ia,ai->', f_ov, t2_11, optimize=True)
    res_t1_01 += 1.0 * np.einsum('ia,ai->', d_ov, t2_12, optimize=True)
    res_t1_01 += 1.0 * t2_02 * np.einsum('ia,ai->', d_ov, t1_10, optimize=True)
    res_t1_01 += 0.25 * np.einsum('ijab,baji->', g_oovv, t2_21, optimize=True)
    res_t1_01 += 1.0 * t1_01 * np.einsum('ia,ai->', d_ov, t2_11, optimize=True)
    res_t1_01 += -1.0 * np.einsum('ijab,bi,aj->', g_oovv, t1_10, t2_11, optimize=True)

    if w == 0:
        t1_01 = 0
    else:
        t1_01 += -res_t1_01 / w

    return t1_01

def ccsd_t2_02(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:]

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]
    d_ov = -dip[:nocc, nocc:]

    res_t2_02 = 0

    res_t2_02 += 1.0 * w * t2_02
    res_t2_02 += 1.0 * w * t2_02
    res_t2_02 += 1.0 * np.einsum('ia,ai->', f_ov, t2_12)
    res_t2_02 += 1.0 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += 1.0 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += 0.25 * np.einsum('ijab,baji->', g_oovv, t2_22)
    res_t2_02 += 1.0 * t1_01 * np.einsum('ia,ai->', d_ov, t2_12)
    res_t2_02 += 1.0 * t2_02 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += 1.0 * t2_02 * np.einsum('ia,ai->', d_ov, t2_11)
    res_t2_02 += -1.0 * np.einsum('ijab,bi,aj->', g_oovv, t1_10, t2_12)
    res_t2_02 += -1.0 * np.einsum('ijab,bi,aj->', g_oovv, t2_11, t2_11)

    if w == 0:
        t2_02 = 0
    else:
        t2_02 += -res_t2_02 / (2 * w)

    return t2_02

def ccsd_t2_11(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:] + w
    e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]
    d_vo = -dip[nocc:, :nocc]

    res_t2_11 = np.zeros((nvir, nocc))

    res_t2_11 += 1.0 * np.einsum('ai->ai', d_vo, optimize = True)
    res_t2_11 += -1.0 * np.einsum('ji,aj->ai', d_oo, t1_10, optimize = True)
    res_t2_11 += 1.0 * t2_02 * np.einsum('ai->ai', d_vo, optimize = True)
    res_t2_11 += 1.0 * np.einsum('ab,bi->ai', d_vv, t1_10, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_20, optimize = True)
    res_t2_11 += -1.0 * np.einsum('ji,aj->ai', f_oo, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('ab,bi->ai', f_vv, t2_11, optimize = True)
    res_t2_11 += 1.0 * w * np.einsum('ai->ai', t2_11, optimize = True)
    #res_t2_11 += 1.0 * G * np.einsum('ai->ai', t2_12, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,abji->ai', f_ov, t2_21, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jaib,bj->ai', g_ovov, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_12, optimize = True)
    res_t2_11 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_12, optimize = True)
    res_t2_11 += -1.0 * t2_02 * np.einsum('ji,aj->ai', d_oo, t1_10, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t1_10, optimize = True)
    res_t2_11 += 1.0 * t2_02 * np.einsum('ab,bi->ai', d_vv, t1_10, optimize = True)
    res_t2_11 += 0.5 * np.einsum('jkib,abkj->ai', g_ooov, t2_21, optimize = True)
    res_t2_11 += -0.5 * np.einsum('jabc,cbji->ai', g_ovvv, t2_21, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_22, optimize = True)
    res_t2_11 += -1.0 * t2_02 * np.einsum('jb,abji->ai', d_ov, t2_20, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,bi,aj->ai', f_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,aj,bi->ai', f_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_11 += 1.0 * t1_01 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t1_10, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkib,bj,ak->ai', g_ooov, t1_10, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jabc,cj,bi->ai', g_ovvv, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_11 += -1.0 * t2_02 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t1_10, optimize = True)
    res_t2_11 += -0.5 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t1_10, t2_21, optimize = True)
    res_t2_11 += -0.5 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t1_10, t2_21, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t1_10, t2_21, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkbc,acji,bk->ai', g_oovv, t2_20, t2_11, optimize = True)
    res_t2_11 += 0.5 * np.einsum('jkbc,cbji,ak->ai', g_oovv, t2_20, t2_11, optimize = True)
    res_t2_11 += 0.5 * np.einsum('jkbc,ackj,bi->ai', g_oovv, t2_20, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jb,ai,bj->ai', d_ov, t2_11, t2_11, optimize = True)
    res_t2_11 += -2.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_11, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * t1_01 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_11 += 1.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jkbc,ci,bj,ak->ai', g_oovv, t1_10, t1_10, t2_11, optimize = True)
    res_t2_11 += -1.0 * np.einsum('jkbc,aj,ck,bi->ai', g_oovv, t1_10, t1_10, t2_11, optimize = True)

    t2_11 += np.einsum('ai,ia -> ai', res_t2_11, e_denom, optimize=True)

    return t2_11

def ccsd_t2_12(f_so, g_so, dip, G, w, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22):
    nvir = t2_20.shape[0]
    nocc = t2_20.shape[2]

    eps = f_so.diagonal()
    eps_occ = eps[:nocc]
    eps_vir = eps[nocc:] + 2 * w
    e_denom = 1 / (eps_occ.reshape(-1, 1) - eps_vir)

    g_oovv = g_so[:nocc, :nocc, nocc:, nocc:]
    g_ovov = g_so[:nocc, nocc:, :nocc, nocc:]
    g_ooov = g_so[:nocc, :nocc, :nocc, nocc:]
    g_ovvv = g_so[:nocc, nocc:, nocc:, nocc:]

    f_oo = f_so[:nocc, :nocc]
    f_vv = f_so[nocc:, nocc:]
    f_ov = f_so[:nocc, nocc:]

    d_oo = -dip[:nocc, :nocc]
    d_vv = -dip[nocc:, nocc:]
    d_ov = -dip[:nocc, nocc:]

    res_t2_12 = np.zeros((nvir, nocc))

    res_t2_12 += -1.0 * np.einsum('ji,aj->ai', f_oo, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('ab,bi->ai', f_vv, t2_12, optimize = True)
    res_t2_12 += 1.0 * w * np.einsum('ai->ai', t2_12, optimize = True)
    res_t2_12 += 1.0 * w * np.einsum('ai->ai', t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += 1.0 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,abji->ai', f_ov, t2_22, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jaib,bj->ai', g_ovov, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += 0.5 * np.einsum('jkib,abkj->ai', g_ooov, t2_22, optimize = True)
    res_t2_12 += -0.5 * np.einsum('jabc,cbji->ai', g_ovvv, t2_22, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', f_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', f_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('ji,aj->ai', d_oo, t2_12, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('ji,aj->ai', d_oo, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += 1.0 * t1_01 * np.einsum('ab,bi->ai', d_vv, t2_12, optimize = True)
    res_t2_12 += 1.0 * t2_02 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += 1.0 * t2_02 * np.einsum('ab,bi->ai', d_vv, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t1_10, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkib,bj,ak->ai', g_ooov, t1_10, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jabc,cj,bi->ai', g_ovvv, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('jb,abji->ai', d_ov, t2_22, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,abji->ai', d_ov, t2_21, optimize = True)
    res_t2_12 += -0.5 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t1_10, t2_22, optimize = True)
    res_t2_12 += -0.5 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t1_10, t2_22, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t1_10, t2_22, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkbc,acji,bk->ai', g_oovv, t2_20, t2_12, optimize = True)
    res_t2_12 += 0.5 * np.einsum('jkbc,cbji,ak->ai', g_oovv, t2_20, t2_12, optimize = True)
    res_t2_12 += 0.5 * np.einsum('jkbc,ackj,bi->ai', g_oovv, t2_20, t2_12, optimize = True)
    res_t2_12 += -2.0 * np.einsum('jb,bi,aj->ai', f_ov, t2_11, t2_11, optimize = True)
    res_t2_12 += -2.0 * np.einsum('jkib,aj,bk->ai', g_ooov, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jabc,ci,bj->ai', g_ovvv, t2_11, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jb,bj,ai->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jb,ai,bj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jb,aj,bi->ai', d_ov, t2_11, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t1_01 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,bi,aj->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * t2_02 * np.einsum('jb,aj,bi->ai', d_ov, t1_10, t2_11, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,ci,abkj->ai', g_oovv, t2_11, t2_21, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,aj,cbki->ai', g_oovv, t2_11, t2_21, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,cj,abki->ai', g_oovv, t2_11, t2_21, optimize = True)
    res_t2_12 += 1.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,ci,bj,ak->ai', g_oovv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_12 += -1.0 * np.einsum('jkbc,aj,ck,bi->ai', g_oovv, t1_10, t1_10, t2_12, optimize = True)
    res_t2_12 += -2.0 * t1_01 * np.einsum('jb,bi,aj->ai', d_ov, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,ci,aj,bk->ai', g_oovv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,aj,ci,bk->ai', g_oovv, t1_10, t2_11, t2_11, optimize = True)
    res_t2_12 += 2.0 * np.einsum('jkbc,cj,bi,ak->ai', g_oovv, t1_10, t2_11, t2_11, optimize = True)

    t2_12 += np.einsum('ai,ia -> ai', res_t2_12, e_denom, optimize=True)

    return t2_12

# ---------------------------------------------------------------------------
# Helper functions (numpy — these run once per QED-CCSD calculation, not on
# the inner contraction loop, so there's no benefit to going through JAX).
# ---------------------------------------------------------------------------

def _spin_block_tei(I):
    """Spin-block a chemist-notation (pq|rs) tensor into spin orbitals."""
    identity = _np.eye(2)
    I = _np.kron(identity, I)
    return _np.kron(identity, I.T)


def _ao_to_mo_transform_full(h_core, g_ao, C):
    """Transform 1- and 2-electron integrals AO → MO (chemist notation)."""
    tmp = _np.einsum('pi,pqrs->iqrs', C, g_ao, optimize=True)
    tmp = _np.einsum('iqrs,rj->iqjs', tmp, C, optimize=True)
    tmp = _np.einsum('qa,iqjs->iajs', C, tmp, optimize=True)
    g_mo = _np.einsum('iajs,sb->iajb', tmp, C, optimize=True)
    h_mo = _np.einsum('pi,pq,qj -> ij', C, h_core, C, optimize=True)
    return h_mo, g_mo


def _sf_to_so(g_mo_sf):
    """Antisymmetrize a spatial chemist tensor into spin orbitals.

    Spin index convention: even = alpha, odd = beta. The output is the
    antisymmetrized chemist-notation tensor ``<pq|rs> - <ps|rq>`` with
    spin selection rules built in. Vectorised replacement for the
    numba inner-loop sf_to_so in the reference code.
    """
    g_mo_sf = _np.asarray(g_mo_sf)
    nmo = g_mo_sf.shape[0]
    nso = 2 * nmo
    idx = _np.arange(nso)
    p = idx[:, None, None, None]
    q = idx[None, :, None, None]
    r = idx[None, None, :, None]
    s = idx[None, None, None, :]
    sp, sq, sr, ss = p % 2, q % 2, r % 2, s % 2
    p2, q2, r2, s2 = p // 2, q // 2, r // 2, s // 2
    g1 = g_mo_sf[p2, q2, r2, s2] * (sp == sq) * (sr == ss)
    g2 = g_mo_sf[p2, s2, r2, q2] * (sp == ss) * (sq == sr)
    return (g1 - g2).astype(_np.float64)


def _block_diag_spin(M_sf):
    """Map a spatial 2-index matrix into spin orbitals (αα/ββ blocks)."""
    return _np.kron(_np.asarray(M_sf), _np.eye(2))


def _dse_ao(qedhf, lambda_x, lambda_y, lambda_z):
    """DSE-dressed two-electron AO tensor g_ao = eri_ao + Σ_X λ_X² μ_X⊗μ_X."""
    mu_x_ao = qedhf['mu_x_ao']
    mu_y_ao = qedhf['mu_y_ao']
    mu_z_ao = qedhf['mu_z_ao']
    dse_ao = (lambda_x * lambda_x * _np.einsum('pq,rs->pqrs',
                                               mu_x_ao, mu_x_ao, optimize=True)
              + lambda_y * lambda_y * _np.einsum('pq,rs->pqrs',
                                                 mu_y_ao, mu_y_ao, optimize=True)
              + lambda_z * lambda_z * _np.einsum('pq,rs->pqrs',
                                                 mu_z_ao, mu_z_ao, optimize=True))
    return qedhf['eri_ao'] + dse_ao


def _build_rhf_so(qedhf):
    """Spin-orbital QED-CCSD tensors from a restricted QED-HF reference.

    This is the original closed-shell build: a single MO coefficient matrix
    ``C``, ``nocc = 2·nocc_spatial`` and the interleaved (even=α/odd=β)
    spin ordering produced by ``_sf_to_so`` / ``_block_diag_spin``.
    """
    omega = qedhf['omega']
    lambda_x, lambda_y, lambda_z = qedhf['lambda_cav']
    C = qedhf['C']
    F_ao = qedhf['F']
    H_core = qedhf['H_core']
    oei = qedhf['oei']
    mu_x_ao = qedhf['mu_x_ao']
    mu_y_ao = qedhf['mu_y_ao']
    mu_z_ao = qedhf['mu_z_ao']
    nocc = 2 * qedhf['nocc_spatial']     # closed shell → 2×nα occupied spin orbs
    nso = 2 * qedhf['nmo_spatial']

    g_ao = _dse_ao(qedhf, lambda_x, lambda_y, lambda_z)
    h_ao = H_core + oei
    _, g_mo_sf = _ao_to_mo_transform_full(h_ao, g_ao, C)
    g_so_chem = _sf_to_so(g_mo_sf)
    g_so = g_so_chem.transpose(0, 2, 1, 3)

    cf_x = lambda_x * math.sqrt(omega / 2.0)
    cf_y = lambda_y * math.sqrt(omega / 2.0)
    cf_z = lambda_z * math.sqrt(omega / 2.0)
    dip_x_mo = C.T @ mu_x_ao @ C
    dip_y_mo = C.T @ mu_y_ao @ C
    dip_z_mo = C.T @ mu_z_ao @ C
    dip_sf = cf_x * dip_x_mo + cf_y * dip_y_mo + cf_z * dip_z_mo
    f_sf = C.T @ F_ao @ C
    f_mo_np = _block_diag_spin(f_sf)
    dip_np = _block_diag_spin(dip_sf)

    return (np.asarray(f_mo_np), np.asarray(g_so), np.asarray(dip_np),
            nocc, nso)


def _build_uhf_so(qedhf):
    """Spin-orbital QED-CCSD tensors from a spin-unrestricted QED-UHF
    reference.

    The α and β orbitals differ, so the antisymmetrised spin-orbital ERI is
    assembled from the four spatial chemist blocks (αα/αβ/ββ/βα). The
    Fock and dipole are spin-block-diagonal (α-block from ``Ca``/``Fa``,
    β-block from ``Cb``/``Fb``). Spin orbitals are ordered occupied-first,
    ``[α-occ, β-occ, α-vir, β-vir]``, which is exactly what the
    reference-agnostic residual equations' ``[:nocc]`` / ``[nocc:]``
    slicing expects. Reduces to :func:`_build_rhf_so` (up to a spin-orbital
    relabelling, which leaves the CCSD energy invariant) when Ca = Cb.
    """
    omega = qedhf['omega']
    lambda_x, lambda_y, lambda_z = qedhf['lambda_cav']
    Ca = _np.asarray(qedhf['Ca'])
    Cb = _np.asarray(qedhf['Cb'])
    Fa = _np.asarray(qedhf['Fa'])
    Fb = _np.asarray(qedhf['Fb'])
    mu_x_ao = qedhf['mu_x_ao']
    mu_y_ao = qedhf['mu_y_ao']
    mu_z_ao = qedhf['mu_z_ao']
    nocc_a = qedhf['nocc_a']
    nocc_b = qedhf['nocc_b']
    nmo = qedhf['nmo_spatial']

    nocc = nocc_a + nocc_b
    nso = 2 * nmo

    # Four spatial chemist MO blocks of the DSE-dressed 2e tensor.
    g_ao = _dse_ao(qedhf, lambda_x, lambda_y, lambda_z)

    def _trans(Cp, Cq, Cr, Cs):
        return _np.einsum('pi,qj,rk,sl,pqrs->ijkl',
                          Cp, Cq, Cr, Cs, g_ao, optimize=True)

    # Gstack[s1, s2] = chemist (i j | k l) with electron-1 spin s1, e-2 s2.
    Gstack = _np.zeros((2, 2, nmo, nmo, nmo, nmo))
    Gstack[0, 0] = _trans(Ca, Ca, Ca, Ca)
    Gstack[1, 1] = _trans(Cb, Cb, Cb, Cb)
    Gstack[0, 1] = _trans(Ca, Ca, Cb, Cb)   # (αα|ββ)
    Gstack[1, 0] = _trans(Cb, Cb, Ca, Ca)   # (ββ|αα)

    # Occupied-first spin-orbital → (spin, spatial) maps.
    order = ([(0, p) for p in range(nocc_a)]
             + [(1, p) for p in range(nocc_b)]
             + [(0, p) for p in range(nocc_a, nmo)]
             + [(1, p) for p in range(nocc_b, nmo)])
    spins = _np.array([s for s, _ in order], dtype=int)
    spat = _np.array([p for _, p in order], dtype=int)

    idx = _np.arange(nso)
    P = idx[:, None, None, None]
    Q = idx[None, :, None, None]
    R = idx[None, None, :, None]
    S = idx[None, None, None, :]
    sP, sQ, sR, sS = spins[P], spins[Q], spins[R], spins[S]
    pP, pQ, pR, pS = spat[P], spat[Q], spat[R], spat[S]

    # Antisymmetrised chemist tensor: direct (P Q | R S) − exchange (P S | R Q),
    # with the spatial block selected by (electron-1 spin, electron-2 spin).
    g1 = Gstack[sP, sR, pP, pQ, pR, pS] * (sP == sQ) * (sR == sS)
    g2 = Gstack[sP, sR, pP, pS, pR, pQ] * (sP == sS) * (sR == sQ)
    g_so = (g1 - g2).transpose(0, 2, 1, 3)   # chemist → physicist

    # Spin-block-diagonal Fock and dipole.
    fa = Ca.T @ Fa @ Ca
    fb = Cb.T @ Fb @ Cb
    cf_x = lambda_x * math.sqrt(omega / 2.0)
    cf_y = lambda_y * math.sqrt(omega / 2.0)
    cf_z = lambda_z * math.sqrt(omega / 2.0)
    dipa = (cf_x * (Ca.T @ mu_x_ao @ Ca) + cf_y * (Ca.T @ mu_y_ao @ Ca)
            + cf_z * (Ca.T @ mu_z_ao @ Ca))
    dipb = (cf_x * (Cb.T @ mu_x_ao @ Cb) + cf_y * (Cb.T @ mu_y_ao @ Cb)
            + cf_z * (Cb.T @ mu_z_ao @ Cb))
    Fstack = _np.stack([fa, fb])
    Dstack = _np.stack([dipa, dipb])

    P2 = idx[:, None]
    Q2 = idx[None, :]
    same = (spins[P2] == spins[Q2])
    f_mo = Fstack[spins[P2], spat[P2], spat[Q2]] * same
    dip = Dstack[spins[P2], spat[P2], spat[Q2]] * same

    return np.asarray(f_mo), np.asarray(g_so), np.asarray(dip), nocc, nso


def _build_ccsd_so(qedhf):
    """Build the spin-orbital QED-CCSD tensors, dispatching on the
    reference type: a restricted QED-HF dict (key ``'C'``) or an
    unrestricted QED-UHF dict (key ``'Ca'``). Returns
    ``(f_mo, g_mo, dip, nocc, nso)``.
    """
    if 'Ca' in qedhf:
        return _build_uhf_so(qedhf)
    return _build_rhf_so(qedhf)


# ---------------------------------------------------------------------------
# QED-CCSD driver (the QED-HF reference comes from OmegaQMC.qed_hf.run_qed_hf).
# ---------------------------------------------------------------------------

def run_qed_ccsd(qedhf, do_t1_01=True, do_t2_11=True, do_t2_21=True,
                 do_t2_02=False, do_t2_12=False, do_t2_22=False,
                 max_iter=50, tol=1e-8, max_diis=20, verbose=True):
    """DIIS-accelerated QED-CCSD on a QED-HF reference.

    The set of active photonic amplitudes selects the flavour:

    * none of do_t1_01..do_t2_22 → conventional CCSD
    * do_t1_01, do_t2_11, do_t2_21 → QED-CCSD-1 / Deprince (a.k.a. QED-CCSD-21)
    * do_t1_01, do_t2_11, do_t2_02, do_t2_12 → QED-CCSD-12 / White
    * all flags True → QED-CCSD-22 (full)

    Args:
        qedhf: dict returned by :func:`run_qed_hf`.
        do_*: enable individual photonic excitation classes.
        max_iter: max CCSD iterations.
        tol: energy convergence threshold.
        max_diis: DIIS history depth.
        verbose: print per-iteration progress.

    Returns:
        dict with the correlation and total QED-CCSD energy, the
        converged amplitudes, and the QED-HF reference energy.
    """
    omega = qedhf['omega']
    lambda_x, lambda_y, lambda_z = qedhf['lambda_cav']
    # Reference energies — keys differ between QED-HF (restricted) and
    # QED-UHF (unrestricted) reference dicts.
    E_qed_hf_ref = qedhf.get('E_qed_hf', qedhf.get('E_qed_uhf'))
    E_hf_ref = qedhf.get('E_rhf', qedhf.get('E_uhf'))

    # Spin-orbital Fock, antisymmetrised ERI and dipole. The build differs
    # for restricted (QED-HF) vs unrestricted (QED-UHF) references; the
    # residual equations below are spin-orbital and reference-agnostic.
    f_mo, g_mo, dip, nocc, nso = _build_ccsd_so(qedhf)
    nvir = nso - nocc

    if verbose:
        flavour = "QED-CCSD"
        if not (do_t1_01 or do_t2_11 or do_t2_21
                or do_t2_02 or do_t2_12 or do_t2_22):
            flavour = "conventional CCSD"
        elif (do_t1_01 and do_t2_11 and do_t2_21
              and not do_t2_02 and not do_t2_12 and not do_t2_22):
            flavour = "QED-CCSD-1 (QED-CCSD-21, Deprince)"
        elif (do_t1_01 and do_t2_11 and not do_t2_21
              and do_t2_02 and do_t2_12 and not do_t2_22):
            flavour = "QED-CCSD-White (QED-CCSD-12)"
        elif (do_t1_01 and do_t2_11 and do_t2_21
              and do_t2_02 and do_t2_12 and do_t2_22):
            flavour = "QED-CCSD-Full (QED-CCSD-22)"
        ref_kind = "QED-UHF" if 'Ca' in qedhf else "QED-HF"
        print(f"\nQED-CCSD: flavour = {flavour}  (reference = {ref_kind})")
        print(f"  nocc (spin) = {nocc}, nvir (spin) = {nvir},"
              f"  ω = {omega:.6f} Ha,  λ = ({lambda_x},{lambda_y},{lambda_z})")

    # --- Initial amplitudes ---
    G = 0.0  # the G-flag in the Wick-derived equations is always 0 for now
    t1_10 = np.zeros((nvir, nocc))
    t1_01 = 0.0
    t2_20 = np.zeros((nvir, nvir, nocc, nocc))
    t2_02 = 0.0
    t2_11 = np.zeros((nvir, nocc))
    t2_21 = np.zeros((nvir, nvir, nocc, nocc))
    t2_12 = np.zeros((nvir, nocc))
    t2_22 = np.zeros((nvir, nvir, nocc, nocc))

    # --- DIIS bookkeeping (uses real numpy; small linear-algebra solve) ---
    diis_vals_t1_10 = [_np.asarray(t1_10)]
    diis_vals_t2_20 = [_np.asarray(t2_20)]
    diis_vals_t2_11 = [_np.asarray(t2_11)] if do_t2_11 else None
    diis_vals_t2_21 = [_np.asarray(t2_21)] if do_t2_21 else None
    diis_vals_t2_12 = [_np.asarray(t2_12)] if do_t2_12 else None
    diis_vals_t2_22 = [_np.asarray(t2_22)] if do_t2_22 else None
    diis_errors = []

    E_CCSD_old = 0.0
    E_CCSD_new = 0.0
    time_total = 0.0

    if verbose:
        print('\nStarting QED-CCSD iteration:')
        print('Iter   E(QED-CCSD corr)        |dE|         time (s)')

    for ccsd_iter in range(1, max_iter + 1):
        t_start = time.time()

        old_t1_10 = _np.asarray(t1_10)
        old_t2_20 = _np.asarray(t2_20)
        old_t2_11 = _np.asarray(t2_11) if do_t2_11 else None
        old_t2_21 = _np.asarray(t2_21) if do_t2_21 else None
        old_t2_12 = _np.asarray(t2_12) if do_t2_12 else None
        old_t2_22 = _np.asarray(t2_22) if do_t2_22 else None

        # Singles
        t1_10 = ccsd_t1_10(f_mo, g_mo, dip, G, omega,
                           t1_10, t1_01, t2_20, t2_02,
                           t2_11, t2_21, t2_12, t2_22)
        if do_t1_01:
            t1_01 = ccsd_t1_01(f_mo, g_mo, dip, G, omega,
                               t1_10, t1_01, t2_20, t2_02,
                               t2_11, t2_21, t2_12, t2_22)
        # Pure doubles
        t2_20 = ccsd_t2_20(f_mo, g_mo, dip, G, omega,
                           t1_10, t1_01, t2_20, t2_02,
                           t2_11, t2_21, t2_12, t2_22)
        if do_t2_02:
            t2_02 = ccsd_t2_02(f_mo, g_mo, dip, G, omega,
                               t1_10, t1_01, t2_20, t2_02,
                               t2_11, t2_21, t2_12, t2_22)
        # Mixed doubles
        if do_t2_11:
            t2_11 = ccsd_t2_11(f_mo, g_mo, dip, G, omega,
                               t1_10, t1_01, t2_20, t2_02,
                               t2_11, t2_21, t2_12, t2_22)
        if do_t2_21:
            t2_21 = ccsd_t2_21(f_mo, g_mo, dip, G, omega,
                               t1_10, t1_01, t2_20, t2_02,
                               t2_11, t2_21, t2_12, t2_22)
        if do_t2_12:
            t2_12 = ccsd_t2_12(f_mo, g_mo, dip, G, omega,
                               t1_10, t1_01, t2_20, t2_02,
                               t2_11, t2_21, t2_12, t2_22)
        if do_t2_22:
            t2_22 = ccsd_t2_22(f_mo, g_mo, dip, G, omega,
                               t1_10, t1_01, t2_20, t2_02,
                               t2_11, t2_21, t2_12, t2_22)

        # CCSD correlation energy (same expression as the reference; the
        # first assignment intentionally overwrites because that is what
        # the reference code does — only the t2_20 + perturbative terms
        # contribute for closed-shell singlet QED-HF references)
        f_ov_block = f_mo[:nocc, nocc:]
        g_oovv_block = g_mo[:nocc, :nocc, nocc:, nocc:]
        d_ov_block = dip[:nocc, nocc:]

        E_CCSD_new = 1.0 * np.einsum('ia,ai->', f_ov_block, t1_10)
        E_CCSD_new = 0.25 * np.einsum('ijab,baji->', g_oovv_block, t2_20)
        E_CCSD_new = E_CCSD_new - 1.0 * np.einsum('ia,ai->', d_ov_block, t2_11)
        E_CCSD_new = (E_CCSD_new
                      - 1.0 * t1_01 * np.einsum('ia,ai->', d_ov_block, t1_10))
        E_CCSD_new = (E_CCSD_new
                      - 0.5 * np.einsum('ijab,bi,aj->',
                                        g_oovv_block, t1_10, t1_10))

        E_CCSD_new = float(E_CCSD_new)

        t_total = time.time() - t_start
        time_total += t_total
        if verbose:
            print('%3d:  %20.12f  %1.5E   %.3f'
                  % (ccsd_iter, E_CCSD_new,
                     abs(E_CCSD_new - E_CCSD_old), t_total))

        if abs(E_CCSD_new - E_CCSD_old) < tol:
            break

        # --- DIIS ---
        diis_vals_t1_10.append(_np.asarray(t1_10))
        diis_vals_t2_20.append(_np.asarray(t2_20))
        if do_t2_11:
            diis_vals_t2_11.append(_np.asarray(t2_11))
        if do_t2_21:
            diis_vals_t2_21.append(_np.asarray(t2_21))
        if do_t2_12:
            diis_vals_t2_12.append(_np.asarray(t2_12))
        if do_t2_22:
            diis_vals_t2_22.append(_np.asarray(t2_22))

        err_pieces = [(_np.asarray(t1_10) - old_t1_10).ravel(),
                      (_np.asarray(t2_20) - old_t2_20).ravel()]
        if do_t2_11:
            err_pieces.append((_np.asarray(t2_11) - old_t2_11).ravel())
        if do_t2_21:
            err_pieces.append((_np.asarray(t2_21) - old_t2_21).ravel())
        if do_t2_12:
            err_pieces.append((_np.asarray(t2_12) - old_t2_12).ravel())
        if do_t2_22:
            err_pieces.append((_np.asarray(t2_22) - old_t2_22).ravel())
        diis_errors.append(_np.concatenate(err_pieces))

        E_CCSD_old = E_CCSD_new

        # Trim history
        if len(diis_vals_t1_10) > max_diis:
            del diis_vals_t1_10[0]
            del diis_vals_t2_20[0]
            if do_t2_11:
                del diis_vals_t2_11[0]
            if do_t2_21:
                del diis_vals_t2_21[0]
            if do_t2_12:
                del diis_vals_t2_12[0]
            if do_t2_22:
                del diis_vals_t2_22[0]
            del diis_errors[0]

        diis_size = len(diis_vals_t1_10) - 1
        if diis_size < 1:
            continue

        # Pulay 1980, eqn 6
        B = -_np.ones((diis_size + 1, diis_size + 1))
        B[-1, -1] = 0.0
        for n1, e1 in enumerate(diis_errors):
            for n2, e2 in enumerate(diis_errors):
                B[n1, n2] = _np.dot(e1, e2)
        B[:-1, :-1] /= _np.abs(B[:-1, :-1]).max()

        resid = _np.zeros(diis_size + 1)
        resid[-1] = -1.0
        ci = _np.linalg.solve(B, resid)

        t1_10 = np.zeros_like(t1_10)
        t2_20 = np.zeros_like(t2_20)
        if do_t2_11:
            t2_11 = np.zeros_like(t2_11)
        if do_t2_21:
            t2_21 = np.zeros_like(t2_21)
        if do_t2_12:
            t2_12 = np.zeros_like(t2_12)
        if do_t2_22:
            t2_22 = np.zeros_like(t2_22)
        for num in range(diis_size):
            t1_10 = t1_10 + ci[num] * np.asarray(diis_vals_t1_10[num + 1])
            t2_20 = t2_20 + ci[num] * np.asarray(diis_vals_t2_20[num + 1])
            if do_t2_11:
                t2_11 = t2_11 + ci[num] * np.asarray(diis_vals_t2_11[num + 1])
            if do_t2_21:
                t2_21 = t2_21 + ci[num] * np.asarray(diis_vals_t2_21[num + 1])
            if do_t2_12:
                t2_12 = t2_12 + ci[num] * np.asarray(diis_vals_t2_12[num + 1])
            if do_t2_22:
                t2_22 = t2_22 + ci[num] * np.asarray(diis_vals_t2_22[num + 1])
    else:
        if verbose:
            print('Warning: QED-CCSD did not converge in %d iterations'
                  % max_iter)

    return {
        'E_qed_ccsd_corr': float(E_CCSD_new),
        'E_qed_ccsd_total': float(E_CCSD_new) + E_qed_hf_ref,
        'E_qed_hf': E_qed_hf_ref,
        'E_rhf': E_hf_ref,
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
# Demo: reproduce the glycolaldehyde / STO-3G QED-CCSD-21 example from the
# psi4 reference (E(QED-CCSD-21) = -262.416986187232396 with the geometry
# below, ω = 3 eV, λ = (0, 0, 0.1)).
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Same geometry as the reference (psi4 default unit = Angstrom).
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
        basis='STO-3G',
        unit='Angstrom',
        symmetry=False,
        verbose=0,
    )

    # Cavity parameters
    omega_eV = 3.0
    omega = omega_eV / 27.211386245988
    lambda_cav = (0.0, 0.0, 0.1)

    print(f"omega = {omega_eV} eV  ({omega:.6f} Ha)")
    print(f"lambda = {lambda_cav}")

    # QED-HF
    qedhf = run_qed_hf(mol, omega, lambda_cav, verbose=True)
    print(f"\nE_QED_HF = {qedhf['E_qed_hf']:.15f}")
    print(f"E_RHF    = {qedhf['E_rhf']:.15f}  (pyscf, no cavity)")

    # QED-CCSD-21 (Deprince flavour)
    result = run_qed_ccsd(
        qedhf,
        do_t1_01=True, do_t2_11=True, do_t2_21=True,
        do_t2_02=False, do_t2_12=False, do_t2_22=False,
        verbose=True,
    )
    print(f"\nE_QED_CCSD corr  = {result['E_qed_ccsd_corr']:.15f}")
    print(f"E_QED_CCSD total = {result['E_qed_ccsd_total']:.15f}")
    print(f"converged in {result['iterations']} iterations,"
          f" {result['time_total']:.2f} s total")
    print("Reference QED-CCSD-21/STO-3G for this molecule:"
          " E(QED-CCSD) = -262.416986187232396")

