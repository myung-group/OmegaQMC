"""
Full-frequency QED-GWΓ vertex self-energy: the dynamically screened
SOSEX (one frequency-dependent W line) on top of cavity QED-G0W0.

================================================================
VALIDATION STATUS: EXPERIMENTAL / NOT VALIDATED — DO NOT USE FOR
PRODUCTION OR PAPER NUMBERS.

The static-limit regression test (this routine in ``mode='static'``
must reproduce the exact real-axis  Σ^SOSEX_static − Σ^2OX  of
:func:`qed_gw.run_qed_gw_sosex`) currently FAILS: the analytic static
value (~+0.0019 Ha on the STO-3G water HOMO) does not match the code
reference (~+0.0044 Ha), and the Padé analytic continuation is
numerically unstable from mid-gap imaginary samples to the real
quasiparticle energy. Two known issues remain to be fixed before this
is trustworthy:
  (i) the index mapping of the bare *exchange* integral (qw|uv) in
      Eq. 16 must be reconciled with the exchange convention of
      ``qed_gw._sosex_static_pieces`` (the two-electron integral
      ordering of the second, bare line differs);
  (ii) the imaginary-axis + Padé route should be replaced by the exact
      analytic real-frequency ω' integral (the single-Wp ω' integral
      is elementary — 4 simple poles — and removes the continuation
      entirely).
The recommended robust path is to promote the *existing, exact* static
SOSEX contraction of ``_sosex_static_pieces`` to dynamical in place,
replacing the static screening factor −2/Ω_m on the screened line by
the analytically integrated dynamical factor R_m(ω; ε_i,ε_a,ε_b)
(which → −2/Ω_m /(ω+ε_i−ε_a−ε_b) as Ω_m → ∞), so the static limit is
exact by construction.
================================================================


Background.  The vertex rungs already in :mod:`qed_gw` freeze the
screened line: ``G0W0+SOSEX`` uses the static W^stat = W(ω=0) and
``G0W0+2OX`` uses the bare interaction v̄. As the benchmark shows,
their static residues carry none of the cavity-induced charged-
excitation shift — the missing physics lives at the polariton
frequency. This module restores the full frequency dependence of the
screened line.

Following Bruneval, Förster et al. (the fully dynamic G3W2 self-energy,
arXiv:2401.12892), the second-order self-energy in W splits as
(their Eq. 7)

    Σ_G3W2 = Σ_SOX            (both lines bare v̄                 = 2OX)
           + 2 Σ_(W-v),v      (one dynamical Wp = W−v, one v̄    = SOSEX corr.)
           + Σ_(W-v),(W-v)    (both dynamical Wp = full G3Wp2).

The existing real-frequency code already provides Σ_SOX exactly
(``run_qed_gw_sosex(screened=False)``) and the statically screened
single-Wp correction (``screened=True`` minus ``screened=False``).
Here we evaluate the *dynamical* single-Wp correction — the genuine
full-frequency vertex at the SOSEX level — from the source paper's
imaginary-frequency SOSEX expression (their Eq. 16):

    Σ^SOSEX_pp(μ+iω) = −(1/2π) ∫dω' Σ_{uvw} (f_v − f_w)
        (wv|Wp(iω')|pu)(pw|uv)
        / [ (iω' + ε_v − ε_w) (μ+iω+iω' − ε_u) ],

with the QED-augmented spectral screening (the M_{pq,m}, Ω_m of
:func:`qed_gw._rpa_at_eps`, which already carry the photon channel):

    (wv|Wp(iω')|pu) = Σ_m M_{wv,m} M_{pu,m} [ 1/(iω'−Ω_m) − 1/(iω'+Ω_m) ]
                    = −2 Σ_m M_{wv,m} M_{pu,m} Ω_m / (ω'^2 + Ω_m^2).

f_v is 1 for occupied, 0 for virtual orbitals. The ω' integral is
performed by Gauss-Legendre quadrature on the (tan-mapped) real line —
the poles of Wp(iω') lie off the imaginary axis, so no η is needed —
and the result, computed on a set of imaginary frequencies μ+iω_k, is
analytically continued to the real axis with a Thiele/Vidberg-Serene
Padé approximant. This is the practical scheme of MOLGW.

Validation hooks (see ``__main__`` / the test script):

* ``mode='static'`` replaces Wp(iω') by Wp(0); after continuation it
  must reproduce  Σ^SOSEX_static − Σ^2OX  evaluated on the real axis by
  the existing :func:`qed_gw.run_qed_gw_sosex` (both the static-SOSEX
  and the 2OX paths are exact real-frequency closed forms).
* ``mode='dyn'`` is the new full-frequency single-Wp correction.

The total full-frequency vertex-corrected quasiparticle equation is

    ω = ε^HF_p + Re Σ^GW_c(ω) + Re Σ^2OX(ω) + Re Σ^{SOSEX,dyn-corr}(ω),

i.e. ``G0W0 + SOX + dynamical single-Wp = G0W0Γ^(1)`` at the SOSEX
level. The remaining double-Wp term Σ_(W-v),(W-v) (full G3W2, N^7) is
not yet included; see the module note.

Note (scope). This is the *single-Wp* (SOSEX-level) full-frequency
vertex. The complete G3W2 additionally contains the double-dynamical-W
term with its 3-occupied / 3-virtual time orderings (arXiv:2401.12892
Eqs. 13a-13f); implementing it is the natural next step and reuses the
same spectral screening data and quadrature machinery.
"""

import math

import numpy as np
from pyscf import gto

from .qed_hf import run_qed_hf
from .qed_gw import (_build_static_quantities, _rpa_at_eps, _eval_sigma,
                     _sosex_static_pieces, _eval_sosex)

EV = 27.211386245988


def _imag_freq_nodes(n_grid=200, w0=1.0):
    """Gauss-Legendre nodes/weights for ∫_{-∞}^{∞} dω' f(ω'), via the
    tan map ω' = w0 tan(π t / 2), t ∈ (−1, 1)."""
    t, wt = np.polynomial.legendre.leggauss(n_grid)
    theta = 0.5 * np.pi * t
    omega_p = w0 * np.tan(theta)
    jac = w0 * 0.5 * np.pi / np.cos(theta) ** 2
    return omega_p, wt * jac


def _sosex_corr_imag(static, Omega, M_full, eps, p_so, Z_pts,
                     nodes, node_w, mode='dyn'):
    """Single-Wp SOSEX self-energy correction Σ^SOSEX_pp(Z) at the
    complex points ``Z_pts`` (source-paper Eq. 16), diagonal p = q.

    ``mode='dyn'`` uses the full Wp(iω'); ``mode='static'`` uses Wp(0).
    Returns a complex array of the same length as ``Z_pts``.
    """
    so = static['so']
    nocc = so['nocc']
    nso = so['nso']
    g = so['g_phys_d']
    d = so['d_so']
    include_dse = static.get('include_dse', True)

    f_occ = (np.arange(nso) < nocc).astype(float)           # occupations
    mask_vw = f_occ[:, None] - f_occ[None, :]               # (v, w): f_v − f_w
    eps = np.asarray(eps)

    Mp = M_full[p_so]                                       # (u, m) = M_{p u, m}

    # B'[w, v, u] = (pw|uv)_vbar = (pw|uv)_chem + DSE d_pw d_uv.
    # (pw|uv)_chem = ⟨pu|wv⟩ = g[p, u, w, v]; g[p_so] is indexed (u, w, v).
    Bp = g[p_so].transpose(1, 2, 0).copy()                 # (w, v, u) <- (u,w,v)
    if include_dse:
        Bp = Bp + np.einsum('w,uv->wvu', d[p_so], d, optimize=True)

    out = np.zeros(len(Z_pts), dtype=complex)
    for wp, weight in zip(nodes, node_w):
        iwp = 1j * wp
        if mode == 'dyn':
            Wpf = -2.0 * Omega / (wp * wp + Omega * Omega)   # (m,)
        elif mode == 'static':
            Wpf = -2.0 / Omega                               # (m,), ω'-independent
        else:
            raise ValueError(f"unknown mode {mode!r}")

        # A[w, v, u] = Σ_m M_{wv,m} M_{pu,m} Wpf_m
        MpW = Mp * Wpf[None, :]                              # (u, m)
        A = np.einsum('wvm,um->wvu', M_full, MpW, optimize=True)
        T = (mask_vw[:, :, None].transpose(1, 0, 2)) * A * Bp
        # mask is (v, w); we need (w, v, u) → transpose mask to (w, v)
        # (done above: mask_vw[v,w] -> [:, :, None].transpose(1,0,2) = (w,v,1))

        denom1 = 1.0 / (iwp + eps[None, :] - eps[:, None])   # (w, v): iω'+ε_v−ε_w
        # contract w, v first → vector over u
        Tu = np.einsum('wvu,wv->u', T, denom1, optimize=True)
        for k, Z in enumerate(Z_pts):
            denom2 = 1.0 / (Z + iwp - eps)                   # (u,)
            out[k] += weight * np.dot(Tu, denom2)

    return -(1.0 / (2.0 * np.pi)) * out


def _pade_coeffs(z, u):
    """Thiele reciprocal-difference coefficients for a Padé continued
    fraction through points (z_i, u_i) (Vidberg-Serene)."""
    n = len(z)
    a = np.empty(n, dtype=complex)
    g_prev = np.array(u, dtype=complex).copy()
    a[0] = g_prev[0]
    for i in range(1, n):
        gi = np.zeros(n, dtype=complex)
        for j in range(i, n):
            gi[j] = (g_prev[i - 1] - g_prev[j]) / \
                    ((z[j] - z[i - 1]) * g_prev[j])
        a[i] = gi[i]
        g_prev = gi
    return a


def _pade_eval(z_nodes, a, x):
    """Evaluate the Thiele continued fraction with coefficients ``a``
    (from :func:`_pade_coeffs` on nodes ``z_nodes``) at point ``x``."""
    n = len(a)
    A_prev, A_cur = 1.0 + 0j, a[0]
    B_prev, B_cur = 0.0 + 0j, 1.0 + 0j
    # build via the standard recurrence A_k = A_{k-1} + (x - z_{k-1}) a_k A_{k-2}
    A = [1.0 + 0j, a[0] + 0j]
    B = [0.0 + 0j, 1.0 + 0j]
    for i in range(1, n):
        Ai = A[i] + (x - z_nodes[i - 1]) * a[i] * A[i - 1]
        Bi = B[i] + (x - z_nodes[i - 1]) * a[i] * B[i - 1]
        A.append(Ai)
        B.append(Bi)
    return A[-1] / B[-1]


def sosex_corr_real(static, Omega, M_full, eps, p_so, omega_real,
                    mode='dyn', n_grid=200, w0=1.0, n_pade=14,
                    omega_max_imag=4.0, mu=None):
    """Single-Wp SOSEX correction Σ^SOSEX_pp(ω) at the *real* frequency
    ``omega_real`` (or array), via imaginary-axis Eq. 16 + Padé."""
    nocc = static['so']['nocc']
    if mu is None:
        mu = 0.5 * (eps[nocc - 1] + eps[nocc])              # mid-gap
    # imaginary sampling points Z_k = μ + i ω_k (ω_k > 0)
    wk = np.linspace(1e-3, omega_max_imag, n_pade)
    Z_pts = mu + 1j * wk
    nodes, node_w = _imag_freq_nodes(n_grid, w0)
    Sig_imag = _sosex_corr_imag(static, Omega, M_full, eps, p_so, Z_pts,
                                nodes, node_w, mode=mode)
    a = _pade_coeffs(Z_pts, Sig_imag)
    xs = np.atleast_1d(omega_real).astype(complex)
    vals = np.array([_pade_eval(Z_pts, a, x) for x in xs])
    return vals if np.ndim(omega_real) else complex(vals[0])


if __name__ == '__main__':
    half = math.radians(104.5 / 2.0)
    rOH = 1.0
    hx, hz = rOH * math.sin(half), -rOH * math.cos(half)
    mol = gto.M(atom=[['O', (0, 0, 0)], ['H', (hx, 0, hz)],
                      ['H', (-hx, 0, hz)]],
                basis='sto-3g', unit='Angstrom', symmetry=False, verbose=0)
    qedhf = run_qed_hf(mol, 0.415668, (0, 0, 0.05), verbose=False)
    static = _build_static_quantities(qedhf, direct=True)
    eps = static['eps_HF']
    Omega, M_full = _rpa_at_eps(static, eps)
    nocc = static['so']['nocc']
    homo = nocc - 1
    w_eval = float(eps[homo])

    # static single-Wp from imaginary+Pade
    s_dyn = sosex_corr_real(static, Omega, M_full, eps, homo, w_eval, 'dyn')
    s_sta = sosex_corr_real(static, Omega, M_full, eps, homo, w_eval, 'static')
    # reference: code static-SOSEX minus code 2OX (both exact real-axis)
    p_sosex = _sosex_static_pieces(static, Omega, M_full, eps, homo,
                                   screened=True)
    p_2ox = _sosex_static_pieces(static, Omega, M_full, eps, homo,
                                 screened=False)
    sig_sosex, _ = _eval_sosex(w_eval, p_sosex, 1e-3)
    sig_2ox, _ = _eval_sosex(w_eval, p_2ox, 1e-3)
    ref_static = sig_sosex - sig_2ox
    print(f"HOMO ω = {w_eval:.5f} Ha")
    print(f"  static single-Wp (imag+Pade) Re = {s_sta.real:+.6f} Ha")
    print(f"  static single-Wp (real, code) Re = {ref_static.real:+.6f} Ha")
    print(f"  -> abs diff = {abs(s_sta.real - ref_static.real):.2e} Ha")
    print(f"  dynamical single-Wp (imag+Pade) Re = {s_dyn.real:+.6f} Ha")
