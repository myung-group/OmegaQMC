"""
Quasiparticle spectral function A_p(ω) from the QED-GW self-energy.

The diagonal spectral function of orbital p is built from the same
spectral-form correlation self-energy Σ_c(ω) used by the quasiparticle
solvers of :mod:`qed_gw` (paper Eq. 33),

    A_p(ω) = (1/π) |Im G_pp(ω)|,
    G_pp(ω) = 1 / (ω − ε^HF_p − Σ_c,pp(ω)),

evaluated on a frequency grid at the one-shot (G0W0) level: the
QED-dRPA screening is built once at the QED-HF orbital energies. The
poles of Σ_c at ε_i − Ω_m (occupied branch) and ε_a + Ω_m (virtual
branch) generate, besides the quasiparticle peak, satellite structures.
In a cavity the modes Ω_m are *polaritonic*, so the photoemission
satellites of a resonantly coupled molecule sit at the lower/upper
polariton energies — the cavity analogue of phonon sidebands, with
anomalously large weight because ω_cav is tuned into the electronic
spectrum (no Migdal suppression).

The broadening η plays the role of the inverse quasiparticle lifetime
used for plotting; satellite positions and weights are independent of
it for small η.
"""

import numpy as np

from .qed_gw import (_build_static_quantities, _rpa_at_eps, _eval_sigma)

HARTREE_TO_EV = 27.211386245988


def qed_gw_spectral_function(qedhf, orbs=None, omega_grid=None,
                             eta=5e-3, verbose=True):
    """Diagonal G0W0-level spectral function A_p(ω) on a QED-HF reference.

    Args:
        qedhf: dict from :func:`OmegaQMC.addons.qed_hf.run_qed_hf`.
        orbs: list of *spatial* orbital indices; None → HOMO and LUMO.
        omega_grid: 1D array of frequencies (Ha). None → an automatic
            grid spanning [ε_HOMO − 2 max(Ω_m), ε_LUMO + 2 max(Ω_m)].
        eta: Lorentzian broadening (Ha) of the poles.
        verbose: print a summary of the dominant poles per orbital.

    Returns:
        dict with 'omega_grid' (Ha), 'A' (norbs × ngrid, 1/Ha),
        'orbs', 'eps_HF' (per orbital, Ha), 'Omega_RPA' (Ha),
        'photon_weight' (per RPA mode, |(M+N)_m|²), and 'poles'
        (per orbital: list of (position, weight, mode index, photon
        weight) for the self-energy poles, sorted by weight).
    """
    static = _build_static_quantities(qedhf, direct=True)
    eps_HF = static['eps_HF']
    nocc, nso = static['so']['nocc'], static['so']['nso']
    nocc_spatial = qedhf['nocc_spatial']

    if orbs is None:
        orbs = [nocc_spatial - 1, nocc_spatial]
    orbs = list(orbs)
    orbs_so = [2 * p for p in orbs]

    Omega, M_full, XplusY_p = _rpa_at_eps(static, eps_HF, full_output=True)
    photon_weight = XplusY_p ** 2

    if omega_grid is None:
        span = 2.0 * float(np.max(Omega))
        lo = float(eps_HF[nocc - 1]) - span
        hi = float(eps_HF[nocc]) + span
        omega_grid = np.linspace(lo, hi, 4001)
    omega_grid = np.asarray(omega_grid, dtype=float)

    A = np.zeros((len(orbs), len(omega_grid)))
    poles = {}
    for k, (p_spatial, p_so) in enumerate(zip(orbs, orbs_so)):
        M_ms = M_full[p_so]
        eps_p = float(eps_HF[p_so])
        for j, w in enumerate(omega_grid):
            sigma, _ = _eval_sigma(w, M_ms, Omega, eps_HF, nocc, eta)
            G = 1.0 / (w - eps_p - sigma)
            A[k, j] = abs(G.imag) / np.pi

        # Pole table of Σ_c for this orbital: positions ε_q ∓ Ω_m with
        # weights |M_{pq,m}|² (occupied branch −, virtual branch +).
        M_sq = M_ms * M_ms
        pole_list = []
        for q in range(nso):
            sign = -1.0 if q < nocc else +1.0
            for m in range(len(Omega)):
                wgt = float(M_sq[q, m])
                if wgt < 1e-8:
                    continue
                pos = float(eps_HF[q] + sign * Omega[m])
                pole_list.append((pos, wgt, m, float(photon_weight[m])))
        pole_list.sort(key=lambda t: -t[1])
        poles[p_spatial] = pole_list

        if verbose:
            eps_ev = eps_p * HARTREE_TO_EV
            print(f"\n  orbital {p_spatial} (ε_HF = {eps_ev:+9.3f} eV): "
                  f"strongest Σ_c poles")
            print("      ω_pole (eV)   |M|² (Ha²)   mode   photon wgt")
            for pos, wgt, m, pw in pole_list[:8]:
                print(f"    {pos * HARTREE_TO_EV:+10.3f}   {wgt:10.3e}"
                      f"   {m:4d}   {pw:8.3f}")

    return {
        'omega_grid': omega_grid,
        'A': A,
        'orbs': orbs,
        'eps_HF': {p: float(eps_HF[2 * p]) for p in orbs},
        'Omega_RPA': Omega,
        'photon_weight': photon_weight,
        'poles': poles,
        'eta': float(eta),
    }


def satellite_weights(result, orb, window):
    """Integrated spectral weight of A_orb(ω) inside a window (Ha).

    Args:
        result: dict returned by :func:`qed_gw_spectral_function`.
        orb: spatial orbital index (must be in result['orbs']).
        window: (ω_min, ω_max) in Ha.

    Returns:
        float — ∫ A(ω) dω over the window (trapezoidal).
    """
    k = result['orbs'].index(orb)
    w = result['omega_grid']
    mask = (w >= window[0]) & (w <= window[1])
    return float(np.trapezoid(result['A'][k][mask], w[mask]))
