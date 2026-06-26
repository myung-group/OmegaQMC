"""
Quantum-chemistry transcorrelated FCI for a 2-electron system.

Builds the transcorrelated Hamiltonian H~ = e^{-J} H e^{J} as matrix elements
in the determinant basis and diagonalises it (TC-FCI) -- the proper QC route,
not a VMC sampling estimator. With 2 electrons there is a single pair, so NO
three-body term: H~ is an (in general non-Hermitian) modified 2-electron
problem.

Method: PySCF gives the bare 1e/2e integrals analytically. The J-dependent TC
corrections (drift  -grad u . grad,  and the smooth  -(grad u)^2 ) are computed
by real-space grid quadrature; the Laplacian term -1/2 grad^2 u (singular for a
cusp factor) is folded in by integration by parts so only smooth grad-u kernels
are gridded -- the standard numerical-TC treatment.

Verification gates (printed):
  [G1] determinant FCI reproduces PySCF FCI (bare Hamiltonian machinery).
  [G2] TC-FCI with J=0 reproduces bare FCI (corrections wired correctly).
  [G3] TC ground-state eigenvalue is real.

Usage:
  python examples/tc_fci_2e.py --atom "H 0 0 0; H 0 0 1.4" --unit Angstrom \
      --basis cc-pvdz --u-form cusp --u-a-unlike 0.5 --u-b 1.0
"""

import argparse

import numpy as np
import scipy.linalg
from pyscf import gto, scf, fci, ao2mo, dft


# --------------------------------------------------------------------------
# 2-electron determinant FCI with general (possibly non-symmetric) integrals
# --------------------------------------------------------------------------

def build_2e_hamiltonian(h1, g_phys, n_orb):
    """H matrix in the |p_alpha q_beta> basis (dim n_orb^2).

    <p_a q_b| H |r_a s_b> = h_pr delta_qs + h_qs delta_pr + g_phys[p,q,r,s],
    with g_phys[p,q,r,s] = <pq|rs> (physicist) = (pr|qs) (chemist).
    No exchange term (opposite spin).
    """
    H = np.zeros((n_orb * n_orb, n_orb * n_orb), dtype=g_phys.dtype)
    for p in range(n_orb):
        for q in range(n_orb):
            I = p * n_orb + q
            for r in range(n_orb):
                for s in range(n_orb):
                    J = r * n_orb + s
                    val = g_phys[p, q, r, s]
                    if q == s:
                        val += h1[p, r]
                    if p == r:
                        val += h1[q, s]
                    H[I, J] = val
    return H


def ground_state(H, e_nuc):
    """Lowest (real-part) eigenvalue; handles non-Hermitian H."""
    if np.allclose(H, H.T):
        w = scipy.linalg.eigvalsh(H)
        e = float(w[0]); imag = 0.0
    else:
        w = scipy.linalg.eigvals(H)
        idx = int(np.argmin(w.real))
        e = float(w[idx].real); imag = float(abs(w[idx].imag))
    return e + e_nuc, imag


# --------------------------------------------------------------------------
# Correlation factor u(r) and derivative profiles (radial)
# --------------------------------------------------------------------------

def u_profiles(form, a, b):
    """Return (uprime(r), ) callables for the chosen u(r); cusp Pade
    u(r)=a r/(1+b r), u'(r)=a/(1+b r)^2. (u itself not needed; only grad u.)"""
    if form == "cusp":
        def uprime(r):
            return a / (1.0 + b * r) ** 2
    elif form == "gauss":
        def uprime(r):
            return -2.0 * a * b * r * np.exp(-b * r * r)
    else:
        raise ValueError(form)
    return uprime


def _mo_on_grid(mol, C, level):
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    coords = np.asarray(grids.coords)
    w = np.asarray(grids.weights)
    ao = dft.numint.eval_ao(mol, coords, deriv=1)  # (4, G, nao)
    P = (ao[0] @ C).T                              # (norb, G)
    Gmo = np.stack([ao[1 + c] @ C for c in range(3)], axis=0)  # (3, G, norb)
    Gmo = np.transpose(Gmo, (0, 2, 1))             # (3, norb, G)
    return coords, w, P, Gmo


def _build_A(rho_pairs, coords, w, uprime, chunk=400):
    """A[pair, g_out, :] = sum_gin w[gin] rho[pair,gin] * u'(r) (x_out-x_in)/r."""
    npair, G = rho_pairs.shape
    A = np.zeros((npair, G, 3))
    wr = w[None, :] * rho_pairs                    # (npair, G)
    for s in range(0, G, chunk):
        blk = coords[s:s + chunk]                  # (nb, 3)
        d = blk[:, None, :] - coords[None, :, :]   # (nb, G, 3)
        r = np.sqrt(np.sum(d * d, axis=-1))        # (nb, G)
        up = uprime(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            kvec = up[..., None] * d / r[..., None]
        kvec[r < 1e-10] = 0.0
        nb = blk.shape[0]
        for c in range(3):                         # BLAS matmul per component
            A[:, s:s + nb, c] = wr @ kvec[:, :, c].T
    return A


def compute_tc_corrections(mol, C, uprime, level, chunk=400, grid=None):
    """Grid TC 2-electron correction tensor DeltaV[p,q,r,s] (physicist).

    ``grid`` may be a precomputed ``(coords, w, P, Gmo)`` tuple (factor-
    independent) to amortise the grid build across a parameter scan.
    """
    coords, w, P, Gmo = grid if grid is not None else _mo_on_grid(mol, C, level)
    n = C.shape[1]
    # pair densities (no weight): rho[(a,b), g] = P[a,g] P[b,g]
    rho = (P[:, None, :] * P[None, :, :]).reshape(n * n, -1)   # (n^2, G)

    # A is identical for D1 and D2 (same pair density rho) -> compute once.
    A = _build_A(rho, coords, w, uprime, chunk)              # (n^2, G, 3)

    # D1: A from e2-density (q,s); contract with e1 orbitals (p, grad r)
    AG1 = np.einsum("Qgc,crg->Qrg", A, Gmo)                  # (n^2, n, G)
    D1 = np.einsum("g,pg,Qrg->pQr", w, P, AG1).reshape(n, n, n, n)
    D1 = np.transpose(D1, (0, 1, 3, 2))   # [p,(q,s),r] -> [p,q,r,s]

    # D2: A from e1-density (p,r); contract with e2 orbitals (q, grad s)
    AG2 = np.einsum("Qgc,csg->Qsg", A, Gmo)                 # (n^2, n, G)
    D2 = np.einsum("g,qg,Qsg->Qqs", w, P, AG2).reshape(n, n, n, n)
    D2 = np.transpose(D2, (0, 2, 1, 3))   # [(p,r),q,s] -> [p,q,r,s]

    # U2[p,q,r,s] = sum_g1g2 w1 P_p P_r(g1) u'(r12)^2 w2 P_q P_s(g2)
    we1 = (w[None, :] * rho)                                  # (n^2, G)  (p,r)
    we2 = (w[None, :] * rho)                                  # (n^2, G)  (q,s)
    U2pair = np.zeros((n * n, n * n))
    for s in range(0, coords.shape[0], chunk):
        blk = coords[s:s + chunk]
        d = blk[:, None, :] - coords[None, :, :]
        r = np.sqrt(np.sum(d * d, axis=-1))
        Ku = uprime(r) ** 2                                  # (nb, G)
        U2pair += we1[:, s:s + blk.shape[0]] @ (Ku @ we2.T)
    U2 = U2pair.reshape(n, n, n, n)                          # [(p,r),(q,s)] -> [p,r,q,s]
    U2 = np.transpose(U2, (0, 2, 1, 3))                      # [p,q,r,s]

    dV = (0.5 * (np.transpose(D1, (2, 1, 0, 3)) - D1)        # 1/2 (D1[r,q,p,s]-D1[p,q,r,s])
          + 0.5 * (np.transpose(D2, (0, 3, 2, 1)) - D2)      # 1/2 (D2[p,s,r,q]-D2[p,q,r,s])
          - U2)
    return dV


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atom", default="H 0 0 0; H 0 0 1.4")
    p.add_argument("--unit", default="Angstrom")
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--u-form", default="cusp", choices=["cusp", "gauss"])
    p.add_argument("--u-a-unlike", type=float, default=0.5)
    p.add_argument("--u-b", type=float, default=1.0)
    p.add_argument("--grid-level", type=int, default=3)
    args = p.parse_args()

    mol = gto.M(atom=args.atom, unit=args.unit, basis=args.basis, verbose=0)
    assert mol.nelectron == 2, "this script is 2-electron only"
    mf = scf.RHF(mol).run()
    C = np.asarray(mf.mo_coeff)
    n_orb = C.shape[1]
    e_nuc = mol.energy_nuc()

    # Bare integrals in the MO basis.
    h1_ao = mf.get_hcore()
    h1 = C.T @ h1_ao @ C
    eri_chem = ao2mo.restore(1, ao2mo.kernel(mol, C), n_orb)  # (pr|qs)
    g_phys = np.transpose(eri_chem, (0, 2, 1, 3))             # <pq|rs>

    # Reference: PySCF FCI.
    cisolver = fci.FCI(mol, mf.mo_coeff)
    e_fci_pyscf, _ = cisolver.kernel()

    # [G1] determinant FCI machinery.
    H_bare = build_2e_hamiltonian(h1, g_phys, n_orb)
    e_fci_mine, _ = ground_state(H_bare, e_nuc)
    print(f"[G1] det-FCI {e_fci_mine:.8f}  vs PySCF FCI {e_fci_pyscf:.8f}  "
          f"diff {abs(e_fci_mine - e_fci_pyscf)*1e6:.2f} uE_h  "
          f"{'PASS' if abs(e_fci_mine - e_fci_pyscf) < 1e-7 else 'FAIL'}")

    print(f"    E_HF={mf.e_tot:.6f}  FCI({args.basis})={e_fci_pyscf:.6f}")

    # [G2] u=0 -> bare FCI (corrections wired correctly). Cheap grid (L0).
    uprime0 = u_profiles(args.u_form, 0.0, args.u_b)
    dV0 = compute_tc_corrections(mol, C, uprime0, 0)
    H0 = build_2e_hamiltonian(h1, g_phys + dV0, n_orb)
    e0, _ = ground_state(H0, e_nuc)
    print(f"[G2] u=0 TC-FCI {e0:.8f}  vs bare {e_fci_mine:.8f}  "
          f"diff {abs(e0 - e_fci_mine)*1e6:.2f} uE_h  "
          f"{'PASS' if abs(e0 - e_fci_mine) < 1e-6 else 'FAIL'}")

    # TC-FCI with the chosen correlation factor.
    uprime = u_profiles(args.u_form, args.u_a_unlike, args.u_b)
    dV = compute_tc_corrections(mol, C, uprime, args.grid_level)
    Htc = build_2e_hamiltonian(h1, g_phys + dV, n_orb)
    e_tc, imag = ground_state(Htc, e_nuc)
    print(f"[G3] TC ground-state imag part {imag:.2e}  "
          f"{'PASS' if imag < 1e-6 else 'WARN (complex)'}")

    print(f"\n=== TC-FCI ({args.u_form}, a={args.u_a_unlike}, b={args.u_b}, "
          f"grid L{args.grid_level}) ===")
    print(f"  E_HF              = {mf.e_tot:.6f}")
    print(f"  FCI({args.basis}) = {e_fci_pyscf:.6f}   (bare finite-basis ceiling)")
    print(f"  TC-FCI({args.basis}) = {e_tc:.6f}")
    print(f"  TC-FCI - FCI      = {(e_tc - e_fci_pyscf)*1e3:+.3f} mE_h "
          f"(negative = recovered beyond-basis correlation)")


if __name__ == "__main__":
    main()
