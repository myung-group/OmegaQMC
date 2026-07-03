"""Kernel-level validation: qed_ccsd_rhf spatial kernels must equal the
alpha / alpha-beta blocks of the qed_ccsd_new spin-orbital kernels on
random closed-shell-consistent amplitudes.

NOTE: qed_ccsd_new.py was removed from OmegaQMC/addons in the
2026-07 cleanup (this validation last passed 2026-07-02, ~1e-15
per kernel). To rerun, regenerate the spin-orbital module from
eqs/ via gen_module.py -- see README.md.
"""
import numpy as np
from pyscf import gto
from OmegaQMC.addons.qed_hf import run_qed_hf
from OmegaQMC.addons import qed_ccsd_new as so
from OmegaQMC.addons import qed_ccsd_rhf as sf

mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            basis="sto-3g", verbose=0)
omega = 3.0 / 27.211386245988
qedhf = run_qed_hf(mol, omega, (0.0, 0.0, 0.05), verbose=False)

f_so, B_so, dip_so, nocc_so, nso = so._build_ccsd_so(qedhf)
ints, nocc, nvir = sf._build_ints(qedhf)

rng = np.random.default_rng(3)
def sym4(x):
    return 0.5 * (x + x.transpose(1, 0, 3, 2))

T1 = 0.05 * rng.standard_normal((nvir, nocc))
U11 = 0.05 * rng.standard_normal((nvir, nocc))
U12 = 0.05 * rng.standard_normal((nvir, nocc))
T2 = 0.05 * sym4(rng.standard_normal((nvir, nvir, nocc, nocc)))
U21 = 0.05 * sym4(rng.standard_normal((nvir, nvir, nocc, nocc)))
U22 = 0.05 * sym4(rng.standard_normal((nvir, nvir, nocc, nocc)))
s1, s2 = 0.031, 0.017

def to_so2(X):
    return np.kron(X, np.eye(2))

def to_so4(T):
    A = np.arange(2 * nvir)[:, None, None, None]
    Bx = np.arange(2 * nvir)[None, :, None, None]
    I = np.arange(2 * nocc)[None, None, :, None]
    J = np.arange(2 * nocc)[None, None, None, :]
    d1 = (A % 2 == I % 2) * (Bx % 2 == J % 2)
    d2 = (A % 2 == J % 2) * (Bx % 2 == I % 2)
    return (T[A // 2, Bx // 2, I // 2, J // 2] * d1
            - T[A // 2, Bx // 2, J // 2, I // 2] * d2)

amps_so = [to_so2(T1), s1, to_so4(T2), s2, to_so2(U11), to_so4(U21),
           to_so2(U12), to_so4(U22)]
amps_sf = [T1, s1, T2, s2, U11, U21, U12, U22]

args_so = [f_so, B_so, dip_so, 0.0, omega] + amps_so
args_sf = [ints, omega] + amps_sf

checks = [
    ('energy', 'scalar'), ('t1_10', '2'), ('t1_01', 'scalar'),
    ('t2_20', '4'), ('t2_02', 'scalar'), ('t2_11', '2'),
    ('t2_21', '4'), ('t2_12', '2'), ('t2_22', '4'),
]
ok = True
for tgt, kind in checks:
    fn_so = getattr(so, 'ccsd_energy' if tgt == 'energy' else f'ccsd_{tgt}')
    fn_sf = getattr(sf, 'ccsd_energy' if tgt == 'energy' else f'ccsd_{tgt}')
    r_so = fn_so(*args_so)
    r_sf = fn_sf(*args_sf)
    if kind == 'scalar':
        d = abs(float(r_so) - float(r_sf))
    elif kind == '2':
        d = abs(np.asarray(r_so)[0::2, 0::2] - r_sf).max()
    else:
        d = abs(np.asarray(r_so)[0::2, 1::2, 0::2, 1::2] - r_sf).max()
    flag = 'OK ' if d < 1e-11 else 'FAIL'
    ok = ok and d < 1e-11
    print(f"{flag} {tgt:7s} max|so_block - sf| = {d:.3e}")
print("ALL OK" if ok else "FAILURES PRESENT")
