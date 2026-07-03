"""Blockwise validation of qed_ccsd_uhf kernels against the spin-orbital
qed_ccsd_new kernels on a QED-UHF reference (H2O+ doublet, STO-3G) with
random blocked amplitudes.

NOTE: qed_ccsd_new.py was removed from OmegaQMC/addons in the
2026-07 cleanup (this validation last passed 2026-07-02, ~1e-15
per kernel). To rerun, regenerate the spin-orbital module from
eqs/ via gen_module.py -- see README.md.
"""
import numpy as np
from pyscf import gto
from OmegaQMC.addons.qed_uhf import run_qed_uhf
from OmegaQMC.addons import qed_ccsd_new as so
from OmegaQMC.addons import qed_ccsd_uhf as ub

mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            basis="sto-3g", charge=1, spin=1, verbose=0)
omega = 3.0 / 27.211386245988
qeduhf = run_qed_uhf(mol, omega, (0.0, 0.0, 0.05), verbose=False)

f_so, B_so, dip_so, nocc_so, nso = so._build_ccsd_so(qeduhf)
ints, nocc, nvir = ub._build_ints(qeduhf)
noa, nob = nocc['a'], nocc['b']
nva, nvb = nvir['a'], nvir['b']

rng = np.random.default_rng(5)
def anti4(x):
    return 0.25 * (x - x.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2)
                   + x.transpose(1, 0, 3, 2))

amps = ub._Amps()
amps.t1_01 = 0.031
amps.t2_02 = 0.017
for base in ('t1_10', 't2_11', 't2_12'):
    setattr(amps, f'{base}_a', 0.05 * rng.standard_normal((nva, noa)))
    setattr(amps, f'{base}_b', 0.05 * rng.standard_normal((nvb, nob)))
for base in ('t2_20', 't2_21', 't2_22'):
    setattr(amps, f'{base}_aa',
            0.05 * anti4(rng.standard_normal((nva, nva, noa, noa))))
    setattr(amps, f'{base}_bb',
            0.05 * anti4(rng.standard_normal((nvb, nvb, nob, nob))))
    setattr(amps, f'{base}_ab',
            0.05 * rng.standard_normal((nva, nvb, noa, nob)))

# ---- map blocked -> spin-orbital arrays (occupied-first ordering:
#      occ = [occ_a; occ_b], vir = [vir_a; vir_b]) ----
def so2(Xa, Xb):
    out = np.zeros((nva + nvb, noa + nob))
    out[:nva, :noa] = Xa
    out[nva:, noa:] = Xb
    return out

def so4(Taa, Tab, Tbb):
    nv, no = nva + nvb, noa + nob
    out = np.zeros((nv, nv, no, no))
    out[:nva, :nva, :noa, :noa] = Taa
    out[nva:, nva:, noa:, noa:] = Tbb
    out[:nva, nva:, :noa, noa:] = Tab
    out[nva:, :nva, noa:, :noa] = Tab.transpose(1, 0, 3, 2)
    out[:nva, nva:, noa:, :noa] = -Tab.transpose(0, 1, 3, 2)
    out[nva:, :nva, :noa, noa:] = -Tab.transpose(1, 0, 2, 3)
    return out

so_amps = [so2(amps.t1_10_a, amps.t1_10_b), amps.t1_01,
           so4(amps.t2_20_aa, amps.t2_20_ab, amps.t2_20_bb), amps.t2_02,
           so2(amps.t2_11_a, amps.t2_11_b),
           so4(amps.t2_21_aa, amps.t2_21_ab, amps.t2_21_bb),
           so2(amps.t2_12_a, amps.t2_12_b),
           so4(amps.t2_22_aa, amps.t2_22_ab, amps.t2_22_bb)]
args_so = [f_so, B_so, dip_so, 0.0, omega] + so_amps

oA = slice(0, noa); oB = slice(noa, noa + nob)
vA = slice(0, nva); vB = slice(nva, nva + nvb)

ok = True
def check(label, d):
    global ok
    flag = 'OK ' if d < 1e-11 else 'FAIL'
    ok = ok and d < 1e-11
    print(f"{flag} {label:12s} {d:.3e}")

for tgt in ('energy', 't1_01', 't2_02'):
    fn_so = getattr(so, 'ccsd_energy' if tgt == 'energy' else f'ccsd_{tgt}')
    fn_ub = getattr(ub, 'ccsd_energy' if tgt == 'energy' else f'ccsd_{tgt}')
    check(tgt, abs(float(fn_so(*args_so))
                   - float(fn_ub(ints, omega, amps))))

for tgt in ('t1_10', 't2_11', 't2_12'):
    r_so = np.asarray(getattr(so, f'ccsd_{tgt}')(*args_so))
    check(f'{tgt}_a', abs(r_so[vA, oA]
          - getattr(ub, f'ccsd_{tgt}_a')(ints, omega, amps)).max())
    check(f'{tgt}_b', abs(r_so[vB, oB]
          - getattr(ub, f'ccsd_{tgt}_b')(ints, omega, amps)).max())

for tgt in ('t2_20', 't2_21', 't2_22'):
    r_so = np.asarray(getattr(so, f'ccsd_{tgt}')(*args_so))
    check(f'{tgt}_aa', abs(r_so[vA, vA, oA, oA]
          - getattr(ub, f'ccsd_{tgt}_aa')(ints, omega, amps)).max())
    check(f'{tgt}_ab', abs(r_so[vA, vB, oA, oB]
          - getattr(ub, f'ccsd_{tgt}_ab')(ints, omega, amps)).max())
    check(f'{tgt}_bb', abs(r_so[vB, vB, oB, oB]
          - getattr(ub, f'ccsd_{tgt}_bb')(ints, omega, amps)).max())

print("ALL OK" if ok else "FAILURES PRESENT")
