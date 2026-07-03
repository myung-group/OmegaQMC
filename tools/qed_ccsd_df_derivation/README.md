# QED-CCSD density-fitted equation derivation

This directory derives and validates the QED-CCSD residual equations used
by `OmegaQMC/addons/qed_ccsd_rhf.py` and `OmegaQMC/addons/qed_ccsd_uhf.py`.

NOTE (2026-07 cleanup): the intermediate spin-orbital DF module
`OmegaQMC/addons/qed_ccsd_new.py` referenced throughout this README was
removed from the package — it is superseded by the spin-adapted
`qed_ccsd_rhf` / `qed_ccsd_uhf` backends (dispatched by
`qed_ccsd.run_qed_ccsd`), with the shared helpers (`_vvvv_ladder`,
`_ao_df_factor`, `_DiisHistory`, ...) kept in
`OmegaQMC/addons/qed_ccsd_utils.py`. The `validate_*_kernels.py` scripts
below therefore no longer run as-is; to rerun them, reassemble the
spin-orbital module from the derived equations in `eqs/` via
`gen_module.py` as described below.

`derive_qed_ccsd_df.py` derives the QED-CCSD-22 residual equations with the
improved Wick (`/Users/willow/Python/wick`, i.e. awhite862/wick plus the
`two_e_df` density-fitted two-electron operator), so the equations contain
only the 3-index DF factor `B_{x,pq}` — no `nmo^4` integral tensor. It adds
the cluster operators and bra projections missing from `wick.convenience`
(boson x double-excitation `u21`/`u22`, and the `<0|b (ij->ab)` /
`<0|bb (ij->ab)` projections).

```sh
# one residual per invocation; all nine run in parallel (~1 min each,
# t2_22 takes ~40 min):
for t in energy t1_10 t1_01 t2_02 t2_11 t2_12 t2_20 t2_21 t2_22; do
  PYTHONPATH=~/Python/wick python derive_qed_ccsd_df.py $t > eqs/$t.py.txt &
done
wait
```

`gen_module.py` converts the raw einsum output in `eqs/` into the residual
kernels (`gen_out/*.py`): it slices `f`/`d`/`B` into occ/vir blocks, drops
the single-cavity-mode boson indices, renames amplitudes to the
`qed_ccsd.py` convention, guards optional-amplitude terms with the `active`
set, applies chunked in-place energy denominators, and routes the
particle-ladder terms (two `B_vv` factors against a 4-index amplitude)
through the batched `_vvvv_ladder` helper (peak intermediate `nvir^3`).

The final module is the concatenation of the static header (docstring +
`_vvvv_ladder`), the nine generated kernels, and the static driver
(builders, disk-capable DIIS, `run_qed_ccsd`) — see the scratchpad build
used on 2026-07-02, or reassemble by splitting qed_ccsd_new.py at the
generated-kernel boundaries.

Validation performed (2026-07-02):

* lambda=0 reproduces plain (frozen-core) CCSD from pyscf;
* all residual kernels except t2_21/t2_22 agree with qed_ccsd.py to
  machine precision on random amplitudes;
* t2_21/t2_22 differ ONLY by the quartic BCH terms (e.g. g*t1^3*u11) that
  the old psi4-reference equations are missing — see the qed_ccsd_new.py
  docstring;
* glycolaldehyde/STO-3G QED-CCSD-21: -262.416985787 (complete equations)
  vs -262.416986187 (published number from the incomplete equations).

## Closed-shell spin adaptation (`qed_ccsd_rhf.py`)

`spin_trace.py` exactly spin-traces the spin-orbital equations in `eqs/`
to closed-shell spatial form (doubles = mixed-spin blocks
`T[a,b,i,j] = t2_so[a-alpha, b-beta, i-alpha, j-beta]`; union-find over
spin delta constraints; factor 2 per free spin group; term merging using
amplitude/integral permutational symmetries). `gen_rhf.py` emits the
spatial kernels of `OmegaQMC/addons/qed_ccsd_rhf.py`, substituting each
non-(vv|vv) `B.B` pair by a precomputed 4-index chemist block
(`v_oooo`..`v_ovvv`, built once per run) and grouping the all-virtual
ladder terms into a single batched `_vvvv_ladder` call per guard group.

`validate_rhf_kernels.py` checks every spatial kernel against the
corresponding spin block of the qed_ccsd_new kernels on random
closed-shell-consistent amplitudes (agreement to ~1e-15, 2026-07-02).
Converged energies agree with qed_ccsd_new to ~6e-13 (H2O -21/-22);
glycolaldehyde/STO-3G -21 gives -262.416985790 in 0.16 s/iteration
(~25x faster than the spin-orbital module).

## Spin-blocked UHF version for open shells (`qed_ccsd_uhf.py`)

`spin_trace_uhf.py` traces the same spin-orbital equations to
alpha/beta-blocked form for QED-UHF references (independent spins,
nocc_a != nocc_b): explicit spin enumeration per delta-constraint group,
blocked amplitudes (t1_10_a/b, antisymmetric t2_20_aa/bb, mixed
t2_20_ab, ...), signed canonical merging. `gen_uhf.py` emits one kernel
per residual spin block (18 kernels) with spin-resolved chemist blocks
(pq|rs)_{s1 s2} precomputed once per run and a two-factor batched
(vv|vv) ladder (`_vvvv_ladder2`).

`validate_uhf_kernels.py` checks all 18 kernels blockwise against
qed_ccsd_new on a QED-UHF H2O+ reference (~1e-15, 2026-07-02).
Converged: H2O+ -21/-22 match qed_ccsd_new to ~3e-13; OH/lambda=0
matches pyscf UCCSD to 6e-12; closed-shell limit matches qed_ccsd_rhf
to 3e-12. Cost is ~3-4x the closed-shell module (vs ~16x for the
spin-orbital code); for tiny (STO-3G) systems the per-term overhead
makes it slower than qed_ccsd_new — the advantage appears at
double-zeta molecule sizes.

## Singlet-adapted QED-RPA / polaritonic QED-BSE (`qed_polariton_singlet.py`)

Closed-shell spin adaptation of qed_rpa / qed_gw(evGW) /
qed_bse_polaritonic: singlet kernels 2*Coulomb - exchange over spatial ov
pairs, photon vertex sqrt(2)*g in the singlet channel only, dRPA
screening tensor M with the sqrt(2) spin sum folded in, everything built
from a 3-index DF factor (no nmo^4). `validate_polariton_singlet.py`
(2026-07-02, H2O/6-31g): RPA/dRPA correlation energies and full/TDA
spectra (singlet + 3x triplet = spin-orbital spectrum), evGW QP energies
for all orbitals, and pol-BSE bright-root energies / oscillator
strengths / photon weights all match the spin-orbital modules to
machine precision. Naphthalene/cc-pVDZ figure:
examples/run_qed_absorption_naph_singlet.py (~3 GB working set).
