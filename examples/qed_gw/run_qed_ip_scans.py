"""Coupling-strength and detuning scans of the cavity-induced IP shift.

Referee checks (Sec.~gw-bench):

1. *lambda-scan.*  Every IP/EA number of Tables I-III sits at the single
   point (lambda, omega_cav) = (0.05, 0.415668 Ha).  The claim that the
   ~40% GW overestimate of delta_lambda IP is a property of the method
   rather than of one data point rests on a lambda^2 argument.  This
   scan turns the assertion into data: delta_lambda IP for H2O/cc-pVDZ
   at lambda = 0.02, 0.05, 0.075, 0.10 for Koopmans, Delta-QED-HF,
   G0W0, evGW and Delta-QED-CCSD, together with the fractional error
   evGW/Delta-QED-CCSD - 1 at each lambda.

2. *Detuning scan.*  The manuscript performs a detuning null test for the
   screening-folded BSE but not for the quasiparticle shift.  Here
   omega_cav is scanned at fixed lambda = 0.05.  The mean-field DSE is
   omega_cav-independent by construction [the (lambda.mu)^2 term of
   Eq. (1)], so any omega_cav dependence of delta_lambda IP is
   resonance-driven and lives entirely in the correlation part.

At lambda = 0 the photon decouples (g = 0, DSE = 0), so the reference
energies are omega_cav-independent; this is verified numerically and the
lambda = 0 pass is then reused for every detuning point.

Results -> qed_ip_scans_results.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_qed_ipea_benchmark as bench                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'qed_ip_scans_results.json')

BASIS = 'cc-pVDZ'
MOL = 'H2O'
LAMBDAS = [0.02, 0.05, 0.075, 0.10]
OMEGA0 = 0.415668                                    # 11.31 eV, RPA-resonant
OMEGAS = [0.20, 0.29166, 0.35, OMEGA0, 0.50, 0.60]
KEYS = ('koop', 'dHF', 'g0w0', 'evgw', 'dCC')


def ipea(lam, omega):
    bench.OMEGA = omega
    return bench.run_molecule(MOL, BASIS, lam)


def shifts(ref, cur):
    return {k: cur[f'{k}_IP'] - ref[f'{k}_IP'] for k in KEYS}


def main():
    out = {'mol': MOL, 'basis': BASIS, 'omega0': OMEGA0}

    print('reference lambda = 0 (checking omega_cav independence)')
    ref = ipea(0.0, OMEGA0)
    ref_alt = ipea(0.0, 0.60)
    dev = max(abs(ref[f'{k}_IP'] - ref_alt[f'{k}_IP']) for k in KEYS)
    print(f'  max |IP(w=0.4157) - IP(w=0.60)| at lambda=0 : {dev:.2e} eV')
    out['lambda0_omega_independence_eV'] = dev
    out['lambda0_IP'] = {k: ref[f'{k}_IP'] for k in KEYS}
    out['lambda0_EA'] = {k: ref[f'{k}_EA'] for k in KEYS}

    print(f'\nlambda scan at omega_cav = {OMEGA0} Ha   '
          'delta_lambda IP (eV)')
    hdr = ' '.join(f'{k:>9s}' for k in KEYS)
    print(f"{'lambda':>7s} {hdr} {'evGW/dCC-1':>11s}")
    out['lambda_scan'] = {}
    for lam in LAMBDAS:
        cur = ipea(lam, OMEGA0)
        d = shifts(ref, cur)
        rel = d['evgw'] / d['dCC'] - 1.0
        d['rel_evgw'] = rel
        d['rel_g0w0'] = d['g0w0'] / d['dCC'] - 1.0
        d['EA'] = {k: cur[f'{k}_EA'] - ref[f'{k}_EA'] for k in KEYS}
        out['lambda_scan'][f'{lam:g}'] = d
        row = ' '.join(f'{d[k]:9.4f}' for k in KEYS)
        print(f'{lam:7.3f} {row} {rel:+10.1%}', flush=True)
        with open(OUT, 'w') as f:
            json.dump(out, f, indent=1)

    print('\ndetuning scan at lambda = 0.05   delta_lambda IP (eV)')
    print(f"{'w (Ha)':>7s} {'w (eV)':>8s} {hdr} {'evGW/dCC-1':>11s}")
    out['detuning_scan'] = {}
    for w in OMEGAS:
        cur = ipea(0.05, w)
        d = shifts(ref, cur)
        rel = d['evgw'] / d['dCC'] - 1.0
        d['rel_evgw'] = rel
        d['EA'] = {k: cur[f'{k}_EA'] - ref[f'{k}_EA'] for k in KEYS}
        out['detuning_scan'][f'{w:g}'] = d
        row = ' '.join(f'{d[k]:9.4f}' for k in KEYS)
        print(f'{w:7.4f} {w * 27.211386245988:8.2f} {row} {rel:+10.1%}',
              flush=True)
        with open(OUT, 'w') as f:
            json.dump(out, f, indent=1)

    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
