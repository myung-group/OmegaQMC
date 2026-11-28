"""Error of the PGCS correlated-sampling energy offset.

``_cs_diagnostic_lines`` prints ``<w E_L(TR)>/<w> - E_V ± err`` for
each secondary state and warns beyond ``CS_OFFSET_WARN_Z`` errors.
The error used to treat blocks as independent and E_V as exact; it
now bins the delta-method series of the difference of the two ratios
(``E_V`` is the mean of the same blocks' ``E_L(R)``) with
:func:`OmegaQMC.utils.binned_block_error`.  With weighted synthetic
blocks whose offset and reference energy follow AR(1) processes
(rho = 0.5) these tests check

* that the printed error matches the analytic error of the offset,
  with and without a block-level fluctuation common to E_L(TR) and
  E_L(R), which the old error missed by a factor of about sqrt(3)
  (too small) and counted as offset noise (too large), respectively;
* that an exact symmetry gives a zero offset and error and no warning;
* that a genuine offset still warns.
"""
import re
import warnings

import numpy as np
import pytest

from OmegaQMC.observables.force import (
    CS_OFFSET_WARN_Z,
    _cs_accumulate,
    _cs_diagnostic_lines,
    _new_cs_accumulator,
)

RHO = 0.5
N_BLOCKS, N_SAMPLES = 400, 200
SIG_C = 0.05          # block-level offset fluctuation, AR(1)
SIG_REF = 0.1         # per-sample noise common to E_L(TR) and E_L(R)
SIG_T = 0.01          # per-sample noise of E_L(TR) alone
SIG_LOGW = 0.3


def _ar1(rng, n):
    x = np.empty(n)
    x[0] = rng.normal()
    for b in range(1, n):
        x[b] = RHO * x[b - 1] + np.sqrt(1 - RHO ** 2) * rng.normal()
    return x


def _blocks(seed, sig_a, shift=0.0):
    """Weighted blocks; E_L(TR) = E_L(R) + c_b + noise."""
    rng = np.random.default_rng(seed)
    a = sig_a * _ar1(rng, N_BLOCKS)       # common block fluctuation
    c = SIG_C * _ar1(rng, N_BLOCKS) + shift
    out = []
    for b in range(N_BLOCKS):
        w = np.exp(SIG_LOGW * rng.normal(size=N_SAMPLES))
        e_ref = -1.0 + a[b] + SIG_REF * rng.normal(size=N_SAMPLES)
        e_t = e_ref + c[b] + SIG_T * rng.normal(size=N_SAMPLES)
        out.append((w, e_t, e_ref))
    return out


def _analytic_err(blocks):
    """Error of mean_b [sum w E_t / sum w - mean E_ref] for the model:
    AR(1) part plus each block's white-noise variance."""
    white = 0.0
    for w, _, _ in blocks:
        p = w / w.sum()
        white += (SIG_REF ** 2 * np.sum((p - 1 / N_SAMPLES) ** 2)
                  + SIG_T ** 2 * np.sum(p ** 2))
    var = SIG_C ** 2 * (1 + RHO) / (1 - RHO) / N_BLOCKS
    return np.sqrt(var + white / N_BLOCKS ** 2)


def _diagnose(blocks):
    e_v = np.mean([e_ref.mean() for _, _, e_ref in blocks])
    acc = _new_cs_accumulator(True)
    for w, e_t, e_ref in blocks:
        _cs_accumulate(acc, w, e_t, e_ref, e_v)
    stats = {'n': N_BLOCKS * N_SAMPLES, 'mean_w': 1.0,
             'max_frac': 1e-3, 'ess': 0.9}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lines = _cs_diagnostic_lines('1:Rz180', acc, stats)
    m = re.search(r'- E_V = (\S+) ± (\S+) Ha', lines[1])
    old = np.sqrt(
        np.sum((np.array(acc['y_blk'])
                - float(m.group(1)) * np.array(acc['w_blk'])) ** 2)
        / (N_BLOCKS * (N_BLOCKS - 1) * np.mean(acc['w_blk']) ** 2))
    return (float(m.group(1)), float(m.group(2)), old,
            [str(w.message) for w in caught])


@pytest.mark.parametrize('sig_a, old_lo, old_hi', [
    (0.0, 0.0, 0.7),       # old: autocorrelation missed, too small
    (0.5, 1.5, np.inf),    # old: E_V's fluctuation counted, too large
])
def test_offset_error_matches_analytic(sig_a, old_lo, old_hi):
    blocks = _blocks(1, sig_a)
    offset, err, old, msgs = _diagnose(blocks)
    exact = _analytic_err(blocks)
    # over seeds the binned error is 0.90 +- 0.08 of exact (BLUE's
    # ceil(2 kappa) bins are a little short); the old one 0.58 / 5.8
    assert 0.8 < err / exact < 1.15
    assert old_lo < old / exact < old_hi
    assert abs(offset) < CS_OFFSET_WARN_Z * err and not msgs


def test_exact_symmetry_zero_error_no_warning():
    rng = np.random.default_rng(2)
    blocks = []
    for b in range(50):
        e_ref = -1.0 + rng.normal(size=N_SAMPLES)
        blocks.append((np.ones(N_SAMPLES), e_ref.copy(), e_ref))
    offset, err, old, msgs = _diagnose(blocks)
    assert abs(offset) < 1e-14 and err < 1e-14 and not msgs
    assert old > 1e-3        # the old error was E_V's noise


def test_genuine_offset_warns():
    blocks = _blocks(3, 0.5, shift=0.1)
    offset, err, _, msgs = _diagnose(blocks)
    assert offset > CS_OFFSET_WARN_Z * err
    assert len(msgs) == 1 and 'standard errors from zero' in msgs[0]
