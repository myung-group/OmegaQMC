"""
Shared helpers for the QED-CCSD backends (:mod:`OmegaQMC.addons.qed_ccsd_rhf`
and :mod:`OmegaQMC.addons.qed_ccsd_uhf`):

* :func:`_vvvv_ladder` — batched all-virtual particle-ladder contraction
  from the 3-index DF factor (peak intermediate ``nvir**3``; the
  ``nvir**4`` tensor is never formed);
* :func:`_ao_df_factor` / :func:`_eigh_factor_ao` — AO-basis 3-index
  factor with the dipole self-energy folded in; a dense (non-DF) QED-HF
  reference is factorised exactly by eigendecomposition, so both
  backends reproduce dense-integral results to machine precision;
* :class:`_DiisHistory` — bounded DIIS history (Pulay extrapolation)
  that can optionally live on disk as ``.npy`` files.
"""

import os
import shutil
import tempfile

import numpy as np


# ---------------------------------------------------------------------------
# Batched particle-ladder helper
# ---------------------------------------------------------------------------
def _vvvv_ladder(B_vv, W, out=None, alpha=1.0):
    """``out[a,b,i,j] += alpha * sum_{x,c,d} B_vv[x,a,c] B_vv[x,b,d]
    W[c,d,i,j]``.

    Batched over ``b`` so the largest intermediate is ``nvir**3`` — the
    all-virtual 4-index tensor (and the even larger ``naux nvir^2 nocc^2``
    einsum intermediate a naive contraction would create) never exists.
    Accumulating into ``out`` in place avoids allocating a second
    ``o^2 v^2`` array at the peak-memory point of the doubles kernels.
    ``W`` must be C-contiguous (the doubles amplitudes are).
    """
    naux, nvir = B_vv.shape[0], B_vv.shape[1]
    nocc = W.shape[2]
    Wm = W.reshape(nvir * nvir, nocc * nocc)
    # One contiguous (a*c, x) copy up front so the per-b dgemm below never
    # re-copies the strided B_vv slice.
    Bm = np.ascontiguousarray(B_vv.transpose(1, 2, 0)).reshape(
        nvir * nvir, naux)
    if out is None:
        out = np.zeros((nvir, nvir, nocc, nocc))
    for b in range(nvir):
        # g_b[(a, c), d] = sum_x B_vv[x, a, c] * B_vv[x, b, d]
        g_b = Bm @ B_vv[:, b, :]
        out[:, b] += alpha * (g_b.reshape(nvir, nvir * nvir) @ Wm).reshape(
            nvir, nocc, nocc)
    return out


# ---------------------------------------------------------------------------
# AO-basis 3-index factor (DSE folded in)
# ---------------------------------------------------------------------------
def _eigh_factor_ao(eri_ao, tol=1e-12):
    """Exact 3-index factorisation of a dense AO ERI by eigendecomposition:
    ``(pq|rs) = sum_x B[x,p,q] B[x,r,s]``. Used when the QED-HF reference was
    run without density fitting, so the DF backends reproduce dense-integral
    results to machine precision. Only sensible for small AO dimensions.
    """
    nao = eri_ao.shape[0]
    M = np.asarray(eri_ao).reshape(nao * nao, nao * nao)
    M = 0.5 * (M + M.T)
    evals, evecs = np.linalg.eigh(M)
    keep = evals > tol * max(evals.max(), 1.0)
    B = (evecs[:, keep] * np.sqrt(evals[keep])).T
    return np.ascontiguousarray(B.reshape(-1, nao, nao))


def _ao_df_factor(qedhf):
    """AO DF factor with the dipole self-energy folded in as three extra
    auxiliary vectors (λ_X μ_X), exactly as in qed_hf.eri_mo_transform."""
    lam = qedhf['lambda_cav']
    mu = (qedhf['mu_x_ao'], qedhf['mu_y_ao'], qedhf['mu_z_ao'])
    dse = np.stack([lam[a] * mu[a] for a in range(3)])
    if 'eri_df' in qedhf:
        B_raw = qedhf['eri_df']
    else:
        B_raw = _eigh_factor_ao(qedhf['eri_ao'])
    return np.concatenate([B_raw, dse], axis=0)


# ---------------------------------------------------------------------------
# Bounded DIIS history, optionally stored on disk
# ---------------------------------------------------------------------------
class _DiisHistory:
    """Bounded DIIS bookkeeping (amplitude vectors + step-error vectors,
    Pulay extrapolation over vals[1:]) whose history can live on disk as
    ``.npy`` files. Error vectors are kept as per-amplitude pieces (never
    concatenated) and the Gram matrix is cached incrementally, so disk mode
    reads each stored piece once per iteration and the peak RAM cost of
    DIIS is one amplitude array."""

    def __init__(self, names, max_diis, on_disk=False):
        self.names = list(names)
        self.max_diis = max_diis
        self.on_disk = on_disk
        self.dir = tempfile.mkdtemp(prefix='qed_ccsd_diis_') if on_disk \
            else None
        self.vals = []   # each: dict name -> array (or .npy path)
        self.errs = []   # each: dict name -> array (or .npy path)
        self.gram = np.zeros((0, 0))
        self._seq = 0

    def _store(self, arr, tag):
        if not self.on_disk:
            return np.asarray(arr)
        path = os.path.join(self.dir, f'{tag}_{self._seq}.npy')
        np.save(path, np.asarray(arr))
        return path

    @staticmethod
    def _load(x):
        return np.load(x, mmap_mode='r') if isinstance(x, str) else x

    def _drop(self, entry):
        if self.on_disk:
            for p in entry.values():
                if isinstance(p, str):
                    os.unlink(p)

    def _err_dot(self, e1, e2):
        # np.dot streams memmaps through the page cache — no full copy
        return sum(float(np.dot(self._load(e1[k]).ravel(),
                                self._load(e2[k]).ravel()))
                   for k in self.names)

    def append(self, amps, err):
        """amps/err: dict name -> array (post-update amplitudes and the
        per-amplitude step pieces)."""
        self._seq += 1
        row = [self._err_dot(e, err) for e in self.errs]
        n = len(self.errs)
        gram = np.empty((n + 1, n + 1))
        gram[:n, :n] = self.gram
        gram[:n, n] = row
        gram[n, :n] = row
        gram[n, n] = self._err_dot(err, err)
        self.gram = gram
        self.vals.append({k: self._store(v, f'val_{k}')
                          for k, v in amps.items()})
        self.errs.append({k: self._store(v, f'err_{k}')
                          for k, v in err.items()})
        if len(self.vals) > self.max_diis:
            self._drop(self.vals.pop(0))
            self._drop(self.errs.pop(0))
            self.gram = self.gram[1:, 1:]

    def extrapolate(self):
        """Return dict name -> extrapolated amplitude, or None if the
        history is too short (needs >= 2 stored iterations)."""
        m = len(self.errs)
        if m < 2:
            return None
        # Pulay 1980, eqn 6
        B = -np.ones((m + 1, m + 1))
        B[-1, -1] = 0.0
        B[:m, :m] = self.gram
        B[:m, :m] /= np.abs(B[:m, :m]).max()
        rhs = np.zeros(m + 1)
        rhs[-1] = -1.0
        ci = np.linalg.solve(B, rhs)
        out = {}
        for k in self.names:
            acc = None
            # vals[0] is the pre-history zero entry (extrapolation runs
            # over vals[1:] only)
            for w_c, val in zip(ci[:m], self.vals[1:]):
                piece = self._load(val[k]) * w_c   # one allocation per piece
                if acc is None:
                    acc = piece
                else:
                    acc += piece
            out[k] = acc
        return out

    def cleanup(self):
        if self.on_disk and self.dir and os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
