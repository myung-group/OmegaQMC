"""
Cholesky decomposition of ERIs and integral preparation
for AFQMC.

Functions here run at setup time (not inside JIT hot loops).
"""

import numpy as np
import jax
import jax.numpy as jnp

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


# ---------------------------------------------------------------------------
# Disk‑backed Cholesky tensor
# ---------------------------------------------------------------------------

class DiskChol:
    """File‑backed (naux, nbasis, nbasis) Cholesky tensor.

    Wraps an HDF5 dataset so the rest of the AFQMC code can consume
    ``chol`` without holding it in RAM. Slabs along the auxiliary axis
    are read on demand via ``iter_g_chunks`` or ``__getitem__``.

    The HDF5 file is opened lazily and kept open between reads. Call
    :meth:`close` when finished (or use ``with`` block).
    """

    def __init__(self, path, dataset='chol_mo', chunk_g=128):
        if h5py is None:
            raise ImportError(
                "h5py is required for DiskChol — install with `pip install h5py`"
            )
        self.path = path
        self.dataset_name = dataset
        self.chunk_g = chunk_g
        self._file = None
        # Read shape/dtype without keeping the file open
        with h5py.File(path, 'r') as f:
            ds = f[dataset]
            self.shape = tuple(ds.shape)
            self.dtype = ds.dtype

    @property
    def naux(self):
        return self.shape[0]

    @property
    def nbasis(self):
        return self.shape[1]

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.path, 'r')
        return self._file[self.dataset_name]

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getitem__(self, key):
        return self._ensure_open()[key]

    def __repr__(self):
        return (
            f"DiskChol(path='{self.path}', dataset='{self.dataset_name}', "
            f"shape={self.shape}, chunk_g={self.chunk_g})"
        )

    def iter_g_chunks(self, chunk_g=None):
        """Yield ``(g0, g1, np.ndarray)`` slabs along the auxiliary axis."""
        cg = chunk_g if chunk_g is not None else self.chunk_g
        ds = self._ensure_open()
        for g0 in range(0, self.naux, cg):
            g1 = min(g0 + cg, self.naux)
            yield g0, g1, np.asarray(ds[g0:g1])


def _iter_chol_g_chunks(chol, chunk_g):
    """Yield ``(g0, g1, chunk)`` along auxiliary axis, for either backing.

    Works transparently for :class:`DiskChol` or any array‑like with
    ``shape`` and slicing. ``chunk_g=None`` yields the whole tensor as
    a single chunk.
    """
    if isinstance(chol, DiskChol):
        yield from chol.iter_g_chunks(chunk_g)
        return
    naux = chol.shape[0]
    cg = chunk_g if chunk_g is not None else naux
    for g0 in range(0, naux, cg):
        g1 = min(g0 + cg, naux)
        yield g0, g1, chol[g0:g1]


def chunked_cholesky(mol, chol_cut=1e-5, max_vecs=None, verbose=False,
                     dense_eri_gb_limit=8.0):
    """Modified Cholesky decomposition of AO ERIs.

    Dispatches between two implementations based on whether the full
    (pq|rs) tensor fits in ``dense_eri_gb_limit`` GB:

    * dense path  (`_cholesky_dense_pivoted`): build the full ERI once,
        then pivot on the supermatrix with a vectorized subtraction.
        Fast for moderate nao (≲ ~200 AO).
    * direct path (`_cholesky_direct_pivoted`): build pivot columns on
        demand via shell‑sliced integral calls. Memory O(nchol·nao^2),
        no O(nao^4) intermediate. Required for e.g. naphthalene cc‑pVTZ.

    Both share an identical pivoted Cholesky loop with a single GEMV
    subtraction (no Python‑level for‑loop over prior vectors).

    Args:
        mol: PySCF Mole.
        chol_cut: Convergence threshold (max diagonal residual).
        max_vecs: Cap on Cholesky vectors. Defaults to ``15*nbasis``.
        verbose: Print path selection and progress.
        dense_eri_gb_limit: Route to the dense path when the full ERI
            would fit in this many GB. Default 8.0.

    Returns:
        chol_vecs: np.ndarray, shape (naux, nbasis, nbasis).
    """
    nbasis = mol.nao_nr()
    eri_gb = (nbasis ** 4 * 8) / 1024 ** 3
    use_dense = eri_gb <= dense_eri_gb_limit
    if verbose:
        path = 'dense' if use_dense else 'integral-direct'
        print(
            f"  Cholesky: {path} path "
            f"(full ERI would be {eri_gb:.2f} GB)"
        )
    if use_dense:
        return _cholesky_dense_pivoted(mol, chol_cut, max_vecs)
    return _cholesky_direct_pivoted(mol, chol_cut, max_vecs, verbose)


def _cholesky_dense_pivoted(mol, chol_cut, max_vecs):
    """Dense ERI + vectorized pivoted Cholesky (fast path)."""
    nbasis = mol.nao_nr()
    nao2 = nbasis * nbasis
    if max_vecs is None:
        max_vecs = 15 * nbasis

    eri_2d = mol.intor('int2e', aosym='s1').reshape(nao2, nao2)
    diag = np.diag(eri_2d).copy()

    chol_arr = np.empty((max_vecs, nao2))
    n_vec = 0
    for _ in range(max_vecs):
        delta_max = float(np.max(diag))
        if delta_max < chol_cut:
            break
        nu = int(np.argmax(diag))
        col = eri_2d[:, nu].copy()
        if n_vec > 0:
            # One GEMV: previously a Python for‑loop over chol_list
            col -= chol_arr[:n_vec].T @ chol_arr[:n_vec, nu]
        col /= np.sqrt(delta_max)
        chol_arr[n_vec] = col
        n_vec += 1
        diag -= col * col
        np.maximum(diag, 0.0, out=diag)

    return chol_arr[:n_vec].reshape(n_vec, nbasis, nbasis)


def _cholesky_direct_pivoted(mol, chol_cut, max_vecs, verbose=False):
    """Integral‑direct pivoted Cholesky (memory path)."""
    nbasis = mol.nao_nr()
    nao2 = nbasis * nbasis
    if max_vecs is None:
        max_vecs = 15 * nbasis

    ao_loc = mol.ao_loc_nr()
    nshell = mol.nbas

    # AO index -> (shell, offset_within_shell)
    ao_to_shell = np.zeros(nbasis, dtype=np.int32)
    for sh in range(nshell):
        ao_to_shell[ao_loc[sh]:ao_loc[sh + 1]] = sh
    ao_to_offset = (
        np.arange(nbasis, dtype=np.int32) - ao_loc[ao_to_shell]
    )

    # Diagonal D[p,q] = (pq|pq) by shell‑pair quartets
    diag2d = np.empty((nbasis, nbasis))
    for sh1 in range(nshell):
        p0, p1 = ao_loc[sh1], ao_loc[sh1 + 1]
        for sh2 in range(nshell):
            q0, q1 = ao_loc[sh2], ao_loc[sh2 + 1]
            buf = mol.intor(
                'int2e', aosym='s1',
                shls_slice=(sh1, sh1 + 1, sh2, sh2 + 1,
                            sh1, sh1 + 1, sh2, sh2 + 1),
            )
            diag2d[p0:p1, q0:q1] = np.einsum('ijij->ij', buf)
    diag = diag2d.reshape(-1)

    # Pivoted Cholesky
    chol_arr = np.empty((max_vecs, nao2))
    n_vec = 0
    print_step = max(1, max_vecs // 20)
    delta_max = float(np.max(diag))
    for _ in range(max_vecs):
        delta_max = float(np.max(diag))
        if delta_max < chol_cut:
            break
        nu = int(np.argmax(diag))
        p_star, q_star = divmod(nu, nbasis)
        sh_p = int(ao_to_shell[p_star])
        sh_q = int(ao_to_shell[q_star])
        off_p = int(ao_to_offset[p_star])
        off_q = int(ao_to_offset[q_star])

        # Single ERI column on demand: shape (nbasis, nbasis, d_p, d_q)
        buf = mol.intor(
            'int2e', aosym='s1',
            shls_slice=(0, nshell, 0, nshell,
                        sh_p, sh_p + 1, sh_q, sh_q + 1),
        )
        col = np.ascontiguousarray(
            buf[:, :, off_p, off_q]
        ).reshape(-1)

        if n_vec > 0:
            col -= chol_arr[:n_vec].T @ chol_arr[:n_vec, nu]

        col /= np.sqrt(delta_max)
        chol_arr[n_vec] = col
        n_vec += 1
        diag -= col * col
        np.maximum(diag, 0.0, out=diag)

        if verbose and (n_vec % print_step == 0):
            print(
                f"    Cholesky iter {n_vec}: "
                f"delta_max = {delta_max:.3e}"
            )

    if verbose:
        print(
            f"    Cholesky converged in {n_vec} vectors "
            f"(final delta_max = {delta_max:.3e})"
        )

    return chol_arr[:n_vec].reshape(n_vec, nbasis, nbasis)


def prepare_afqmc_integrals(
    mf, chol_cut=1e-5, mo_coeff=None,
    chol_h5_path=None, chol_chunk_g=128,
):
    """Prepare all integrals for AFQMC.

    Computes:
    1. Cholesky-decomposed ERIs in MO basis
    2. Modified one-body Hamiltonian (h1e_mod)
    3. Nuclear repulsion energy

    Args:
        mf: PySCF mean-field object (RHF or UHF,
            must have run kernel()).
        chol_cut: Cholesky decomposition threshold.
        mo_coeff: MO coefficient matrix to use instead
            of mf.mo_coeff.  Useful for CASSCF trials.
        chol_h5_path: If set, write the MO‑basis Cholesky tensor to this
            HDF5 file (dataset name ``chol_mo``) and return a
            :class:`DiskChol` handle as ``chol`` instead of an in‑memory
            array. AO→MO transform and v0 accumulation are done in
            ``chol_chunk_g`` slabs so peak RAM is independent of naux.
        chol_chunk_g: Slab size along the auxiliary axis. Default 128.

    Returns:
        dict with keys:
            'h1e' : shape (nbasis, nbasis)
            'h1e_mod' : shape (nbasis, nbasis)
            'chol' : shape (naux, nbasis, nbasis); either jnp.ndarray (default) or :class:`DiskChol` if ``chol_h5_path`` set
            'enuc' : float
            'nbasis', 'nup', 'ndown' : int
            'mo_coeff' : jnp.array
    """
    mol = mf.mol
    nbasis = mol.nao_nr()

    if mo_coeff is None:
        mo_coeff = np.asarray(mf.mo_coeff)
    else:
        mo_coeff = np.asarray(mo_coeff)

    nup = mol.nelec[0]
    ndown = mol.nelec[1]

    hcore_ao = np.asarray(mf.get_hcore())
    h1e = mo_coeff.T @ hcore_ao @ mo_coeff

    # Cholesky decomposition in AO basis
    chol_ao = chunked_cholesky(mol, chol_cut=chol_cut)
    naux = chol_ao.shape[0]
    nmo = mo_coeff.shape[1]

    if chol_h5_path is None:
        # In‑memory path: AO→MO transform as two BLAS calls
        tmp = chol_ao.reshape(naux * nbasis, nbasis) @ mo_coeff
        tmp = tmp.reshape(naux, nbasis, nmo)
        chol_mo = (
            mo_coeff.T
            @ tmp.transpose(1, 0, 2).reshape(nbasis, naux * nmo)
        ).reshape(nmo, naux, nmo).transpose(1, 0, 2)

        # v0[i,k] = -0.5 sum_{g,j} L[g,i,j] L[g,k,j]
        A = chol_mo.transpose(0, 2, 1).reshape(naux * nmo, nmo)
        v0 = -0.5 * (A.T @ A)
        chol_out = jnp.array(chol_mo)

    else:
        # Disk‑backed path: stream AO→MO transform and v0 by g‑chunks
        if h5py is None:
            raise ImportError(
                "h5py is required for chol_h5_path — install with "
                "`pip install h5py`"
            )
        v0 = np.zeros((nmo, nmo))
        with h5py.File(chol_h5_path, 'w') as f:
            ds = f.create_dataset(
                'chol_mo',
                shape=(naux, nmo, nmo),
                dtype=np.float64,
                chunks=(min(chol_chunk_g, naux), nmo, nmo),
            )
            for g0 in range(0, naux, chol_chunk_g):
                g1 = min(g0 + chol_chunk_g, naux)
                chunk_ao = chol_ao[g0:g1]
                ng = g1 - g0
                # AO→MO: two GEMMs per chunk
                tmp = chunk_ao.reshape(ng * nbasis, nbasis) @ mo_coeff
                tmp = tmp.reshape(ng, nbasis, nmo)
                chunk_mo = (
                    mo_coeff.T
                    @ tmp.transpose(1, 0, 2).reshape(nbasis, ng * nmo)
                ).reshape(nmo, ng, nmo).transpose(1, 0, 2)
                ds[g0:g1] = chunk_mo
                # Accumulate v0 contribution from this chunk
                A = chunk_mo.transpose(0, 2, 1).reshape(ng * nmo, nmo)
                v0 += A.T @ A
                del tmp, chunk_mo, A
        v0 *= -0.5
        chol_out = DiskChol(chol_h5_path, dataset='chol_mo',
                            chunk_g=chol_chunk_g)

    h1e_mod = h1e + v0
    enuc = mol.energy_nuc()

    return {
        'h1e': jnp.array(h1e),
        'h1e_mod': jnp.array(h1e_mod),
        'chol': chol_out,
        'enuc': float(enuc),
        'nbasis': nbasis,
        'nup': nup,
        'ndown': ndown,
        'mo_coeff': jnp.array(mo_coeff),
    }


def half_rotate_cholesky(chol, trial_up, trial_dn, chunk_g=None):
    """Half-rotate Cholesky vectors with trial wavefunction.

    Precomputes rchol[g,i,q] = sum_p trial[p,i]* L^g_{pq}
    to reduce cost from O(M^2 * naux) to O(nocc * M * naux).

    Accepts either an in-memory ``chol`` (np.ndarray / jnp.ndarray) or a
    :class:`DiskChol`. The output ``rchol_a/b`` is (naux, nocc, nbasis)
    — much smaller than ``chol`` — and stays in RAM.

    Args:
        chol: Cholesky vectors,
            shape (naux, nbasis, nbasis).
        trial_up: Trial alpha orbitals,
            shape (nbasis, nup).
        trial_dn: Trial beta orbitals,
            shape (nbasis, ndown).
        chunk_g: g-axis slab size when streaming a DiskChol. Defaults
            to the DiskChol's chunk_g, or no chunking for arrays.

    Returns:
        rchol_a: shape (naux, nup, nbasis).
        rchol_b: shape (naux, ndown, nbasis).
    """
    if isinstance(chol, DiskChol):
        naux = chol.naux
        nbasis = chol.nbasis
        nup = trial_up.shape[1]
        ndn = trial_dn.shape[1]
        tu = np.asarray(trial_up).conj()
        td = np.asarray(trial_dn).conj()
        rchol_a = np.empty((naux, nup, nbasis), dtype=tu.dtype)
        rchol_b = np.empty((naux, ndn, nbasis), dtype=td.dtype)
        for g0, g1, chunk in chol.iter_g_chunks(chunk_g):
            rchol_a[g0:g1] = np.einsum('pi,gpq->giq', tu, chunk)
            rchol_b[g0:g1] = np.einsum('pi,gpq->giq', td, chunk)
        return jnp.array(rchol_a), jnp.array(rchol_b)

    rchol_a = jnp.einsum('pi,gpq->giq', trial_up.conj(), chol)
    rchol_b = jnp.einsum('pi,gpq->giq', trial_dn.conj(), chol)
    return rchol_a, rchol_b


def half_rotate_cholesky_multidet(
    chol, trials_up, trials_dn, chunk_g=None,
):
    """Half-rotate Cholesky vectors for multi-det trial.

    Accepts in-memory chol or :class:`DiskChol`. With many determinants
    the output ``rchols_a/b`` can become huge (ndet × naux × nocc × M);
    that does NOT get spilled to disk here.

    Args:
        chol: Cholesky vectors,
            shape (naux, nbasis, nbasis).
        trials_up: Trial alpha orbitals,
            shape (ndet, nbasis, nup).
        trials_dn: Trial beta orbitals,
            shape (ndet, nbasis, ndown).
        chunk_g: g-axis slab size when streaming a DiskChol.

    Returns:
        rchols_a: shape (ndet, naux, nup, nbasis).
        rchols_b: shape (ndet, naux, ndown, nbasis).
    """
    if isinstance(chol, DiskChol):
        ndet = trials_up.shape[0]
        naux = chol.naux
        nbasis = chol.nbasis
        nup = trials_up.shape[2]
        ndn = trials_dn.shape[2]
        tu = np.asarray(trials_up).conj()
        td = np.asarray(trials_dn).conj()
        rchols_a = np.empty((ndet, naux, nup, nbasis), dtype=tu.dtype)
        rchols_b = np.empty((ndet, naux, ndn, nbasis), dtype=td.dtype)
        for g0, g1, chunk in chol.iter_g_chunks(chunk_g):
            rchols_a[:, g0:g1] = np.einsum('dpi,gpq->dgiq', tu, chunk)
            rchols_b[:, g0:g1] = np.einsum('dpi,gpq->dgiq', td, chunk)
        return jnp.array(rchols_a), jnp.array(rchols_b)

    _hr = lambda trial: jnp.einsum(
        'pi,gpq->giq', trial.conj(), chol,
    )
    rchols_a = jax.vmap(_hr)(trials_up)
    rchols_b = jax.vmap(_hr)(trials_dn)
    return rchols_a, rchols_b
