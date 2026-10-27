"""Multi-determinant trial wavefunctions from PySCF solvers.

:func:`trial_from_pyscf` turns the result of a PySCF correlated
calculation into the ``trial`` dict that the GTO VMC, GTO VMC
optimisation and AFQMC drivers accept.  Supported sources:

- CASCI / CASSCF on RHF or ROHF orbitals, including state-averaged
  and multi-root runs (pick a root with ``state``), with the default
  FCI solver or a selected-CI ``fcisolver``;
- FCI solvers (``pyscf.fci.FCI``, ``direct_spin0``/``direct_spin1``
  and their symmetry variants, ``selected_ci.SCI``), given the MO
  coefficients the solver was run in;
- RCISD (``pyscf.ci.CISD`` on RHF), with frozen orbitals;
- a bare CI vector in PySCF's FCI layout (alpha strings x beta
  strings, or an ``SCIvector``), given the MO coefficients and the
  active electron count.

UHF-based expansions (UCASCI/UCASSCF, UCISD, ``direct_uhf``) need
separate alpha and beta orbitals, which the drivers do not support,
and coupled-cluster wavefunctions are not finite determinant
expansions; both raise an error.

Determinant convention: determinant ``I`` is the product of an alpha
and a beta Slater determinant whose columns are the MOs
``mo_coeff[:, occ_up[I]]`` and ``mo_coeff[:, occ_dn[I]]`` in the
listed order, with alpha electrons before beta electrons.  For the
sorted occupation lists of PySCF's CI strings this is PySCF's own
sign convention, so CI coefficients carry over unchanged.  CISD
excitations are listed in place (each hole replaced by its particle
in the reference ordering), which absorbs the excitation sign.
"""

import numpy as np
import jax.numpy as jnp

_UHF_MSG = ("UHF-based multi-determinant trials (separate alpha and "
            "beta orbitals) are not supported")


def _select_state(ci, state, what):
    """Return one CI vector out of a (possibly multi-root) result."""
    if ci is None:
        raise ValueError(f"{what} has no CI vector; run kernel() first")
    if isinstance(ci, (list, tuple)):
        if state is None:
            if len(ci) == 1:
                return ci[0]
            raise ValueError(
                f"{what} holds {len(ci)} CI vectors (multi-root or "
                f"state-averaged); choose one with `state`")
        return ci[state]
    if state not in (None, 0):
        raise ValueError(f"{what} holds a single CI vector; "
                         f"state={state} is out of range")
    return ci


def _occ_from_strs(strs, norb):
    """Occupation lists (ascending) from integer bit strings."""
    strs = np.asarray(strs, dtype=np.int64)
    bits = (strs[:, None] >> np.arange(norb)) & 1
    nel = int(bits[0].sum()) if len(strs) else 0
    return np.nonzero(bits)[1].reshape(len(strs), nel)


def _dets_from_fcivec(ci, norb, nelec, ncore, threshold):
    """Determinants of a CI vector in PySCF's FCI string layout.

    Args:
        ci: CI vector, shape (na, nb) or flat; an ``SCIvector`` uses
            its own string lists.
        norb: Number of correlated (active) orbitals.
        nelec: (nalpha, nbeta) active electrons.
        ncore: Doubly occupied orbitals below the active space.
        threshold: Keep determinants with ``|c| > threshold``.

    Returns:
        (coeffs, occ_up, occ_dn, total_weight), occupations in the
        full MO index space.
    """
    from pyscf.fci import cistring
    nela, nelb = nelec
    strs = getattr(ci, '_strs', None)
    if strs is not None:
        occa = _occ_from_strs(strs[0], norb)
        occb = _occ_from_strs(strs[1], norb)
    else:
        occa = np.asarray(cistring.gen_occslst(range(norb), nela))
        occb = np.asarray(cistring.gen_occslst(range(norb), nelb))
        occa = occa.reshape(-1, nela)
        occb = occb.reshape(-1, nelb)
    ci = np.asarray(ci).reshape(len(occa), len(occb))
    ia, ib = np.nonzero(np.abs(ci) > threshold)
    core = np.broadcast_to(np.arange(ncore), (len(ia), ncore))
    occ_up = np.hstack([core, ncore + occa[ia]])
    occ_dn = np.hstack([core, ncore + occb[ib]])
    return ci[ia, ib], occ_up, occ_dn, np.sum(np.abs(ci) ** 2)


def _dets_from_cisd(civec, nmo_full, nocc_full, frozen, threshold):
    """Determinants of an RCISD vector, built excitation by excitation.

    The cost scales with the number of CISD amplitudes, not with the
    FCI space, so there is no limit on the number of orbitals.

    Args:
        civec: RCISD vector (``myci.ci``).
        nmo_full: Number of MOs including frozen ones.
        nocc_full: Number of doubly occupied MOs including frozen.
        frozen: ``myci.frozen`` (None, an int or a list of indices).
        threshold: Keep determinants with ``|c| > threshold``.

    Returns:
        (coeffs, occ_up, occ_dn, total_weight).
    """
    from pyscf.ci import cisd
    mask = np.zeros(nmo_full, dtype=bool)
    if frozen is not None:
        if isinstance(frozen, (int, np.integer)):
            mask[:frozen] = True
        else:
            mask[np.asarray(frozen, dtype=int)] = True
    occ = np.nonzero(~mask[:nocc_full])[0]
    vir = nocc_full + np.nonzero(~mask[nocc_full:])[0]
    nocc, nvir = len(occ), len(vir)
    c0, c1, c2 = cisd.cisdvec_to_amplitudes(
        np.asarray(civec), nocc + nvir, nocc, copy=False)
    ref = np.arange(nocc_full)

    def singles(ii, aa):
        # Reference rows with hole occ[i] replaced in place by vir[a]
        rows = np.tile(ref, (len(ii), 1))
        rows[np.arange(len(ii)), occ[ii]] = vir[aa]
        return rows

    def doubles(ii, jj, aa, bb):
        # i > j and a > b: occ[j] -> vir[b] and occ[i] -> vir[a], so
        # the in-place determinant has the sign of PySCF's c2aa
        rows = singles(ii, aa)
        rows[np.arange(len(ii)), occ[jj]] = vir[bb]
        return rows

    coeffs = [np.atleast_1d(c0)]
    ups, dns = [ref[None]], [ref[None]]
    w = [c0 ** 2]

    i, a = np.nonzero(np.abs(c1) > threshold)
    s = singles(i, a)
    coeffs += [c1[i, a], c1[i, a]]
    ups += [s, np.tile(ref, (len(i), 1))]
    dns += [np.tile(ref, (len(i), 1)), s]
    w.append(2 * np.sum(c1 ** 2))

    # Opposite-spin doubles: alpha i -> a, beta j -> b
    i, j, a, b = np.nonzero(np.abs(c2) > threshold)
    coeffs.append(c2[i, j, a, b])
    ups.append(singles(i, a))
    dns.append(singles(j, b))
    w.append(np.sum(c2 ** 2))

    # Same-spin doubles
    c2aa = c2 - c2.transpose(1, 0, 2, 3)
    pair = (np.arange(nocc)[:, None] > np.arange(nocc))[:, :, None, None]
    pair = pair & (np.arange(nvir)[:, None] > np.arange(nvir))
    i, j, a, b = np.nonzero(pair & (np.abs(c2aa) > threshold))
    d = doubles(i, j, a, b)
    coeffs += [c2aa[i, j, a, b], c2aa[i, j, a, b]]
    ups += [d, np.tile(ref, (len(i), 1))]
    dns += [np.tile(ref, (len(i), 1)), d]
    w.append(2 * np.sum(c2aa[np.broadcast_to(pair, c2aa.shape)] ** 2))

    coeffs = np.concatenate(coeffs)
    keep = np.ones(len(coeffs), dtype=bool)
    keep[0] = abs(c0) > threshold
    return (coeffs[keep], np.concatenate(ups)[keep],
            np.concatenate(dns)[keep], float(np.sum(w)))


def _build_trial(coeffs, occ_up, occ_dn, mo_coeff, max_ndet,
                 total_weight):
    """Sort by |c| (descending), truncate and pack the trial dict."""
    if len(coeffs) == 0:
        raise ValueError("no determinant has |c| above coeff_threshold")
    order = np.argsort(-np.abs(coeffs), kind='stable')
    if max_ndet is not None:
        order = order[:max_ndet]
    coeffs = np.asarray(coeffs)[order]
    return {
        'ci_coeffs': jnp.asarray(coeffs),
        'occ_up': jnp.asarray(np.asarray(occ_up)[order], dtype=jnp.int32),
        'occ_dn': jnp.asarray(np.asarray(occ_dn)[order], dtype=jnp.int32),
        'ndet': len(order),
        'mo_coeff': np.asarray(mo_coeff),
        'ci_weight': float(np.sum(np.abs(coeffs) ** 2) / total_weight),
    }


def trial_from_pyscf(obj, coeff_threshold=1e-4, max_ndet=None,
                     state=None, mo_coeff=None, nelec=None, ncore=0,
                     norb=None):
    """Multi-determinant ``trial`` dict from a PySCF calculation.

    Args:
        obj: A converged PySCF object or CI vector:

            - CASCI / CASSCF (RHF/ROHF orbitals); orbitals, core and
              active space are read from it.
            - RCISD (``pyscf.ci.CISD`` on RHF); frozen orbitals are
              honoured.
            - An FCI solver after ``kernel()``, e.g. from
              ``pyscf.fci.FCI(mf)`` or ``pyscf.fci.SCI(mol)``;
              ``mo_coeff`` is required.
            - A CI vector in PySCF's FCI layout (alpha x beta
              strings, or an ``SCIvector``); ``mo_coeff`` and
              ``nelec`` are required.

        coeff_threshold: Keep determinants with ``|c| > threshold``.
        max_ndet: Keep at most this many determinants (the largest
            ``|c|``), after the threshold.  None keeps all.
        state: Root to use when ``obj`` holds several CI vectors
            (``nroots > 1`` or state-averaged CASSCF).  Required in
            that case; None or 0 for a single vector.
        mo_coeff: MO coefficients, shape (nao, nmo), that the FCI
            solver or bare CI vector refers to, e.g. ``mf.mo_coeff``.
            Only for FCI solvers and bare CI vectors.
        nelec: Active (alpha, beta) electrons, or their total, for a
            bare CI vector.  FCI solvers report their own.
        ncore: Doubly occupied MOs below the orbitals the FCI solver
            or bare CI vector correlates (default 0).
        norb: Correlated orbitals of a bare CI vector; default
            ``mo_coeff.shape[1] - ncore``.

    Returns:
        dict: The trial wavefunction, with keys

        - ``ci_coeffs``: jnp.array, shape (ndet,), sorted by
          decreasing ``|c|``, not renormalised.
        - ``occ_up``: jnp.array int32, shape (ndet, nup); MO indices
          (into ``mo_coeff`` columns) of the alpha electrons.
        - ``occ_dn``: jnp.array int32, shape (ndet, ndown).
        - ``ndet``: Number of determinants retained.
        - ``mo_coeff``: np.ndarray, shape (nao, nmo).
        - ``ci_weight``: Fraction of the CI norm retained,
          ``sum_kept |c|^2 / sum_all |c|^2``.

        Determinant 0 has the largest ``|c|``: AFQMC starts its
        walkers from it and uses it as the Wick reference, and GTO
        VMC builds the MO density for SWCT ``scheme1`` from it.

    Raises:
        NotImplementedError: UHF-based objects (UCASCI, UCASSCF,
            UCISD, GCISD, ``direct_uhf`` FCI).
        TypeError: Unsupported object (e.g. coupled cluster), or
            ``mo_coeff``/``nelec``/``ncore``/``norb`` given for an
            object that defines them itself.
        ValueError: Missing inputs, no CI vector, or no determinant
            above the threshold.
    """
    from pyscf.mcscf.casci import CASBase
    from pyscf.mcscf.ucasci import UCASBase
    from pyscf.ci.cisd import CISD
    from pyscf.ci.ucisd import UCISD
    from pyscf.ci.gcisd import GCISD
    from pyscf.fci.direct_spin1 import FCIBase
    from pyscf.fci.direct_uhf import FCISolver as UHFFCISolver

    if max_ndet is not None and max_ndet < 1:
        raise ValueError(f"max_ndet must be positive, got {max_ndet}")
    if isinstance(obj, (UCASBase, UCISD, GCISD, UHFFCISolver)):
        raise NotImplementedError(f"{type(obj).__name__}: {_UHF_MSG}")
    name = type(obj).__name__
    own = isinstance(obj, (CASBase, CISD))
    if own and (mo_coeff is not None or nelec is not None
                or ncore != 0 or norb is not None):
        raise TypeError(f"{name} defines its own orbitals and electron "
                        "count; do not pass mo_coeff/nelec/ncore/norb")

    if isinstance(obj, CASBase):
        ci = _select_state(obj.ci, state, name)
        nelec = obj.nelecas
        args = (obj.ncas, (int(nelec[0]), int(nelec[1])), obj.ncore)
        mo_coeff = obj.mo_coeff
        dets = _dets_from_fcivec(ci, *args, coeff_threshold)
    elif isinstance(obj, CISD):
        ci = _select_state(obj.ci, state, name)
        mo_coeff = obj.mo_coeff
        nocc_full = int(np.count_nonzero(np.asarray(obj.mo_occ) > 0))
        dets = _dets_from_cisd(ci, mo_coeff.shape[1], nocc_full,
                               obj.frozen, coeff_threshold)
    else:
        if isinstance(obj, FCIBase):
            ci = _select_state(getattr(obj, 'ci', None), state, name)
            if nelec is not None or norb is not None:
                raise TypeError(f"{name} defines its own norb and "
                                "nelec; do not pass them")
            norb, nelec = obj.norb, obj.nelec
        elif isinstance(obj, (np.ndarray, list, tuple)):
            ci = _select_state(obj, state, 'CI vector')
            if nelec is None:
                raise ValueError("a bare CI vector needs `nelec`")
        else:
            raise TypeError(
                f"unsupported object {name}; expected CASCI/CASSCF, "
                "RCISD, an FCI solver or a CI vector")
        if mo_coeff is None:
            raise ValueError(
                f"{name} does not store its orbitals; pass the MO "
                "coefficients it was run in as `mo_coeff`")
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise NotImplementedError(_UHF_MSG)
        if norb is None:
            norb = mo_coeff.shape[1] - ncore
        if np.ndim(nelec) == 0:
            nelec = (int(nelec) - int(nelec) // 2, int(nelec) // 2)
        nelec = (int(nelec[0]), int(nelec[1]))
        if ncore + norb > mo_coeff.shape[1]:
            raise ValueError(
                f"ncore + norb = {ncore + norb} exceeds the "
                f"{mo_coeff.shape[1]} MOs in mo_coeff")
        dets = _dets_from_fcivec(ci, norb, nelec, ncore,
                                 coeff_threshold)

    return _build_trial(*dets[:3], mo_coeff, max_ndet, dets[3])
