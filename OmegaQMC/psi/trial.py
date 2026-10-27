"""Multi-determinant trial wavefunctions from PySCF solvers.

:func:`trial_from_pyscf` turns the result of a PySCF correlated
calculation into the ``trial`` dict that the GTO VMC, GTO VMC
optimisation and AFQMC drivers accept.  Supported sources:

- CASCI / CASSCF on RHF or ROHF orbitals, and UCASCI / UCASSCF on UHF
  orbitals, including state-averaged and multi-root runs (pick a root
  with ``state``), with the default FCI solver or a selected-CI
  ``fcisolver``;
- FCI solvers (``pyscf.fci.FCI``, ``direct_spin0``/``direct_spin1``,
  ``direct_uhf`` and their symmetry variants, ``selected_ci.SCI``),
  given the MO coefficients the solver was run in;
- RCISD and UCISD (``pyscf.ci.CISD`` on RHF, ROHF or UHF), with
  frozen orbitals;
- a bare CI vector in PySCF's FCI layout (alpha strings x beta
  strings, or an ``SCIvector``), given the MO coefficients and the
  active electron count.

GCISD (generalized spin orbitals) and coupled-cluster wavefunctions,
which are not finite determinant expansions, raise an error.

Determinant convention: determinant ``I`` is the product of an alpha
and a beta Slater determinant whose columns are the alpha MOs
``occ_up[I]`` and the beta MOs ``occ_dn[I]`` in the listed order,
with alpha electrons before beta electrons.  Restricted trials share
one MO set (``mo_coeff`` of shape (nao, nmo)); UHF-based trials have
``mo_coeff`` of shape (2, nao, nmo) and index the alpha and beta MOs
separately.  For the sorted occupation lists of PySCF's CI strings
this is PySCF's own sign convention, so CI coefficients carry over
unchanged.  CISD excitations are listed in place (each hole replaced
by its particle in the reference ordering), which absorbs the
excitation sign.
"""

import numpy as np
import jax.numpy as jnp


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


def _spin_pair(x):
    """(alpha, beta) from a per-spin pair or a value shared by both."""
    if np.ndim(x) == 0:
        return int(x), int(x)
    return int(x[0]), int(x[1])


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
        ncore: (alpha, beta) occupied orbitals below the active space.
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
    nca, ncb = ncore
    core_a = np.broadcast_to(np.arange(nca), (len(ia), nca))
    core_b = np.broadcast_to(np.arange(ncb), (len(ib), ncb))
    occ_up = np.hstack([core_a, nca + occa[ia]])
    occ_dn = np.hstack([core_b, ncb + occb[ib]])
    return ci[ia, ib], occ_up, occ_dn, np.sum(np.abs(ci) ** 2)


def _frozen_masks(frozen, nmo_full, unrestricted):
    """Frozen-orbital masks (alpha, beta) from PySCF's ``frozen``."""
    masks = np.zeros((2, nmo_full), dtype=bool)
    if frozen is None:
        return masks
    if isinstance(frozen, (int, np.integer)):
        masks[:, :frozen] = True
        return masks
    frozen = list(frozen)
    per_spin = (unrestricted and len(frozen) == 2
                and all(np.ndim(f) == 1 for f in frozen))
    for s in range(2):
        idx = frozen[s] if per_spin else frozen
        masks[s, np.asarray(idx, dtype=int)] = True
    return masks


def _dets_from_cisd(c0, c1, c2ab, c2ss, nocc_full, masks, threshold):
    """Determinants of a CISD vector, built excitation by excitation.

    The cost scales with the number of CISD amplitudes, not with the
    FCI space, so there is no limit on the number of orbitals.  The
    amplitudes follow PySCF's UCISD layout (RCISD maps onto it).

    Args:
        c0: Reference coefficient.
        c1: (c1a, c1b), shapes (nocc_s, nvir_s).
        c2ab: Opposite-spin doubles, alpha i -> a and beta j -> b,
            shape (nocca, noccb, nvira, nvirb).
        c2ss: (c2aa, c2bb), antisymmetric same-spin doubles.
        nocc_full: (alpha, beta) occupied MOs including frozen ones.
        masks: Frozen-orbital masks, shape (2, nmo_full).
        threshold: Keep determinants with ``|c| > threshold``.

    Returns:
        (coeffs, occ_up, occ_dn, total_weight).
    """
    occ, vir, ref = [], [], []
    for s in range(2):
        n = nocc_full[s]
        occ.append(np.nonzero(~masks[s, :n])[0])
        vir.append(n + np.nonzero(~masks[s, n:])[0])
        ref.append(np.arange(n))

    def tile(s, n):
        return np.tile(ref[s], (n, 1))

    def singles(s, ii, aa):
        # Reference rows with hole occ[i] replaced in place by vir[a]
        rows = tile(s, len(ii))
        rows[np.arange(len(ii)), occ[s][ii]] = vir[s][aa]
        return rows

    def doubles(s, ii, jj, aa, bb):
        # i > j and a > b: occ[j] -> vir[b] and occ[i] -> vir[a], so
        # the in-place determinant has the sign of PySCF's c2aa
        rows = singles(s, ii, aa)
        rows[np.arange(len(ii)), occ[s][jj]] = vir[s][bb]
        return rows

    coeffs = [np.atleast_1d(c0)]
    ups, dns = [ref[0][None]], [ref[1][None]]
    w = [abs(c0) ** 2]

    for s in range(2):
        i, a = np.nonzero(np.abs(c1[s]) > threshold)
        exc, other = singles(s, i, a), tile(1 - s, len(i))
        coeffs.append(c1[s][i, a])
        ups.append(exc if s == 0 else other)
        dns.append(other if s == 0 else exc)
        w.append(np.sum(np.abs(c1[s]) ** 2))

    i, j, a, b = np.nonzero(np.abs(c2ab) > threshold)
    coeffs.append(c2ab[i, j, a, b])
    ups.append(singles(0, i, a))
    dns.append(singles(1, j, b))
    w.append(np.sum(np.abs(c2ab) ** 2))

    for s in range(2):
        c2 = c2ss[s]
        no, nv = c2.shape[0], c2.shape[2]
        pair = (np.arange(no)[:, None] > np.arange(no))[:, :, None, None]
        pair = pair & (np.arange(nv)[:, None] > np.arange(nv))
        i, j, a, b = np.nonzero(pair & (np.abs(c2) > threshold))
        exc, other = doubles(s, i, j, a, b), tile(1 - s, len(i))
        coeffs.append(c2[i, j, a, b])
        ups.append(exc if s == 0 else other)
        dns.append(other if s == 0 else exc)
        w.append(np.sum(np.abs(c2[np.broadcast_to(pair, c2.shape)]) ** 2))

    coeffs = np.concatenate(coeffs)
    keep = np.ones(len(coeffs), dtype=bool)
    keep[0] = abs(c0) > threshold
    return (coeffs[keep], np.concatenate(ups)[keep],
            np.concatenate(dns)[keep], float(np.sum(w)))


def _cisd_dets(obj, civec, unrestricted, threshold):
    """Determinants of an RCISD or UCISD object's CI vector."""
    from pyscf.ci import cisd, ucisd
    mo_occ = np.asarray(obj.mo_occ)
    nmo_full = np.asarray(obj.mo_coeff).shape[-1]
    masks = _frozen_masks(obj.frozen, nmo_full, unrestricted)
    if unrestricted:
        nocc_full = tuple(int(np.count_nonzero(mo_occ[s] > 0))
                          for s in range(2))
    else:
        nocc_full = (int(np.count_nonzero(mo_occ > 0)),) * 2
    nocc = tuple(int(np.count_nonzero(~masks[s, :nocc_full[s]]))
                 for s in range(2))
    nmo = tuple(int(np.count_nonzero(~masks[s])) for s in range(2))
    civec = np.asarray(civec)
    if unrestricted:
        c0, c1, (c2aa, c2ab, c2bb) = ucisd.cisdvec_to_amplitudes(
            civec, nmo, nocc, copy=False)
    else:
        c0, c1r, c2r = cisd.cisdvec_to_amplitudes(
            civec, nmo[0], nocc[0], copy=False)
        c1, c2ab = (c1r, c1r), c2r
        c2aa = c2bb = c2r - c2r.transpose(1, 0, 2, 3)
    return _dets_from_cisd(c0, c1, c2ab, (c2aa, c2bb), nocc_full,
                           masks, threshold)


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

            - CASCI / CASSCF (RHF/ROHF orbitals) or UCASCI / UCASSCF
              (UHF orbitals); orbitals, core and active space are
              read from it.
            - RCISD or UCISD (``pyscf.ci.CISD`` on RHF, ROHF or UHF);
              frozen orbitals are honoured.
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
        mo_coeff: MO coefficients that the FCI solver or bare CI
            vector refers to, e.g. ``mf.mo_coeff``: shape (nao, nmo),
            or (alpha, beta) of shape (2, nao, nmo) for UHF orbitals.
            Only for FCI solvers and bare CI vectors.
        nelec: Active (alpha, beta) electrons, or their total, for a
            bare CI vector.  FCI solvers report their own.
        ncore: Occupied MOs below the orbitals the FCI solver or bare
            CI vector correlates (default 0); an (alpha, beta) pair
            for different cores.
        norb: Correlated orbitals of a bare CI vector; default
            ``nmo - max(ncore)``.

    Returns:
        dict: The trial wavefunction, with keys

        - ``ci_coeffs``: jnp.array, shape (ndet,), sorted by
          decreasing ``|c|``, not renormalised.
        - ``occ_up``: jnp.array int32, shape (ndet, nup); alpha MO
          indices (``mo_coeff`` columns) of the alpha electrons.
        - ``occ_dn``: jnp.array int32, shape (ndet, ndown); beta MO
          indices of the beta electrons.
        - ``ndet``: Number of determinants retained.
        - ``mo_coeff``: np.ndarray, shape (nao, nmo), or
          (2, nao, nmo) with the alpha and beta MOs of a UHF-based
          trial.
        - ``ci_weight``: Fraction of the CI norm retained,
          ``sum_kept |c|^2 / sum_all |c|^2``.

        Determinant 0 has the largest ``|c|``: AFQMC starts its
        walkers from it and uses it as the Wick reference, and GTO
        VMC builds the MO density for SWCT ``scheme1`` from it.

    Raises:
        NotImplementedError: GCISD (generalized spin orbitals).
        TypeError: Unsupported object (e.g. coupled cluster), or
            ``mo_coeff``/``nelec``/``ncore``/``norb`` given for an
            object that defines them itself.
        ValueError: Missing inputs, no CI vector, or no determinant
            above the threshold.
    """
    from pyscf.mcscf.casci import CASBase
    from pyscf.ci.cisd import CISD
    from pyscf.ci.ucisd import UCISD
    from pyscf.ci.gcisd import GCISD
    from pyscf.fci.direct_spin1 import FCIBase

    if max_ndet is not None and max_ndet < 1:
        raise ValueError(f"max_ndet must be positive, got {max_ndet}")
    if isinstance(obj, GCISD):
        raise NotImplementedError(
            "GCISD: generalized (spin-mixing) orbitals are not supported")
    name = type(obj).__name__
    own = isinstance(obj, (CASBase, CISD))
    if own and (mo_coeff is not None or nelec is not None
                or ncore != 0 or norb is not None):
        raise TypeError(f"{name} defines its own orbitals and electron "
                        "count; do not pass mo_coeff/nelec/ncore/norb")

    if isinstance(obj, CASBase):
        ci = _select_state(obj.ci, state, name)
        mo_coeff = np.asarray(obj.mo_coeff)
        dets = _dets_from_fcivec(ci, obj.ncas, _spin_pair(obj.nelecas),
                                 _spin_pair(obj.ncore), coeff_threshold)
    elif isinstance(obj, CISD):
        ci = _select_state(obj.ci, state, name)
        mo_coeff = np.asarray(obj.mo_coeff)
        dets = _cisd_dets(obj, ci, isinstance(obj, UCISD),
                          coeff_threshold)
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
                "CISD, an FCI solver or a CI vector")
        if mo_coeff is None:
            raise ValueError(
                f"{name} does not store its orbitals; pass the MO "
                "coefficients it was run in as `mo_coeff`")
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim not in (2, 3) or (mo_coeff.ndim == 3
                                           and len(mo_coeff) != 2):
            raise ValueError(
                "mo_coeff must have shape (nao, nmo) or (2, nao, nmo), "
                f"got {mo_coeff.shape}")
        ncore = _spin_pair(ncore)
        nmo = mo_coeff.shape[-1]
        if norb is None:
            norb = nmo - max(ncore)
        if np.ndim(nelec) == 0:
            nelec = (int(nelec) - int(nelec) // 2, int(nelec) // 2)
        nelec = _spin_pair(nelec)
        if max(ncore) + norb > nmo:
            raise ValueError(
                f"ncore + norb = {max(ncore) + norb} exceeds the "
                f"{nmo} MOs in mo_coeff")
        dets = _dets_from_fcivec(ci, norb, nelec, ncore,
                                 coeff_threshold)

    return _build_trial(*dets[:3], mo_coeff, max_ndet, dets[3])
