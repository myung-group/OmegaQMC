"""Backend-agnostic fragment helpers for PGCS.

These helpers are shared by :class:`OmegaQMC.vmc_gto._VMCDriverGTO`
and :class:`OmegaQMC.vmc_nn._VMCDriverNN` so that
Point Group Correlated Sampling uses a single source
of truth.  None of the helpers depend on a
mean-field / SCF object — they only require a
:class:`OmegaQMC.utils.Mole_custom` that has been
through :func:`~OmegaQMC.utils.parse_molecular_inspheres`
and
:func:`~OmegaQMC.symm.operations.populate_fragment_symmops`.
"""

import warnings

import jax.numpy as jnp
import numpy as np

from .operations import (
    POINT_GROUP_OP_ALIASES,
    POINT_GROUP_OPS,
)

# Union of all canonical ops across supported point
# groups.  Used to distinguish "invalid symbol" from
# "valid symbol, but outside this fragment's PG".
_ALL_PG_OPS = frozenset(
    op for ops in POINT_GROUP_OPS.values() for op in ops
)

# Operations that, if requested, mean the fragment has a
# principal axis higher than C2 or more than one C2 axis
# (C4v / D4h / D2h).  For those the single-C2 geometric
# frame below is not sufficient, so we leave the existing
# ``detect_symm`` / SVD frame in place.
_MULTI_AXIS_OPS = frozenset(
    {'Rz90', 'Rz270', 'Rx180', 'Ry180'}
)


def _geometric_c2_frame(frag_nuc, charges, centroid):
    """Local frame whose z-axis is the fragment's C2 axis.

    The axis is derived directly from the geometry so that it
    varies *continuously* with distortion.  This is the fix
    for the ``detect_symm``-based frame, which switches local-z
    discontinuously from the true C2 axis to the molecular-plane
    normal the instant a fragment falls below PySCF's exact
    point-group tolerance — turning ``Rz180`` into a rotation
    about the wrong axis (a ~1 A electron displacement instead
    of a self-mapping).

    Local-z is the charge-weighted principal direction whose
    180-degree rotation best maps the nuclei onto themselves
    (the true C2; the plane normal and the in-plane
    perpendicular both fail this test).  Local-x is the
    molecular-plane normal and local-y completes a right-handed
    frame.  Rows of the returned matrix are the local axes in
    the lab frame, matching the ``centered @ Vh.T`` convention
    of the reflection kernels.
    """
    X = np.asarray(frag_nuc, dtype=float) - np.asarray(
        centroid, dtype=float,
    )
    Z = np.asarray(charges, dtype=float)
    _, _, Vt = np.linalg.svd(
        np.sqrt(Z)[:, None] * X, full_matrices=True,
    )
    eye3 = np.eye(3)

    best_u, best_err = Vt[0], np.inf
    for u in Vt:
        # Householder-style 180-degree rotation about ``u``.
        R = 2.0 * np.outer(u, u) - eye3
        Xr = X @ R.T
        err = 0.0
        for i in range(len(X)):
            same = [j for j in range(len(X)) if Z[j] == Z[i]]
            err = max(
                err,
                min(np.linalg.norm(Xr[i] - X[j]) for j in same),
            )
        if err < best_err:
            best_err, best_u = err, u
    z = best_u / np.linalg.norm(best_u)

    # Plane normal = least-spread principal direction, made
    # orthogonal to z (defines the C2v sigma_v frame).
    pn = Vt[2] - np.dot(Vt[2], z) * z
    if np.linalg.norm(pn) < 1e-8:
        pn = np.cross(z, Vt[0])
    x = pn / np.linalg.norm(pn)
    y = np.cross(z, x)
    Vh = np.stack([x, y, z])
    if np.linalg.det(Vh) < 0:
        Vh[0] = -Vh[0]
    return jnp.asarray(Vh)


def build_frag_transform_data(mol, nuc_crds, frag_symmops=None):
    """Precompute per-fragment data for symmetry transforms.

    Covers rotations (e.g. ``Rz180``), reflections, and
    improper operations — the per-fragment frame returned
    here is applied as a basis change for any of them.

    Parameters
    ----------
    mol : Mole_custom
        Must have ``map_frag_ctr``, ``map_nuc_frag``,
        ``inradii`` populated; ``map_frag_symmops`` is
        optional.
    nuc_crds : jnp.ndarray
        Nuclear coordinates in Bohr, shape ``(natm, 3)``.
    frag_symmops : dict, optional
        Maps fragment id to the list of operations actually
        requested for this run.  When given, the rotation
        frame is built to match these ops rather than the
        detected point-group ops (which may differ for a
        distorted fragment — e.g. a slightly asymmetric
        water is detected as Cs yet still has ``Rz180``
        applied).

    Returns
    -------
    tuple
        ``(frag_centroids, frag_inradii, frag_Vh,
        frag_is_planar)`` — always a 4-tuple of JAX
        arrays with leading dimension equal to the
        number of fragments.
    """
    if hasattr(mol, 'map_frag_symmops'):
        frag_ids = sorted(mol.map_frag_ctr.keys())
        frag_axes_map = getattr(mol, 'map_frag_axes', None)

        centroids_list = []
        inradii_list = []
        Vh_list = []
        is_planar_list = []

        for fid in frag_ids:
            # Prefer the operations actually requested for
            # this run (``frag_symmops``); fall back to the
            # detected point-group ops.  The frame must match
            # what is applied during sampling — e.g. a
            # distorted water is detected as Cs (no ``Rz180``)
            # yet ``Rz180`` is still the requested operation.
            if frag_symmops is not None:
                frag_ops = list(frag_symmops.get(fid, ['E']))
            else:
                frag_ops = mol.map_frag_symmops.get(fid, ['E'])
            frag_atom_indices = [
                i for i, f in enumerate(mol.map_nuc_frag)
                if f == fid
            ]
            is_planar = (
                frag_ops != ['E']
                and len(frag_atom_indices) >= 3
            )

            centroid = jnp.array(mol.map_frag_ctr[fid])
            centroids_list.append(centroid)
            inradii_list.append(mol.inradii[fid])

            # Prefer PySCF's standard-orientation axes
            # (populate_fragment_symmops stashes them on
            # ``mol.map_frag_axes``) so operations like
            # ``Rz180`` rotate about the fragment's actual
            # principal C_n axis.  SVD's principal-moments
            # frame is only a fallback — for C2v / C2h /
            # D2h fragments it puts the plane normal
            # along local-z and turns ``Rz180`` into an
            # axis-misaligned rotation that is *not* a
            # self-symmetry of the fragment.
            # For a single-C2 (C2v / C2h / Cs) fragment that
            # requests ``Rz180``, derive the C2 axis straight
            # from the geometry.  This is correct for *any*
            # distortion, unlike the ``detect_symm`` axes
            # (``map_frag_axes``) and the SVD principal-moments
            # frame, both of which place the molecular-plane
            # normal — not the C2 axis — along local-z once the
            # fragment is no longer exactly C2v.
            wants_c2_frame = (
                is_planar
                and 'Rz180' in frag_ops
                and not (_MULTI_AXIS_OPS & set(frag_ops))
            )

            frag_Vh = None
            if frag_axes_map is not None and fid in frag_axes_map:
                frag_Vh = jnp.asarray(frag_axes_map[fid])

            if wants_c2_frame:
                frag_nuc = nuc_crds[
                    jnp.array(frag_atom_indices)
                ]
                charges = mol.atom_charges()[
                    np.asarray(frag_atom_indices)
                ]
                Vh_list.append(
                    _geometric_c2_frame(
                        frag_nuc, charges, centroid,
                    )
                )
            elif frag_Vh is not None:
                Vh_list.append(frag_Vh)
            elif is_planar:
                frag_nuc = nuc_crds[
                    jnp.array(frag_atom_indices)
                ]
                centered = frag_nuc - centroid
                _, _, Vh = jnp.linalg.svd(
                    centered, full_matrices=True,
                )
                Vh_list.append(Vh)
            else:
                Vh_list.append(jnp.eye(3))
            is_planar_list.append(is_planar)
    else:
        centroids_list = [jnp.mean(nuc_crds, axis=0)]
        inradii_list = [jnp.inf]
        Vh_list = [jnp.eye(3)]
        is_planar_list = [False]

    return (
        jnp.stack(centroids_list),
        jnp.array(inradii_list),
        jnp.stack(Vh_list),
        jnp.array(is_planar_list, dtype=jnp.bool_),
    )


def build_frag_symmops(mol, symmop_list, frag_ids) -> dict:
    """Process *symmop_list* (``None`` / ``"auto"`` /
    ``list`` / ``dict``) into a per-fragment dict."""
    if symmop_list is None:
        return {fid: ['E'] for fid in frag_ids}

    has_map = (
        hasattr(mol, 'map_frag_symmops')
        and mol.map_frag_symmops
    )

    if symmop_list == "auto":
        if has_map:
            frag_symmops = {
                fid: list(
                    mol.map_frag_symmops.get(fid, ['E'])
                )
                for fid in frag_ids
            }
        else:
            frag_symmops = {fid: ['E'] for fid in frag_ids}
        if mol.verbose >= 2:
            print(
                f"Auto-derived symmetry operations: "
                f"{frag_symmops}"
            )
        return frag_symmops

    if isinstance(symmop_list, list):
        frag_symmops = {}
        # Normalize user-supplied aliases to the
        # canonical ``POINT_GROUP_OPS`` spelling so
        # equivalent synonyms (e.g. 'sigma_x' vs 'sx',
        # '-1' vs 'i') compare correctly below.
        normalized = []
        for op in symmop_list:
            canon = POINT_GROUP_OP_ALIASES.get(op, op)
            if canon != op:
                warnings.warn(
                    'Normalizing point group operation symbol '
                    f'"{op}" → "{canon}"',
                    stacklevel=2,
                )
            normalized.append(canon)
        symm_level = getattr(
            mol, 'symmetrization_level', 1,
        )
        for fid in frag_ids:
            allowed = set(
                mol.map_frag_symmops.get(fid, ['E'])
                if has_map else ['E']
            )
            requested = set(normalized)
            elements_pg = requested & allowed
            nonelements_pg = (
                (requested - allowed) & _ALL_PG_OPS
            )
            rejected = requested - _ALL_PG_OPS
            if rejected:
                rejected_raw = {
                    raw for raw, norm
                    in zip(symmop_list, normalized)
                    if norm in rejected
                }
                warnings.warn(
                    f"Fragment {fid}: operations {rejected_raw} are not valid "
                    "symmetry operations and will be removed",
                    stacklevel=2,
                )
            if nonelements_pg:
                if symm_level < 2:
                    print(
                        f"ℹ️\tFragment {fid}: "
                        "including operations outside its point group "
                        f"{sorted(nonelements_pg)} (symmetrization_level < 2)"
                    )
                    frag_symmops[fid] = sorted(
                        elements_pg | nonelements_pg,
                    )
                else:
                    warnings.warn(
                        f"Fragment {fid}: operations {sorted(nonelements_pg)} "
                        "are not in the fragment's point group "
                        "and will be dropped (symmetrization_level >= 2)",
                        stacklevel=2,
                    )
                    frag_symmops[fid] = sorted(
                        elements_pg,
                    )
            else:
                frag_symmops[fid] = sorted(elements_pg)
            if 'E' not in frag_symmops[fid]:
                frag_symmops[fid].insert(0, 'E')
        if mol.verbose >= 2:
            print(
                f"Symmetry operations filtered from "
                f"input: {frag_symmops}"
            )
        return frag_symmops

    if isinstance(symmop_list, dict):
        frag_symmops = {}
        symm_level = getattr(
            mol, 'symmetrization_level', 1,
        )
        for fid in frag_ids:
            if fid in symmop_list:
                allowed = set(
                    mol.map_frag_symmops.get(fid, ['E'])
                    if has_map else ['E']
                )
                raw = list(symmop_list[fid])
                normalized = []
                for op in raw:
                    canon = POINT_GROUP_OP_ALIASES.get(
                        op, op,
                    )
                    if canon != op:
                        warnings.warn(
                            'Normalizing point group operation symbol '
                            f'"{op}" → "{canon}"',
                            stacklevel=2,
                        )
                    normalized.append(canon)
                requested = set(normalized)
                elements_pg = requested & allowed
                nonelements_pg = (
                    (requested - allowed) & _ALL_PG_OPS
                )
                rejected = requested - _ALL_PG_OPS
                if rejected:
                    rejected_raw = {
                        r for r, n in zip(raw, normalized)
                        if n in rejected
                    }
                    warnings.warn(
                        f"Fragment {fid}: operations {rejected_raw} "
                        "are not valid symmetry operations "
                        "and will be removed", stacklevel=2,
                    )
                if nonelements_pg:
                    if symm_level < 2:
                        print(
                            f"ℹ️\tFragment {fid}: including operations "
                            "outside its point group {sorted(nonelements_pg)} "
                            "(symmetrization_level < 2)"
                        )
                        frag_symmops[fid] = sorted(
                            elements_pg | nonelements_pg,
                        )
                    else:
                        warnings.warn(
                            f"Fragment {fid}: "
                            f"operations {sorted(nonelements_pg)} "
                            "are not in the fragment's "
                            "point group and will be dropped "
                            "(symmetrization_level >= 2)", stacklevel=2,
                        )
                        frag_symmops[fid] = sorted(
                            elements_pg,
                        )
                else:
                    frag_symmops[fid] = sorted(
                        elements_pg,
                    )
            else:
                frag_symmops[fid] = ['E']
            if 'E' not in frag_symmops[fid]:
                frag_symmops[fid].insert(0, 'E')
        if mol.verbose >= 2:
            print(
                f"Input-specified symmetry operations: "
                f"{frag_symmops}"
            )
        return frag_symmops

    raise TypeError(
        f"symmop_list must be None, \"auto\", "
        f"list[str], or dict[int, list[str]], got "
        f"{type(symmop_list)}"
    )


def build_single_frag_combos(frag_ids, frag_symmops) -> list:
    """Enumerate ``(frag_pos, op, label)`` tuples for
    single-fragment correlated sampling."""
    single_frag_combos = []
    for frag_pos, fid in enumerate(frag_ids):
        for op in frag_symmops[fid]:
            if op == 'E':
                continue
            parts = [
                f"{fid2}:{op if fid2 == fid else 'E'}"
                for fid2 in frag_ids
            ]
            label = ",".join(parts)
            single_frag_combos.append(
                (frag_pos, op, label),
            )
    return single_frag_combos


def make_apply_single_frag_symmop(
    frag_centroids, frag_Vh, frag_inradii,
):
    """Return a JAX callable that applies a symmetry
    operation to a single fragment.

    The returned function has signature
    ``(batch_samples, frag_pos, s_op_fn) -> batch``
    and only transforms electrons whose distance from
    the fragment centroid is within the fragment's
    in-radius.  Both ``_VMCDriverGTO`` and
    ``_VMCDriverNN`` use this factory so the JAX
    closure is compiled once per driver from the same
    source.
    """
    def _apply_single_frag_symmop(
        batch_samples, frag_pos, s_op_fn,
    ):
        centroid = frag_centroids[frag_pos]
        Vh = frag_Vh[frag_pos]
        inradius = frag_inradii[frag_pos]

        centered = batch_samples - centroid
        rotated = centered @ Vh.T
        operated = s_op_fn(rotated)
        proposed = operated @ Vh + centroid

        dist = jnp.linalg.norm(centered, axis=-1)
        mask = dist <= inradius
        return jnp.where(
            mask[:, :, None], proposed, batch_samples,
        )

    return _apply_single_frag_symmop
