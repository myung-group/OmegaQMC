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
    symmetry_operations_map,
)

# Union of all canonical ops across supported point
# groups.  Used to distinguish "invalid symbol" from
# "valid symbol, but outside this fragment's PG".
_ALL_PG_OPS = frozenset(
    op for ops in POINT_GROUP_OPS.values() for op in ops
)


def _op_matrix_local(op):
    """3x3 lab-action matrix of ``op`` in the local frame.

    ``symmetry_operations_map[op]`` transforms each row of its
    input, so applying it to the identity returns the images
    of the basis vectors as rows, i.e. the transpose of the
    matrix ``M`` with ``op(v) = M @ v``.
    """
    return np.asarray(
        symmetry_operations_map[op](jnp.eye(3))
    ).T


def _selfmap_error(Xc, M, Z):
    """Max charge-matched nuclear displacement under op ``M``.

    ``Xc`` are centroid-centred coordinates; ``M`` is a 3x3
    lab-frame operation matrix.  Returns the largest distance
    from each transformed nucleus to the nearest original
    nucleus of equal charge — zero iff ``M`` is an exact
    self-symmetry of the fragment.
    """
    Xr = Xc @ M.T
    err = 0.0
    for i in range(len(Xc)):
        same = [j for j in range(len(Xc)) if Z[j] == Z[i]]
        err = max(
            err,
            min(np.linalg.norm(Xr[i] - Xc[j]) for j in same),
        )
    return err


def _geometric_symmetry_frame(frag_nuc, charges, centroid,
                              frag_ops):
    """Local frame aligned to a fragment's symmetry elements.

    Generalises the C2 fix to every supported point group
    (C2v / C2h / Cs / D2h / C4v / D4h and the linear
    Coov/Dooh mappings).  The frame is found purely from the
    geometry — independent of PySCF ``detect_symm`` — so the
    rotation / reflection / improper axes vary *continuously*
    with distortion and never flip to the wrong axis when a
    fragment falls below the exact point-group tolerance.

    The principal axis (local-z) and the in-plane orientation
    are chosen together to minimise the total residual of
    *every* requested operation applied in the frame.  Because
    a C_n / S_n rotation self-maps the nuclei only about its
    true axis, this lands local-z on that axis (the long axis
    for a linear fragment, the plane normal for a square-planar
    one) and aligns local-x / y with the sigma_v / C2' elements
    that the reflections and secondary rotations require.

    Rows of the returned matrix are the local axes in the lab
    frame, matching the ``centered @ Vh.T`` convention of the
    transform kernels.
    """
    Xc = np.asarray(frag_nuc, dtype=float) - np.asarray(
        centroid, dtype=float,
    )
    Z = np.asarray(charges)
    ops = [
        o for o in frag_ops
        if o in symmetry_operations_map and o not in ('E', 'i')
    ]
    if not ops:
        return jnp.eye(3)
    mats = {o: _op_matrix_local(o) for o in ops}

    # Charge-weighted principal directions are the candidate
    # symmetry axes (true for any symmetric top).
    _, _, Vt = np.linalg.svd(
        np.sqrt(Z.astype(float))[:, None] * Xc,
        full_matrices=True,
    )

    def total_error(V):
        return sum(
            _selfmap_error(Xc, V.T @ mats[o] @ V, Z)
            for o in ops
        )

    def inplane_basis(z):
        seed = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(seed, z)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        e1 = seed - np.dot(seed, z) * z
        e1 = e1 / np.linalg.norm(e1)
        return e1, np.cross(z, e1)

    best_V, best_err = jnp.eye(3), np.inf
    for zc in Vt:
        z = zc / np.linalg.norm(zc)
        e1, e2 = inplane_basis(z)
        # Coarse 1-degree sweep, then refine the in-plane angle
        # that aligns x with a sigma_v / C2' element.
        thetas = np.linspace(0.0, np.pi, 181)
        for _ in range(2):
            errs = [
                total_error(
                    np.stack([
                        np.cos(t) * e1 + np.sin(t) * e2,
                        np.cross(
                            z, np.cos(t) * e1 + np.sin(t) * e2,
                        ),
                        z,
                    ])
                )
                for t in thetas
            ]
            k = int(np.argmin(errs))
            t0, dt = thetas[k], thetas[1] - thetas[0]
            if errs[k] < best_err:
                x = np.cos(t0) * e1 + np.sin(t0) * e2
                best_V = np.stack([x, np.cross(z, x), z])
                best_err = errs[k]
            thetas = np.linspace(t0 - dt, t0 + dt, 41)
    return jnp.asarray(best_V)


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

            # Build the operation frame geometrically for any
            # multi-atom fragment that applies symmetry ops.
            # ``_geometric_symmetry_frame`` aligns local-z with
            # the true C_n / S_n axis and local-x / y with the
            # sigma_v / C2' elements, continuously in the
            # distortion.  This replaces the ``detect_symm``
            # axes (``map_frag_axes``) and the SVD
            # principal-moments frame, which both put the
            # molecular-plane normal along local-z once a
            # fragment drops below its exact point-group
            # tolerance — silently rotating about the wrong
            # axis (see ``operations.populate_fragment_symmops``).
            # Diatomics / single atoms (not ``is_planar``) are
            # collinear, so their detected axes are robust and
            # are kept as-is.
            frag_Vh = None
            if frag_axes_map is not None and fid in frag_axes_map:
                frag_Vh = jnp.asarray(frag_axes_map[fid])

            if is_planar:
                frag_nuc = nuc_crds[
                    jnp.array(frag_atom_indices)
                ]
                charges = mol.atom_charges()[
                    np.asarray(frag_atom_indices)
                ]
                Vh_list.append(
                    _geometric_symmetry_frame(
                        frag_nuc, charges, centroid, frag_ops,
                    )
                )
            elif frag_Vh is not None:
                Vh_list.append(frag_Vh)
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
