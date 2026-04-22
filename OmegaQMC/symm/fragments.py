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

import jax
import jax.numpy as jnp

from .operations import POINT_GROUP_OP_ALIASES


def build_frag_reflect_data(mol, nuc_crds):
    """Precompute per-fragment data for reflections/rotations.

    Parameters
    ----------
    mol : Mole_custom
        Must have ``map_frag_ctr``, ``map_nuc_frag``,
        ``inradii`` populated; ``map_frag_symmops`` is
        optional.
    nuc_crds : jnp.ndarray
        Nuclear coordinates in Bohr, shape ``(natm, 3)``.

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

        centroids_list = []
        inradii_list = []
        Vh_list = []
        is_planar_list = []

        for fid in frag_ids:
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

            if is_planar:
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
        for fid in frag_ids:
            allowed = set(
                mol.map_frag_symmops.get(fid, ['E'])
                if has_map else ['E']
            )
            requested = set(normalized)
            rejected = requested - allowed - {'E'}
            if rejected:
                rejected_raw = {
                    raw for raw, norm
                    in zip(symmop_list, normalized)
                    if norm in rejected
                }
                warnings.warn(
                    f"Fragment {fid}: operations "
                    f"{rejected_raw} are not valid "
                    "symmetry operations and will be "
                    "removed",
                    stacklevel=2,
                )
            frag_symmops[fid] = sorted(requested & allowed)
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
                rejected = requested - allowed - {'E'}
                if rejected:
                    rejected_raw = {
                        r for r, n in zip(raw, normalized)
                        if n in rejected
                    }
                    warnings.warn(
                        f"Fragment {fid}: operations "
                        f"{rejected_raw} are not valid "
                        "symmetry operations and will be "
                        "removed",
                        stacklevel=2,
                    )
                frag_symmops[fid] = sorted(
                    requested & allowed,
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
