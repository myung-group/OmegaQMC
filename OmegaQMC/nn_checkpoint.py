"""HDF5 checkpoint I/O for NN VMC parameters.

Save/load routines for the NNX parameter pytree
produced by
:func:`~OmegaQMC.psi.nn.adapter.make_nn_log_psi`.
Checkpoint files use the ``.chk.h5`` suffix.

HDF5 layout::

    /params/0          # first leaf array (gzip)
    /params/1          # second leaf array
    ...
    /params.attrs['num_leaves']
    /meta.attrs['config_name', 'epoch', ...]
    /meta/charges      # (natom,) float64
    /meta/coords       # (natom, 3) float64
    /vmc/              # appended by vmc_nn (optional)
        .attrs['E_mean', 'E_serr']
        E_blocks       # (num_blocks,) float64
"""

import h5py
import jax
import jax.numpy as jnp
import numpy as np


def save_nn_checkpoint(
    filepath, params, epoch, config_name,
    mol_info, energy=None,
):
    """Save NN parameters to an HDF5 checkpoint.

    Overwrites any existing file at *filepath*.

    Args:
        filepath: Path to ``.chk.h5`` file.
        params: NNX parameter pytree from
            :func:`~OmegaQMC.psi.nn.adapter\
.make_nn_log_psi`.
        epoch: Completed epoch number (int).
        config_name: NN config name (str).
        mol_info:
            :class:`~OmegaQMC.utils.Mole_custom`.
        energy: Optional energy estimate (float).
    """
    leaves = jax.tree.leaves(params)
    with h5py.File(filepath, 'w') as f:
        pg = f.create_group('params')
        for i, leaf in enumerate(leaves):
            arr = np.asarray(leaf)
            kw = {'compression': 'gzip'} if arr.ndim > 0 else {}
            pg.create_dataset(
                str(i), data=arr, **kw,
            )
        pg.attrs['num_leaves'] = len(leaves)

        mg = f.create_group('meta')
        mg.attrs['config_name'] = config_name
        mg.attrs['epoch'] = int(epoch)
        mg.attrs['n_up'] = int(mol_info.n_up)
        mg.attrs['n_down'] = int(mol_info.n_down)
        if energy is not None:
            mg.attrs['energy'] = float(energy)
        mg.create_dataset(
            'charges',
            data=np.asarray(mol_info.charges),
        )
        mg.create_dataset(
            'coords',
            data=np.asarray(mol_info.coords),
        )


def load_nn_checkpoint(filepath, template_params):
    """Load NN parameters from an HDF5 checkpoint.

    The *template_params* pytree (from a freshly
    initialised model) provides the tree structure;
    leaf values are replaced with checkpoint data.

    Args:
        filepath: Path to ``.chk.h5`` file.
        template_params: NNX parameter pytree with
            the correct tree structure.

    Returns:
        Tuple ``(params, meta)`` where *params* is
        the restored parameter pytree and *meta* is
        a dict of checkpoint metadata.
    """
    _, treedef = jax.tree.flatten(template_params)
    with h5py.File(filepath, 'r') as f:
        pg = f['params']
        n = int(pg.attrs['num_leaves'])
        leaves = [jnp.asarray(pg[str(i)][()]) for i in range(n)]
        meta = {}
        if 'meta' in f:
            mg = f['meta']
            for k in mg.attrs:
                meta[k] = mg.attrs[k]
            if 'charges' in mg:
                meta['charges'] = np.asarray(mg['charges'])
            if 'coords' in mg:
                meta['coords'] = np.asarray(mg['coords'])
    return jax.tree.unflatten(treedef, leaves), meta


def load_nn_checkpoint_partial(filepath, template_params, log_fn=print):
    """Load NN parameters with shape-mismatch tolerance.

    For each leaf in ``template_params`` (from a fresh
    initialisation of the TARGET model), copy the value
    from the corresponding leaf in *filepath* IF the
    shapes match.  Otherwise keep the template's random
    init (skip the source leaf).

    Designed for transfer learning between different N
    values: MPNN body / deep Jastrow / coord BF have
    N-independent shapes (will copy), while orbital
    coefficients have N-dependent shapes (will skip,
    keeping fresh init).

    Args:
        filepath: Path to source ``.chk.h5`` file.
        template_params: TARGET pytree with random init.
            Must have the same tree STRUCTURE as the
            source (same number of leaves in same order).
        log_fn: Callable to print/log messages.  Pass
            ``log_fn=lambda *a, **k: None`` to silence.

    Returns:
        Tuple ``(params, meta)`` — params is the merged
        pytree, meta includes per-leaf load status.
    """
    target_leaves, treedef = jax.tree.flatten(template_params)
    n_target = len(target_leaves)

    with h5py.File(filepath, 'r') as f:
        pg = f['params']
        n_source = int(pg.attrs['num_leaves'])
        meta = {}
        if 'meta' in f:
            mg = f['meta']
            for k in mg.attrs:
                meta[k] = mg.attrs[k]
            if 'charges' in mg:
                meta['charges'] = np.asarray(mg['charges'])
            if 'coords' in mg:
                meta['coords'] = np.asarray(mg['coords'])

        if n_source > n_target:
            log_fn(
                f"  [load_partial] source has {n_source} leaves, "
                f"target has only {n_target}.  Skipping all -- "
                f"target tree is missing leaves the source has."
            )
            meta['leaves_copied'] = 0
            meta['leaves_skipped'] = n_target
            return jax.tree.unflatten(treedef, target_leaves), meta

        merged_leaves = []
        n_copied = 0
        n_skipped = 0
        copied_total_params = 0
        skipped_total_params = 0
        # When target has MORE leaves than source (e.g., a new
        # submodule was added in the target), only the first
        # n_source positions can be copied; the rest must keep
        # their fresh init.  This relies on the new submodule's
        # leaves landing at the END of the target's flat order
        # (i.e., the new attribute being declared after the
        # existing ones in the target module's __init__).
        for i in range(n_target):
            tgt = target_leaves[i]
            if i >= n_source:
                merged_leaves.append(tgt)
                n_skipped += 1
                skipped_total_params += int(np.prod(tgt.shape))
                continue
            src = jnp.asarray(pg[str(i)][()])
            if src.shape == tgt.shape and src.dtype == tgt.dtype:
                merged_leaves.append(src)
                n_copied += 1
                copied_total_params += int(np.prod(src.shape))
            else:
                merged_leaves.append(tgt)
                n_skipped += 1
                skipped_total_params += int(np.prod(tgt.shape))

    log_fn(
        f"  [load_partial] copied {n_copied}/{n_target} leaves "
        f"({copied_total_params} params), skipped {n_skipped} "
        f"({skipped_total_params} params kept at fresh init)."
    )
    meta['leaves_copied'] = n_copied
    meta['leaves_skipped'] = n_skipped
    meta['params_copied'] = copied_total_params
    meta['params_skipped'] = skipped_total_params
    return jax.tree.unflatten(treedef, merged_leaves), meta


def load_nn_checkpoint_partial_by_path(
    filepath, target_template_params, source_template_params,
    log_fn=print,
):
    """Path-based partial loader for cross-architecture continuation.

    Unlike :func:`load_nn_checkpoint_partial` (which copies by flat
    INDEX and assumes target's first n_source positions match the
    source), this variant matches by KEY PATH.  Use when adding a
    submodule whose name sorts BEFORE existing modules in the
    flat order (e.g., ``coord_backflow`` < ``envelope`` < ``omni``):
    nnx flattens attributes in ATTRIBUTE-NAME-SORTED order, so a new
    early-letter attribute shifts every existing leaf's flat position.

    Caller responsibility: provide ``source_template_params`` — a
    fresh-init pytree with the SAME structure as the saved chkpt
    (typically built from the target's config with the new modules'
    flags disabled).

    Args:
        filepath: Path to source ``.chk.h5`` file.
        target_template_params: Target pytree (with the new modules)
            at fresh init.  Output preserves this structure.
        source_template_params: Pytree matching the chkpt's structure
            (i.e., target minus the new modules).  Used only to derive
            the chkpt's path for each leaf — values are ignored.
        log_fn: Logger.

    Returns:
        Tuple ``(params, meta)`` — params has source values at every
        path that exists in both source and target (with matching
        shapes); target's fresh init for paths that exist only in
        target (the new modules).
    """
    src_paths_leaves = jax.tree_util.tree_flatten_with_path(
        source_template_params,
    )[0]
    tgt_paths_leaves, tgt_treedef = (
        jax.tree_util.tree_flatten_with_path(target_template_params)
    )
    n_source_paths = len(src_paths_leaves)

    with h5py.File(filepath, 'r') as f:
        pg = f['params']
        n_source_chkpt = int(pg.attrs['num_leaves'])
        if n_source_paths != n_source_chkpt:
            raise ValueError(
                f"source_template_params has {n_source_paths} leaves "
                f"but chkpt has {n_source_chkpt}.  Source template's "
                f"structure must match the saved chkpt exactly."
            )
        src_dict = {
            path: jnp.asarray(pg[str(i)][()])
            for i, (path, _) in enumerate(src_paths_leaves)
        }
        meta = {}
        if 'meta' in f:
            mg = f['meta']
            for k in mg.attrs:
                meta[k] = mg.attrs[k]
            if 'charges' in mg:
                meta['charges'] = np.asarray(mg['charges'])
            if 'coords' in mg:
                meta['coords'] = np.asarray(mg['coords'])

    merged_leaves = []
    n_copied = 0
    n_skipped_shape = 0
    n_new = 0
    copied_total = 0
    skipped_total = 0
    new_total = 0
    for path, tgt_leaf in tgt_paths_leaves:
        if path in src_dict:
            src_leaf = src_dict[path]
            if (src_leaf.shape == tgt_leaf.shape
                    and src_leaf.dtype == tgt_leaf.dtype):
                merged_leaves.append(src_leaf)
                n_copied += 1
                copied_total += int(np.prod(src_leaf.shape))
            else:
                merged_leaves.append(tgt_leaf)
                n_skipped_shape += 1
                skipped_total += int(np.prod(tgt_leaf.shape))
        else:
            merged_leaves.append(tgt_leaf)
            n_new += 1
            new_total += int(np.prod(tgt_leaf.shape))

    log_fn(
        f"  [load_partial_by_path] copied {n_copied} leaves "
        f"({copied_total} params), {n_new} new leaves "
        f"({new_total} params at fresh init), "
        f"{n_skipped_shape} shape-mismatched."
    )
    meta['leaves_copied'] = n_copied
    meta['leaves_new'] = n_new
    meta['leaves_shape_mismatch'] = n_skipped_shape
    meta['params_copied'] = copied_total
    meta['params_new'] = new_total
    return jax.tree_util.tree_unflatten(tgt_treedef, merged_leaves), meta


def zero_module_leaves(params, module_name, log_fn=print):
    """Zero out all leaves in ``params`` whose key path contains ``module_name``.

    Used after partial transfer to reset specific submodules (e.g.,
    coord_backflow) to their natural zero-init, so the optimizer can
    re-learn them from scratch even when the rest of the network is
    transferred.

    Args:
        params: NNX parameter pytree.
        module_name: Substring matched against each leaf's key path.
        log_fn: Callable for status messages.

    Returns:
        Modified pytree with matching leaves replaced by zeros.
    """
    n_zeroed = 0

    def fn(path, leaf):
        nonlocal n_zeroed
        path_str = jax.tree_util.keystr(path)
        if module_name in path_str:
            n_zeroed += 1
            return jnp.zeros_like(leaf)
        return leaf

    out = jax.tree_util.tree_map_with_path(fn, params)
    log_fn(
        f"  [zero_module] reset {n_zeroed} leaves matching "
        f"{module_name!r} to zero."
    )
    return out


def append_vmc_results(filepath, results):
    """Append VMC results to an existing checkpoint.

    Opens the file in append mode and writes (or
    overwrites) a ``/vmc`` group with the VMC energy
    statistics.

    Args:
        filepath: Path to ``.chk.h5`` file.
        results: Dict with keys ``'E_mean'``,
            ``'E_serr'``, and optionally
            ``'E_blocks'``.
    """
    with h5py.File(filepath, 'a') as f:
        if 'vmc' in f:
            del f['vmc']
        vg = f.create_group('vmc')
        vg.attrs['E_mean'] = float(results['E_mean'])
        vg.attrs['E_serr'] = float(results['E_serr'])
        if 'E_blocks' in results:
            vg.create_dataset(
                'E_blocks',
                data=np.asarray(
                    results['E_blocks'],
                    dtype=np.float64,
                ),
            )
