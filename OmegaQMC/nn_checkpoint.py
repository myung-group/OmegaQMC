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
            :class:`~OmegaQMC.psi.nn.wf\
.MoleculeInfo`.
        energy: Optional energy estimate (float).
    """
    leaves = jax.tree.leaves(params)
    with h5py.File(filepath, 'w') as f:
        pg = f.create_group('params')
        for i, leaf in enumerate(leaves):
            arr = np.asarray(leaf)
            kw = (
                {'compression': 'gzip'}
                if arr.ndim > 0 else {}
            )
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
        leaves = [
            jnp.asarray(pg[str(i)][()])
            for i in range(n)
        ]
        meta = {}
        if 'meta' in f:
            mg = f['meta']
            for k in mg.attrs:
                meta[k] = mg.attrs[k]
            if 'charges' in mg:
                meta['charges'] = np.asarray(
                    mg['charges']
                )
            if 'coords' in mg:
                meta['coords'] = np.asarray(
                    mg['coords']
                )
    return jax.tree.unflatten(treedef, leaves), meta


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
        vg.attrs['E_mean'] = float(
            results['E_mean']
        )
        vg.attrs['E_serr'] = float(
            results['E_serr']
        )
        if 'E_blocks' in results:
            vg.create_dataset(
                'E_blocks',
                data=np.asarray(
                    results['E_blocks'],
                    dtype=np.float64,
                ),
            )
