"""Post-processing of VMC gradient data for nuclear forces.

Reads the gradient HDF5 file written by
:meth:`~OmegaQMC.vmc_gto._VMCDriverGTO.__call__` and
applies Point Group Correlated Sampling (PGCS) to
obtain symmetry-averaged nuclear force estimates.
"""

import sys

import h5py
import jax.numpy as jnp
from pyscf import gto

from ..utils import (
    batched_binning_analysis_grds,
    compute_torque_with_error,
)


def postproc_h5_pgcs(
        prefix: str = "vmc",
        logfile: bool | str = False,
        walker_based_batch_size: int = 10
        ) -> jnp.ndarray:
    """Post-process VMC gradient data to obtain \
nuclear forces using PGCS.

    Reads the gradient HDF5 file written by
    :meth:`_VMCDriverGTO.__call__` and applies Point
    Group Correlated Sampling (PGCS) to obtain
    symmetry-averaged estimates of the nuclear forces
    and their statistical errors.

    Parameters
    ----------
    prefix : str, optional
        File-name stem used when the VMC run was set
        up (the ``prefix`` argument of
        :func:`get_vmc_gto_func`).  The function looks
        for ``<prefix>.grd.h5``; trailing ``.chk.h5``
        or ``.grd.h5`` suffixes are stripped
        automatically.  Default is ``"vmc"``.
    logfile : bool or str, optional
        Controls logging output.  ``False`` (default)
        suppresses logging.  ``True`` writes to
        ``<prefix>.log``.  A string is used as the log
        file path directly (a ``.log`` extension is
        appended if absent).
    walker_based_batch_size : int, optional
        Number of walker blocks to load into memory at
        once when summing gradient contributions.
        Reduce this value if GPU memory is tight.
        Default is 10.

    Returns
    -------
    grd : jnp.ndarray, shape (num_atoms, 3)
        Mean nuclear forces (negative energy gradient)
        in Hartree/Bohr.
    grd_err : jnp.ndarray, shape (num_atoms, 3)
        Statistical error (standard error of the mean)
        of each force component.
    """
    suffixes_checked = [".chk.h5", ".grd.h5"]
    for s in suffixes_checked:
        if prefix.endswith(s):
            prefix = prefix[:-len(s)]
    # ofname_chkpt = prefix + ".chk.h5"
    ofname_grd = prefix + ".grd.h5"
    if not logfile or (
        isinstance(logfile, str) and logfile == ""
    ):
        ofname_log = None
    else:
        ofname_log = logfile.strip() \
            if logfile.endswith(".log") \
            else logfile.strip() + ".log"

    with h5py.File(ofname_grd, 'r') as f:
        atom_symbols = (
            f["system"]["atom_symbols"][()].split()
        )
        atom_coords = f["system"]["atom_coords"]
        myUnits = (
            f["system"]["units"][()].decode()
        )
        mole_data = [
            (atom_symbols[i].decode(),
             atom_coords[i, :])
            for i in range(len(atom_symbols))
        ]
        myMol = gto.M(
            atom=mole_data, basis="mini",
            unit=myUnits,
        )

        if "atom_fragment_map" in f["system"]:
            atom_frag_map = list(
                f["system"]["atom_fragment_map"][:]
            )
        else:
            atom_frag_map = None

        dict_grd_samples = {}
        for key, val in f.items():
            if isinstance(val, h5py.Group):
                dict_grd_samples[key] = {}
                for key2, val2 in val.items():
                    if isinstance(
                        val2, h5py.Group
                    ):
                        dict_grd_samples[key][key2] \
                            = {}
                        for key3, val3 in (
                            val2.items()
                        ):
                            if not val3.shape:
                                dict_grd_samples[
                                    key][key2][key3] \
                                    = val3[()].decode()
                            else:
                                dict_grd_samples[
                                    key][key2][key3] \
                                    = jnp.array(val3)
                    elif not val2.shape:
                        dict_grd_samples[key][key2] \
                            = val2[()].decode()
                    else:
                        dict_grd_samples[key][key2] \
                            = jnp.array(val2)
            elif val.ndim == 0:
                dict_grd_samples[key] = val[()]
            else:
                dict_grd_samples[key] = (
                    jnp.array(val[:])
                )

        block_nums = [
            int(k) for k in
            dict_grd_samples["local_energies"]
            .keys()
            if k.isdigit()
        ]
        block_nums.sort()

        loc_e_list = []
        for block_cnt in block_nums:
            local_energies = dict_grd_samples[
                "local_energies"
            ][f'{block_cnt}']
            loc_e_list.append(
                jnp.array(local_energies)
            )
        enr_mean = jnp.vstack(loc_e_list).mean()

        grd_nn = dict_grd_samples['grd_nn']

        # Identify combo labels from fragment_weights
        combo_labels = []
        if 'fragment_weights' in dict_grd_samples:
            combo_labels = sorted(
                k for k in dict_grd_samples[
                    'fragment_weights'
                ]
                if isinstance(
                    dict_grd_samples[
                        'fragment_weights'
                    ][k],
                    dict,
                )
            )
        states = [None] + combo_labels

        if ofname_log is None:
            fout = sys.stdout
        else:
            fout = open(ofname_log, 'w', 1)

        ref_grd_tot = None
        ref_grd_err = None
        all_state_results = {}

        for state_label in states:
            valid_samples_count = 0
            grd_ke_sum = 0.0
            grd_ee_en_sum = 0.0
            grd_pulay_sum = 0.0

            grd_tot_list = []
            grd_err_list = []

            for block_cnt in block_nums:
                if state_label is None:
                    grd_ee_en = dict_grd_samples[
                        'grd_ee_en'
                    ][f'{block_cnt}']
                    grd_ke = dict_grd_samples[
                        'grd_ke'
                    ][f'{block_cnt}']
                    grd_logpsi = dict_grd_samples[
                        'grd_logpsi'
                    ][f'{block_cnt}']
                else:
                    grd_ee_en = dict_grd_samples[
                        'grd_ee_en'
                    ][state_label][f'{block_cnt}']
                    grd_ke = dict_grd_samples[
                        'grd_ke'
                    ][state_label][f'{block_cnt}']
                    grd_logpsi = dict_grd_samples[
                        'grd_logpsi'
                    ][state_label][f'{block_cnt}']
                local_energies = jnp.array(
                    dict_grd_samples[
                        'local_energies'
                    ][f'{block_cnt}']
                )

                # Pulay force contribution
                d_enr = local_energies - enr_mean

                _, num_nuc, _ = grd_ee_en.shape
                num_steps_per_block, num_walkers = (
                    local_energies.shape
                )

                # Regroup
                grd_ee_en = grd_ee_en.reshape(
                    num_steps_per_block,
                    num_walkers, num_nuc, 3,
                )
                grd_ke = grd_ke.reshape(
                    num_steps_per_block,
                    num_walkers, num_nuc, 3,
                )
                grd_logpsi = grd_logpsi.reshape(
                    num_steps_per_block,
                    num_walkers, num_nuc, 3,
                )
                grd_pulay = 2.0 * jnp.einsum(
                    'sw,swnK->swnK',
                    d_enr, grd_logpsi,
                )

                grd_nn_sw = jnp.broadcast_to(
                    grd_nn[
                        jnp.newaxis, jnp.newaxis,
                        :, :
                    ],
                    (num_steps_per_block,
                     num_walkers, num_nuc, 3),
                )
                grd_arrays = [
                    grd_nn_sw,
                    grd_ee_en, grd_ke,
                    grd_pulay,
                ]
                grd_tot_sw = (
                    jnp.stack(grd_arrays, axis=0)
                    .sum(axis=0)
                )

                # Load fragment weights for
                # secondary states
                if state_label is not None:
                    frag_w = dict_grd_samples[
                        'fragment_weights'
                    ][state_label][f'{block_cnt}']
                    frag_w = frag_w.reshape(
                        num_steps_per_block,
                        num_walkers,
                    )
                else:
                    frag_w = None

                # Compute forces and error
                xbar, serr, sdev, kappa = (
                    batched_binning_analysis_grds(
                        grd_tot_sw,
                        walker_based_batch_size,
                        weights=frag_w,
                    )
                )
                grd_tot_list.append(
                    xbar[None, :, :, :]
                )
                grd_err_list.append(
                    serr[None, :, :, :]
                )

                grd_ee_en_sum += (
                    grd_ee_en.sum(axis=0)
                )
                grd_ke_sum += grd_ke.sum(axis=0)
                grd_pulay_sum += (
                    grd_pulay.sum(axis=0)
                )

                valid_samples_count += (
                    local_energies.shape[0]
                )

            # Compute averages
            if valid_samples_count > 0:
                grd_ee_en = (
                    grd_ee_en_sum
                    / valid_samples_count
                )
                grd_ke = (
                    grd_ke_sum
                    / valid_samples_count
                )
                grd_pulay = (
                    grd_pulay_sum
                    / valid_samples_count
                )

                grd_tot_bw = jnp.concatenate(
                    grd_tot_list, axis=0,
                )
                grd_err_bw = jnp.concatenate(
                    grd_err_list, axis=0,
                )

                # mean over blocks, then walkers
                grd_tot = (
                    grd_tot_bw.mean(axis=0)
                    .squeeze()
                )
                grd_tot = grd_tot.mean(axis=0)

                grd_err = (
                    jnp.linalg.norm(
                        grd_err_bw, axis=0,
                    ).squeeze()
                    / grd_err_bw.shape[0]
                )
                grd_err = (
                    jnp.linalg.norm(
                        grd_err, axis=0,
                    )
                    / grd_err.shape[0]
                )

                # Compute torques and error
                torque, dtau = (
                    compute_torque_with_error(
                        myMol, grd_tot, grd_err,
                    )
                )

                grd_ee_en = jnp.mean(
                    grd_ee_en, axis=0,
                )
                grd_ke = jnp.mean(
                    grd_ke, axis=0,
                )
                grd_pulay = jnp.mean(
                    grd_pulay, axis=0,
                )
            else:
                grd_ee_en = jnp.zeros_like(grd_nn)
                grd_ke = jnp.zeros_like(grd_nn)
                grd_pulay = jnp.zeros_like(grd_nn)

            # Save reference results for return
            if state_label is None:
                ref_grd_tot = grd_tot
                ref_grd_err = grd_err

            all_state_results[state_label] = (
                grd_tot, grd_err,
            )

            # Write results
            with jnp.printoptions(
                precision=12, suppress=True,
            ):
                if state_label is not None:
                    print(
                        "\n--- Secondary state:"
                        f" {state_label} ---",
                        file=fout,
                    )

                print(
                    'NN gradients\n',
                    grd_nn, file=fout,
                )
                print(
                    'ee+eN gradients\n',
                    grd_ee_en, file=fout,
                )
                print(
                    'KE gradients\n',
                    grd_ke, file=fout,
                )
                print(
                    'Pulay gradients\n',
                    grd_pulay, file=fout,
                )
                print(
                    'Total gradients\n',
                    grd_tot, file=fout,
                )

                fout.write(
                    "Total forces (-gradients)\n"
                )
                for i in range(num_nuc):
                    fout.write(
                        "{:4s}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}\n"
                        .format(
                            myMol.atom_symbol(i),
                            -grd_tot[i, 0],
                            grd_err[i, 0],
                            -grd_tot[i, 1],
                            grd_err[i, 1],
                            -grd_tot[i, 2],
                            grd_err[i, 2],
                        )
                    )
                fout.write("Total torque\n")
                fout.write(
                    "    {:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}\n"
                    .format(
                        torque[0], dtau[0],
                        torque[1], dtau[1],
                        torque[2], dtau[2],
                    )
                )
                fout.write("\n")

        # --- Fragment-wise averaging of PGCS ---
        if (
            atom_frag_map is not None
            and combo_labels
        ):
            frag_to_states = {}
            for fid in set(atom_frag_map):
                frag_to_states[fid] = [None]

            for label in combo_labels:
                for part in label.split(','):
                    fid_str, op = part.split(':')
                    if op != 'E':
                        frag_to_states[
                            int(fid_str)
                        ].append(label)
                        break

            avg_grd_tot = jnp.zeros((num_nuc, 3))
            avg_grd_err = jnp.zeros((num_nuc, 3))

            for i in range(num_nuc):
                fid = atom_frag_map[i]
                relevant_states = (
                    frag_to_states.get(fid, [None])
                )
                forces = jnp.stack([
                    all_state_results[s][0][i]
                    for s in relevant_states
                ])
                errors = jnp.stack([
                    all_state_results[s][1][i]
                    for s in relevant_states
                ])
                N = len(relevant_states)
                avg_grd_tot = (
                    avg_grd_tot.at[i]
                    .set(forces.mean(axis=0))
                )
                avg_grd_err = (
                    avg_grd_err.at[i].set(
                        jnp.sqrt(
                            jnp.sum(
                                errors**2, axis=0,
                            )
                        )
                        / N
                    )
                )

            avg_torque, avg_dtau = (
                compute_torque_with_error(
                    myMol, avg_grd_tot, avg_grd_err,
                )
            )

            with jnp.printoptions(
                precision=12, suppress=True,
            ):
                fout.write(
                    "\n\tFragment-wise averaged"
                    " forces\n"
                )
                fout.write(
                    "Total gradients (averaged)\n"
                )
                fout.write(f" {avg_grd_tot}\n")
                fout.write(
                    "Total forces"
                    " (-gradients, averaged)\n"
                )
                for i in range(num_nuc):
                    fid = atom_frag_map[i]
                    n_states = len(
                        frag_to_states.get(
                            fid, [None],
                        )
                    )
                    fout.write(
                        "{:4s}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}"
                        "{:>16.6g} ± {:>12.6g}"
                        "  (fragment {},"
                        " over {} states)\n"
                        .format(
                            myMol.atom_symbol(i),
                            -avg_grd_tot[i, 0],
                            avg_grd_err[i, 0],
                            -avg_grd_tot[i, 1],
                            avg_grd_err[i, 1],
                            -avg_grd_tot[i, 2],
                            avg_grd_err[i, 2],
                            fid, n_states,
                        )
                    )
                fout.write(
                    "Total torque (averaged)\n"
                )
                fout.write(
                    "    {:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}"
                    "{:>16.6g} ± {:>12.6g}\n"
                    .format(
                        avg_torque[0], avg_dtau[0],
                        avg_torque[1], avg_dtau[1],
                        avg_torque[2], avg_dtau[2],
                    )
                )
                fout.write("\n")

            ref_grd_tot = avg_grd_tot
            ref_grd_err = avg_grd_err

        if ofname_log is not None:
            fout.close()

        return -ref_grd_tot, ref_grd_err
