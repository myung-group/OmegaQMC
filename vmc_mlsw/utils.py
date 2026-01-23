import sys
import h5py
from pyscf import gto
from pyscf.gto.basis import _format_basis_name
import jax
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve
# from jax import lax

jax.config.update("jax_enable_x64", True)


@jax.jit
def do_binning_analysis(a):
    num_steps = a.shape[0]
    xbar = jnp.mean(a)
    sdev = jnp.std(a)
    # ac = jnp.correlate(a-xbar, a-xbar, mode='full')
    x = a - xbar
    ac = fftconvolve(x, x[::-1], mode="full")
    ac /= ac[ac.shape[0]//2]

    def compute_kappa(ac):
        """
        kappa = 1.0
        for i1 in range(int(len(ac) >> 1) + 1, int(len(ac))):
            if ac[i1] < 0:
                break
            else:
                kappa += (2.0*ac[i1])
        serr = s*(kappa/num_steps)**(0.5)
        """
        kappa_init = 1.0
        N = ac.shape[0]
        start = (N >> 1) + 1

        def cond_func(carry):
            i, kappa = carry
            return jnp.logical_and(i < N, ac[i] >= 0)

        def body_func(carry):
            i, kappa = carry
            return (i+1, kappa + 2.0*ac[i])
        _, kappa = jax.lax.while_loop(
            cond_func, body_func, (start, kappa_init)
        )
        return kappa

    kappa = compute_kappa(ac)
    serr = sdev*(kappa/num_steps)**(0.5)
    return (xbar, serr, sdev, kappa)


@jax.jit
def do_binning_analysis_grds(grd_tot_ls):
    # grd_tot_ls.shape == (samples, num_walkers, num_nuc, xyz)
    return jax.vmap(      # walker axis
                jax.vmap(     # nuclear axis
                    jax.vmap(     # xyz axis
                        do_binning_analysis,
                        in_axes=1, out_axes=(0, 0, 0, 0)
                    ),
                    in_axes=1, out_axes=(0, 0, 0, 0)
                ),
                in_axes=1, out_axes=(0, 0, 0, 0)
    )(grd_tot_ls)


def batched_binning_analysis(x, batch_size=100):
    n_walkers = x.shape[1]      # (samples, walkers)
    results = []
    for i in range(0, n_walkers, batch_size):
        x_chunk = x[:, i:i+batch_size]
        xbar, serr, s, kappa = jax.vmap(
            do_binning_analysis, in_axes=1, out_axes=(0, 0, 0, 0)
            )(x_chunk)
        results.append((xbar, serr, s, kappa))

    xbar_all = jnp.concatenate([r[0] for r in results])
    serr_all = jnp.concatenate([r[1] for r in results])
    s_all = jnp.concatenate([r[2] for r in results])
    kappa_all = jnp.concatenate([r[3] for r in results])

    return xbar_all, serr_all, s_all, kappa_all


def batched_binning_analysis_grds(grd_tot_ls, batch_size=100):
    # grd_tot_ls.shape == (num_steps_per_block, num_walkers, num_nuc, xyz)
    n_walkers = grd_tot_ls.shape[1]
    results = []
    for i in range(0, n_walkers, batch_size):
        sub = grd_tot_ls[:, i:i+batch_size]
        xbar, serr, s, kappa = do_binning_analysis_grds(sub)
        results.append((xbar, serr, s, kappa))
    xbar_all = jnp.concatenate([r[0] for r in results], axis=0)
    serr_all = jnp.concatenate([r[1] for r in results], axis=0)
    s_all = jnp.concatenate([r[2] for r in results], axis=0)
    kappa_all = jnp.concatenate([r[3] for r in results], axis=0)
    return xbar_all, serr_all, s_all, kappa_all


def compute_torque(mol, grd):
    coords = jnp.array(mol.atom_coords())
    masses = jnp.array(mol.atom_mass_list())
    # Center of mass
    ref = jnp.average(coords, axis=0, weights=masses)
    # Compute torque
    r = coords - ref
    torque = jnp.sum(jnp.cross(r, -grd), axis=0)

    return torque


def compute_torque_with_error(mol, grd, grd_err):
    coords = jnp.array(mol.atom_coords())
    masses = jnp.array(mol.atom_mass_list())
    # Center of mass
    ref = jnp.average(coords, axis=0, weights=masses)
    # Compute torque
    r = coords - ref
    torque = jnp.sum(jnp.cross(r, -grd), axis=0)
    # Compute torque errors
    x, y, z = coords.T
    dFx, dFy, dFz = grd_err.T
    dtau_sq = jnp.stack([
        (y * dFz)**2 + (z * dFy)**2,
        (z * dFx)**2 + (x * dFz)**2,
        (x * dFy)**2 + (y * dFx)**2,
    ])
    dtau_sq = jnp.sum(dtau_sq, axis=1)
    dtau = jnp.sqrt(dtau_sq)

    return torque, dtau


def compute_energy_with_error(chkfile):
    with h5py.File(chkfile, 'r') as f:
        dict_grd_samples = {}

        for key, val in f.items():
            if key in ["E_w"]:
                dict_grd_samples[key] = jnp.array(val[:])

        walkers_energies = dict_grd_samples["E_w"]
        xbar, serr, s, kappa = batched_binning_analysis(walkers_energies)

        e_mean = jnp.array(xbar).mean()
        e_err = jnp.linalg.norm(serr) / len(serr)

    return e_mean, e_err


def format_basis_name(basisname: str | dict):
    if isinstance(basisname, dict):
        # Get all string values and find the longest one
        string_values = [v for v in basisname.values() if isinstance(v, str)]
        if string_values:
            return _format_basis_name(max(string_values,
                                          key=len)).replace('*', 's')
        else:
            return "gen"
    else:
        return _format_basis_name(basisname).replace('*', 's')


# --- Gradient post-processing ---
def vmc_forces_with_space_warping(
        prefix: str = "vmc",
        logfile: bool | str = False,
        walker_based_batch_size: int = 10
        ) -> jnp.ndarray:
    suffixes_checked = [".chk.h5", ".grd.h5"]
    for s in suffixes_checked:
        if prefix.endswith(s):
            prefix = prefix[:-len(s)]
    # ofname_chkpt = prefix + ".chk.h5"
    ofname_grd = prefix + ".grd.h5"
    if not logfile or (isinstance(logfile, str) and logfile == ""):
        ofname_log = None
    else:
        ofname_log = logfile.strip() \
            if logfile.endswith(".log") \
            else logfile.strip() + ".log"

    with h5py.File(ofname_grd, 'r') as f:
        atom_symbols = f["system"]["atom_symbols"][()].split()
        atom_coords = f["system"]["atom_coords"]
        myUnits = f["system"]["units"][()].decode()
        mole_data = [(atom_symbols[i].decode(), atom_coords[i, :])
                     for i in range(len(atom_symbols))]
        myMol = gto.M(atom=mole_data, basis="mini", unit=myUnits)

        dict_grd_samples = {}
        for key, val in f.items():
            if isinstance(val, h5py.Group):
                dict_grd_samples[key] = {}
                for key2, val2 in val.items():
                    if not val2.shape:
                        dict_grd_samples[key][key2] = val2[()].decode()
                    else:
                        dict_grd_samples[key][key2] = jnp.array(val2)
            elif val.ndim == 0:
                dict_grd_samples[key] = val[()]    # scalar
            else:
                dict_grd_samples[key] = jnp.array(val[:])

        block_nums = [int(k)
                      for k in dict_grd_samples["local_energies"].keys()]
        block_nums.sort()
        # num_blocks = len(block_nums)
        # block_cnt_start = block_nums[0]

        loc_e_list = []
        for block_cnt in block_nums:
            local_energies \
                = dict_grd_samples["local_energies"][f'{block_cnt}']
            loc_e_list.append(jnp.array(local_energies))
        enr_mean = jnp.vstack(loc_e_list).mean()

        # enr_std = dict_grd_samples['enr_std']
        grd_nn = dict_grd_samples['grd_nn']

        valid_samples_count = 0
        grd_ke_sum = 0.0
        grd_ee_en_sum = 0.0
        grd_pulay_sum = 0.0

        grd_tot_list = []
        grd_err_list = []

        for block_cnt in block_nums:
            grd_ee_en = dict_grd_samples['grd_ee_en'][f'{block_cnt}']
            grd_ke = dict_grd_samples['grd_ke'][f'{block_cnt}']
            grd_logpsi = dict_grd_samples['grd_logpsi'][f'{block_cnt}']
            local_energies \
                = jnp.array(dict_grd_samples['local_energies'][f'{block_cnt}'])

            # Pulay force contribution
            d_enr = local_energies - enr_mean

            _, num_nuc, _ = grd_ee_en.shape
            num_steps_per_block, num_walkers = local_energies.shape

            # Regroup
            grd_ee_en = grd_ee_en.reshape(num_steps_per_block,
                                          num_walkers, num_nuc, 3)
            grd_ke = grd_ke.reshape(num_steps_per_block,
                                    num_walkers, num_nuc, 3)
            grd_logpsi = grd_logpsi.reshape(num_steps_per_block,
                                            num_walkers, num_nuc, 3)
            grd_pulay = 2.0 * jnp.einsum('sw,swnK->swnK',
                                         d_enr, grd_logpsi)

            grd_nn_sw = jnp.broadcast_to(
                grd_nn[jnp.newaxis, jnp.newaxis, :, :],
                (num_steps_per_block, num_walkers, num_nuc, 3)
                )
            grd_arrays = [grd_nn_sw,
                          grd_ee_en, grd_ke,
                          grd_pulay]
            grd_tot_sw = jnp.stack(grd_arrays, axis=0).sum(axis=0)

            # Compute forces and error
            xbar, serr, sdev, kappa = batched_binning_analysis_grds(
                grd_tot_sw, walker_based_batch_size
            )
            grd_tot_list.append(xbar[None, :, :, :])
            grd_err_list.append(serr[None, :, :, :])

            grd_ee_en_sum += grd_ee_en.sum(axis=0)
            grd_ke_sum += grd_ke.sum(axis=0)
            grd_pulay_sum += grd_pulay.sum(axis=0)

            valid_samples_count += local_energies.shape[0]
            # do not include num_walkers factor

        # Compute averages
        if valid_samples_count > 0:
            grd_ee_en = grd_ee_en_sum / valid_samples_count
            grd_ke = grd_ke_sum / valid_samples_count
            grd_pulay = grd_pulay_sum / valid_samples_count

            grd_tot_bw = jnp.concatenate(grd_tot_list, axis=0)
            grd_err_bw = jnp.concatenate(grd_err_list, axis=0)

            # mean over blocks, then walkers
            grd_tot = grd_tot_bw.mean(axis=0).squeeze()
            grd_tot = grd_tot.mean(axis=0)

            grd_err = jnp.linalg.norm(grd_err_bw, axis=0).squeeze() \
                / grd_err_bw.shape[0]
            grd_err = jnp.linalg.norm(grd_err, axis=0) \
                / grd_err.shape[0]

            # Compute torques and error
            torque, dtau \
                = compute_torque_with_error(myMol, grd_tot, grd_err)

            grd_ee_en = jnp.mean(grd_ee_en, axis=0)
            grd_ke = jnp.mean(grd_ke, axis=0)
            grd_pulay = jnp.mean(grd_pulay, axis=0)
        else:
            grd_ee_en = jnp.zeros_like(grd_nn)
            grd_ke = jnp.zeros_like(grd_nn)
            grd_pulay = jnp.zeros_like(grd_nn)

        # Write results
        with jnp.printoptions(precision=12, suppress=True):
            if ofname_log is None:
                fout = sys.stdout
            else:
                fout = open(ofname_log, 'w', 1)

            print('NN gradients\n', grd_nn, file=fout)
            print('ee+eN gradients\n', grd_ee_en, file=fout)
            print('KE gradients\n', grd_ke, file=fout)
            print('Pulay gradients\n', grd_pulay, file=fout)
            print('Total gradients\n', grd_tot, file=fout)

            fout.write("Total forces (-gradients)\n")
            for i in range(num_nuc):
                fout.write("{:4s}{:>16.6g} ± {:>12.6g}"
                           "{:>16.6g} ± {:>12.6g}"
                           "{:>16.6g} ± {:>12.6g}\n"
                           .format(myMol.atom_symbol(i),
                                   -grd_tot[i, 0], grd_err[i, 0],
                                   -grd_tot[i, 1], grd_err[i, 1],
                                   -grd_tot[i, 2], grd_err[i, 2]))
            fout.write("Total torque\n")
            fout.write("    {:>16.6g} ± {:>12.6g}"
                       "{:>16.6g} ± {:>12.6g}"
                       "{:>16.6g} ± {:>12.6g}\n"
                       .format(torque[0], dtau[0],
                               torque[1], dtau[1],
                               torque[2], dtau[2]))
            fout.write("\n")

            if ofname_log is not None:
                fout.close()

        return -grd_tot, grd_err
