import h5py
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

        for key, data in f.items():
            if key in ["E_w"]:
                dict_grd_samples[key] = jnp.array(data[:])

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
            return _format_basis_name(max(string_values, key=len)).replace('*', 's')
        else:
            return "gen"
    else:
        return _format_basis_name(basisname).replace('*', 's')