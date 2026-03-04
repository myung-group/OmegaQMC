import os
import sys
import h5py
import numpy as np
from pyscf import __config__, gto
from pyscf.lib import logger, param
from pyscf.gto.basis import _format_basis_name
from pyscf.data.elements import MASSES, ELEMENTS_PROTON, \
        is_ghost_atom, _atom_symbol
import jax
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve
# from jax import lax



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


def batched_binning_analysis_grds(grd_tot_ls, batch_size=100, weights=None):
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

    if weights is not None:
        # Weighted mean per walker: (S, W, N, 3) weighted by (S, W)
        w_sum = weights.sum(axis=0)                       # (W,)
        xbar_all = jnp.einsum('sw,swnk->wnk', weights, grd_tot_ls) \
            / w_sum[:, None, None]

    return xbar_all, serr_all, s_all, kappa_all


def compute_center_of_mass(coords: np.ndarray, symbols: list[str],
                           ignore_hydrogen_mass: bool = False) \
        -> np.ndarray:
    """
    Calculate the center of mass for a set of atomic coordinates.

    Args:
        coords: Atomic coordinates (N, 3)
        symbols: List of element symbols (e.g., ['O', 'H', 'H'])

    Returns:
        Center of mass coordinates (3,)
    """
    if len(coords) != len(symbols):
        raise ValueError("Number of coordinates and symbols must match")

    total_mass = 0.0
    weighted_coords = np.zeros(3)

    # TODO: np.average(coords, axis=0, weights=masses) 형식으로 단순화
    for coord, symbol in zip(coords, symbols):
        # Get atomic number from element symbol
        if symbol not in ELEMENTS_PROTON:
            raise ValueError(f"Unknown element symbol: {symbol}")

        atomic_number = ELEMENTS_PROTON[symbol]
        mass = 0 \
            if atomic_number == 1 and ignore_hydrogen_mass \
            else MASSES[atomic_number]

        total_mass += mass
        weighted_coords += mass * coord

    return weighted_coords / total_mass


def parse_molecular_inspheres(mol: gto.Mole):
    assert hasattr(mol, "ignore_hydrogen_mass")

    if isinstance(mol._atom, str):
        if mol._atom.endswith(".xyz"):
            with open(mol._atom, 'r') as f:
                Z = f.readlines()
            if "molecule:I:1" not in Z[1]:
                print("⚠️ WARNING! Line 2 of {} should have a properties "
                      "block indicating a column with molecular "
                      "fragment indices "
                      "e.g. 'Properties=species:S:1:pos:R:3:molecule:I:1'"
                      .format(mol._atom))
            Z = list(filter(lambda x: x.strip() != "", Z[2:]))
        else:
            Z = mol._atom.strip().split('\n')
    else:
        Z = []
        for a in mol._atom:
            assert len(a) == 3
            Z.append(f"{a[0]} {a[1][0]} {a[1][1]} {a[1][2]} {a[2]}")
    # natm = len(Z)

    Y = []
    mol.map_nuc_frag = []
    for line in Z:
        z = line.strip().split()
        if len(z) >= 5:
            m = int(z[4])
            z = [z[0], float(z[1]), float(z[2]), float(z[3]), m]
        elif len(z) == 4:
            m = 0
            z = [z[0], float(z[1]), float(z[2]), float(z[3]), m]
        else:
            m = 0
            z = [z[0], 0, 0, 0, m]
        mol.map_nuc_frag.append(m)
        Y.append(z)

    mol.map_frag_ctr = dict[int, np.array]()
    for k in set(mol.map_nuc_frag):
        symbols = []
        X = []
        for y in Y:
            if y[4] == k:
                symbols.append(y[0])
                X.append(y[1:4])
        mol.map_frag_ctr[k] = compute_center_of_mass(np.array(X), symbols)

    seed_points = list(mol.map_frag_ctr.values())

    # Compute Voronoi in-radii (half-distance to nearest neighbor)
    mol.inradii = dict[int, float]()
    if len(seed_points) <= 1:
        # Single fragment: no Voronoi boundaries, return infinite radius
        mol.inradii[0] = mol.inradii[1] = np.inf
    else:
        seed_array = np.array(seed_points)
        # Compute pairwise distances
        diff = seed_array[:, np.newaxis, :] - seed_array[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        # Set diagonal to inf to exclude self-distances
        np.fill_diagonal(dist_matrix, np.inf)
        # In-radius is half the distance to nearest neighbor
        min_distances = dist_matrix.min(axis=1)

        for i, r in zip(mol.map_frag_ctr.keys(),
                        (min_distances / 2.0).tolist()):
            mol.inradii[i] = r


def _parse_default_basis(basis, uniq_atoms):
    if isinstance(basis, (str, tuple, list)):
        # default basis for all atoms
        _basis = {a: basis for a in uniq_atoms}
    elif 'default' in basis:
        default_basis = basis['default']
        _basis = {a: default_basis for a in uniq_atoms}
        _basis.update(basis)
        del _basis['default']
    else:
        _basis = basis
    return _basis


def _length_in_au(unit):
    '''Converts the input unit string into its length in A.U.'''
    if isinstance(unit, str):
        if gto.is_au(unit):
            unit = 1.
        else:
            unit = 1/param.BOHR
    else:
        unit = 1./unit
    return unit


class Mole_custom(gto.Mole):
    def format_atom(self, atoms, origin=0, axes=None,
                    unit=getattr(__config__, 'UNIT', 'Ang')):
        def str2atm(line):
            dat = line.split()
            try:
                coords = [float(x) for x in dat[1:4]]
            except ValueError:
                if gto.DISABLE_EVAL:
                    raise ValueError('Failed to parse geometry %s' % line)
                else:
                    coords = list(eval(','.join(dat[1:4])))

            if len(dat) >= 5:
                frag = int(dat[4])
            else:
                frag = 0

            if len(coords) != 3:
                raise ValueError('Coordinates error in %s' % line)
            return [_atom_symbol(dat[0]), coords, frag]

        if isinstance(atoms, str):
            # The input atoms points to a geometry file
            if os.path.isfile(atoms):
                try:
                    atoms = gto.fromfile(atoms)
                except ValueError:
                    sys.stderr.write('\nFailed to parse geometry file  %s\n\n'
                                     % atoms)
                    raise

            atoms = atoms.replace(';', '\n').replace(',', ' ') \
                .replace('\t', ' ')
            fmt_atoms = []
            for dat in atoms.split('\n'):
                dat = dat.strip()
                if dat and dat[0] != '#':
                    fmt_atoms.append(dat)

            if len(fmt_atoms[0].split()) < 4:
                fmt_atoms = gto.from_zmatrix('\n'.join(fmt_atoms))
            else:
                fmt_atoms = [str2atm(line) for line in fmt_atoms]
        else:
            fmt_atoms = []
            for atom in atoms:
                if isinstance(atom, str):
                    if atom.lstrip()[0] != '#':
                        fmt_atoms.append(str2atm(atom.replace(',', ' ')))
                else:
                    frag = int(atom[4]) if len(atom) >= 5 \
                        else int(atom[2]) if len(atom) == 3 \
                        else 0

                    if isinstance(atom[1], (int, float)):
                        fmt_atoms.append([_atom_symbol(atom[0]), atom[1:4],
                                          frag])
                    else:
                        fmt_atoms.append([_atom_symbol(atom[0]), atom[1],
                                          frag])

        if len(fmt_atoms) == 0:
            return []

        if axes is None:
            axes = np.eye(3)

        unit = _length_in_au(unit)
        c = np.array([a[1] for a in fmt_atoms], dtype=np.double)
        c = np.einsum('ix,kx->ki', axes * unit, c - origin)
        z = [a[0] for a in fmt_atoms]
        f = [a[2] for a in fmt_atoms]
        return list(zip(z, c.tolist(), f))

    def check_sanity(self):
        if isinstance(self.ecp, str):
            return self

        if isinstance(self.basis, str) and not self.ecp:
            elements = [x[0] for x in self._atom]
            ecp, ecp_atoms = gto.bse_predefined_ecp(self.basis, elements)
            if ecp_atoms:
                logger.warn(self, 'ECP not specified. '
                            f'The basis set {self.basis} include an ECP. '
                            f'Recommended ECP: {ecp}.')
        elif isinstance(self.basis, dict) and isinstance(self.ecp, dict):
            _basis = self.basis
            if 'default' in _basis:
                uniq_atoms = {a[0] for a in self._atom}
                basis = _parse_default_basis(_basis, uniq_atoms)
            else:
                basis = _basis
            for element, basname in basis.items():
                if isinstance(basname, str) and not self.ecp.get(element):
                    ecp, ecp_atoms = gto.bse_predefined_ecp(basname, element)
                    if ecp_atoms:
                        logger.warn(self, f'ECP for {element} not specified. '
                                    f'The basis set {basname} include an ECP. '
                                    f'Recommended ECP: {ecp}.')
            basis = None
        return self

    def set_geom_(self, atoms_or_coords, unit=None, symmetry=None,
                  inplace=True):
        if inplace:
            mol = self
        else:
            mol = self.copy(deep=False)
            mol._env = mol._env.copy()

        if unit is None:
            _unit = mol.unit
        else:
            _unit = _length_in_au(unit)
            if _unit != _length_in_au(self.unit):
                logger.warn(mol, 'Mole.unit (%s) is changed to %s',
                            self.unit, unit)
                mol.unit = unit

        if symmetry is None:
            symmetry = mol.symmetry

        if isinstance(atoms_or_coords, np.ndarray):
            mol.atom = [[a[0], b, a[2]]
                        for a, b in zip(mol._atom, atoms_or_coords.tolist())]
        else:
            mol.atom = atoms_or_coords

        if isinstance(atoms_or_coords, np.ndarray) and not symmetry:
            _unit = _length_in_au(mol.unit)
            mol._atom = list(zip([x[0] for x in mol._atom],
                                 (atoms_or_coords * _unit).tolist()))
            ptr = mol._atm[:, gto.PTR_COORD]
            mol._env[ptr+0] = _unit * atoms_or_coords[:, 0]
            mol._env[ptr+1] = _unit * atoms_or_coords[:, 1]
            mol._env[ptr+2] = _unit * atoms_or_coords[:, 2]
            # reset nuclear energy
            mol.enuc = None
        else:
            mol.symmetry = symmetry
            mol.build(False, False)

        if mol.verbose >= logger.INFO:
            logger.info(mol, 'New geometry')
            for ia, atom in enumerate(mol._atom):
                coorda = tuple([x * param.BOHR for x in atom[1]])
                coordb = tuple(atom[1])
                coords = coorda + coordb
                logger.info(mol, ' %3d %-4s %16.12f %16.12f %16.12f AA  '
                            '%16.12f %16.12f %16.12f Bohr',
                            ia+1, mol.atom_symbol(ia), *coords)
        return mol


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
def vmc_forces_with_pgcs(
        prefix: str = "vmc",
        logfile: bool | str = False,
        walker_based_batch_size: int = 10
        ) -> jnp.ndarray:
    """ PGCS = point group correlated sampling """
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

        if "atom_fragment_map" in f["system"]:
            atom_frag_map = list(f["system"]["atom_fragment_map"][:])
        else:
            atom_frag_map = None

        dict_grd_samples = {}
        for key, val in f.items():
            if isinstance(val, h5py.Group):
                dict_grd_samples[key] = {}
                for key2, val2 in val.items():
                    if isinstance(val2, h5py.Group):
                        dict_grd_samples[key][key2] = {}
                        for key3, val3 in val2.items():
                            if not val3.shape:
                                dict_grd_samples[key][key2][key3] \
                                    = val3[()].decode()
                            else:
                                dict_grd_samples[key][key2][key3] \
                                    = jnp.array(val3)
                    elif not val2.shape:
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

        # Identify combo labels from fragment_weights
        combo_labels = []
        if 'fragment_weights' in dict_grd_samples:
            combo_labels = sorted(
                k for k in dict_grd_samples['fragment_weights']
                if isinstance(dict_grd_samples['fragment_weights'][k],
                              dict))
        states = [None] + combo_labels   # None = reference

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
                    grd_ee_en \
                        = dict_grd_samples['grd_ee_en'][f'{block_cnt}']
                    grd_ke \
                        = dict_grd_samples['grd_ke'][f'{block_cnt}']
                    grd_logpsi \
                        = dict_grd_samples['grd_logpsi'][f'{block_cnt}']
                else:
                    grd_ee_en \
                        = dict_grd_samples['grd_ee_en'][state_label][f'{block_cnt}']
                    grd_ke \
                        = dict_grd_samples['grd_ke'][state_label][f'{block_cnt}']
                    grd_logpsi \
                        = dict_grd_samples['grd_logpsi'][state_label][f'{block_cnt}']
                local_energies = jnp.array(
                    dict_grd_samples['local_energies'][f'{block_cnt}'])

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

                # Load fragment weights for secondary states
                if state_label is not None:
                    frag_w = dict_grd_samples[
                        'fragment_weights'][state_label][f'{block_cnt}']
                    frag_w = frag_w.reshape(num_steps_per_block, num_walkers)
                else:
                    frag_w = None

                # Compute forces and error
                xbar, serr, sdev, kappa = batched_binning_analysis_grds(
                    grd_tot_sw, walker_based_batch_size, weights=frag_w
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

            # Save reference results for return value
            if state_label is None:
                ref_grd_tot = grd_tot
                ref_grd_err = grd_err

            all_state_results[state_label] = (grd_tot, grd_err)

            # Write results
            with jnp.printoptions(precision=12, suppress=True):
                if state_label is not None:
                    print(f'\n--- Secondary state: {state_label} ---',
                          file=fout)

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

        # --- Fragment-wise averaging of PGCS force estimates ---
        if atom_frag_map is not None and combo_labels:
            frag_to_states = {}
            for fid in set(atom_frag_map):
                frag_to_states[fid] = [None]  # all include reference

            for label in combo_labels:
                for part in label.split(','):
                    fid_str, op = part.split(':')
                    if op != 'E':
                        frag_to_states[int(fid_str)].append(label)
                        break

            avg_grd_tot = jnp.zeros((num_nuc, 3))
            avg_grd_err = jnp.zeros((num_nuc, 3))

            for i in range(num_nuc):
                fid = atom_frag_map[i]
                relevant_states = frag_to_states.get(fid, [None])
                forces = jnp.stack(
                    [all_state_results[s][0][i] for s in relevant_states])
                errors = jnp.stack(
                    [all_state_results[s][1][i] for s in relevant_states])
                N = len(relevant_states)
                avg_grd_tot = avg_grd_tot.at[i].set(forces.mean(axis=0))
                avg_grd_err = avg_grd_err.at[i].set(
                    jnp.sqrt(jnp.sum(errors**2, axis=0)) / N)

            avg_torque, avg_dtau \
                = compute_torque_with_error(myMol, avg_grd_tot, avg_grd_err)

            with jnp.printoptions(precision=12, suppress=True):
                fout.write("\n=== Fragment-wise averaged forces ===\n")
                fout.write("Total gradients (averaged)\n")
                fout.write(f" {avg_grd_tot}\n")
                fout.write("Total forces (-gradients, averaged)\n")
                for i in range(num_nuc):
                    fid = atom_frag_map[i]
                    n_states = len(frag_to_states.get(fid, [None]))
                    fout.write("{:4s}{:>16.6g} ± {:>12.6g}"
                               "{:>16.6g} ± {:>12.6g}"
                               "{:>16.6g} ± {:>12.6g}"
                               "  (fragment {}, over {} states)\n"
                               .format(myMol.atom_symbol(i),
                                       -avg_grd_tot[i, 0], avg_grd_err[i, 0],
                                       -avg_grd_tot[i, 1], avg_grd_err[i, 1],
                                       -avg_grd_tot[i, 2], avg_grd_err[i, 2],
                                       fid, n_states))
                fout.write("Total torque (averaged)\n")
                fout.write("    {:>16.6g} ± {:>12.6g}"
                           "{:>16.6g} ± {:>12.6g}"
                           "{:>16.6g} ± {:>12.6g}\n"
                           .format(avg_torque[0], avg_dtau[0],
                                   avg_torque[1], avg_dtau[1],
                                   avg_torque[2], avg_dtau[2]))
                fout.write("\n")

            ref_grd_tot = avg_grd_tot
            ref_grd_err = avg_grd_err

        if ofname_log is not None:
            fout.close()

        return -ref_grd_tot, ref_grd_err
