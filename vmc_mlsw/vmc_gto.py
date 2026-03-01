import sys
import pathlib
from collections.abc import Collection
from datetime import datetime
import warnings

from pyscf import scf, symm
import jax
import jax.numpy as jnp
# from functools import partial
import h5py

from .psi_gto import get_psi_fun
from .cusp import get_cusp_params
from .utils import parse_molecular_inspheres, Mole_custom, _length_in_au
# from .symm.water_rotation_matrix import symmetrize_water_molecule
from .symm.operations import (symmetry_operations_map,
                              populate_fragment_symmops,
                              apply_reflection_x,
                              apply_reflection_y,
                              apply_reflection_z,
                              apply_rotation_x180,
                              apply_rotation_y180,
                              apply_rotation_z180)
from .symm.point_groups import (auto_symmetrize_molecule,
                                detect_symmetry_quality)
# from .constants import CHEMICAL_ACCURACY
from .constants import MIN_DIST_THRESHOLD

# VMC hyperparameters
TARGET_ACCEPTANCE_RATE = 0.4
STEP_SIZE_ADAPTATION_RATE = 0.05
PSI2_RATIO_THRESHOLD = 1e-4  # screen symmetrized gradient samples


def _get_electron_displacement_fn(Z_charges: jnp.ndarray,
                                  nuc_crds: jnp.ndarray,
                                  cluster_idx: Collection[int] | None,
                                  mol=None) -> tuple:
    """Precompute per-fragment data for Metropolis moves.

    Returns:
        frag_reflect_data: (frag_centroids, frag_inradii, frag_Vh,
        frag_is_planar) — always a 4-tuple of JAX arrays.
    """
    if mol is not None and hasattr(mol, 'map_frag_symmops'):
        frag_ids = sorted(mol.map_frag_ctr.keys())

        centroids_list = []
        inradii_list = []
        Vh_list = []
        is_planar_list = []

        for fid in frag_ids:
            frag_ops = mol.map_frag_symmops.get(fid, ['E'])
            frag_atom_indices = [i for i, f in enumerate(mol.map_nuc_frag)
                                 if f == fid]
            is_planar = (frag_ops != ['E'] and len(frag_atom_indices) >= 3)

            centroid = jnp.array(mol.map_frag_ctr[fid])
            centroids_list.append(centroid)
            inradii_list.append(mol.inradii[fid])

            if is_planar:
                frag_nuc = nuc_crds[jnp.array(frag_atom_indices)]
                centered = frag_nuc - centroid
                _, _, Vh = jnp.linalg.svd(centered, full_matrices=True)
                Vh_list.append(Vh)
            else:
                Vh_list.append(jnp.eye(3))
            is_planar_list.append(is_planar)
    else:
        # Fallback: single pseudo-fragment covering entire molecule
        centroids_list = [jnp.mean(nuc_crds, axis=0)]
        inradii_list = [jnp.inf]
        Vh_list = [jnp.eye(3)]
        is_planar_list = [False]

    frag_reflect_data = (
        jnp.stack(centroids_list),                  # (num_frags, 3)
        jnp.array(inradii_list),                    # (num_frags,)
        jnp.stack(Vh_list),                         # (num_frags, 3, 3)
        jnp.array(is_planar_list, dtype=jnp.bool_)  # (num_frags,)
    )

    return frag_reflect_data


def _initialize_walkers(rng_key: jax.Array,
                        num_walkers: int,
                        nelec: int,
                        Z_charges: jnp.ndarray,
                        nuc_crds: jnp.ndarray,
                        mol_charge: int) -> jnp.ndarray:
    """Initialize walker positions near nuclear centers."""
    # Assign electrons to atoms based on atomic number
    idx_cnt = []
    for ia, iz in enumerate(Z_charges):
        idx_cnt.extend([ia] * iz)

    # Adjust for molecular charge
    if mol_charge < 0:
        idx_cnt.extend([0] * abs(mol_charge))
    elif mol_charge > 0:
        idx_cnt = idx_cnt[:-mol_charge]

    idx_cnt = jnp.array(idx_cnt)
    centers = nuc_crds[idx_cnt]

    # Initialize with small Gaussian noise around centers
    noise = jax.random.normal(rng_key, (num_walkers, nelec, 3))
    walkers = centers[jnp.newaxis, :, :] + 0.05 * noise

    return walkers


def _adapt_step_size(step_size: float, acceptance_ratio: float) -> float:
    """Adapt step size based on acceptance ratio."""
    log_step = jnp.log(step_size) + STEP_SIZE_ADAPTATION_RATE * (
        acceptance_ratio - TARGET_ACCEPTANCE_RATE
    )
    return jnp.exp(log_step)


def generate_molecular_orbitals(astr: str,
                                unit: str = None, units: str = "Bohr",
                                spin: int = 0,
                                basis: str | dict = "aug-cc-pVTZ",
                                postHF: str = None,
                                ignore_hydrogen_mass: bool = False,
                                symmetrization_level: int = 1):
    """ PySCF wrapper """
    if unit is not None:
        units = unit
        # "unit" (sic) takes precedence

    if astr.endswith(".xyz"):
        if units.upper().startswith("B") or units.upper().startswith("AU"):
            print("⚠️ WARNING! XYZ input uses Å units by default, "
                  "but the user has specified Bohrs.")
        elif units is None or units == "":
            units = "angstroms"

    # mol = gto.M(atom=astr, basis=basis, unit=units)
    mol = Mole_custom()
    mol.build(atom=astr, basis=basis, unit=units)
    # see pyscf.gto.mole.is_au(unit)

    if symmetrization_level >= 1:
        # Detect symmetry and get principal axes transformation
        gpname, centroid, axes = symm.geom.detect_symm(mol._atom)
        # Apply symmetry-based transformation:
        # center and rotate to principal axes
        mol.atom = [[a[0], a[1] / _length_in_au(units), b[2]]
                    for a, b in zip(symm.geom.shift_atom(mol._atom,
                                                         centroid, axes),
                                    mol._atom)]
        mol.build()

    # TODO: 여기를 포함하여 모든 centroid 계산을
    # utils.compute_center_of_mass 호출하는 것으로 일원화 할 것
    if ignore_hydrogen_mass:
        import numpy as np
        Z = mol.atom_charges()
        masses_adjusted = np.where(Z == 1, 0.0, mol.atom_mass_list())
        if masses_adjusted.sum() > 0.0:
            centroid = np.average(mol.atom_coords(), axis=0,
                                  weights=masses_adjusted)
            mol.set_geom_((mol.atom_coords() - centroid)
                          / _length_in_au(units))
            mol.build()
        else:
            warnings.warn(
                "ignore_hydrogen_mass parameter will be disabled "
                "for systems that consist of only hydrogen atoms.",
                stacklevel=2
            )

    # Apply additional symmetrization for improved numerical precision
    if symmetrization_level >= 2:
        if ignore_hydrogen_mass:
            warnings.warn(
                "ignore_hydrogen_mass parameter will be disabled "
                "for symmetrization_level >= 2.",
                stacklevel=2
            )

        # Check if symmetrization is beneficial for this molecule
        try:
            quality = detect_symmetry_quality(mol.atom, gpname)

            if mol.verbose >= 3:
                print("Symmetry quality check - Max deviation: "
                      f"{quality['max_deviation']:.2e}")
                print("Needs symmetrization: "
                      f"{quality['needs_symmetrization']}")

            if quality['needs_symmetrization']:
                # Apply automatic symmetrization
                symmetrized_atoms = auto_symmetrize_molecule(mol.atom, gpname)

                # Apply symmetrization regardless
                mol.atom = symmetrized_atoms
                mol.build()

                if mol.verbose >= 3:
                    print(f"Applied {gpname} symmetrization "
                          "for improved numerical precision")
        except Exception as e:
            if mol.verbose >= 2:
                print(f"Symmetrization skipped due to error: {e}")

    if mol.verbose >= 3:
        print(mol._atom)
        # mol._atom and mol.atom_coords() in au (internal)
        # mol.atom in input units

    mol.ignore_hydrogen_mass = ignore_hydrogen_mass
    parse_molecular_inspheres(mol)
    assert hasattr(mol, "map_nuc_frag")
    assert hasattr(mol, "map_frag_ctr") and isinstance(mol.map_frag_ctr, dict)
    assert hasattr(mol, "inradii") and isinstance(mol.inradii, dict)

    populate_fragment_symmops(mol)
    # TODO: later use symmetrization_level option

    mf = scf.UHF(mol) if spin & 1 else scf.RHF(mol)
    mf.kernel()
    # mf_grad = mf.nuc_grad_method()
    # grad = mf_grad.kernel()

    if postHF is not None:
        if postHF == "CCSD":
            from pyscf import cc
            postmf = cc.CCSD(mf).run()
            # cc_grad = postmf.nuc_grad_method()
            # cc_grad.kernel()
            postmf.mol.groupname = gpname
            return postmf
    else:
        # XXX: hack - merely tag the molecule afterwards
        #      without doing any symmetry-adapted SCF
        mf.mol.groupname = gpname
        return mf


def get_vmc_func(mf,
                 params_corr: dict | None,
                 cusp_scheme='Quady2025',
                 gr_scheme='scheme1',
                 prefix='vmc',
                 symmop_list: list[str] | dict[int, list[str]] | None = None,
                 cluster_idx: Collection[int] = None):
    # Build per-fragment symmetry operations dict
    if hasattr(mf.mol, 'map_frag_symmops') and mf.mol.map_frag_symmops:
        frag_ids = sorted(mf.mol.map_frag_ctr.keys())
    else:
        frag_ids = [0]

    if symmop_list is None:
        # Auto-derive: use all allowed operations for each fragment
        if hasattr(mf.mol, 'map_frag_symmops') and mf.mol.map_frag_symmops:
            frag_symmops = {fid: list(mf.mol.map_frag_symmops.get(fid, ['E']))
                            for fid in frag_ids}
        else:
            frag_symmops = {fid: ['E'] for fid in frag_ids}
        if mf.mol.verbose >= 2:
            print(f"Auto-derived symmetry operations: {frag_symmops}")
    elif isinstance(symmop_list, list):
        # List of strings: intersect with each fragment's allowed operations
        frag_symmops = {}
        for fid in frag_ids:
            allowed = set(mf.mol.map_frag_symmops.get(fid, ['E'])
                          if hasattr(mf.mol, 'map_frag_symmops') else ['E'])
            requested = set(symmop_list)
            invalid = requested - allowed - {'E'}
            if invalid:
                warnings.warn(
                    f"Fragment {fid}: operations {invalid} are not "
                    "valid symmetry operations and will be removed",
                    stacklevel=2)
            frag_symmops[fid] = sorted(requested & allowed)
            if 'E' not in frag_symmops[fid]:
                frag_symmops[fid].insert(0, 'E')
        if mf.mol.verbose >= 2:
            print(f"Input-processed symmetry operations: {frag_symmops}")
    elif isinstance(symmop_list, dict):
        # Dict: per-fragment specification
        frag_symmops = {}
        for fid in frag_ids:
            if fid in symmop_list:
                allowed = set(mf.mol.map_frag_symmops.get(fid, ['E'])
                              if hasattr(mf.mol, 'map_frag_symmops')
                              else ['E'])
                requested = set(symmop_list[fid])
                invalid = requested - allowed - {'E'}
                if invalid:
                    warnings.warn(
                        f"Fragment {fid}: operations {invalid} are not "
                        "valid symmetry operations and will be removed",
                        stacklevel=2)
                frag_symmops[fid] = sorted(requested & allowed)
            else:
                frag_symmops[fid] = ['E']
            if 'E' not in frag_symmops[fid]:
                frag_symmops[fid].insert(0, 'E')
        if mf.mol.verbose >= 2:
            print(f"Symmetry operations corrected from input: {frag_symmops}")
    else:
        raise TypeError(
            f"symmop_list must be None, list[str], or dict[int, list[str]], "
            f"got {type(symmop_list)}")

    # Derived data structures for gradient computation
    frag_ops_sets = [set(frag_symmops[fid]) for fid in frag_ids]
    all_symmops = sorted(set(op for ops in frag_symmops.values()
                             for op in ops))

    assert all_symmops != []

    # Enumerate single-fragment operation combos for correlated sampling
    single_frag_combos = []
    for frag_pos, fid in enumerate(frag_ids):
        for op in frag_symmops[fid]:
            if op == 'E':
                continue
            parts = [f"{fid2}:{op if fid2 == fid else 'E'}"
                     for fid2 in frag_ids]
            label = ",".join(parts)
            single_frag_combos.append((frag_pos, op, label))

    if mf.mol.groupname == 'C1' \
            and any(len(ops) > 1 for ops in frag_symmops.values()):
        warnings.warn(
            "Calculating symmetry-adapted forces "
            "on a system with no symmetry (C1)",
            stacklevel=2
        )

    # check prefix
    suffixes_checked = [".chk.h5", ".grd.h5"]
    for s in suffixes_checked:
        if prefix.endswith(s):
            prefix = prefix[:-len(s)]
    ofname_chkpt = prefix + ".chk.h5"
    ofname_grd = prefix + ".grd.h5"

    # check params_corr
    if params_corr is None:
        params_corr = dict()
    else:
        kList = []
        for k, v in params_corr.items():
            assert isinstance(k, str)
            if isinstance(v, dict):
                # J1_params: per-element dict
                for sk in v:
                    assert isinstance(sk, str)
                if len(v) == 0:
                    kList.append(k)
            else:
                assert isinstance(v, jnp.ndarray)
                if k == "J2_params" and params_corr[k].shape[0] < 2:
                    jax.debug.print(
                        f"⚠️ WARNING! Correlation parameter set \"{k}\" "
                        "requires 2 elements, but the user provided fewer.  "
                        "Deleting..."
                        )
                    kList.append(k)
        for k in kList:
            del params_corr[k]

    nuc_crds = jnp.array(mf.mol.atom_coords(unit='Bohr'))
    eps = jnp.finfo(nuc_crds.dtype).eps     # softwired epsilon
    nelec = mf.mol.tot_electrons()
    num_nuc = mf.mol.natm
    Z_charges = mf.mol.atom_charges()
    mol_charge = mf.mol.charge

    # Precompute electron pair indices for distance calculations
    i_e, j_e = jnp.triu_indices(nelec, k=1)

    # Precompute per-fragment data for Metropolis moves
    frag_reflect_data \
        = _get_electron_displacement_fn(Z_charges, nuc_crds, cluster_idx,
                                        mol=mf.mol)
    frag_centroids, frag_inradii, frag_Vh, frag_is_planar \
        = frag_reflect_data

    # atomic_masses = mf.mol.atom_mass_list()
    # mass_center = jnp.einsum('i,ij->j',
    #                          atomic_masses, nuc_crds)/atomic_masses.sum()
    # relative_nuc_pos = nuc_crds - mass_center
    timestamp_init = datetime.now()
    print("Begin time: {}".format(timestamp_init))

    if cusp_scheme == "Quady2025":
        params_cusp = {}
        for i in range(num_nuc):
            atom_symbol = mf.mol.atom_symbol(i)
            if atom_symbol not in params_cusp:
                if isinstance(mf.mol.basis, str):
                    p = get_cusp_params(atom_symbol, mf.mol.basis)
                else:
                    p = get_cusp_params(atom_symbol, mf.mol.basis[atom_symbol])
                params_cusp[atom_symbol] = p[atom_symbol]
    else:
        params_cusp = None

    # Get wavefunction and energy functions
    log_trial_wavefunction, local_energy, get_psi_mo \
        = get_psi_fun(mf, params_cusp=params_cusp)

    # Precompute nuclear-nuclear energy and gradient
    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke \
        = local_energy
    enr_nn = local_energy_nn(nuc_crds)
    grd_nn = jax.grad(local_energy_nn)(nuc_crds)

    # --- Gradient sample redistribution schemes for space warping ---
    @jax.jit
    def redistribute_scheme1(elec_crds: jnp.ndarray) -> jnp.ndarray:
        _, mo_val_s = get_psi_mo(elec_crds, nuc_crds)
        weight = jnp.einsum('neo,neo->en', mo_val_s, mo_val_s)  # **(1.0/4)
        return weight / jnp.sum(weight, axis=-1, keepdims=True)

    @jax.jit
    def redistribute_scheme2(elec_crds: jnp.ndarray) -> jnp.ndarray:
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
        dist = jnp.where(dist < eps, eps, dist)
        weight = dist**(-4.0)
        return weight / jnp.sum(weight, axis=-1, keepdims=True)

    rescale_fn = redistribute_scheme2 \
        if 'scheme2' in gr_scheme \
        else redistribute_scheme1
    jac_rescale_fn = jax.jacobian(rescale_fn, argnums=0)

    @jax.jit
    def grad_fn_ee(e_pos: jnp.ndarray) -> jnp.ndarray:
        return jax.grad(local_energy_ee)(e_pos)

    @jax.jit
    def grad_fn_en(e_pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.grad(local_energy_en, argnums=(0, 1))(e_pos, nuc_crds)

    @jax.jit
    def grad_fn_ke(e_pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.grad(local_energy_ke,
                        argnums=(0, 1))(e_pos, nuc_crds, params_corr)

    @jax.jit
    def grad_fn_logpsi(e_pos: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.grad(log_trial_wavefunction,
                        argnums=(0, 1))(e_pos, nuc_crds, params_corr)

    @jax.jit
    def _log_psi_batch(batch):
        """Evaluate log|ψ| for a batch of walker configurations."""
        return jax.vmap(
            lambda x: log_trial_wavefunction(x, nuc_crds, params_corr)
        )(batch)

    # @jax.jit
    # def total_local_energy_fn(elec_crds):
    #     return (local_energy_ee(elec_crds)
    #             + local_energy_en(elec_crds, nuc_crds)
    #             + local_energy_ke(elec_crds, nuc_crds, params_corr)
    #             + enr_nn)

    def metropolis_move_alle(rng_key: jax.Array,
                             elec_crds: jnp.ndarray,
                             step_size: float,
                             frag_idx: int = -1) \
            -> tuple[jnp.ndarray, float]:
        """Gaussian proposal. Returns (proposed_crds, proposal_ratio).
        `frag_idx` unused (present for lax.switch signature compat)."""
        key_displace, _ = jax.random.split(rng_key)

        proposed_crds = elec_crds \
            + step_size * jax.random.normal(key_displace, elec_crds.shape)

        # Check for singularities (electron-electron and electron-nuclei)
        diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
        dists_ee = jnp.linalg.norm(diffs_ee, axis=-1)

        diffs_en = proposed_crds[:, None, :] - nuc_crds[None, :, :]
        dists_en = jnp.linalg.norm(diffs_en, axis=-1)

        # Whether this is a valid move
        is_invalid_move = (dists_en.min() < MIN_DIST_THRESHOLD) \
            | (dists_ee.min() < MIN_DIST_THRESHOLD)
        return jax.lax.cond(is_invalid_move,
                            lambda: (proposed_crds, 0.0),
                            lambda: (proposed_crds, 1.0))

    def metropolis_fragment_reflection_z(rng_key: jax.Array,
                                         elec_crds: jnp.ndarray,
                                         step_size: float,
                                         frag_idx: int) \
            -> tuple[jnp.ndarray, float]:
        """Fragment-level planar reflection proposal.

        Reflects all electrons within frag_idx's inradius through
        the fragment's molecular plane. `rng_key` and `step_size`
        unused (present for lax.switch signature compat).
        """
        Vh = frag_Vh[frag_idx]               # (3, 3)
        centroid = frag_centroids[frag_idx]   # (3,)
        inradius = frag_inradii[frag_idx]     # scalar
        is_planar = frag_is_planar[frag_idx]  # bool

        elec_centered = elec_crds - centroid
        elec_rotated = elec_centered @ Vh.T
        elec_reflected = apply_reflection_z(elec_rotated)
        elec_proposed = elec_reflected @ Vh + centroid

        dist_from_centroid = jnp.linalg.norm(elec_centered, axis=-1)
        mask = is_planar & (dist_from_centroid <= inradius)
        proposed_crds = jnp.where(mask[:, None],
                                  elec_proposed, elec_crds)

        return proposed_crds, 1.0

    def metropolis_fragment_reflection_x(rng_key: jax.Array,
                                         elec_crds: jnp.ndarray,
                                         step_size: float,
                                         frag_idx: int) \
            -> tuple[jnp.ndarray, float]:
        """Fragment-level x-reflection proposal.

        Reflects all electrons within frag_idx's inradius through
        the fragment's yz-plane. `rng_key` and `step_size`
        unused (present for lax.switch signature compat).
        """
        Vh = frag_Vh[frag_idx]               # (3, 3)
        centroid = frag_centroids[frag_idx]   # (3,)
        inradius = frag_inradii[frag_idx]     # scalar
        is_planar = frag_is_planar[frag_idx]  # bool

        elec_centered = elec_crds - centroid
        elec_rotated = elec_centered @ Vh.T
        elec_reflected = apply_reflection_x(elec_rotated)
        elec_proposed = elec_reflected @ Vh + centroid

        dist_from_centroid = jnp.linalg.norm(elec_centered, axis=-1)
        mask = is_planar & (dist_from_centroid <= inradius)
        proposed_crds = jnp.where(mask[:, None],
                                  elec_proposed, elec_crds)

        return proposed_crds, 1.0

    def metropolis_fragment_reflection_y(rng_key: jax.Array,
                                         elec_crds: jnp.ndarray,
                                         step_size: float,
                                         frag_idx: int) \
            -> tuple[jnp.ndarray, float]:
        """Fragment-level y-reflection proposal.

        Reflects all electrons within frag_idx's inradius through
        the fragment's xz-plane. `rng_key` and `step_size`
        unused (present for lax.switch signature compat).
        """
        Vh = frag_Vh[frag_idx]               # (3, 3)
        centroid = frag_centroids[frag_idx]   # (3,)
        inradius = frag_inradii[frag_idx]     # scalar
        is_planar = frag_is_planar[frag_idx]  # bool

        elec_centered = elec_crds - centroid
        elec_rotated = elec_centered @ Vh.T
        elec_reflected = apply_reflection_y(elec_rotated)
        elec_proposed = elec_reflected @ Vh + centroid

        dist_from_centroid = jnp.linalg.norm(elec_centered, axis=-1)
        mask = is_planar & (dist_from_centroid <= inradius)
        proposed_crds = jnp.where(mask[:, None],
                                  elec_proposed, elec_crds)

        return proposed_crds, 1.0

    def metropolis_fragment_rotation_x180(rng_key: jax.Array,
                                          elec_crds: jnp.ndarray,
                                          step_size: float,
                                          frag_idx: int) \
            -> tuple[jnp.ndarray, float]:
        """Fragment-level 180-degree rotation about x-axis proposal.

        Rotates all electrons within frag_idx's inradius by 180 degrees
        about the fragment's first principal axis. `rng_key` and
        `step_size` unused (present for lax.switch signature compat).
        """
        Vh = frag_Vh[frag_idx]               # (3, 3)
        centroid = frag_centroids[frag_idx]   # (3,)
        inradius = frag_inradii[frag_idx]     # scalar
        is_planar = frag_is_planar[frag_idx]  # bool

        elec_centered = elec_crds - centroid
        elec_rotated = elec_centered @ Vh.T
        elec_operated = apply_rotation_x180(elec_rotated)
        elec_proposed = elec_operated @ Vh + centroid

        dist_from_centroid = jnp.linalg.norm(elec_centered, axis=-1)
        mask = is_planar & (dist_from_centroid <= inradius)
        proposed_crds = jnp.where(mask[:, None],
                                  elec_proposed, elec_crds)

        return proposed_crds, 1.0

    def metropolis_fragment_rotation_y180(rng_key: jax.Array,
                                          elec_crds: jnp.ndarray,
                                          step_size: float,
                                          frag_idx: int) \
            -> tuple[jnp.ndarray, float]:
        """Fragment-level 180-degree rotation about y-axis proposal.

        Rotates all electrons within frag_idx's inradius by 180 degrees
        about the fragment's second principal axis. `rng_key` and
        `step_size` unused (present for lax.switch signature compat).
        """
        Vh = frag_Vh[frag_idx]               # (3, 3)
        centroid = frag_centroids[frag_idx]   # (3,)
        inradius = frag_inradii[frag_idx]     # scalar
        is_planar = frag_is_planar[frag_idx]  # bool

        elec_centered = elec_crds - centroid
        elec_rotated = elec_centered @ Vh.T
        elec_operated = apply_rotation_y180(elec_rotated)
        elec_proposed = elec_operated @ Vh + centroid

        dist_from_centroid = jnp.linalg.norm(elec_centered, axis=-1)
        mask = is_planar & (dist_from_centroid <= inradius)
        proposed_crds = jnp.where(mask[:, None],
                                  elec_proposed, elec_crds)

        return proposed_crds, 1.0

    def metropolis_fragment_rotation_z180(rng_key: jax.Array,
                                          elec_crds: jnp.ndarray,
                                          step_size: float,
                                          frag_idx: int) \
            -> tuple[jnp.ndarray, float]:
        """Fragment-level 180-degree rotation about z-axis proposal.

        Rotates all electrons within frag_idx's inradius by 180 degrees
        about the fragment's normal axis. `rng_key` and `step_size`
        unused (present for lax.switch signature compat).
        """
        Vh = frag_Vh[frag_idx]               # (3, 3)
        centroid = frag_centroids[frag_idx]   # (3,)
        inradius = frag_inradii[frag_idx]     # scalar
        is_planar = frag_is_planar[frag_idx]  # bool

        elec_centered = elec_crds - centroid
        elec_rotated = elec_centered @ Vh.T
        elec_operated = apply_rotation_z180(elec_rotated)
        elec_proposed = elec_operated @ Vh + centroid

        dist_from_centroid = jnp.linalg.norm(elec_centered, axis=-1)
        mask = is_planar & (dist_from_centroid <= inradius)
        proposed_crds = jnp.where(mask[:, None],
                                  elec_proposed, elec_crds)

        return proposed_crds, 1.0

    def metropolis_fragment_noop(rng_key: jax.Array,
                                 elec_crds: jnp.ndarray,
                                 step_size: float,
                                 frag_idx: int) \
            -> tuple[jnp.ndarray, float]:
        """No-op move for non-planar fragments."""
        return elec_crds, 1.0

    @jax.jit
    def metropolis_move_1w(rng_key: jax.Array,
                           elec_crds: jnp.ndarray,
                           _step_size: float,
                           step_count: int) \
            -> tuple[jnp.ndarray, bool, bool]:
        """Single-walker Metropolis step with per-fragment branching.

        1. Pick a random electron and find its fragment (Voronoi).
        2. Per-fragment alternation:
             planar    -> deterministic (step_count % 7)
             non-planar -> random
        3. Eight-way dispatch:
             0 = Gaussian, 1 = z-reflection, 2 = x-reflection,
             3 = y-reflection, 4 = C2x rotation, 5 = C2y rotation,
             6 = C2z rotation, 7 = no-op
        """
        key_elec, key_disp, key_prop, key_accept \
            = jax.random.split(rng_key, 4)

        # # 1. Pick random electron, find its fragment
        # elec_idx = jax.random.randint(key_elec, (), 0, nelec)
        # elec_pos = elec_crds[elec_idx]  # (3,)
        # dists = jnp.linalg.norm(
        #     frag_centroids - elec_pos[None, :], axis=-1)
        # frag_idx = jnp.argmin(dists)
        # is_planar = frag_is_planar[frag_idx]

        # # 2. Per-fragment alternation type
        # displacement_idx = jnp.where(
        #     is_planar,
        #     step_count % 7,
        #     jax.random.randint(key_disp, (), 0, 2))

        # # 3. Eight-way dispatch
        # switch_idx = jnp.where(
        #     displacement_idx == 0, 0,
        #     jnp.where(~is_planar, 7,         # noop for non-planar
        #               displacement_idx))      # 1-6 for planar

        # proposed_crds, proposal_ratio = jax.lax.switch(
        #     switch_idx,
        #     [metropolis_move_alle,                  # 0: Gaussian
        #      metropolis_fragment_reflection_z,       # 1: z-reflection
        #      metropolis_fragment_reflection_x,       # 2: x-reflection
        #      metropolis_fragment_reflection_y,       # 3: y-reflection
        #      metropolis_fragment_rotation_x180,      # 4: C2x rotation
        #      metropolis_fragment_rotation_y180,      # 5: C2y rotation
        #      metropolis_fragment_rotation_z180,      # 6: C2z rotation
        #      metropolis_fragment_noop],              # 7: no-op
        #     key_prop, elec_crds, _step_size, frag_idx)
        # 1.-3. XXX: Disable per-fragment displacements for now.
        displacement_idx = 0
        proposed_crds, proposal_ratio \
            = metropolis_move_alle(key_prop, elec_crds, _step_size)

        # 4. Metropolis accept/reject
        log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds,
                                             params_corr)
        log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds,
                                             params_corr)

        accept = jax.random.uniform(key_accept) \
            < jnp.exp(2.0 * (log_psi_new - log_psi_old)) \
            * proposal_ratio
        new_crds = jnp.where(accept, proposed_crds, elec_crds)

        is_gaussian = (displacement_idx == 0)
        return new_crds, accept, is_gaussian

    metropolis_move_allw = jax.vmap(metropolis_move_1w,
                                    in_axes=(0, 0, None, None))

    # --- Gradient batch computation ---
    @jax.jit
    def vmc_gradient_batch(batch_samples: jnp.ndarray) \
            -> tuple[jnp.ndarray, ...]:
        grd_ee_elc = jax.vmap(grad_fn_ee)(batch_samples)
        grd_en_elc, grd_en_nuc = jax.vmap(grad_fn_en)(batch_samples)
        grd_ke_elc, grd_ke_nuc = jax.vmap(grad_fn_ke)(batch_samples)
        grd_logpsi_elc, grd_logpsi_nuc \
            = jax.vmap(grad_fn_logpsi)(batch_samples)

        rescale = jax.vmap(rescale_fn)(batch_samples)
        jac_rescale_elc = jax.vmap(jac_rescale_fn)(batch_samples)
        novel_correction = 0.5 * jnp.einsum('beneK->bnK', jac_rescale_elc)

        grd_ee = jnp.einsum('beK,ben->bnK', grd_ee_elc, rescale)
        grd_en = grd_en_nuc + jnp.einsum('beK,ben->bnK', grd_en_elc, rescale)
        grd_ke = grd_ke_nuc + jnp.einsum('beK,ben->bnK', grd_ke_elc, rescale)

        grd_logpsi = grd_logpsi_nuc + jnp.einsum('beK,ben->bnK',
                                                 grd_logpsi_elc, rescale)
        grd_logpsi += novel_correction

        return grd_ee, grd_en, grd_ke, grd_logpsi

    # --- Per-fragment symmetry operation for gradient batches ---
    def _apply_frag_symmop(batch_samples, s_op_fn, s_op_name):
        """Apply symmetry operation per-fragment.

        For each fragment that has `s_op_name` in its allowed operations:
        translate to centroid, rotate to principal axes, apply operation,
        rotate back, translate back. Only electrons within the fragment's
        inradius are transformed.

        Uses the original (unmodified) electron positions for all
        fragments to avoid cross-fragment contamination when a
        reflection moves an electron near another fragment's in-sphere.
        """
        num_frags = frag_centroids.shape[0]
        result = batch_samples  # (batch_size, nelec, 3)

        for fid in range(num_frags):
            if s_op_name not in frag_ops_sets[fid]:
                continue

            centroid = frag_centroids[fid]       # (3,)
            Vh = frag_Vh[fid]                    # (3, 3)
            inradius = frag_inradii[fid]         # scalar

            centered = batch_samples - centroid  # (batch, nelec, 3)
            rotated = centered @ Vh.T            # (batch, nelec, 3)
            operated = s_op_fn(rotated)          # (batch, nelec, 3)
            proposed = operated @ Vh + centroid   # (batch, nelec, 3)

            dist = jnp.linalg.norm(centered, axis=-1)   # (batch, nelec)
            mask = dist <= inradius                     # (batch, nelec)
            result = jnp.where(mask[:, :, None], proposed, result)

        return result

    def _apply_single_frag_symmop(batch_samples, frag_pos, s_op_fn):
        """Apply a symmetry operation to a single fragment only.

        Args:
            batch_samples: (batch_size, nelec, 3)
            frag_pos: fragment array index (position in frag_ids)
            s_op_fn: JAX function implementing the symmetry operation

        Returns:
            Transformed coordinates with only the target fragment modified.
        """
        centroid = frag_centroids[frag_pos]
        Vh = frag_Vh[frag_pos]
        inradius = frag_inradii[frag_pos]

        centered = batch_samples - centroid
        rotated = centered @ Vh.T
        operated = s_op_fn(rotated)
        proposed = operated @ Vh + centroid

        dist = jnp.linalg.norm(centered, axis=-1)
        mask = dist <= inradius
        return jnp.where(mask[:, :, None], proposed, batch_samples)

    # --- Gradient saving ---
    def vmc_gradient_save(block_cnt: int,
                          sampled_walkers: jnp.ndarray,
                          local_energies: jnp.ndarray,
                          batch_size: int, num_batches: int):
        # sampled_walkers enter flattened along the first two axes.
        # ie. num_samples_per_block == num_steps_per_block * num_walkers
        # sampled_walkers.shape
        #   == (num_steps_per_block * num_walkers, nelec, 3)
        # local_energies.shape == (num_steps_per_block, num_walkers)
        num_samples_per_block = sampled_walkers.shape[0]

        # Reference gradient accumulators
        w_grd_ee_en = []
        w_grd_ke = []
        w_grd_logpsi = []

        # Per-combo accumulators
        combo_grd_ee_en = {label: [] for _, _, label in single_frag_combos}
        combo_grd_ke = {label: [] for _, _, label in single_frag_combos}
        combo_grd_logpsi = {label: [] for _, _, label in single_frag_combos}
        combo_weights = {label: [] for _, _, label in single_frag_combos}

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples_per_block)
            batch_orig = sampled_walkers[start_idx:end_idx, :, :]

            # Reference: gradients at original (untransformed) positions
            g_ee, g_en, g_ke, g_logpsi = vmc_gradient_batch(batch_orig)
            w_grd_ee_en.append(g_ee + g_en)
            w_grd_ke.append(g_ke)
            w_grd_logpsi.append(g_logpsi)

            # Evaluate log|ψ| at original positions (once per batch)
            log_psi_orig = _log_psi_batch(batch_orig)  # (batch_size,)

            # Single-fragment operation combos
            for frag_pos, op, label in single_frag_combos:
                batch_trans = _apply_single_frag_symmop(
                    batch_orig, frag_pos,
                    symmetry_operations_map[op])

                # Screen: fall back to original where |ψ|² drops
                log_psi_trans = _log_psi_batch(batch_trans)
                psi2_ratio = jnp.exp(
                    2.0 * (log_psi_trans - log_psi_orig))
                safe = psi2_ratio > PSI2_RATIO_THRESHOLD
                batch_trans = jnp.where(
                    safe[:, None, None], batch_trans, batch_orig)

                # Weight: J * |ψ(r')|² / |ψ(r)|²
                # J = 1 for orthogonal point group operations
                weight = jnp.where(safe, psi2_ratio, 1.0)

                g_ee, g_en, g_ke, g_logpsi \
                    = vmc_gradient_batch(batch_trans)
                combo_grd_ee_en[label].append(g_ee + g_en)
                combo_grd_ke[label].append(g_ke)
                combo_grd_logpsi[label].append(g_logpsi)
                combo_weights[label].append(weight)

        # Stack all batches
        w_grd_ee_en = jnp.vstack(w_grd_ee_en)
        w_grd_ke = jnp.vstack(w_grd_ke)
        w_grd_logpsi = jnp.vstack(w_grd_logpsi)

        # Save to HDF5
        with h5py.File(ofname_grd, "a") as f:
            block_cnt_str = f'{block_cnt}'

            # Ensure top-level groups exist
            grp_names = ['grd_ee_en', 'grd_ke', 'grd_logpsi',
                         'local_energies']
            if single_frag_combos:
                grp_names.append('fragment_weights')
            for k in grp_names:
                if k not in f.keys():
                    f.create_group(k)

            # Clean up existing block data (restart case)
            if block_cnt_str in f['grd_ee_en'].keys():
                del f['grd_ee_en'][block_cnt_str]
                del f['grd_ke'][block_cnt_str]
                del f['grd_logpsi'][block_cnt_str]
                del f['local_energies'][block_cnt_str]
                for _, _, label in single_frag_combos:
                    if label in f['grd_ee_en'] \
                            and block_cnt_str in f['grd_ee_en'][label]:
                        del f['grd_ee_en'][label][block_cnt_str]
                        del f['grd_ke'][label][block_cnt_str]
                        del f['grd_logpsi'][label][block_cnt_str]
                        del f['fragment_weights'][label][block_cnt_str]

            # A. Reference gradients
            f['grd_ee_en'].create_dataset(block_cnt_str,
                                          data=w_grd_ee_en)
            f['grd_ke'].create_dataset(block_cnt_str, data=w_grd_ke)
            f['grd_logpsi'].create_dataset(block_cnt_str,
                                           data=w_grd_logpsi)
            f['local_energies'].create_dataset(block_cnt_str,
                                               data=local_energies)

            # B. Per-combo secondary gradients and weights
            for _, _, label in single_frag_combos:
                c_ee_en = jnp.vstack(combo_grd_ee_en[label])
                c_ke = jnp.vstack(combo_grd_ke[label])
                c_logpsi = jnp.vstack(combo_grd_logpsi[label])
                c_w = jnp.concatenate(combo_weights[label])

                for grp, data in [('grd_ee_en', c_ee_en),
                                  ('grd_ke', c_ke),
                                  ('grd_logpsi', c_logpsi)]:
                    if label not in f[grp]:
                        f[grp].create_group(label)
                    f[grp][label].create_dataset(block_cnt_str,
                                                 data=data)

                if label not in f['fragment_weights']:
                    f['fragment_weights'].create_group(label)
                f['fragment_weights'][label].create_dataset(
                    block_cnt_str, data=c_w)

    # --- Main VMC run ---
    def vmc_run(rng_key: int | jnp.ndarray,
                num_walkers: int = 1000,
                num_steps_per_block: int = 100, num_steps_decorr: int = 1,
                num_blocks: int = 1000, num_blocks_equil: int = 100,
                mc_timestep: float = 0.1,
                fname_log: str = None,
                mode_restart: bool = False,
                compute_gradients: bool = False) -> None:
        """VMC run with better memory management."""
        # tolerance_enr_std_per_elec=CHEMICAL_ACCURACY,
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        rng_key_to_restart = rng_key.copy()

        # Initialize electron positions more efficiently
        rng_key, init_key = jax.random.split(rng_key)
        walkers = _initialize_walkers(init_key,
                                      num_walkers, nelec,
                                      Z_charges, nuc_crds, mol_charge)
        mc_stepsize = (3 * mc_timestep)**0.5

        # Equilibration phase
        @jax.jit
        def equilibration_step(state, step_idx):
            rng_key, walkers, step_size = state
            rng_key, key = jax.random.split(rng_key)
            walker_keys = jax.random.split(key, num_walkers)
            new_walkers, accepted, is_gaussian \
                = metropolis_move_allw(walker_keys, walkers, step_size,
                                       step_idx)

            # Only use Gaussian acceptance for step size adaptation
            n_gauss = jnp.sum(is_gaussian)
            gauss_accept_sum = jnp.where(is_gaussian, accepted, 0.0).sum()
            gauss_ratio = jnp.where(
                n_gauss > 0,
                gauss_accept_sum / n_gauss,
                TARGET_ACCEPTANCE_RATE)
            new_step_size = _adapt_step_size(step_size, gauss_ratio)

            return (rng_key, new_walkers, new_step_size), gauss_ratio
        # walkers.shape == (num_walkers, nelec, 3)

        if mode_restart:
            with h5py.File(ofname_chkpt, 'r') as f:
                # Load metadata
                block_cnt_start = int(f['block_count'][()])
                mc_stepsize = f['mc_stepsize'][()]
                mc_timestep = mc_stepsize * mc_stepsize / 3
                rng_key = jax.random.key(int(f['rng_key'][()]))
                rng_key_to_restart = rng_key.copy()
                rng_key, init_key = jax.random.split(rng_key)
                walkers = jnp.array(f['walkers'][:])
                E_b = list(f['E_blocks'][:])
                print("Restarting ...")

            # for _ in range(num_blocks_equil):
            #     initial_state = (rng_key, walkers, mc_stepsize)
            #     final_state, ratios = jax.lax.scan(equilibration_step,
            #                                        initial_state,
            #                                        jnp.arange(num_steps_per_block))
            #     rng_key, walkers, mc_stepsize = final_state
        else:
            for _ in range(num_blocks_equil):
                initial_state = (rng_key, walkers, mc_stepsize)
                final_state, ratios \
                    = jax.lax.scan(equilibration_step,
                                   initial_state,
                                   jnp.arange(num_steps_per_block))
                rng_key, walkers, mc_stepsize = final_state
            block_cnt_start = 1
            mc_timestep = mc_stepsize * mc_stepsize / 3
            E_b = []
            # std_E_b = []

        ratio = ratios[-1]

        print(f"ℹ️ Equilibration acceptance rate: {ratio:.2f}")
        print(f"ℹ️ Adjusted step size: {mc_stepsize:.4f} bohr "
              f"~ {mc_timestep:.4f} Ha⁻¹ in Brownian time")

        # Production phase
        @jax.jit
        def production_step(state, step_number):
            rng_key, walkers, step_size = state

            for d in range(num_steps_decorr):
                rng_key, key_displace = jax.random.split(rng_key)
                walker_keys = jax.random.split(key_displace, num_walkers)
                step_count = step_number * num_steps_decorr + d
                new_walkers, accepted, _is_gaussian \
                    = metropolis_move_allw(walker_keys, walkers, step_size,
                                           step_count)
                walkers = new_walkers

            ratio = accepted.mean()
            # new_step_size = step_size * (0.5 + ratio)

            # calculate energy
            # energies = jax.vmap(total_local_energy_fn)(new_walkers)
            enr_ee = jax.vmap(local_energy_ee)(new_walkers)
            enr_en = jax.vmap(local_energy_en,
                              in_axes=(0, None))(new_walkers,
                                                 nuc_crds)
            enr_ke = jax.vmap(local_energy_ke,
                              in_axes=(0, None, None))(new_walkers,
                                                       nuc_crds,
                                                       params_corr)

            return (rng_key, walkers, step_size), \
                (ratio, enr_ee, enr_en, enr_ke, new_walkers)
        # walkers.shape == (num_walkers, nelec, 3)
        # ratios.shape == (num_steps_per_block,)
        # energies.shape == (num_steps_per_block, num_walkers)

        if fname_log is None \
                or (isinstance(fname_log, str) and fname_log == ""):
            fout = sys.stdout
        else:
            fout = open(fname_log, 'w', 1)

        # could turn this into a one-liner if requirement becomes Python 3.8+
        p = pathlib.Path(ofname_chkpt)
        if p.exists():
            p.unlink()
        with h5py.File(ofname_chkpt, 'a') as f:
            f.create_dataset("version", data="0.1.0")
            # TODO: take versionm info from pyproject.toml using tomli/tomllib

            g = f.create_group("timestamps")
            g.create_dataset("start",
                             data=str(timestamp_init))

            g = f.create_group("system")
            asym = [mf.mol.atom_symbol(i) for i in range(num_nuc)]
            g.create_dataset("atom_symbols", data=" ".join(asym))
            g.create_dataset("atom_coords", data=mf.mol.atom_coords())
            aobs = [mf.mol.basis] * num_nuc \
                if isinstance(mf.mol.basis, str) \
                else [mf.mol.basis[asym[i]] for i in range(num_nuc)]
            # TODO: add pseudopotentials ~"atom_pp" here as needed
            g.create_dataset("ao_basis", data=" ".join(aobs))
            g.create_dataset("units", data=mf.mol.unit.upper())

        if compute_gradients:
            p = pathlib.Path(ofname_grd)
            if p.exists():
                p.unlink()
            with h5py.File(ofname_grd, 'a') as f:
                f.create_dataset('grd_nn', data=grd_nn)
                g = f.create_group("system")
                asym = [mf.mol.atom_symbol(i) for i in range(num_nuc)]
                g.create_dataset("atom_symbols", data=" ".join(asym))
                g.create_dataset("atom_coords", data=mf.mol.atom_coords())
                g.create_dataset("units", data=mf.mol.unit.upper())

        base_batch_size = 500
        memory_factor = max(1, nelec * num_nuc // 1000)
        batch_size = min(50, base_batch_size // memory_factor)
        num_batches = (num_steps_per_block * num_walkers + batch_size - 1) \
            // batch_size
        # mark_samples = ((jnp.arange(num_steps_per_block)+1) == 0)
        print("ℹ️ Adjusted batch size, number of batches: "
              f"{batch_size}, {num_batches}")
        print("# block_cnt        E_loc_mean      E_loc_std"
              "       eePotential     enPotential     Kinetic"
              "          ∆t_block",
              file=fout)

        timestamp_prev = datetime.now()
        # Main sampling phase with pre-allocated arrays
        for block_cnt in range(block_cnt_start,
                               block_cnt_start+num_blocks):
            initial_state = (rng_key, walkers, mc_stepsize)
            final_state, result \
                = jax.lax.scan(production_step,
                               initial_state,
                               jnp.arange(num_steps_per_block))

            rng_key, walkers, _ = final_state
            ratios, enr_ee_sw, enr_en_sw, enr_ke_sw, sampled_walkers = result
            E_loc_sw = enr_ee_sw + enr_en_sw + enr_ke_sw + enr_nn
            # sampled_walkers.shape
            #   == (num_steps_per_block, num_walkers, nelec, 3)
            # E_loc_sw.shape == (num_steps_per_block, num_walkers)

            # mean over walkers
            enr_ee_s = enr_ee_sw.mean(axis=1)
            enr_en_s = enr_en_sw.mean(axis=1)
            enr_ke_s = enr_ke_sw.mean(axis=1)
            E_loc_s = enr_ee_s + enr_en_s + enr_ke_s + enr_nn

            # mean over steps in block
            enr_ee = enr_ee_s.mean()
            enr_en = enr_en_s.mean()
            enr_ke = enr_ke_s.mean()

            E_mean = E_loc_s.mean()             # mean over steps
            std_E_s = E_loc_s.std()             # std over steps

            E_b.append(E_mean)              # append to block data
            # std_E_b.append(std_E_s)

            timestamp_curr = datetime.now()
            tdelta_block = (timestamp_curr - timestamp_prev).total_seconds()
            print(f"{block_cnt:>8d}{E_mean:>24.8e}{std_E_s:>16.8e}"
                  f"{enr_ee:>16.8e}{enr_en:>16.8e}{enr_ke:>16.8e}"
                  f"{tdelta_block:>16.6f}",
                  file=fout)

            if compute_gradients:
                vmc_gradient_save(block_cnt,
                                  sampled_walkers.reshape(-1, nelec, 3),
                                  E_loc_sw,
                                  batch_size, num_batches)

            # if std_E_s < tolerance_enr_std_per_elec * nelec:
            #     break

            timestamp_prev = timestamp_curr

        if not (fname_log is None
                or (isinstance(fname_log, str) and fname_log == "")):
            fout.close()

        timestamp_fin = datetime.now()
        print("End time: {}\t({:.6f} seconds total)"
              .format(timestamp_fin,
                      (timestamp_fin-timestamp_init).total_seconds()))

        with h5py.File(ofname_chkpt, 'a') as f:
            f.create_dataset('E_blocks', data=E_b)
            f.create_dataset('rng_key',
                             data=jax.random.key_data(rng_key_to_restart))
            f.create_dataset('block_count', data=block_cnt,
                             dtype=jnp.int32)
            f.create_dataset('walkers', data=sampled_walkers[-1, :, :, :])
            f["timestamps"].create_dataset("end", data=str(timestamp_fin))

    return vmc_run
