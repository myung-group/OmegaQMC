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
from jax.sharding import NamedSharding, PartitionSpec

from .psi.gto import (
    get_psi_fun, _sanitize_J3_eeI_params, compute_orbital_response,
)
from .psi.cusp import get_cusp_params
from .utils import (parse_molecular_inspheres,
                    Mole_custom,
                    _length_in_au,
                    do_binning_analysis,
                    _make_sharding,
                    _autotune_prod_walkers)
# from .symm.water_rotation_matrix import symmetrize_water_molecule
from .symm.operations import populate_fragment_symmops
from .symm.fragments import (
    build_frag_transform_data,
    build_frag_symmops,
    build_single_frag_combos,
    make_apply_single_frag_symmop,
)
from .observables.force import (
    vmc_gto_gradients,
    _build_param_response_fns,
    save_gto_gradients,
)
from .symm.point_groups import (auto_symmetrize_molecule,
                                detect_symmetry_quality)
# from .constants import CHEMICAL_ACCURACY
from .constants import MIN_DIST_THRESHOLD

# VMC hyperparameters
TARGET_ACCEPTANCE_RATE = 0.4
STEP_SIZE_ADAPTATION_RATE = 0.05

# Initial basin-thermalization (charge-transfer ergodicity).
# The local Gaussian move cannot carry an electron across the
# low-density gap between well-separated fragments, so without
# this step every walker is stuck in the electron-to-nucleus
# partition it was initialized in.  A single-electron move that
# relocates one electron to a Gaussian blob around a randomly
# chosen nucleus restores ergodicity over charge-transfer
# (ionic / covalent) basins; see ``_VMCDriverGTO`` __init__.
BASIN_RELOC_SIGMA = 1.5          # bohr; nucleus-blob proposal width
N_BASIN_THERMAL_STEPS = 800      # single-electron relocation steps


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
    """Run a PySCF mean-field calculation and return the result object.

    Wraps PySCF to build a molecule, detect its point-group symmetry,
    run a restricted Hartree-Fock (or post-HF) calculation, and symmetrize
    the resulting molecular orbitals.  The returned object is passed directly
    to :func:`get_vmc_gto_func` and :func:`get_vmcopt_gto_func`.

    Parameters
    ----------
    astr : str
        Atom specification accepted by PySCF: an inline string such as
        ``"H 0 0 0; H 0 0 1.4"`` **or** a path to an ``.xyz`` file.
    unit : str, optional
        Deprecated alias for *units*.  If provided it takes precedence.
    units : str, optional
        Length unit of *astr* coordinates.  Accepts ``"Bohr"`` / ``"au"``
        (default) or ``"angstrom"`` / ``"ang"``.  ``.xyz`` files are always
        in ångströms regardless of this setting.
    spin : int, optional
        Number of unpaired electrons (2S).  Default is 0 (singlet).
    basis : str or dict, optional
        Basis-set name (e.g. ``"aug-cc-pVTZ"``) or an element-keyed dict for
        mixed basis sets.  Default is ``"aug-cc-pVTZ"``.
    postHF : str, optional
        Post-HF method to run on top of the RHF reference, e.g. ``"MP2"``,
        ``"CCSD"``.  If *None* (default) only RHF natural orbitals are used.
    ignore_hydrogen_mass : bool, optional
        Replace hydrogen masses with a small value so that H nuclei do not
        dominate center-of-mass translations.  Default is ``False``.
    symmetrization_level : int, optional
        How aggressively to symmetrize the MOs.  ``0`` = no symmetrization,
        ``1`` = symmetrize degenerate blocks (default), ``2`` = full
        projection onto irreducible representations.

    Returns
    -------
    mf : pyscf.scf.RHF
        Converged mean-field object with the molecule (``mf.mol``) and MO
        coefficients (``mf.mo_coeff``) attached.  Point-group information is
        stored on ``mf.mol`` via custom attributes set by this function.
    """
    if unit is not None:
        units = unit
        # "unit" (sic) takes precedence

    if astr.endswith(".xyz"):
        if units.upper().startswith("B") or units.upper().startswith("AU"):
            warnings.warn("XYZ input uses Å units by default, "
                          "but the user has specified Bohrs.")

        elif units is None or units == "":
            units = "angstroms"

    # mol = gto.M(atom=astr, basis=basis, unit=units)
    mol = Mole_custom()
    mol.build(atom=astr, basis=basis, unit=units)
    # see pyscf.gto.mole.is_au(unit)

    gpname = "C1"
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
    mol.symmetrization_level = symmetrization_level
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


# ---------------------------------------------------------------------------
# Module-level helpers (no JAX, called once during get_vmc_gto_func setup)
# ---------------------------------------------------------------------------

def _validate_params_corr(params_corr, mf) -> dict:
    """Validate and clean params_corr in-place; return it."""
    eps = jnp.finfo(jnp.array(mf.mol.atom_coords(unit='Bohr')).dtype).eps
    if params_corr is None:
        return dict()
    kList = []
    for k, v in params_corr.items():
        assert isinstance(k, str)
        if isinstance(v, dict):
            # J1_pade: per-element dict; J2_pade: "like"/"unlike" dict
            for sk in v:
                assert isinstance(sk, str)
            if len(v) == 0:
                kList.append(k)
        else:
            assert isinstance(v, jnp.ndarray)
            if k == "J2_pade" and params_corr[k].shape[0] < 2:
                warnings.warn(f"Correlation parameter set \"{k}\" "
                              "requires 2 elements, "
                              "but the user provided fewer.  Deleting...")
                kList.append(k)
    for k in kList:
        del params_corr[k]

    # Check J2 cusp coefficients
    if "J2_pade" in params_corr:
        j2 = params_corr["J2_pade"]
        if isinstance(j2, dict):
            if "like" in j2 and abs(float(j2["like"][0]) - 0.25) > eps:
                warnings.warn(
                    f"J2_pade['like'][0] = {float(j2['like'][0]):.8f}, "
                    "expected 0.25 (same-spin cusp condition)")
            if "unlike" in j2 and abs(float(j2["unlike"][0]) - 0.5) > eps:
                warnings.warn(
                    f"J2_pade['unlike'][0] = {float(j2['unlike'][0]):.8f}, "
                    "expected 0.5 (opposite-spin cusp condition)")

    # Validate B-spline params
    for bkey in ("J1_bspline", "J2_bspline"):
        if bkey not in params_corr:
            continue
        v = params_corr[bkey]
        if not isinstance(v, dict):
            raise TypeError(
                f"{bkey} must be a dict, got {type(v)}"
            )
        for sk, sv in v.items():
            if not hasattr(sv, 'shape') or sv.shape[0] < 2:
                raise ValueError(
                    f"{bkey}['{sk}'] must have >= 2 "
                    "coefficients"
                )

    # Validate and sanitize J3_eeI params
    if "J3_eeI" in params_corr:
        v = params_corr["J3_eeI"]
        if not isinstance(v, dict):
            raise TypeError(
                "J3_eeI params must be a dict, "
                f"got {type(v)}"
            )
        for sk in v:
            if not (sk.startswith("like+")
                    or sk.startswith("unlike+")):
                raise ValueError(
                    f"J3_eeI sub-key '{sk}' must "
                    "start with 'like+' or 'unlike+'"
                )
        params_corr["J3_eeI"] = (
            _sanitize_J3_eeI_params(v)
        )

    # Warn if both pade + bspline present for same body
    if "J1_pade" in params_corr \
            and "J1_bspline" in params_corr:
        warnings.warn(
            "Both J1_pade and J1_bspline are present; "
            "their contributions will be summed.")
    if "J2_pade" in params_corr \
            and "J2_bspline" in params_corr:
        warnings.warn(
            "Both J2_pade and J2_bspline are present; "
            "their contributions will be summed.")

    return params_corr


def _build_cusp_params(mf, cusp_scheme, num_nuc) -> dict | None:
    """Return cusp params dict for cusp_scheme='Quady2025', else None."""
    if cusp_scheme != "Quady2025":
        return None
    params_cusp = {}
    for i in range(num_nuc):
        atom_symbol = mf.mol.atom_symbol(i)
        if atom_symbol not in params_cusp:
            if isinstance(mf.mol.basis, str):
                p = get_cusp_params(atom_symbol, mf.mol.basis)
            else:
                p = get_cusp_params(atom_symbol, mf.mol.basis[atom_symbol])
            params_cusp[atom_symbol] = p[atom_symbol]
    return params_cusp


def _apply_fragment_op(elec_crds: jnp.ndarray,
                       centroid: jnp.ndarray,
                       Vh: jnp.ndarray,
                       inradius: float,
                       is_planar: bool,
                       op_fn) -> tuple[jnp.ndarray, float]:
    """Generic fragment-level symmetry operation proposal.

    Applies `op_fn` to all electrons within `inradius` of `centroid`
    in the fragment's principal-axis frame. `is_planar` gates the mask.
    Returns (proposed_crds, proposal_ratio=1.0).
    """
    elec_centered = elec_crds - centroid
    elec_rotated = elec_centered @ Vh.T
    elec_operated = op_fn(elec_rotated)
    elec_proposed = elec_operated @ Vh + centroid

    dist_from_centroid = jnp.linalg.norm(elec_centered, axis=-1)
    mask = is_planar & (dist_from_centroid <= inradius)
    proposed_crds = jnp.where(mask[:, None], elec_proposed, elec_crds)
    return proposed_crds, 1.0


# ---------------------------------------------------------------------------
# _VMCDriverGTO: holds all precompiled VMC kernels and runs the simulation
# ---------------------------------------------------------------------------

class _VMCDriverGTO:
    """Holds all precompiled VMC computation kernels
    and runs the simulation."""

    def __init__(self, mf, params_corr, params_cusp, mo_relax,
                 nuc_crds, frag_transform_data, single_frag_combos,
                 frag_symmops, frag_ops_sets, frag_ids,
                 ofname_chkpt, ofname_grd, timestamp_init,
                 gr_scheme='scheme1',
                 force_estimator='simple',
                 trial=None,
                 jastrow_config=None):
        # --- Store state ---
        self.mf = mf
        self.params_corr = params_corr
        self.mo_relax = mo_relax
        self.force_estimator = force_estimator
        self.nuc_crds = nuc_crds
        self.single_frag_combos = single_frag_combos
        self.frag_symmops = frag_symmops
        self.frag_ops_sets = frag_ops_sets
        self.frag_ids = frag_ids
        self.ofname_chkpt = ofname_chkpt
        self.ofname_grd = ofname_grd
        self.timestamp_init = timestamp_init

        # Unpack frag_transform_data
        frag_centroids, frag_inradii, frag_Vh, frag_is_planar \
            = frag_transform_data
        self.frag_centroids = frag_centroids
        self.frag_inradii = frag_inradii
        self.frag_Vh = frag_Vh
        self.frag_is_planar = frag_is_planar

        # Derived scalars
        nelec = mf.mol.tot_electrons()
        self.nelec = nelec
        self.Z_charges = mf.mol.atom_charges()
        self.mol_charge = mf.mol.charge
        num_nuc = mf.mol.natm
        self.num_nuc = num_nuc
        eps = jnp.finfo(nuc_crds.dtype).eps
        i_e, j_e = jnp.triu_indices(nelec, k=1)
        self.i_e = i_e
        self.j_e = j_e

        # Get wavefunction and energy functions
        log_trial_wavefunction, local_energy, mo_fns, C_fns \
            = get_psi_fun(mf, params_cusp=params_cusp,
                          trial=trial,
                          jastrow_config=jastrow_config)
        _, get_psi_mo_partition_vg = mo_fns
        local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke \
            = local_energy
        self.local_energy_ee = local_energy_ee
        self.local_energy_en = local_energy_en
        self.local_energy_ke = local_energy_ke

        # Precompute nuclear-nuclear energy and gradient
        self.enr_nn = local_energy_nn(nuc_crds)
        self.grd_nn = jax.grad(local_energy_nn)(nuc_crds)

        # Compute CPHF orbital response (if enabled)
        if mo_relax:
            log_trial_wavefunction_C, local_energy_ke_C = C_fns
            nocc = jnp.count_nonzero(jnp.array(mf.mo_occ) > 0)
            mo1s = compute_orbital_response(mf)  # (natm, 3, nao, nocc)
            C0 = jnp.array(mf.mo_coeff[:, :nocc])

        # --- Build batched gradient function ---
        _mo_kw = {}
        if mo_relax:
            _mo_kw = dict(
                local_energy_ke_C=local_energy_ke_C,
                log_trial_wavefunction_C=(
                    log_trial_wavefunction_C
                ),
                C0=C0,
                mo1s=mo1s,
                num_nuc=num_nuc,
            )
        self.vmc_gradient_batch = vmc_gto_gradients(
            local_energy_ee,
            local_energy_en,
            local_energy_ke,
            log_trial_wavefunction,
            nuc_crds,
            params_corr,
            get_psi_mo_partition_vg,
            eps,
            gr_scheme,
            mo_relax=mo_relax,
            **_mo_kw,
        )

        # Parameter-response functions (ZVZB2)
        # ref. Assaraf and Caffarel. JCP (2003) Eq 79
        if force_estimator == 'zvzb2':
            pr_batch, n_jp = (
                _build_param_response_fns(
                    log_trial_wavefunction,
                    local_energy_ke,
                    local_energy_en,
                    nuc_crds,
                    params_corr,
                )
            )
            self.param_response_batch = pr_batch
            self.n_jastrow_params = n_jp
        else:
            # ref. Assaraf and Caffarel. JCP (2003) Eq 73
            self.param_response_batch = None
            self.n_jastrow_params = 0

        @jax.jit
        def _log_psi_batch(batch):
            """Evaluate log|ψ| for a batch of walker configurations."""
            return jax.vmap(
                lambda x: log_trial_wavefunction(x, nuc_crds, params_corr)
            )(batch)

        self._log_psi_batch = _log_psi_batch

        enr_nn_val = self.enr_nn

        @jax.jit
        def _local_energy_batch(batch):
            """Total local energy for a batch of walker configurations."""
            ee = jax.vmap(local_energy_ee)(batch)
            en = jax.vmap(local_energy_en,
                          in_axes=(0, None))(batch, nuc_crds)
            ke = jax.vmap(local_energy_ke,
                          in_axes=(0, None, None))(
                batch, nuc_crds, params_corr)
            return ee + en + ke + enr_nn_val

        self._local_energy_batch = _local_energy_batch

        # --- Metropolis moves ---
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

            # # 3. Eight-way dispatch via _apply_fragment_op
            # switch_idx = jnp.where(
            #     displacement_idx == 0, 0,
            #     jnp.where(~is_planar, 7,         # noop for non-planar
            #               displacement_idx))      # 1-6 for planar

            # proposed_crds, proposal_ratio = jax.lax.switch(
            #     switch_idx,
            #     [metropolis_move_alle,      # 0: Gaussian
            #      ...specific closures using _apply_fragment_op...
            #      metropolis_fragment_noop], # 7: no-op
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

        self.metropolis_move_allw = jax.vmap(metropolis_move_1w,
                                             in_axes=(0, 0, None, None))

        # --- Basin-hopping move for initial thermalization ---
        # Proposes relocating one randomly chosen electron to a
        # Gaussian blob (width BASIN_RELOC_SIGMA) around a randomly
        # chosen nucleus.  Accepted with the exact Metropolis-
        # Hastings probability, including the asymmetric proposal
        # ratio q(old)/q(new) with q(x) proportional to
        # sum_J exp(-|x - R_J|^2 / (2 sigma^2)), so the stationary
        # distribution is exactly |psi|^2.  Unlike the local move it
        # can teleport an electron between far-apart nuclei, which is
        # required to populate charge-transfer (ionic) basins at
        # large fragment separations.  Used only to thermalize the
        # initial ensemble; the production move set is unchanged.
        _sigma = BASIN_RELOC_SIGMA
        _inv2s2 = 1.0 / (2.0 * _sigma * _sigma)

        def _basin_log_q(x):
            d2 = ((x[jnp.newaxis, :] - nuc_crds) ** 2).sum(-1)
            return jax.scipy.special.logsumexp(-d2 * _inv2s2)

        @jax.jit
        def basin_move_1w(rng_key, elec_crds):
            key_e, key_j, key_pos, key_acc = jax.random.split(rng_key, 4)
            k = jax.random.randint(key_e, (), 0, nelec)
            j = jax.random.randint(key_j, (), 0, num_nuc)
            rk_new = nuc_crds[j] \
                + _sigma * jax.random.normal(key_pos, (3,))
            proposed_crds = elec_crds.at[k].set(rk_new)

            # Reject proposals that hit an electron-nucleus or
            # electron-electron singularity.
            diffs_ee = proposed_crds[i_e] - proposed_crds[j_e]
            dists_ee = jnp.linalg.norm(diffs_ee, axis=-1)
            diffs_en = proposed_crds[:, None, :] - nuc_crds[None, :, :]
            dists_en = jnp.linalg.norm(diffs_en, axis=-1)
            is_invalid = (dists_en.min() < MIN_DIST_THRESHOLD) \
                | (dists_ee.min() < MIN_DIST_THRESHOLD)

            log_qratio = _basin_log_q(elec_crds[k]) \
                - _basin_log_q(rk_new)
            log_psi_old = log_trial_wavefunction(elec_crds, nuc_crds,
                                                 params_corr)
            log_psi_new = log_trial_wavefunction(proposed_crds, nuc_crds,
                                                 params_corr)
            log_alpha = 2.0 * (log_psi_new - log_psi_old) + log_qratio
            accept = (jnp.log(jax.random.uniform(key_acc)) < log_alpha) \
                & (~is_invalid)
            new_crds = jnp.where(accept, proposed_crds, elec_crds)
            return new_crds, accept

        self.basin_move_allw = jax.vmap(basin_move_1w, in_axes=(0, 0))

        # --- Per-fragment symmetry operation for gradient batches ---
        self._apply_single_frag_symmop = (
            make_apply_single_frag_symmop(
                frag_centroids, frag_Vh, frag_inradii,
            )
        )

    def _thermalize_basins(self, rng_key, walkers, num_walkers,
                           walker_keys_sharding, n_steps):
        """Thermalize charge-transfer basins of the initial ensemble.

        Runs ``n_steps`` single-electron relocation moves (see
        ``basin_move_allw``) so the walker ensemble populates the
        electron-to-nucleus partitions in proportion to |psi|^2.
        The local production move cannot do this across well-
        separated fragments, so this is what makes the dissociation
        limit (e.g. ionic configurations of a stretched bond) get
        sampled at all.  Returns ``(rng_key, walkers, mean_accept)``.
        """
        basin_move_allw = self.basin_move_allw

        @jax.jit
        def basin_step(state, _):
            rng_key, walkers = state
            rng_key, key = jax.random.split(rng_key)
            walker_keys = jax.random.split(key, num_walkers)
            if walker_keys_sharding is not None:
                walker_keys = jax.lax.with_sharding_constraint(
                    walker_keys, walker_keys_sharding)
            walkers, accepted = basin_move_allw(walker_keys, walkers)
            return (rng_key, walkers), accepted.mean()

        (rng_key, walkers), accepts = jax.lax.scan(
            basin_step, (rng_key, walkers), None, length=n_steps)
        return rng_key, walkers, accepts.mean()

    def __call__(self, rng_key: int | jnp.ndarray,
                 num_walkers: int = 1000,
                 num_steps_per_block: int = 100, num_steps_decorr: int = 1,
                 num_blocks: int = 1000, num_blocks_equil: int = 100,
                 mc_timestep: float = 0.1,
                 fname_log: str = None,
                 mode_restart: bool = False,
                 num_steps_basin: int = N_BASIN_THERMAL_STEPS,
                 compute_gradients: bool = False) -> None:
        """Execute a VMC run and write results to HDF5 checkpoint files.

        Runs Metropolis-Hastings Monte Carlo sampling of the trial wave
        function, accumulating the local energy (and optionally nuclear-force
        gradients) block by block.  Results are written to ``<prefix>.chk.h5``
        (energies / walker snapshots) and, when *compute_gradients* is
        ``True``, ``<prefix>.grd.h5`` (gradient data).

        Parameters
        ----------
        rng_key : int or jnp.ndarray
            JAX PRNG key used to initialise the walkers and Monte Carlo moves.
            Pass an integer to use ``jax.random.key(rng_key)``.
        num_walkers : int, optional
            Number of independent electron-position walkers.  Default 1000.
        num_steps_per_block : int, optional
            Monte Carlo steps between successive energy measurements.
            Default 100.
        num_steps_decorr : int, optional
            Additional decorrelation steps taken *within* each block.
            Default 1.
        num_blocks : int, optional
            Total number of measurement blocks (including equilibration).
            Default 1000.
        num_blocks_equil : int, optional
            Number of leading blocks discarded as equilibration.  Default 100.
        mc_timestep : float, optional
            Gaussian proposal width for Metropolis moves (in Bohr).
            Adjusted adaptively to reach ~50 % acceptance.  Default 0.1.
        fname_log : str, optional
            Path for the plain-text log file.  Defaults to
            ``<prefix>.log`` when *None*.
        mode_restart : bool, optional
            If ``True``, attempt to resume from an existing checkpoint file
            and continue accumulating blocks.  Default ``False``.
        num_steps_basin : int, optional
            Number of single-electron relocation moves used to thermalize
            charge-transfer basins before equilibration.  Required for an
            unbiased energy when fragments are well separated (e.g. a
            stretched bond); harmless otherwise.  Set to ``0`` to disable.
            Default :data:`N_BASIN_THERMAL_STEPS`.
        compute_gradients : bool, optional
            If ``True``, accumulate and save nuclear-force gradient data
            needed by :func:`~OmegaQMC.observables.force.postproc_h5_pgcs`.
            Default ``False``.

        Returns
        -------
        None
            All output is written to disk.  Use
            :func:`~OmegaQMC.observables.force.postproc_h5_pgcs` to
            post-process the gradient file.
        """
        # tolerance_enr_std_per_elec=CHEMICAL_ACCURACY,
        mf = self.mf
        nuc_crds = self.nuc_crds
        nelec = self.nelec
        Z_charges = self.Z_charges
        mol_charge = self.mol_charge
        num_nuc = self.num_nuc
        enr_nn = self.enr_nn
        grd_nn = self.grd_nn
        params_corr = self.params_corr
        metropolis_move_allw = self.metropolis_move_allw
        ofname_chkpt = self.ofname_chkpt
        ofname_grd = self.ofname_grd
        timestamp_init = self.timestamp_init
        local_energy_ee = self.local_energy_ee
        local_energy_en = self.local_energy_en
        local_energy_ke = self.local_energy_ke

        # --- Informational GPU capacity estimate ---
        # Does NOT modify num_walkers.
        try:
            from .vmcopt_gto_linear import _get_free_gpu_mb
            free_mb = _get_free_gpu_mb()
            n_rec, bpw = _autotune_prod_walkers(
                self._local_energy_batch, nelec, free_mb)
            free_txt = (f"{free_mb:.0f} MiB free"
                        if free_mb is not None
                        else "free GPU mem unknown")
            print(f"ℹ️\tEst. GPU capacity: {n_rec} walkers "
                  f"(user requested {num_walkers}; "
                  f"{bpw / 1e6:.2f} MB/walker, {free_txt})")
        except Exception as e:
            print(f"ℹ️\tGPU capacity estimate unavailable: {e}")

        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        rng_key_to_restart = rng_key.copy()

        # Initialize electron positions more efficiently
        rng_key, init_key = jax.random.split(rng_key)
        walkers = _initialize_walkers(init_key,
                                      num_walkers, nelec,
                                      Z_charges, nuc_crds, mol_charge)
        walkers_sharding, walker_keys_sharding = _make_sharding(num_walkers)
        if walkers_sharding is not None:
            walkers = jax.device_put(walkers, walkers_sharding)

        # Thermalize charge-transfer basins before equilibration so
        # the initial ensemble populates electron-to-nucleus
        # partitions per |psi|^2.  The local move alone cannot reach
        # ionic configurations across well-separated fragments, which
        # otherwise biases the energy below the trial-wavefunction
        # value at large bond lengths.  Skipped on restart (the
        # restored walkers are already thermalized).
        if not mode_restart and num_steps_basin > 0:
            rng_key, basin_key = jax.random.split(rng_key)
            basin_key, walkers, basin_accept = self._thermalize_basins(
                basin_key, walkers, num_walkers, walker_keys_sharding,
                num_steps_basin)
            print(f"ℹ️\tBasin thermalization: {num_steps_basin} "
                  f"relocation steps, acceptance {basin_accept:.2f}")

        mc_stepsize = (3 * mc_timestep)**0.5

        # Equilibration phase
        @jax.jit
        def equilibration_step(state, step_idx):
            rng_key, walkers, step_size = state
            rng_key, key = jax.random.split(rng_key)
            walker_keys = jax.random.split(key, num_walkers)
            if walker_keys_sharding is not None:
                walker_keys = jax.lax.with_sharding_constraint(
                    walker_keys, walker_keys_sharding)
            new_walkers, accepted, is_gaussian \
                = metropolis_move_allw(walker_keys, walkers, step_size,
                                       step_idx)

            # Only use Gaussian acceptance for step size adaptation
            n_gauss = jnp.sum(is_gaussian)
            gauss_accept_sum = jnp.where(is_gaussian, accepted, 0.0).sum()
            gauss_ratio = jnp.where(n_gauss > 0, gauss_accept_sum / n_gauss,
                                    TARGET_ACCEPTANCE_RATE)
            new_step_size = _adapt_step_size(step_size, gauss_ratio)

            return (rng_key, new_walkers, new_step_size), gauss_ratio
        # walkers.shape == (num_walkers, nelec, 3)

        if mode_restart:
            with h5py.File(ofname_chkpt, 'r') as f:
                # Load metadata
                block_cnt_start = int(f['block_count'][()]) + 1
                mc_stepsize = f['mc_stepsize'][()]
                mc_timestep = mc_stepsize * mc_stepsize / 3
                rng_key = jax.random.wrap_key_data(
                    jnp.array(f['rng_key'][:],
                              dtype=jnp.uint32))
                rng_key_to_restart = rng_key.copy()
                rng_key, init_key = jax.random.split(rng_key)
                walkers = jnp.array(f['walkers'][:])
                if walkers_sharding is not None:
                    walkers = jax.device_put(walkers, walkers_sharding)
                E_b = list(f['E_blocks'][:])
                E_cs_b = list(f['E_cs_blocks'][:]) \
                    if 'E_cs_blocks' in f else []
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
            E_cs_b = []    # CS-averaged block energies
            # std_E_b = []

        if not mode_restart:
            ratio = ratios[-1]
            print(f"ℹ️\tEquilibration acceptance rate: "
                  f"{ratio:.2f}")
        print(f"ℹ️\tAdjusted step size: {mc_stepsize:.4f} bohr "
              f"~ {mc_timestep:.4f} Ha⁻¹ in Brownian time")

        # Production phase
        @jax.jit
        def production_step(state, step_number):
            rng_key, walkers, step_size = state

            for d in range(num_steps_decorr):
                rng_key, key_displace = jax.random.split(rng_key)
                walker_keys = jax.random.split(key_displace, num_walkers)
                if walker_keys_sharding is not None:
                    walker_keys = jax.lax.with_sharding_constraint(
                        walker_keys, walker_keys_sharding)
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
            # TODO: take version info from pyproject.toml using tomli/tomllib

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

        if compute_gradients and not mode_restart:
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
                g.create_dataset("atom_fragment_map",
                                 data=mf.mol.map_nuc_frag)

        base_batch_size = 500
        memory_factor = max(1, nelec * num_nuc // 1000)
        batch_size = min(50, base_batch_size // memory_factor)
        num_batches = (num_steps_per_block * num_walkers + batch_size - 1) \
            // batch_size
        # mark_samples = ((jnp.arange(num_steps_per_block)+1) == 0)
        print("ℹ️\tAdjusted batch size, number of batches: "
              f"{batch_size}, {num_batches}")
        print("# block_cnt        E_loc_mean      E_loc_std"
              "       eePotential     enPotential     Kinetic"
              "          ∆t_block", file=fout)

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
                  f"{tdelta_block:>16.6f}", file=fout)

            if compute_gradients:
                if walkers_sharding is not None:
                    sampled_walkers = jax.device_put(
                        sampled_walkers,
                        NamedSharding(walkers_sharding.mesh,
                                      PartitionSpec(None, None, None, None)))
                combo_E = save_gto_gradients(
                    block_cnt,
                    sampled_walkers.reshape(-1, nelec, 3),
                    E_loc_sw,
                    batch_size, num_batches,
                    self.single_frag_combos,
                    self.ofname_grd,
                    self.vmc_gradient_batch,
                    self._log_psi_batch,
                    self._local_energy_batch,
                    self._apply_single_frag_symmop,
                    param_response_batch=(
                        self.param_response_batch
                    ),
                )
                if combo_E:
                    all_E = [float(E_mean)] + list(combo_E.values())
                    E_cs_b.append(sum(all_E) / len(all_E))

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

        E_blocks = jnp.array(E_b)
        e_mean, e_serr, _, e_kappa = do_binning_analysis(E_blocks)
        e_neff = E_blocks.shape[0] / e_kappa
        print(f"ℹ️\tVMC energy: {e_mean:.8f} ± {e_serr:.8f} Ha"
              f" (N_eff = {e_neff:.1f})")

        if compute_gradients and E_cs_b:
            E_cs_blocks = jnp.array(E_cs_b)
            ecs_mean, ecs_serr, _, ecs_kappa \
                = do_binning_analysis(E_cs_blocks)
            ecs_neff = E_cs_blocks.shape[0] / ecs_kappa
            print(f"ℹ️\tCS-averaged VMC energy: {ecs_mean:.8f} "
                  f"± {ecs_serr:.8f} Ha (N_eff = {ecs_neff:.1f})")

        with h5py.File(ofname_chkpt, 'a') as f:
            f.create_dataset('E_blocks', data=E_b)
            if E_cs_b:
                f.create_dataset('E_cs_blocks', data=E_cs_b)
            f.create_dataset('rng_key',
                             data=jax.random.key_data(rng_key_to_restart))
            f.create_dataset('block_count', data=block_cnt,
                             dtype=jnp.int32)
            f.create_dataset('mc_stepsize',
                             data=float(mc_stepsize))
            f.create_dataset('walkers', data=sampled_walkers[-1, :, :, :])
            f["timestamps"].create_dataset("end", data=str(timestamp_fin))


def get_vmc_gto_func(mf,
                     params_corr: dict | None,
                     cusp_scheme='Quady2025',
                     gr_scheme='scheme1',
                     force_estimator='simple',
                     prefix='vmc',
                     symmop_list: str | list[str] | dict[int, list[str]] | None = None,
                     cluster_idx: Collection[int] = None,
                     mo_relax: bool = True,
                     trial: dict | None = None,
                     jastrow_config: dict | None = None):
    """Construct a callable VMC driver for the given mean-field object.

    Assembles the trial wave function (Slater determinant + Jastrow factor
    with optional cusp corrections) and returns a :class:`_VMCDriverGTO`
    instance that can be called directly to run a VMC simulation.

    Parameters
    ----------
    mf : pyscf.scf.RHF
        Converged mean-field object as returned by
        :func:`generate_molecular_orbitals`.
    params_corr : dict or None
        Jastrow-factor parameters.  Pass a dict with key ``"J2_pade"``
        containing a 1-D array of optimizable coefficients, or ``None`` /
        ``{"J2_pade": jnp.array([])}`` to use no Jastrow factor.
    cusp_scheme : str, optional
        Cusp-correction scheme to apply near nuclei.  ``"Quady2025"``
        (default) uses the scheme described in Quady *et al.* (2025).
        Pass ``None`` to disable cusp corrections.
    gr_scheme : str, optional
        Gradient estimator scheme.  Default is ``"scheme1"``.
    force_estimator : str, optional
        Force estimator to use.  ``'simple'`` (default)
        uses the standard ZVZB estimator.  ``'zvzb2'``
        adds a parameter-response variance-reduction
        correction via analytical linear response of
        the Jastrow parameters.
    prefix : str, optional
        Stem used for output file names (``<prefix>.chk.h5``,
        ``<prefix>.grd.h5``, ``<prefix>.log``).  Default is ``"vmc"``.
    symmop_list : str, list of str, dict mapping int to list of str, or None
        Point-group symmetry operations to use for correlated sampling.
        ``None`` (default) applies only the identity ``"E"`` (no correlated
        sampling overhead).  ``"auto"`` auto-derives all allowed operations
        from the molecule's fragment symmetry map.  A plain list applies the
        same operations to every fragment.  A dict maps fragment indices (as
        defined in the atom string by trailing integer labels) to their own
        operation lists.
    cluster_idx : collection of int, optional
        Indices of atoms forming a sub-cluster for gradient calculations.
        ``None`` (default) treats all atoms as one cluster.
    mo_relax : bool, optional
        If ``True`` (default), relax the MO coefficients
        to minimise the energy variance during the
        cusp-correction step.
    jastrow_config : dict or None, optional
        Cutoff radii for B-spline Jastrow factors.
        Example::

            {"J1": {"H": {"r_cut": 5.0}},
             "J2": {"r_cut": 10.0},
             "J3": {"r_cut": 5.0, "N_eI": 3, "N_ee": 3}}

    Returns
    -------
    driver : _VMCDriverGTO
        A callable object.  Call it with ``driver(rng_key, ...)`` to run the
        VMC simulation; see :meth:`_VMCDriverGTO.__call__` for parameters.
    """
    # Build per-fragment symmetry operations dict
    if hasattr(mf.mol, 'map_frag_symmops') and mf.mol.map_frag_symmops:
        frag_ids = sorted(mf.mol.map_frag_ctr.keys())
    else:
        frag_ids = [0]

    frag_symmops = build_frag_symmops(mf.mol, symmop_list, frag_ids)
    single_frag_combos = build_single_frag_combos(frag_ids, frag_symmops)
    frag_ops_sets = [set(frag_symmops[fid]) for fid in frag_ids]

    if mf.mol.groupname == 'C1' \
            and any(len(ops) > 1 for ops in frag_symmops.values()):
        warnings.warn(
            "Calculating symmetry-adapted forces "
            "on a system with no symmetry (C1)",
            stacklevel=2
        )

    # check prefix
    for s in [".chk.h5", ".grd.h5"]:
        if prefix.endswith(s):
            prefix = prefix[:-len(s)]
    ofname_chkpt = prefix + ".chk.h5"
    ofname_grd = prefix + ".grd.h5"

    if trial is not None and mo_relax:
        warnings.warn(
            "MO relaxation is not supported with "
            "multi-determinant trials. "
            "Disabling mo_relax.",
            stacklevel=2)
        mo_relax = False

    params_corr = _validate_params_corr(params_corr, mf)
    params_cusp = _build_cusp_params(mf, cusp_scheme, mf.mol.natm)

    nuc_crds = jnp.array(mf.mol.atom_coords(unit='Bohr'))
    frag_transform_data = build_frag_transform_data(
        mf.mol, nuc_crds, frag_symmops=frag_symmops,
    )

    timestamp_init = datetime.now()
    print("Begin time: {}".format(timestamp_init))

    return _VMCDriverGTO(mf, params_corr, params_cusp, mo_relax,
                         nuc_crds, frag_transform_data,
                         single_frag_combos,
                         frag_symmops, frag_ops_sets, frag_ids,
                         ofname_chkpt, ofname_grd,
                         timestamp_init,
                         gr_scheme=gr_scheme,
                         force_estimator=force_estimator,
                         trial=trial,
                         jastrow_config=jastrow_config)
