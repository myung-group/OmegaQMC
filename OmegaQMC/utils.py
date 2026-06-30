import os
import sys
import warnings
from collections.abc import Callable
import h5py
import numpy as np
from pyscf import __config__, gto
from pyscf.lib import logger, param
from pyscf.gto.basis import _format_basis_name
from pyscf.data.elements import MASSES, ELEMENTS_PROTON, _atom_symbol
import jax
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve
from jax.sharding import Mesh, NamedSharding, PartitionSpec
# from jax import lax


def equilibration_length(x, tail=0.5, bounces=2):
    """Deterministic estimate of a time series' equilibration length.

    A robust band ``median ± ~1 sigma`` is estimated from the last
    ``tail`` fraction of the series (assumed equilibrated), then the
    series is scanned from the start, counting how many times it
    crosses into that band.  The returned cutoff is the midpoint of
    the final bounce interval — i.e. the block index at which the
    series has settled.

    This is a deterministic adaptation of the common
    ``equilibration_length`` heuristic: the optional random placement
    within the bounce interval (useful only when the detector is
    applied across an ensemble, so the choice averages out) is dropped
    in favour of the midpoint, since a post-processing routine emits a
    single cutoff and must be reproducible.

    Parameters
    ----------
    x : array_like, shape (N,)
        One-dimensional series (e.g. per-block mean local energies).
    tail : float, optional
        Fraction of the series, taken from the end, used to estimate
        the equilibrated band.  Default ``0.5``.
    bounces : int, optional
        Number of band re-entries to require before declaring
        equilibration.  Default ``2``.

    Returns
    -------
    int
        Estimated equilibration length (number of leading samples to
        discard).  ``0`` if the series is too short (< 10 tail
        samples) or already starts inside the band.
    """
    x = np.asarray(x)
    bounces = max(1, bounces)
    eqlen = 0
    nx = len(x)
    xt = x[int((1.0 - tail) * nx + 0.5):]
    nxt = len(xt)
    if nxt < 10:
        return eqlen
    xs = np.sort(xt)
    mean = xs[int(0.5 * (nxt - 1) + 0.5)]
    sigma = (
        np.abs(xs[int((0.5 - 0.341) * nxt + 0.5)] - mean)
        + np.abs(xs[int((0.5 + 0.341) * nxt + 0.5)] - mean)
    ) / 2
    crossings = bounces * [0, 0]
    if np.abs(x[0] - mean) > sigma:
        s = -np.sign(x[0] - mean)
        ncrossings = 0
        for i in range(nx):
            dist = s * (x[i] - mean)
            if dist > sigma and dist < 5 * sigma:
                crossings[ncrossings] = i
                s *= -1
                ncrossings += 1
                if ncrossings == 2 * bounces:
                    break
        bounce = crossings[-2:]
        bounce[1] = max(bounce[1], bounce[0])
        eqlen = (bounce[0] + bounce[1]) // 2
    return int(eqlen)


@jax.jit
def do_binning_analysis(a):
    """Compute mean, standard error, standard deviation, and autocorrelation length.

    Estimates the statistical error of a correlated time series using the
    integrated autocorrelation time (binning / blocking analysis).  The
    autocorrelation function is computed via FFT convolution for efficiency.

    Parameters
    ----------
    a : jnp.ndarray, shape (N,)
        One-dimensional array of sequential Monte Carlo samples (e.g.
        local-energy values from a VMC run).

    Returns
    -------
    xbar : jnp.ndarray, scalar
        Sample mean.
    serr : jnp.ndarray, scalar
        Standard error of the mean, corrected for autocorrelation:
        ``serr = std(a) * sqrt(kappa / N)`` where *kappa* is the integrated
        autocorrelation time.
    sdev : jnp.ndarray, scalar
        Uncorrected sample standard deviation.
    kappa : jnp.ndarray, scalar
        Integrated autocorrelation time (dimensionless).  Values close to
        1.0 indicate nearly uncorrelated samples.
    """
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


# A sample covariance of K correlated estimators needs many more
# bins than states to be reliable; with fewer than this many bins
# per state the off-diagonal entries are too noisy to trust, so
# blue_combine_states falls back to the inverse-variance estimator.
MIN_BINS_PER_STATE = 5


def blue_combine_states(
    series_per_label, labels, kappa_per_label=None,
):
    """Combine independent block-mean time series via BLUE.

    Builds the K×K covariance matrix Σ of the per-label
    mean estimators from bin-mean cross-covariance:
    block series are partitioned into bins of size
    ``b = ceil(2 · max_k κ_k)`` so that successive bin
    means are approximately decorrelated, the sample
    covariance of the (N_b, K) bin-mean matrix gives
    ``S``, and ``Σ = S / N_b`` is the resulting
    covariance of the per-label mean estimators —
    capturing both within-series autocorrelation (via
    the bin size) and cross-series correlations at
    matched block index.  The Best Linear Unbiased
    Estimator under the unit-sum linear constraint is
    then ``w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)``, ``μ = wᵀ Ê``,
    ``Var(μ) = 1 / (1ᵀ Σ⁻¹ 1)``.  The full BLUE is used only
    when Σ is positive definite *and* there are enough bins to
    estimate its off-diagonals (``N_b ≥ MIN_BINS_PER_STATE·K``).
    With too few bins the BLUE weights become noise-driven
    leverage and the variance is under-reported, so the combine
    falls back to the inverse-variance (diagonal-Σ) estimator;
    a (near-)degenerate covariance with non-positive variances
    falls back to an equal-weight average.  Every fallback warns.

    Parameters
    ----------
    series_per_label : dict
        Maps each label to a 1-D array-like of length
        ``T`` (per-block scalar estimates of a common
        expectation).
    labels : list
        Ordered labels selecting which series to
        combine.  The output ``weights`` is in this
        order.
    kappa_per_label : dict, optional
        Per-label integrated autocorrelation time used
        to set the bin size.  When ``None`` each κ is
        computed internally via
        :func:`do_binning_analysis`.

    Returns
    -------
    mean : float
        BLUE-combined mean.
    err : float
        Standard error of the BLUE mean.
    neff : float
        Effective sample count relative to the first
        listed label's per-block sample variance,
        matching the ``N_eff = σ²/Var(mean)``
        convention of :func:`do_binning_analysis`.
    weights : np.ndarray, shape (K,)
        BLUE weights in the order of ``labels``; may
        include negative entries when series are
        strongly cross-correlated.
    bin_size : int
        Bin size used to build Σ.
    """
    K = len(labels)
    E = np.column_stack(
        [np.asarray(series_per_label[lbl], dtype=np.float64)
         for lbl in labels]
    )  # (T, K)
    T = E.shape[0]

    if kappa_per_label is None:
        kappas = [
            float(
                do_binning_analysis(
                    jnp.asarray(series_per_label[lbl])
                )[3]
            )
            for lbl in labels
        ]
    else:
        kappas = [float(kappa_per_label[lbl]) for lbl in labels]
    kappa_max = max(kappas) if kappas else 1.0

    bin_size = max(1, int(np.ceil(2.0 * kappa_max)))
    N_b = T // bin_size
    if N_b < K + 1:
        # Not enough bins to estimate a (K×K) covariance
        # — fall back to the smallest viable bin size.
        bin_size = max(1, T // (K + 1))
        N_b = T // bin_size

    truncated = E[:N_b * bin_size]
    bin_means = truncated.reshape(
        N_b, bin_size, K,
    ).mean(axis=1)  # (N_b, K)

    state_means = E.mean(axis=0)
    ones = np.ones(K)

    if N_b < 2:
        # Fewer than two bins: no covariance can be formed.
        # Combine with equal weights; no usable error bar.
        warnings.warn(
            f"blue_combine_states: only N_b={N_b} bin(s) for "
            f"K={K} states; cannot estimate a covariance. "
            "Falling back to an equal-weight average with no "
            "usable error bar.",
            stacklevel=2,
        )
        w = ones / K
        mean = float(w @ state_means)
        return mean, float('inf'), 0.0, w, bin_size

    centered = bin_means - bin_means.mean(
        axis=0, keepdims=True,
    )
    S = centered.T @ centered / (N_b - 1)  # (K, K)
    Sigma = S / N_b  # covariance of the means

    # Minimum-variance (BLUE) weights solve ``Σ w ∝ 1`` and need
    # a positive-definite Σ.  PD alone is not enough, though:
    # estimating the K×K off-diagonal covariance reliably needs
    # many more bins than states.  With too few bins the BLUE
    # weights become noise-driven leverage (large ± values) and
    # the variance is badly under-reported — the failure mode
    # that lets the combine drift outside the range of the
    # inputs.  So the full BLUE is used only when Σ is PD *and*
    # ``N_b >= MIN_BINS_PER_STATE * K``.  Otherwise fall back to
    # the inverse-variance (diagonal-Σ) combine, whose convex
    # weights need only the per-state variances; or, if even
    # those are unusable, to an equal-weight average.  Every
    # fallback warns so the degradation is visible.
    diag = np.diag(Sigma)
    is_pd = False
    try:
        np.linalg.cholesky(Sigma)  # succeeds iff Σ is PD
        sol = np.linalg.solve(Sigma, ones)
        inv_sum = float(ones @ sol)
        is_pd = np.isfinite(inv_sum) and inv_sum > 0.0
    except np.linalg.LinAlgError:
        pass

    enough_bins = N_b >= MIN_BINS_PER_STATE * K

    if is_pd and enough_bins:
        w = sol / inv_sum
        var_mu = 1.0 / inv_sum
    elif np.all(diag > 0.0):
        if is_pd:
            warnings.warn(
                f"blue_combine_states: only N_b={N_b} bins for "
                f"K={K} states (< MIN_BINS_PER_STATE="
                f"{MIN_BINS_PER_STATE} per state); the "
                "off-diagonal covariance is unreliable, so "
                "falling back to the inverse-variance combine.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                "blue_combine_states: covariance is not positive "
                "definite; falling back to the inverse-variance "
                "combine.",
                stacklevel=2,
            )
        w = 1.0 / diag
        w = w / w.sum()
        var_mu = max(float(w @ Sigma @ w), 0.0)
    else:
        warnings.warn(
            "blue_combine_states: degenerate covariance "
            "(non-positive variances); falling back to an "
            "equal-weight average.",
            stacklevel=2,
        )
        w = ones / K
        var_mu = max(float(w @ Sigma @ w), 0.0)

    mean = float(w @ state_means)
    err = float(np.sqrt(max(var_mu, 0.0)))

    ref_var = float(np.var(E[:, 0], ddof=1))
    neff = (
        ref_var / var_mu
        if var_mu > 0 else float('inf')
    )
    return mean, err, neff, w, bin_size


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
                warnings.warn("Line 2 of {} should have a properties "
                              "block indicating a column with molecular "
                              "fragment indices e.g. "
                              "'Properties=species:S:1:pos:R:3:molecule:I:1'"
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


def get_shell(z):
    """Number of (partially) occupied shells for *z*.

    Ported from ``deepqmc/hamil.py``.

    Args:
        z: Number of electrons (integer).

    Returns:
        Shell count ``n`` such that the first *n*
        shells can hold at least *z* electrons.
    """
    from itertools import count as _count
    max_elec = 0
    for n in _count():
        if z <= max_elec:
            break
        max_elec += 2 * (1 + n) ** 2
    return n


class Mole_custom(gto.Mole):
    # Default symmetrization level; overridden by
    # callers such as
    # :func:`OmegaQMC.vmc_gto.generate_molecular_orbitals`.
    # Downstream helpers (e.g.
    # :func:`OmegaQMC.symm.operations.populate_fragment_symmops`,
    # :func:`OmegaQMC.symm.fragments.build_frag_symmops`)
    # may consult this to decide how strictly to trust
    # detected point-group operations.
    symmetrization_level = 1

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

    def dumps(self):
        import numpy as np

        def _json_safe(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    out[str(k)] = v.tolist()
                elif isinstance(v, float) and np.isinf(v):
                    out[str(k)] = None
                else:
                    out[str(k)] = v
            return out

        saved = {}
        for attr in ('map_frag_ctr', 'inradii', 'map_frag_symmops'):
            d = getattr(self, attr, None)
            if isinstance(d, dict):
                saved[attr] = d
                setattr(self, attr, _json_safe(d))
        try:
            return super().dumps()
        finally:
            for attr, d in saved.items():
                setattr(self, attr, d)

    def loads_(self, molstr):
        import numpy as np
        super().loads_(molstr)
        if hasattr(self, 'map_frag_ctr') and isinstance(self.map_frag_ctr, dict):
            self.map_frag_ctr = {
                int(k): np.array(v) for k, v in self.map_frag_ctr.items()
            }
        if hasattr(self, 'inradii') and isinstance(self.inradii, dict):
            self.inradii = {
                int(k): (np.inf if v is None else float(v))
                for k, v in self.inradii.items()
            }
        if hasattr(self, 'map_frag_symmops') and isinstance(self.map_frag_symmops, dict):
            self.map_frag_symmops = {
                int(k): v for k, v in self.map_frag_symmops.items()
            }
        return self

    # ----- NN-compatible properties -----

    @property
    def n_up(self):
        """Number of spin-up electrons."""
        return self.nelec[0]

    @property
    def n_down(self):
        """Number of spin-down electrons."""
        return self.nelec[1]

    @property
    def charges(self):
        """Nuclear charges as a JAX array."""
        return jnp.asarray(
            self.atom_charges(),
            dtype=jnp.float64,
        )

    @property
    def coords(self):
        """Nuclear coordinates (Bohr), JAX array."""
        return jnp.asarray(
            self.atom_coords(),
            dtype=jnp.float64,
        )

    @property
    def mol_shells(self):
        """Occupied shell counts per nucleus."""
        return [
            get_shell(int(z))
            for z in self.atom_charges()
        ]

    @property
    def mol_ecp_shells(self):
        """ECP shell indices per nucleus."""
        return [0] * self.natm

    @classmethod
    def from_arrays(
        cls, charges, coords,
        n_up=None, n_down=None,
        spin=0, charge=0,
        unit='Bohr',
    ):
        """Build from arrays of charges and coords.

        Parameters
        ----------
        charges : array-like, shape ``(natom,)``
            Nuclear charges (atomic numbers).
        coords : array-like, shape ``(natom, 3)``
            Nuclear coordinates.
        n_up, n_down : int, optional
            Electron counts.  When both are given,
            *spin* and *charge* are inferred from
            them (overriding the explicit values).
        spin : int
            Total spin 2S (default 0).
        charge : int
            Molecular charge (default 0).
        unit : str
            Coordinate unit (default ``'Bohr'``).

        Returns
        -------
        Mole_custom
            A built molecule instance.
        """
        import numpy as np
        from pyscf.data.elements import ELEMENTS
        charges_np = np.asarray(
            charges, dtype=int,
        )
        coords_np = np.asarray(
            coords, dtype=float,
        )
        if n_up is not None and n_down is not None:
            spin = n_up - n_down
            charge = (
                int(charges_np.sum())
                - n_up - n_down
            )
        atom_list = [
            (ELEMENTS[int(z)], c.tolist())
            for z, c in zip(
                charges_np, coords_np,
            )
        ]
        mol = cls()
        mol.build(
            atom=atom_list,
            spin=spin,
            charge=charge,
            unit=unit,
        )
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
    """Return a filesystem-safe representation of a basis-set name.

    Delegates to PySCF's internal ``_format_basis_name`` and additionally
    replaces ``*`` characters (which appear in names like ``6-31G**``) with
    the letter ``s`` so that the result can be used safely in file names.

    Parameters
    ----------
    basisname : str or dict
        Basis-set name (e.g. ``"aug-cc-pVTZ"``, ``"6-31G**"``) or an
        element-keyed dict of basis names.  When a dict is supplied the
        longest string value is formatted; if no string values are present
        the literal string ``"gen"`` is returned.

    Returns
    -------
    str
        Formatted basis name suitable for use in file-name prefixes.

    Examples
    --------
    >>> format_basis_name("6-31G**")
    '631gss'
    >>> format_basis_name({"O": "aug-cc-pVTZ", "H": "cc-pVDZ"})
    'augccpvtz'
    """
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


def _make_sharding(num_walkers: int):
    """Return (walkers_sharding, walker_keys_sharding) or (None, None)."""
    devices = jax.devices()
    n = len(devices)
    if n == 1:
        return None, None
    assert num_walkers % n == 0, (
        f"num_walkers ({num_walkers}) must be divisible by device count ({n})")
    mesh = Mesh(np.array(devices), ('w',))
    ws  = NamedSharding(mesh, PartitionSpec('w', None, None))  # (w, nelec, 3)
    wks = NamedSharding(mesh, PartitionSpec('w',))             # (w,) typed keys
    return ws, wks


def _autotune_prod_walkers(prod_batch, nelec, free_mb, mem_frac=0.75):
    """Estimate walker count a batched production kernel fits.

    Compiles *prod_batch* for a single-walker input and reads
    ``alias_size + temp_size`` from JAX's memory analysis to
    estimate bytes per walker.  Falls back to 0.5 MB/walker
    when AOT analysis is unavailable.  Informational only — the
    caller prints the result and does not mutate driver state.

    Parameters
    ----------
    prod_batch : callable
        ``(n_walkers, nelec, 3) -> ...`` — typically the driver's
        batched local-energy evaluator (already vmapped + jit).
        Both ``_VMCDriverGTO`` and ``_VMCDriverNN`` pass their
        ``_local_energy_batch`` closure.
    nelec : int
        Number of electrons.
    free_mb : float or None
        Free GPU memory in MiB; ``None`` assumes 4096.
    mem_frac : float
        Fraction of free memory to target (default 0.75).

    Returns
    -------
    (int, float)
        ``(n_rec, bytes_per_walker)`` — recommended walker count
        at *mem_frac* of free GPU memory, and the per-walker
        byte estimate that produced it.
    """
    bytes_per_walker = None
    try:
        probe = jnp.zeros((1, nelec, 3))
        compiled = jax.jit(prod_batch).lower(probe).compile()
        analysis = compiled.memory_analysis()
        bytes_per_walker = (analysis.alias_size
                            + analysis.temp_size)
    except Exception:
        pass

    if not bytes_per_walker:
        bytes_per_walker = 0.5e6  # 0.5 MB fallback

    free_bytes = (free_mb or 4096.0) * 1e6 * mem_frac
    n_rec = int(free_bytes / bytes_per_walker)
    return max(10, n_rec), bytes_per_walker


def laplacian_linearize(
    f: Callable[[jax.Array], jax.Array],
) -> Callable[
    [jax.Array], tuple[jax.Array, jax.Array]
]:
    """O(N) Laplacian via ``jax.linearize`` + ``fori_loop``.

    Given a scalar function *f* of a flat coordinate vector,
    returns a function that computes ``(nabla^2 f, grad f)``.

    This is more efficient than the full Hessian approach
    ``jax.hessian`` which scales as O(N^2).

    Args:
        f: Scalar function of a 1-D coordinate array.

    Returns:
        Function ``(x) -> (laplacian, gradient)``.
    """
    def lap(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        n_coord = len(x)
        grad_f = jax.grad(f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(n_coord)
        d2f = (
            lambda i, val: val + grad_f_jvp(eye[i])[i]
        )
        d2f_sum = jax.lax.fori_loop(
            0, n_coord, d2f, 0.0,
        )
        return d2f_sum, df
    return lap
