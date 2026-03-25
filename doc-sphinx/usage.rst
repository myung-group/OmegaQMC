Getting Started
===============

This page walks through minimal workflows for the two quantum Monte Carlo
methods provided by the ``OmegaQMC`` package: variational Monte Carlo
(VMC) :cite:`Foulkes2001` with point-group correlated sampling (PGCS)
and auxiliary-field quantum Monte Carlo (AFQMC) :cite:`Zhang2003`.

VMC with PGCS
-------------

The example below is based on the water-dimer test script and runs a short VMC
simulation that computes nuclear forces via point-group correlated sampling.

Step 1 — Build the molecular system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~OmegaQMC.generate_molecular_orbitals` accepts an inline atom string or
the path to an ``.xyz`` file, runs a PySCF mean-field calculation, and returns
the converged mean-field object:

.. code-block:: python

    from OmegaQMC import generate_molecular_orbitals

    atoms_string = '''
    O       8.707172158e-01  -1.720661566e+00  -4.814067179e-01     1
    H       1.689789528e+00  -1.221338203e+00  -3.052747557e-01     1
    H       1.142438797e+00  -2.587969761e+00  -7.799456175e-01     1
    O      -7.398283056e-01   4.040418183e-01  -1.654300203e+00     2
    H      -2.723133426e-01  -4.319081553e-01  -1.528862134e+00     2
    H      -1.614078540e+00   2.476812916e-01  -1.263515900e+00     2
    '''

    mf = generate_molecular_orbitals(atoms_string, units="ang", basis="6-31G")

The trailing integer on each atom line is a *fragment label*, used by the
correlated-sampling machinery.

Step 2 — Construct the VMC driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the mean-field object and Jastrow parameters to
:func:`~OmegaQMC.get_vmc_func`:

.. code-block:: python

    import jax.numpy as jnp
    from OmegaQMC import get_vmc_func
    from OmegaQMC.utils import format_basis_name

    params_jastrow = {"J2_pade": jnp.array([])}   # no Jastrow factor

    data_prefix = "water2_vmc_{}".format(format_basis_name("6-31G"))

    symmetry_ops = {1: ["E"], 2: ["E"]}   # identity only per fragment

    vmc_run = get_vmc_func(
        mf,
        params_jastrow,
        prefix=data_prefix,
        symmop_list=symmetry_ops,
    )

Step 3 — Run the VMC loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call the driver to start the simulation:

.. code-block:: python

    import jax

    rng_key = jax.random.key(888)

    vmc_run(
        rng_key,
        num_walkers=100,
        num_steps_per_block=100,
        num_blocks=10,
        num_blocks_equil=5,
        mc_timestep=0.001,
        compute_gradients=True,
    )

Results are written to ``<prefix>.chk.h5`` (energies and walker snapshots)
and ``<prefix>.grd.h5`` (gradient data).

Step 4 — Post-process forces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~OmegaQMC.utils.vmc_forces_with_pgcs` reads the gradient file and
returns symmetry-averaged nuclear forces with statistical error estimates:

.. code-block:: python

    from OmegaQMC.utils import vmc_forces_with_pgcs

    forces, forces_err = vmc_forces_with_pgcs(prefix=data_prefix)
    print("Forces (Ha/Bohr):\n", forces)
    print("Errors:\n", forces_err)

Optimizing Jastrow parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before a production VMC run, optimize the Jastrow factor
with :func:`~OmegaQMC.get_vmcopt_func`.  The default
implementation uses the **linear method**
:cite:`Umrigar2007,Toulouse2008` (a port of the QMCPACK
:cite:`Kim2018` ``OneShiftOnly`` algorithm), which
typically converges an order of magnitude faster than
gradient descent:

.. code-block:: python

    from OmegaQMC import get_vmcopt_func

    vmcopt_run = get_vmcopt_func(mf)
    params_opt, info = vmcopt_run(
        rng_key,
        num_walkers=1000,
        num_opt_samples=5000,
        num_epochs=50,
    )

``num_opt_samples`` controls how many walker snapshots are
collected per epoch to build the overlap and Hamiltonian
matrices; ``num_epochs`` should be at least
``3–5 × num_params`` for reliable convergence.

Pass the returned *params_opt* as ``params_corr`` to
:func:`~OmegaQMC.get_vmc_func` for the production run.

Two alternative optimizer implementations are also
available directly from their submodules:

- :func:`~OmegaQMC.vmcopt_gto_pssgd.get_vmcopt_func` —
  post-sampling SGD/Adam optimizer
- :func:`~OmegaQMC.vmcopt_gto_naive.get_vmcopt_func` —
  naïve optimizer (differentiates through MC; reference
  implementation only)

Multi-determinant trial wavefunction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A CASSCF multi-determinant expansion can replace the single Slater
determinant.  The helper
:func:`~OmegaQMC.afqmc_gto.extract_casscf_trial` converts a converged
CASSCF object into the ``trial`` dict accepted by
:func:`~OmegaQMC.get_vmc_func`.

Build the mean-field object with
:func:`~OmegaQMC.generate_molecular_orbitals`, then run CASSCF on top:

.. code-block:: python

    from pyscf import mcscf
    from OmegaQMC import generate_molecular_orbitals, get_vmc_func
    from OmegaQMC.afqmc_gto import extract_casscf_trial

    mf = generate_molecular_orbitals(
        "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        units="ang", basis="6-31G",
    )

    mc = mcscf.CASSCF(mf, ncas=4, nelecas=(2, 2))
    mc.kernel()

    trial = extract_casscf_trial(mc, coeff_threshold=1e-2)
    print(f"Determinants retained: {trial['ndet']}")

Pass the trial dict to :func:`~OmegaQMC.get_vmc_func` via the ``trial``
keyword.  MO relaxation is automatically disabled with a warning:

.. code-block:: python

    params_jastrow = {"J1_pade": {"O": 0.0, "H": 0.0}}

    vmc_run = get_vmc_func(
        mf, params_jastrow,
        prefix="h2o_msd_vmc",
        trial=trial,
    )

    vmc_run(
        rng_key,
        num_walkers=200,
        num_steps_per_block=100,
        num_blocks=50,
        num_blocks_equil=10,
        mc_timestep=0.001,
        compute_gradients=False,
    )

The ``coeff_threshold`` argument controls how many determinants are kept
from the CI expansion; smaller values include more determinants and
improve accuracy at higher computational cost.

B-spline Jastrow factors
~~~~~~~~~~~~~~~~~~~~~~~~~

For higher variational freedom, replace the two-parameter Padé
Jastrow :cite:`Drummond2004` with a cubic B-spline Jastrow
(following the QMCPACK :cite:`Kim2018` ``BsplineFunctor``
convention).  Cutoff radii are passed separately via a
``bspline_config`` dict:

.. code-block:: python

    import jax.numpy as jnp
    from OmegaQMC import get_vmcopt_func

    bspline_config = {
        "J1": {"H": {"r_cut": 5.0},
               "O": {"r_cut": 8.0}},
        "J2": {"r_cut": 10.0},
    }

    params_jastrow = {
        "J2_bspline": {
            "like":   jnp.zeros(8),
            "unlike": jnp.zeros(8),
        },
    }

    vmcopt_run = get_vmcopt_func(
        mf, bspline_config=bspline_config
    )
    params_opt, info = vmcopt_run(
        rng_key,
        params_corr_init=params_jastrow,
        num_walkers=500,
    )

The ``bspline_config`` dict specifies cutoff radii only
(structural, not optimized).  The number of variational
parameters is determined by the length of each coefficient
array in ``params_jastrow``.

Cusp constraints :cite:`Kato1957` are enforced
automatically:

- **J2 like-spin**: cusp value = -1/4
- **J2 unlike-spin**: cusp value = -1/2
- **J1**: cusp value = -Z (nuclear charge), or 0
  when cusp-corrected orbitals are used :cite:`Quady2025`

Both Padé and B-spline Jastrows can coexist (their
contributions are summed), though a warning is emitted.

AFQMC
-----

Auxiliary-field quantum Monte Carlo (AFQMC)
:cite:`Zhang2003` provides systematically improvable
correlation energies at polynomial cost.  The
``OmegaQMC`` package exposes a two-step API through
:func:`~OmegaQMC.get_afqmc_func`: first build the
driver, then call it to run the simulation.

Step 1 — Build the molecular system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AFQMC uses the standard PySCF interface directly.  Build and converge a
Hartree-Fock mean-field object:

.. code-block:: python

    from pyscf import gto, scf

    mol = gto.M(
        atom='H 0 0 0; H 0 0 1.4',   # H2 at equilibrium, bond length in Bohr
        basis='sto-6g',
        unit='Bohr',
        verbose=0,
    )

    mf = scf.RHF(mol)
    mf.kernel()
    print(f"E_HF = {mf.e_tot:.10f}")

Step 2 — Construct the AFQMC driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the converged mean-field object to :func:`~OmegaQMC.get_afqmc_func`.
The key algorithmic parameters are the imaginary-time step ``dt`` and the
Cholesky decomposition threshold ``chol_cut``:

.. code-block:: python

    import jax
    from OmegaQMC import get_afqmc_func

    driver = get_afqmc_func(mf, dt=0.005, chol_cut=1e-6)

Step 3 — Run the AFQMC simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call the driver with a JAX random key and walker / block settings:

.. code-block:: python

    result = driver(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=100,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_eqlb_blocks=10,
    )

    e_afqmc = result['energy_mean']
    e_err   = result['energy_err']
    print(f"E_AFQMC = {e_afqmc:.10f} +/- {e_err:.10f}")

The return value is a dict containing at minimum ``energy_mean`` and
``energy_err`` (one-sigma statistical error from block averaging).

Key parameters
~~~~~~~~~~~~~~

``num_eqlb_blocks``
    Number of equilibration blocks discarded before statistics are
    accumulated.  Set this to roughly 10 % of ``num_blocks``.

``stabilize_freq``
    Frequency (in steps) at which the walker overlap matrix is
    re-orthonormalized to prevent numerical instability.

``pop_control_freq``
    Frequency (in steps) at which population control is applied to keep
    the walker weights from diverging.

Comparing with FCI
~~~~~~~~~~~~~~~~~~~

For small systems you can cross-check the AFQMC energy against the exact
full-CI result:

.. code-block:: python

    from pyscf import fci

    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()
    print(f"E_FCI   = {e_fci:.10f}")
    print(f"E_corr (FCI)   = {e_fci   - mf.e_tot:.10f}")
    print(f"E_corr (AFQMC) = {e_afqmc - mf.e_tot:.10f}")

Multi-determinant trial wavefunction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AFQMC accuracy is sensitive to the trial wavefunction quality.
Replacing the single-determinant Hartree-Fock trial with a CASSCF
multi-determinant expansion reduces the phaseless approximation
:cite:`Zhang2003` bias, which is particularly significant in
strongly correlated systems.

Run an RHF calculation followed by CASSCF, then extract the trial
wavefunction using :func:`~OmegaQMC.afqmc_gto.extract_casscf_trial`:

.. code-block:: python

    from pyscf import gto, scf, mcscf
    from OmegaQMC import get_afqmc_func
    from OmegaQMC.afqmc_gto import extract_casscf_trial

    mol = gto.M(
        atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="6-31G", unit="angstrom", verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    mc = mcscf.CASSCF(mf, ncas=4, nelecas=(2, 2))
    mc.kernel()

    trial = extract_casscf_trial(mc, coeff_threshold=1e-2)
    print(f"Determinants retained: {trial['ndet']}")

Build and run the driver by passing the trial dict as the ``trial``
keyword argument:

.. code-block:: python

    driver = get_afqmc_func(
        mf, dt=0.005, chol_cut=1e-6, trial=trial
    )

    result = driver(
        rng_key=jax.random.key(42),
        num_walkers=100,
        num_blocks=200,
        num_steps_per_block=25,
        stabilize_freq=5,
        pop_control_freq=5,
        num_eqlb_blocks=20,
    )

    print(f"E_AFQMC = {result['energy_mean']:.10f} "
          f"+/- {result['energy_err']:.10f}")

The ``trial`` dict is shared between the VMC and AFQMC drivers, so the
same :func:`~OmegaQMC.afqmc_gto.extract_casscf_trial` call can feed
either driver without modification.
