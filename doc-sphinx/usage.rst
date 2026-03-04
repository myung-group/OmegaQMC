Getting Started
===============

This page walks through a minimal VMC workflow using the ``vmc_pgcs`` package.
The example is based on the water-dimer test script and runs a short VMC
simulation that computes nuclear forces via point-group correlated sampling
(PGCS).

Step 1 — Build the molecular system
------------------------------------

:func:`~vmc_pgcs.generate_molecular_orbitals` accepts an inline atom string or
the path to an ``.xyz`` file, runs a PySCF mean-field calculation, and returns
the converged mean-field object:

.. code-block:: python

    from vmc_pgcs import generate_molecular_orbitals

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
-----------------------------------

Pass the mean-field object and Jastrow parameters to
:func:`~vmc_pgcs.get_vmc_func`:

.. code-block:: python

    import jax.numpy as jnp
    from vmc_pgcs import get_vmc_func
    from vmc_pgcs.utils import format_basis_name

    params_jastrow = {"J2_params": jnp.array([])}   # no Jastrow factor

    data_prefix = "water2_vmc_{}".format(format_basis_name("6-31G"))

    symmetry_ops = {1: ["E"], 2: ["E"]}   # identity only per fragment

    vmc_run = get_vmc_func(
        mf,
        params_jastrow,
        prefix=data_prefix,
        symmop_list=symmetry_ops,
    )

Step 3 — Run the VMC loop
--------------------------

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
------------------------------

:func:`~vmc_pgcs.utils.vmc_forces_with_pgcs` reads the gradient file and
returns symmetry-averaged nuclear forces with statistical error estimates:

.. code-block:: python

    from vmc_pgcs.utils import vmc_forces_with_pgcs

    forces, forces_err = vmc_forces_with_pgcs(prefix=data_prefix)
    print("Forces (Ha/Bohr):\n", forces)
    print("Errors:\n", forces_err)

Optimizing Jastrow parameters
-------------------------------

Before a production VMC run you may want to optimize the Jastrow factor using
:func:`~vmc_pgcs.get_vmcopt_func`:

.. code-block:: python

    from vmc_pgcs import get_vmcopt_func

    vmcopt_run = get_vmcopt_func(mf)
    params_opt, info = vmcopt_run(rng_key, num_walkers=500, num_steps=200)

Pass the returned *params_opt* as ``params_corr`` to :func:`~vmc_pgcs.get_vmc_func`
for the production run.
