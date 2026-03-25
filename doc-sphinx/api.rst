API Reference
=============

Top-level functions
--------------------

.. autofunction:: OmegaQMC.generate_molecular_orbitals

   Constructs the mean-field object via PySCF
   :cite:`Sun2020`.

.. autofunction:: OmegaQMC.get_vmc_func

.. autofunction:: OmegaQMC.get_vmcopt_func

.. autofunction:: OmegaQMC.get_afqmc_func

.. autofunction:: OmegaQMC.get_qed_afqmc_func

   Implements the phaseless QED-AFQMC method
   :cite:`Weber2025`.

VMC driver
-----------

.. autoclass:: OmegaQMC.vmc_gto._VMCDriver
   :members: __call__

AFQMC driver
-------------

.. autoclass:: OmegaQMC.afqmc_gto._AFQMCDriver
   :members: __call__

QED-AFQMC driver
-----------------

.. autoclass:: OmegaQMC.qed_afqmc_gto._QEDAFQMCDriver
   :members: __call__

QED-FCI
--------

.. autofunction:: OmegaQMC.qed_fci.qed_fci

Utilities
----------

.. autofunction:: OmegaQMC.utils.vmc_forces_with_pgcs

.. autofunction:: OmegaQMC.utils.format_basis_name

.. autofunction:: OmegaQMC.utils.do_binning_analysis

Cusp corrections
-----------------

.. autofunction:: OmegaQMC.cusp.get_cusp_params

Jastrow optimizers
-------------------

Three optimizer implementations are provided, in order of
decreasing efficiency:

**Linear method** (recommended)

Ports the QMCPACK :cite:`Kim2018` ``OneShiftOnly``
algorithm :cite:`Umrigar2007,Toulouse2008`.  At each
epoch it builds overlap (S) and Hamiltonian (H) matrices
from per-walker log-psi and local-energy derivatives,
then solves a shifted generalized eigenvalue problem
for the parameter update.

.. autofunction:: OmegaQMC.vmcopt_gto_linear.get_vmcopt_func

.. autoclass:: OmegaQMC.vmcopt_gto_linear._VMCOptLinearDriver
   :members: __call__

**Post-sampling SGD**

Collects walker snapshots in a sampling phase, then
minimizes a combined energy-plus-variance loss on those
snapshots with SGD or Adam.

.. autofunction:: OmegaQMC.vmcopt_gto_pssgd.get_vmcopt_func

.. autoclass:: OmegaQMC.vmcopt_gto_pssgd._VMCOptDriver
   :members: __call__

**Naïve (reference)**

Differentiates through the entire MC trajectory at each
epoch.  Memory-intensive; intended as a reference
implementation only.

.. autofunction:: OmegaQMC.vmcopt_gto_naive.get_vmcopt_func

.. autoclass:: OmegaQMC.vmcopt_gto_naive._VMCOptNaiveDriver
   :members: __call__
