API Reference
=============

Top-level functions
--------------------

.. autofunction:: vmc_pgcs.generate_molecular_orbitals

.. autofunction:: vmc_pgcs.get_vmc_func

.. autofunction:: vmc_pgcs.get_vmcopt_func

.. autofunction:: vmc_pgcs.get_afqmc_func

.. autofunction:: vmc_pgcs.get_qed_afqmc_func

VMC driver
-----------

.. autoclass:: vmc_pgcs.vmc_gto._VMCDriver
   :members: __call__

AFQMC driver
-------------

.. autoclass:: vmc_pgcs.afqmc_gto._AFQMCDriver
   :members: __call__

QED-AFQMC driver
-----------------

.. autoclass:: vmc_pgcs.qed_afqmc_gto._QEDAFQMCDriver
   :members: __call__

QED-FCI
--------

.. autofunction:: vmc_pgcs.qed_fci.qed_fci

Utilities
----------

.. autofunction:: vmc_pgcs.utils.vmc_forces_with_pgcs

.. autofunction:: vmc_pgcs.utils.format_basis_name

.. autofunction:: vmc_pgcs.utils.do_binning_analysis

Cusp corrections
-----------------

.. autofunction:: vmc_pgcs.cusp.get_cusp_params
