API Reference
=============

Top-level functions
--------------------

.. autofunction:: OmegaQMC.generate_molecular_orbitals

.. autofunction:: OmegaQMC.get_vmc_func

.. autofunction:: OmegaQMC.get_vmcopt_func

.. autofunction:: OmegaQMC.get_afqmc_func

.. autofunction:: OmegaQMC.get_qed_afqmc_func

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
