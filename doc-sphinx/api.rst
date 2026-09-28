API Reference
=============

Top-level functions
--------------------

.. autofunction:: OmegaQMC.generate_molecular_orbitals

   Constructs the mean-field object via PySCF
   :cite:`Sun2020`.

.. autofunction:: OmegaQMC.get_afqmc_func

.. autofunction:: OmegaQMC.get_qed_afqmc_func

   Implements the phaseless QED-AFQMC method
   :cite:`Weber2025`.

GTO VMC driver
---------------

.. autofunction:: OmegaQMC.vmc_gto.get_vmc_gto_func

.. autoclass:: OmegaQMC.vmc_gto._VMCDriverGTO
   :members: __call__

GTO VMC optimizers
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

.. autofunction:: OmegaQMC.vmcopt_gto_linear.get_vmcopt_gto_func

.. autoclass:: OmegaQMC.vmcopt_gto_linear._VMCOptDriverGTO_Linear
   :members: __call__

**Iteratively-resampled SGD (IRSGD)**

Alternates MCMC resampling with SGD or Adam epochs,
avoiding stale-sample drift while remaining more
memory-efficient than the naïve approach.

.. autofunction:: OmegaQMC.vmcopt_gto_irsgd.get_vmcopt_gto_func

.. autoclass:: OmegaQMC.vmcopt_gto_irsgd._VMCOptDriverGTO_IRSGD
   :members: __call__

**Naïve (reference)**

Differentiates through the entire MC trajectory at each
epoch.  Memory-intensive; intended as a reference
implementation only.

.. autofunction:: OmegaQMC.vmcopt_gto_naive.get_vmcopt_gto_func

.. autoclass:: OmegaQMC.vmcopt_gto_naive._VMCOptDriverGTO_Naive
   :members: __call__

NN VMC driver
--------------

.. autofunction:: OmegaQMC.vmc_nn.get_vmc_nn_func

.. autoclass:: OmegaQMC.vmc_nn._VMCDriverNN
   :members: __call__, load_checkpoint

NN VMC optimizers
------------------

**Stochastic Reconfiguration** (recommended)

Natural-gradient optimizer using Stochastic
Reconfiguration (SR).  Instead of Adam, the parameter
update direction is obtained by solving the
natural-gradient equation via conjugate gradient,
accounting for the Fisher information metric of the
wavefunction.  Scales to ~100K parameters without
forming the full overlap matrix.

.. autofunction:: OmegaQMC.vmcopt_nn_sr.get_vmcopt_nn_func

.. autoclass:: OmegaQMC.vmcopt_nn_sr._VMCOptDriverNN_SR
   :members: __call__

**Iteratively-resampled Adam (IRAdam)**

.. autofunction:: OmegaQMC.vmcopt_nn_iradam.get_vmcopt_nn_func

.. autoclass:: OmegaQMC.vmcopt_nn_iradam._VMCOptDriverNN_IRAdam
   :members: __call__

.. autofunction:: OmegaQMC.vmcopt_nn_iradam.pretrain_to_hf

**Kronecker-factored natural gradient (KFAC)**

Natural-gradient optimizer whose curvature is a per-layer
Kronecker-factored approximation of the Fisher matrix, so its
state does not grow with the walker count.  The step is bounded
by a Fisher-norm trust region (``norm_constraint``), and the
defaults follow DeepQMC's KFAC configuration, including constant
damping.

.. autofunction:: OmegaQMC.vmcopt_nn_kfac.get_vmcopt_nn_func

.. autoclass:: OmegaQMC.vmcopt_nn_kfac._VMCOptDriverNN_KFAC
   :members: __call__

**Excited states: NES-VMC penalty methods**

Optimise a trial orthogonal to a frozen ground state by adding an
overlap penalty to the loss.  The three variants differ in how the
overlap is measured: against a ground-state NN trial, against a
synthetic ground state built from a CI vector, or as the cosine
between CI vectors.  :class:`~OmegaQMC.vmcopt_nn_nes._VMCOptDriverNN_NES`
uses the IRAdam loop
(:meth:`~OmegaQMC.vmcopt_nn_iradam._VMCOptDriverNN_IRAdam.__call__`).

.. autofunction:: OmegaQMC.vmcopt_nn_nes.get_vmcopt_nn_nes_func

.. autoclass:: OmegaQMC.vmcopt_nn_nes._VMCOptDriverNN_NES
   :members: evaluate_overlap

.. autofunction:: OmegaQMC.vmcopt_nn_nes.get_vmcopt_nn_nes_basis_func

.. autoclass:: OmegaQMC.vmcopt_nn_nes._VMCOptDriverNN_NES_Basis
   :members: __call__

.. autofunction:: OmegaQMC.vmcopt_nn_nes.get_vmcopt_nn_nes_ci_func

.. autoclass:: OmegaQMC.vmcopt_nn_nes._VMCOptDriverNN_NES_CIOverlap
   :members: __call__

**Excited states: determinantal K-state optimizers**

Optimise K states jointly through the determinant of single-state
trials evaluated at K configurations, which enforces orthogonality
without a penalty (see :mod:`OmegaQMC.vmcopt_nn_pfau`).  Joint
walkers are updated by stochastic reconfiguration.

.. autofunction:: OmegaQMC.vmcopt_nn_pfau.get_vmcopt_nn_pfau_k2_func

.. autoclass:: OmegaQMC.vmcopt_nn_pfau._VMCOptDriverNN_Pfau_K2
   :members: __call__

.. autofunction:: OmegaQMC.vmcopt_nn_pfau.get_vmcopt_nn_pfau_k_func

.. autoclass:: OmegaQMC.vmcopt_nn_pfau._VMCOptDriverNN_Pfau_K
   :members: __call__

NN checkpoints
---------------

.. autofunction:: OmegaQMC.psi.nn.checkpoint.save_nn_checkpoint

.. autofunction:: OmegaQMC.psi.nn.checkpoint.load_nn_checkpoint

.. autofunction:: OmegaQMC.psi.nn.checkpoint.append_vmc_results

AFQMC driver
-------------

.. autoclass:: OmegaQMC.afqmc_gto._AFQMCDriverGTO
   :members: __call__

**Energy-streaming variant**

.. automodule:: OmegaQMC.afqmc_gto_estream

.. autoclass:: OmegaQMC.afqmc_gto_estream._AFQMCDriverGTO_EStream
   :members: __call__

QED-AFQMC driver
-----------------

.. autoclass:: OmegaQMC.qed_afqmc_gto._QEDAFQMCDriverGTO
   :members: __call__

QED-VMC driver and optimizer
-----------------------------

Joint electron-photon VMC for a molecule coupled to one cavity mode
(dipole-gauge Pauli-Fierz Hamiltonian), sampling electron
coordinates and a photon Fock index together.  The optimizer builds
its own QED-VMC driver (``opt.driver``) and returns the optimised
parameters without writing them back; assign them to a driver's
``params`` to evaluate the trial.

.. autofunction:: OmegaQMC.qed_vmc_nn.get_qed_vmc_nn_func

.. autoclass:: OmegaQMC.qed_vmc_nn._QEDVMCDriverNN
   :members: __call__, initialize_walkers

.. autofunction:: OmegaQMC.qed_vmcopt_nn_sr.get_qed_vmcopt_nn_sr_func

.. autoclass:: OmegaQMC.qed_vmcopt_nn_sr._QEDVMCOptDriverNN_SR
   :members: __call__

HEG drivers
------------

Drivers specialised for the homogeneous electron gas.  Two families
are available:

* **Neural-network VMC** on top of the
  :class:`~OmegaQMC.psi.nn.heg_wf.HEGConfig` ansatz, with optional
  twist averaging.
* **Plane-wave AFQMC** in the jellium plane-wave basis, with
  symmetrised Cholesky vectors and twist averaging.

A supervised Hartree-Fock pre-training stage prepares the NN
ansatz for the energy optimizers.

NN VMC driver
~~~~~~~~~~~~~

.. autofunction:: OmegaQMC.vmc_nn_heg.get_vmc_nn_heg_func

.. autoclass:: OmegaQMC.vmc_nn_heg._VMCDriverNNHEG
   :members: __call__, load_checkpoint, initialize_walkers

.. autofunction:: OmegaQMC.vmc_nn_heg.get_vmc_nn_heg_twist_func

.. autoclass:: OmegaQMC.vmc_nn_heg._VMCDriverNNHEG_Twist
   :members: __call__

.. autofunction:: OmegaQMC.vmc_nn_heg.run_twist_averaged_heg

NN VMC optimizers
~~~~~~~~~~~~~~~~~

**Stochastic Reconfiguration** (recommended for production runs)

.. autofunction:: OmegaQMC.vmcopt_nn_heg_sr.get_vmcopt_nn_heg_sr_func

.. autoclass:: OmegaQMC.vmcopt_nn_heg_sr._VMCOptDriverNNHEG_SR
   :members: __call__

**KFAC** (Kronecker-factored approximate curvature)

.. autofunction:: OmegaQMC.vmcopt_nn_heg_kfac.get_vmcopt_nn_heg_kfac_func

.. autoclass:: OmegaQMC.vmcopt_nn_heg_kfac._VMCOptDriverNNHEG_KFAC
   :members: __call__

**Adam** (reference / lightweight smoke tests)

.. autofunction:: OmegaQMC.vmcopt_nn_heg.get_vmcopt_nn_heg_func

.. autoclass:: OmegaQMC.vmcopt_nn_heg._VMCOptDriverNNHEG_Adam
   :members: __call__

Supervised pre-training
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: OmegaQMC.pretrain_heg

.. autofunction:: OmegaQMC.pretrain_heg.pretrain_heg_psiformer

.. autoclass:: OmegaQMC.pretrain_heg._HEGPreTrainDriver
   :members: __call__

Plane-wave AFQMC driver
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: OmegaQMC.afqmc_pw_heg.build_3deg_system

.. autofunction:: OmegaQMC.afqmc_pw_heg.get_afqmc_3deg_func

.. autoclass:: OmegaQMC.afqmc_pw_heg._AFQMCDriverPWHEG
   :members: __call__

.. autofunction:: OmegaQMC.afqmc_pw_heg.run_twist_averaged_afqmc_3deg

Add-on calculators
-------------------

Non-QMC reference calculators living under :mod:`OmegaQMC.addons`.

.. autofunction:: OmegaQMC.addons.qed_hf.run_qed_hf

.. autofunction:: OmegaQMC.addons.qed_ccsd.run_qed_ccsd

.. autofunction:: OmegaQMC.addons.qed_fci.run_qed_fci

Observables
-----------

Energy estimators
~~~~~~~~~~~~~~~~~

.. autofunction:: OmegaQMC.observables.energy.local_energy_1body

.. autofunction:: OmegaQMC.observables.energy.local_energy_2body

.. autofunction:: OmegaQMC.observables.energy.local_energy

.. autofunction:: OmegaQMC.observables.energy.local_energy_multidet

Nuclear forces
~~~~~~~~~~~~~~

.. autofunction:: OmegaQMC.observables.force.vmc_gto_gradients

.. autofunction:: OmegaQMC.observables.force.save_gto_gradients

.. autofunction:: OmegaQMC.observables.force.postproc_h5_pgcs

.. autofunction:: OmegaQMC.observables.force.vmc_nn_gradients_zvzb

.. autofunction:: OmegaQMC.observables.force.save_nn_gradients

Green's functions
~~~~~~~~~~~~~~~~~

.. autofunction:: OmegaQMC.observables.greens.greens_function

.. autofunction:: OmegaQMC.observables.greens.greens_function_multidet

Integrals
---------

Cholesky decomposition
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: OmegaQMC.integrals.cholesky.chunked_cholesky

.. autofunction:: OmegaQMC.integrals.cholesky.prepare_afqmc_integrals

.. autofunction:: OmegaQMC.integrals.cholesky.half_rotate_cholesky

.. autofunction:: OmegaQMC.integrals.cholesky.half_rotate_cholesky_multidet

QED integrals
~~~~~~~~~~~~~

.. autofunction:: OmegaQMC.integrals.qed.prepare_qed_integrals

Trial wavefunctions
--------------------

Interfaces
~~~~~~~~~~

.. autoclass:: OmegaQMC.psi.VMCTrialState

.. autoclass:: OmegaQMC.psi.AFQMCTrialState

GTO trial
~~~~~~~~~

.. autofunction:: OmegaQMC.psi.gto.get_psi_fun

.. autofunction:: OmegaQMC.psi.gto.extract_casscf_trial

Cusp corrections
~~~~~~~~~~~~~~~~

.. autofunction:: OmegaQMC.psi.cusp.get_cusp_params

Neural network trial
~~~~~~~~~~~~~~~~~~~~~

Ports the DeepQMC architectures (PauliNet, FermiNet, DeepErwin,
PsiFormer) using Flax NNX.  All NN-related code lives under
``OmegaQMC.psi.nn``.

.. autofunction:: OmegaQMC.psi.nn.adapter.make_nn_log_psi

.. autoclass:: OmegaQMC.utils.Mole_custom
   :members: n_up, n_down, charges, coords,
             mol_shells, mol_ecp_shells, from_arrays

.. autoclass:: OmegaQMC.psi.nn.wf.NeuralNetworkWaveFunction
   :members: __call__

Configuration
'''''''''''''

.. autoclass:: OmegaQMC.psi.nn.config.NNAnsatzConfig
   :members: __post_init__

.. autofunction:: OmegaQMC.psi.nn.config.load_nn_config

Types
'''''

.. autoclass:: OmegaQMC.psi.nn.types.PhysicalConfiguration

.. autoclass:: OmegaQMC.psi.nn.types.Psi

Compatibility
'''''''''''''

.. autofunction:: OmegaQMC.psi.nn.compat.param_value

.. autofunction:: OmegaQMC.psi.nn.compat.register_pytree

Layers
''''''

.. autoclass:: OmegaQMC.psi.nn.layers.MLP
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.layers.ResidualConnection
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.layers.GLU
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.layers.SumPool

.. autoclass:: OmegaQMC.psi.nn.layers.Identity

Envelopes and cusp corrections
''''''''''''''''''''''''''''''''

.. autoclass:: OmegaQMC.psi.nn.env.ExponentialEnvelopes
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.cusp.ElectronicCuspAsymptotic
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.cusp.NuclearCuspAsymptotic
   :members: __call__

OmniNet (GNN + Jastrow + Backflow)
''''''''''''''''''''''''''''''''''''

.. autoclass:: OmegaQMC.psi.nn.omni.OmniNet
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.omni.Jastrow
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.omni.Backflow
   :members: __call__

Graph neural network
'''''''''''''''''''''

.. autoclass:: OmegaQMC.psi.nn.gnn.electron_gnn.ElectronGNN
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.gnn.electron_gnn.ElectronGNNLayer
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.gnn.electron_gnn.ElectronEmbedding
   :members: __call__

Edge features
'''''''''''''

.. autoclass:: OmegaQMC.psi.nn.gnn.edge_features.DifferenceEdgeFeature
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.gnn.edge_features.DistancePowerEdgeFeature
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.gnn.edge_features.GaussianEdgeFeature
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.gnn.edge_features.CombinedEdgeFeature
   :members: __call__

Update features
''''''''''''''''

.. autoclass:: OmegaQMC.psi.nn.gnn.update_features.ResidualElectronUpdateFeature
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.gnn.update_features.NodeSumElectronUpdateFeature
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.gnn.update_features.EdgeSumElectronUpdateFeature
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.gnn.update_features.ConvolutionElectronUpdateFeature
   :members: __call__

.. autoclass:: OmegaQMC.psi.nn.gnn.update_features.NodeAttentionElectronUpdateFeature
   :members: __call__

Physics utilities
''''''''''''''''''

.. autofunction:: OmegaQMC.psi.nn.physics.pairwise_diffs

.. autofunction:: OmegaQMC.psi.nn.physics.pairwise_self_distance

.. autofunction:: OmegaQMC.psi.nn.physics.laplacian

Graph utilities
''''''''''''''''

.. autofunction:: OmegaQMC.psi.nn.gnn.graph.GraphEdgeBuilder

.. autofunction:: OmegaQMC.psi.nn.gnn.graph.MolecularGraphEdgeBuilder

.. autofunction:: OmegaQMC.psi.nn.gnn.graph.GraphUpdate

.. autoclass:: OmegaQMC.psi.nn.gnn.graph.SimpleGraphEdges

.. autoclass:: OmegaQMC.psi.nn.gnn.graph.SameGraphEdges

.. autoclass:: OmegaQMC.psi.nn.gnn.graph.AntiGraphEdges

Utility functions
''''''''''''''''''

.. autofunction:: OmegaQMC.psi.nn.utils.norm

.. autofunction:: OmegaQMC.psi.nn.utils.triu_flat

.. autofunction:: OmegaQMC.psi.nn.utils.flatten

.. autofunction:: OmegaQMC.psi.nn.utils.unflatten

Symmetry
--------

Point-group operations, the fragment helpers behind Point Group
Correlated Sampling (PGCS), and the molecular orientation and
symmetrization applied by
:func:`~OmegaQMC.vmc_gto.generate_molecular_orbitals`.

Operations
~~~~~~~~~~

.. automodule:: OmegaQMC.symm.operations

.. autofunction:: OmegaQMC.symm.operations.populate_fragment_symmops

.. autofunction:: OmegaQMC.symm.operations.get_global_symmops

Fragment helpers for PGCS
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: OmegaQMC.symm.fragments

.. autofunction:: OmegaQMC.symm.fragments.build_frag_symmops

.. autofunction:: OmegaQMC.symm.fragments.build_frag_transform_data

.. autofunction:: OmegaQMC.symm.fragments.build_single_frag_combos

.. autofunction:: OmegaQMC.symm.fragments.make_apply_single_frag_symmop

Orientation and symmetrization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: OmegaQMC.symm.point_groups

.. autofunction:: OmegaQMC.symm.point_groups.charge_inertia_axes

.. autofunction:: OmegaQMC.symm.point_groups.canonicalize_symmetry_axes

.. autofunction:: OmegaQMC.symm.point_groups.detect_symmetry_quality

.. autofunction:: OmegaQMC.symm.point_groups.auto_symmetrize_molecule

.. autofunction:: OmegaQMC.symm.point_groups.symmetrize_molecule

.. autofunction:: OmegaQMC.symm.point_groups.get_symmetrizer

.. autoclass:: OmegaQMC.symm.point_groups.PointGroupSymmetrizer

Utilities
----------

.. autofunction:: OmegaQMC.utils.format_basis_name

.. autofunction:: OmegaQMC.utils.do_binning_analysis

