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

**Post-sampling SGD**

Collects walker snapshots in a sampling phase, then
minimizes a combined energy-plus-variance loss on those
snapshots with SGD or Adam.

.. autofunction:: OmegaQMC.vmcopt_gto_pssgd.get_vmcopt_gto_func

.. autoclass:: OmegaQMC.vmcopt_gto_pssgd._VMCOptDriverGTO_PSSGD
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

NN VMC optimizer
-----------------

.. autofunction:: OmegaQMC.vmcopt_nn.get_vmcopt_nn_func

.. autoclass:: OmegaQMC.vmcopt_nn._VMCOptDriverNN
   :members: __call__

.. autofunction:: OmegaQMC.vmcopt_nn.pretrain_to_hf

NN checkpoints
---------------

.. autofunction:: OmegaQMC.nn_checkpoint.save_nn_checkpoint

.. autofunction:: OmegaQMC.nn_checkpoint.load_nn_checkpoint

.. autofunction:: OmegaQMC.nn_checkpoint.append_vmc_results

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

.. autofunction:: OmegaQMC.integrals.cholesky.extract_casscf_trial

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

Cusp corrections
~~~~~~~~~~~~~~~~

.. autofunction:: OmegaQMC.psi.cusp.get_cusp_params

Neural network trial
~~~~~~~~~~~~~~~~~~~~~

Ports the DeepQMC architectures (PauliNet, FermiNet, DeepErwin,
PsiFormer) using Flax NNX.  All NN-related code lives under
``OmegaQMC.psi.nn``.

.. autofunction:: OmegaQMC.psi.nn.adapter.make_nn_log_psi

.. autoclass:: OmegaQMC.psi.nn.wf.MoleculeInfo
   :members:

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

Utilities
----------

.. autofunction:: OmegaQMC.utils.format_basis_name

.. autofunction:: OmegaQMC.utils.do_binning_analysis

