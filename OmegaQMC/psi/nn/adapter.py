"""Adapter bridging NNX model to OmegaQMC convention.

Wraps a :class:`NeuralNetworkWaveFunction` (Flax NNX) into
the ``log_psi(elec_crds, nuc_crds, params) -> float``
calling convention used by OmegaQMC's VMC drivers.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import NNAnsatzConfig, load_nn_config
from .types import PhysicalConfiguration
from .wf import MoleculeInfo, NeuralNetworkWaveFunction
from .build import build_nn_wf


def make_nn_log_psi(config, mol_info, rng_key):
    """Create an NN trial wavefunction.

    Args:
        config: :class:`NNAnsatzConfig` or a string
            (built-in name or YAML path).
        mol_info: :class:`MoleculeInfo` instance.
        rng_key: JAX PRNG key for parameter init.

    Returns:
        Tuple ``(log_psi, init_params, graphdef)``.
        *log_psi* is a callable
        ``(elec_crds, nuc_crds, params) -> float``
        where *params* is an NNX ``State`` pytree.
        *init_params* is the initial parameter pytree.
        *graphdef* is the NNX ``GraphDef`` needed to
        reconstruct the model via ``nnx.merge``.
    """
    if isinstance(config, str):
        config = load_nn_config(config)

    rngs = nnx.Rngs(rng_key)
    model = build_nn_wf(config, mol_info, rngs)

    # Separate trainable params from other state
    # (e.g. Embed indices, RNG state).
    graphdef, params, other = nnx.split(
        model, nnx.Param, ...,
    )
    init_params = params

    n_up = mol_info.n_up

    def log_psi(elec_crds, nuc_crds, params):
        """Evaluate log|psi| for a single walker.

        Args:
            elec_crds: ``(nelec, 3)`` interleaved
                alpha/beta (OmegaQMC convention).
            nuc_crds: ``(natom, 3)``
            params: NNX State pytree (trainable
                ``nnx.Param`` only).

        Returns:
            Scalar log|psi| value.
        """
        # Reorder: interleaved → grouped
        r_up = elec_crds[::2]
        r_dn = elec_crds[1::2]
        r_grouped = jnp.concatenate(
            [r_up, r_dn], axis=0,
        )

        phys_conf = PhysicalConfiguration(
            R=nuc_crds,
            r=r_grouped,
            mol_idx=jnp.array(0),
        )

        mdl = nnx.merge(
            graphdef, params, other,
        )
        psi = mdl(phys_conf)
        return psi.log

    return log_psi, init_params, graphdef
