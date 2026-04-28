"""Adapter bridging NNX model to OmegaQMC convention.

Wraps a :class:`NeuralNetworkWaveFunction` (Flax NNX) into
the ``log_psi(elec_crds, nuc_crds, params) -> float``
calling convention used by OmegaQMC's VMC drivers, and
exposes a forward-Laplacian callable for kinetic energy
evaluation.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import NNAnsatzConfig, load_nn_config
from .forward_lap import log_psi_vgl_psiformer
from .gnn.edge_features import (
    CombinedEdgeFeature,
    DifferenceEdgeFeature,
    DistancePowerEdgeFeature,
    GaussianEdgeFeature,
)
from .gnn.update_features import (
    ConvolutionElectronUpdateFeature,
    EdgeSumElectronUpdateFeature,
    NodeAttentionElectronUpdateFeature,
    NodeSumElectronUpdateFeature,
    ResidualElectronUpdateFeature,
)
from .physics import laplacian
from .types import PhysicalConfiguration
from .wf import NeuralNetworkWaveFunction
from .build import build_nn_wf


def _psiformer_compat(config) -> bool:
    """Return True if ``config`` is supported by the
    PsiFormer VGL forward-Laplacian path."""
    if not config.full_determinant:
        return False
    if config.backflow_transform != 'mult':
        return False
    if config.conf_coeff != 'sum_pool':
        return False
    if not config.envelope_per_orbital_exponent:
        return False
    if config.envelope_spin_restricted:
        return False
    if not config.use_backflow:
        return False
    if config.cusp_nuclei:
        return False
    if (
        config.cusp_electrons
        and config.cusp_electrons_type
        not in ('psiformer', 'deepqmc')
    ):
        return False
    if config.deep_features not in (False, 'shared'):
        return False
    if config.ne_powers != [1]:
        return False
    if not config.update_features:
        return False
    for uf in config.update_features:
        if uf.get('type') not in (
            'node_attention', 'residual', 'node_sum',
            'edge_sum', 'convolution',
        ):
            return False
    return True


def _extract_mlp_layers(mlp):
    out = []
    for layer in mlp.layers:
        w = layer.kernel.value
        b = layer.bias.value if layer.use_bias else None
        out.append((w, b))
    return out


def _mlp_activation_name(mlp):
    """Reverse-lookup the MLP's resolved activation
    callable to a string key understood by
    :data:`_VGL_ACTIVATIONS` in ``forward_lap``."""
    from .layers import _ACTIVATIONS

    fn = mlp.activation
    if fn is None:
        return None
    for name, candidate in _ACTIVATIONS.items():
        if candidate is fn:
            return name
    raise ValueError(
        f"unsupported MLP activation: {fn!r}",
    )


def _extract_update_spec(uf):
    if isinstance(uf, NodeAttentionElectronUpdateFeature):
        attn = uf.attention
        attn_norm = (
            uf.attention_residual.normalize
            if uf.attention_residual is not None
            else None
        )
        mlp_norm = (
            uf.mlp_residual.normalize
            if uf.mlp_residual is not None
            else None
        )
        return {
            'type': 'node_attention',
            'attn_wq': attn.wq.kernel.value,
            'attn_wk': attn.wk.kernel.value,
            'attn_wv': attn.wv.kernel.value,
            'attn_wo': attn.wo.kernel.value,
            'attn_num_heads': attn.num_heads,
            'attn_mlp_layers': _extract_mlp_layers(uf.mlp),
            'attn_mlp_activation': 'tanh',
            'attn_mlp_last_linear': False,
            'attn_residual_normalize': attn_norm,
            'attn_mlp_residual_normalize': mlp_norm,
        }
    if isinstance(uf, ResidualElectronUpdateFeature):
        return {'type': 'residual'}
    if isinstance(uf, NodeSumElectronUpdateFeature):
        return {
            'type': 'node_sum',
            'node_types': uf.node_types,
            'normalize': uf.normalize,
        }
    if isinstance(uf, EdgeSumElectronUpdateFeature):
        return {
            'type': 'edge_sum',
            'edge_types': uf.edge_types,
            'normalize': uf.normalize,
        }
    if isinstance(uf, ConvolutionElectronUpdateFeature):
        def _mlp_spec(net):
            return {
                'layers': _extract_mlp_layers(net),
                'activation': _mlp_activation_name(net),
                'last_linear': net.last_linear,
            }
        w_specs = {
            et: _mlp_spec(uf.w_nets[et])
            for et in uf.edge_types
        }
        h_specs = {
            et: _mlp_spec(uf.h_nets[et])
            for et in uf.edge_types
        }
        return {
            'type': 'convolution',
            'edge_types': uf.edge_types,
            'normalize': uf.normalize,
            'w_specs': w_specs,
            'h_specs': h_specs,
        }
    raise ValueError(
        f"unsupported update feature: {type(uf).__name__}",
    )


def _extract_edge_feature_spec(feat):
    """Reverse-lookup an edge-feature module to a spec dict
    accepted by :func:`vgl_apply_edge_feature`."""
    if isinstance(feat, CombinedEdgeFeature):
        return {
            'type': 'combined',
            'features': [
                _extract_edge_feature_spec(f)
                for f in feat.features
            ],
        }
    if isinstance(feat, DistancePowerEdgeFeature):
        spec = {
            'type': 'distance_power',
            'powers': [int(p) for p in feat.powers],
            'log_rescale': feat.log_rescale,
        }
        if feat.eps:
            spec['eps'] = feat.eps
        return spec
    if isinstance(feat, DifferenceEdgeFeature):
        return {
            'type': 'difference',
            'log_rescale': feat.log_rescale,
        }
    if isinstance(feat, GaussianEdgeFeature):
        return {
            'type': 'gaussian',
            'mus': feat.mus,
            'sigmas': feat.sigmas,
        }
    raise ValueError(
        f"unsupported edge feature: {type(feat).__name__}",
    )


def _extract_layer_spec(layer):
    e_res_norm = (
        layer.electron_residual.normalize
        if layer.electron_residual is not None else None
    )
    update_specs = [
        _extract_update_spec(uf)
        for uf in layer.update_features
    ]
    spec = {
        'update_specs': update_specs,
        'subnet_layers': _extract_mlp_layers(layer.subnet),
        'subnet_activation': 'tanh',
        'subnet_last_linear': False,
        'electron_residual_normalize': e_res_norm,
    }
    if layer.deep_features == 'shared':
        spec['subnet_g_layers'] = _extract_mlp_layers(
            layer.subnet_g,
        )
        spec['subnet_g_activation'] = (
            _mlp_activation_name(layer.subnet_g)
        )
        spec['subnet_g_last_linear'] = (
            layer.subnet_g.last_linear
        )
        tp_res = layer.tp_residual
        spec['tp_residual_normalize'] = (
            tp_res.normalize if tp_res is not None else None
        )
    return spec


def _build_vgl_kwargs(model, *, ne_log_rescale=True):
    """Extract VGL kwargs from a built PsiFormer model."""
    gnn = model.omni.gnn
    embed_mod = gnn.electron_embedding
    proj_W = (
        embed_mod.proj.kernel.value
        if embed_mod.proj is not None else None
    )
    embedding_kwargs = {
        'ne_powers': [1],
        'ne_log_rescale': ne_log_rescale,
        'use_spin': embed_mod.use_spin,
        'proj_W': proj_W,
    }
    layer_specs = [
        _extract_layer_spec(lyr) for lyr in gnn.layers
    ]
    bf_up_layers = _extract_mlp_layers(
        model.omni.backflow['up'].nets[0],
    )
    bf_down_layers = _extract_mlp_layers(
        model.omni.backflow['down'].nets[0],
    )
    env = model.envelope
    envelope = {
        'center_idx': env.center_idx,
        'zetas_up': env.zetas_up.value,
        'zetas_down': env.zetas_down.value,
        'pi_up': env.pi_up.value,
        'pi_down': env.pi_down.value,
        'isotropic': env.isotropic,
        'per_orbital_exponent': env.per_orbital_exponent,
        'softplus_zeta': env.softplus_zeta,
    }
    if model.omni.jastrow is not None:
        jas_mod = model.omni.jastrow
        jastrow = {
            'layers': _extract_mlp_layers(jas_mod.net),
            'activation': _mlp_activation_name(jas_mod.net),
            'last_linear': jas_mod.net.last_linear,
            'sum_first': jas_mod.sum_first,
        }
    else:
        jastrow = None
    if model.cusp_electrons is not None:
        cusp_mod = model.cusp_electrons
        cusp_cls_name = type(
            cusp_mod.cusp_function,
        ).__name__
        if cusp_cls_name == 'PsiformerCusp':
            cusp_type = 'psiformer'
        elif cusp_cls_name == 'DeepQMCCusp':
            cusp_type = 'deepqmc'
        else:
            raise ValueError(
                f"unsupported cusp class: {cusp_cls_name}",
            )
        if cusp_mod.trainable_alpha:
            same_alpha = cusp_mod.same_alpha.value
            anti_alpha = cusp_mod.anti_alpha.value
        else:
            init = jnp.asarray(
                cusp_mod.initial_alpha, dtype=float,
            )
            same_alpha = init
            anti_alpha = init
        cusp = {
            'type': cusp_type,
            'same_scale': cusp_mod.same_scale,
            'anti_scale': cusp_mod.anti_scale,
            'same_alpha': same_alpha,
            'anti_alpha': anti_alpha,
        }
    else:
        cusp = None
    if gnn.edge_features:
        edge_features = {
            et: _extract_edge_feature_spec(feat)
            for et, feat in gnn.edge_features.items()
        }
    else:
        edge_features = None
    deep_features = gnn.layers[0].deep_features
    return {
        'embedding_kwargs': embedding_kwargs,
        'layer_specs': layer_specs,
        'bf_up_layers': bf_up_layers,
        'bf_down_layers': bf_down_layers,
        'envelope': envelope,
        'cusp': cusp,
        'jastrow': jastrow,
        'edge_features': edge_features,
        'self_interaction': gnn.self_interaction,
        'deep_features': deep_features,
    }


def make_nn_log_psi(config, mol_info, rng_key):
    """Create an NN trial wavefunction.

    Args:
        config: :class:`NNAnsatzConfig` or a string
            (built-in name or YAML path).
        mol_info: :class:`~OmegaQMC.utils.Mole_custom`
            instance.
        rng_key: JAX PRNG key for parameter init.

    Returns:
        Tuple ``(log_psi, init_params, graphdef, lap_grad)``.
        *log_psi* is a callable
        ``(elec_crds, nuc_crds, params) -> float``.
        *init_params* is the initial parameter pytree.
        *graphdef* is the NNX ``GraphDef`` needed to
        reconstruct the model via ``nnx.merge``.
        *lap_grad* is a callable
        ``(elec_crds, nuc_crds, params) -> (lap, grad_flat)``
        that uses the analytic forward-Laplacian path when
        the config matches the supported PsiFormer subset,
        and falls back to ``laplacian(...)`` (linearize-
        based) otherwise.
    """
    if isinstance(config, str):
        config = load_nn_config(config)

    rngs = nnx.Rngs(rng_key)
    model = build_nn_wf(config, mol_info, rngs)

    graphdef, params, other = nnx.split(
        model, nnx.Param, ...,
    )
    init_params = params

    n_up = mol_info.n_up
    n_down = mol_info.n_down
    n_e = n_up + n_down
    n_det = config.n_determinants
    use_vgl = _psiformer_compat(config)

    def log_psi(elec_crds, nuc_crds, params):
        """Evaluate log|psi| for a single walker.

        Args:
            elec_crds: ``(nelec, 3)`` interleaved
                alpha/beta (OmegaQMC convention).
            nuc_crds: ``(natom, 3)``.
            params: NNX State pytree.
        """
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
        mdl = nnx.merge(graphdef, params, other)
        return mdl(phys_conf).log

    def _lap_grad_linearize(elec_crds, nuc_crds, params):
        def f_flat(r_flat):
            r = r_flat.reshape(n_e, 3)
            return log_psi(r, nuc_crds, params)
        return laplacian(f_flat)(elec_crds.reshape(-1))

    def _lap_grad_vgl(elec_crds, nuc_crds, params):
        r_up = elec_crds[::2]
        r_dn = elec_crds[1::2]
        r_grouped = jnp.concatenate(
            [r_up, r_dn], axis=0,
        )
        elec_flat = r_grouped.reshape(-1)
        mdl = nnx.merge(graphdef, params, other)
        kwargs = _build_vgl_kwargs(
            mdl, ne_log_rescale=config.ne_log_rescale,
        )
        out = log_psi_vgl_psiformer(
            elec_flat, nuc_crds,
            n_up=n_up, n_down=n_down, n_det=n_det,
            **kwargs,
        )
        return out.lap, out.grad

    lap_grad = (
        _lap_grad_vgl if use_vgl else _lap_grad_linearize
    )

    return log_psi, init_params, graphdef, lap_grad
