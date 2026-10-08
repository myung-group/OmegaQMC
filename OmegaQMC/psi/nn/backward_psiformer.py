"""Hand-written backward pass of the PsiFormer ``log|psi|``.

Evaluates, for one walker and without automatic differentiation,

* ``log|psi|``,
* its gradient with respect to every network parameter, returned
  in the structure of the NNX parameter state, and
* for every ``nnx.Linear`` layer the per-electron inputs ``a`` and
  output cotangents ``g = d log|psi| / d(layer output)``, the
  quantities KFAC builds its Kronecker factors from
  (``a^T a`` and ``g^T g``; the kernel gradient is
  ``sum_e a_e (x) g_e``).

It mirrors, operation for operation, the forward pass of
:func:`OmegaQMC.psi.nn.forward_lap.log_psi_vgl_psiformer` for the
configuration that function supports (``psiformer`` in
``psi/nn/conf``): positional electron-nucleus embedding with a
bias-less projection, ``node_attention`` GNN layers (multi-head
attention, residual, tanh MLP, residual) followed by a tanh
``subnet`` MLP, per-spin linear backflow with the default
``1 + 2 tanh(x/4)`` multiplicative activation, isotropic
exponential envelopes with per-orbital exponents, a sum of full
determinants, and the PsiFormer electron cusp.  Anything else
raises ``NotImplementedError`` when the backward pass is built.

As in :func:`~OmegaQMC.psi.nn.forward_lap.log_psi_vgl_psiformer`,
the Slater stage runs in float64 even when the network runs in
float32 (see ``_vgl_to_slater_dtype`` there): the cotangent of a
determinant near its own node is a vanishing weight times a
diverging inverse.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .adapter import _build_vgl_kwargs, _psiformer_compat, cast_floats
from .build import build_nn_wf
from .config import load_nn_config


def _path_str(path):
    """``'omni/gnn/layers/0/subnet/layers/0/kernel'`` for an NNX
    parameter path (the trailing ``.value`` dropped)."""
    parts = [str(getattr(p, 'key', p)) for p in path]
    if parts and parts[-1] == '.value':
        parts = parts[:-1]
    return '/'.join(parts)


def _check_supported(config, kw):
    """Raise unless *config* is the PsiFormer subset handled here."""
    def need(cond, what):
        if not cond:
            raise NotImplementedError(
                f"backward_psiformer does not support {what}",
            )

    need(_psiformer_compat(config), "this ansatz configuration")
    need(not config.deep_features, "deep_features")
    need(config.use_spin_embedding, "use_spin_embedding=False")
    need(config.project_to_embedding_dim,
         "project_to_embedding_dim=False")
    need(kw['jastrow'] is None, "a Jastrow factor")
    need(kw['cusp'] is None or kw['cusp']['type'] == 'psiformer',
         "this electron cusp type")
    env = kw['envelope']
    need(env['isotropic'] and env['per_orbital_exponent']
         and not env['softplus_zeta'],
         "this envelope variant")
    need(all(float(p) > 0 for p in kw['embedding_kwargs']['ne_powers']),
         "non-positive ne_powers")
    for spec in kw['layer_specs']:
        need(len(spec['update_specs']) == 1
             and spec['update_specs'][0]['type'] == 'node_attention',
             "update features other than one node_attention")
        u = spec['update_specs'][0]
        need(u['attn_mlp_activation'] == 'tanh'
             and not u['attn_mlp_last_linear'],
             "this attention-MLP activation")
        need(u['attn_residual_normalize'] is not None
             and u['attn_mlp_residual_normalize'] is not None,
             "attention updates without residuals")
        need(spec['subnet_activation'] == 'tanh'
             and not spec['subnet_last_linear'],
             "this subnet activation")
        need(spec['electron_residual_normalize'] is None,
             "an electron residual")
    for layers in (kw['bf_up_layers'], kw['bf_down_layers']):
        need(len(layers) == 1 and layers[0][1] is None,
             "a backflow MLP with hidden layers or bias")


def _tanh_bwd(g_out, y):
    """Cotangent of ``z`` for ``y = tanh(z)``."""
    return g_out * (1.0 - y * y)


def make_psiformer_backward(config, mol_info, rng_key=None,
                            compute_dtype=None):
    """Build the hand-written PsiFormer backward pass.

    Args:
        config: :class:`~OmegaQMC.psi.nn.config.NNAnsatzConfig` or a
            string (built-in name or YAML path).
        mol_info: :class:`~OmegaQMC.utils.Mole_custom` instance.
        rng_key: key used to build the template model (only its
            structure is used; the parameters come in at call time).
        compute_dtype: precision of the network, as in
            :func:`~OmegaQMC.psi.nn.adapter.make_nn_log_psi`;
            ``None`` evaluates in the precision of the inputs.

    Returns:
        ``(backward, layer_names)``.  ``backward(elec_crds,
        nuc_crds, params)`` takes one walker in OmegaQMC's
        interleaved spin order and returns ``(log_psi, grads,
        captures)``: ``grads`` has the structure of *params*;
        ``captures`` maps each Linear layer (named as in
        *layer_names*, e.g. ``'omni/gnn/layers/0/subnet/layers/1'``)
        to ``(a, g)`` of shapes ``(n_rows, in)`` and
        ``(n_rows, out)``, one row per electron the layer is
        applied to.  Layers the forward pass never uses (the
        ``subnet_g`` kernels without deep features) get zero
        gradients and no capture.
    """
    if isinstance(config, str):
        config = load_nn_config(config)
    if rng_key is None:
        rng_key = jax.random.key(0)
    model = build_nn_wf(config, mol_info, nnx.Rngs(rng_key))
    kw = _build_vgl_kwargs(model, ne_log_rescale=config.ne_log_rescale)
    _check_supported(config, kw)
    _, params0, _ = nnx.split(model, nnx.Param, ...)
    flat0, treedef = jax.tree_util.tree_flatten_with_path(params0)
    paths = [_path_str(p) for p, _ in flat0]

    n_up, n_down = int(mol_info.n_up), int(mol_info.n_down)
    n_e = n_up + n_down
    n_det = int(config.n_determinants)
    n_layers = len(kw['layer_specs'])
    num_heads = [s['update_specs'][0]['attn_num_heads']
                 for s in kw['layer_specs']]
    ne_powers = [float(p) for p in kw['embedding_kwargs']['ne_powers']]
    ne_log_rescale = bool(kw['embedding_kwargs']['ne_log_rescale'])
    center_idx = jnp.asarray(kw['envelope']['center_idx'])
    cusp = kw['cusp']
    attn_norm = [
        (bool(s['update_specs'][0]['attn_residual_normalize']),
         bool(s['update_specs'][0]['attn_mlp_residual_normalize']))
        for s in kw['layer_specs']
    ]
    n_attn_mlp = [len(s['update_specs'][0]['attn_mlp_layers'])
                  for s in kw['layer_specs']]
    n_subnet = [len(s['subnet_layers']) for s in kw['layer_specs']]

    gnn = 'omni/gnn'
    emb_name = f'{gnn}/electron_embedding/proj'
    bf_names = {'up': 'omni/backflow/up/nets/0/layers/0',
                'down': 'omni/backflow/down/nets/0/layers/0'}

    def lyr(i):
        return f'{gnn}/layers/{i}'

    def attn_name(i, w):
        return f'{lyr(i)}/update_features/0/attention/{w}'

    def amlp_name(i, k):
        return f'{lyr(i)}/update_features/0/mlp/layers/{k}'

    def sub_name(i, k):
        return f'{lyr(i)}/subnet/layers/{k}'

    layer_names = [emb_name]
    for i in range(n_layers):
        layer_names += [attn_name(i, w) for w in ('wq', 'wk', 'wv', 'wo')]
        layer_names += [amlp_name(i, k) for k in range(n_attn_mlp[i])]
        layer_names += [sub_name(i, k) for k in range(n_subnet[i])]
    layer_names += [bf_names['up'], bf_names['down']]
    for name in layer_names:
        if name + '/kernel' not in paths:
            raise NotImplementedError(
                f"unexpected parameter layout: no {name}/kernel",
            )

    cdt = None if compute_dtype is None else jnp.dtype(compute_dtype)
    sdt = jax.dtypes.canonicalize_dtype(jnp.float64)

    def backward(elec_crds, nuc_crds, params):
        out_dt = jnp.promote_types(elec_crds.dtype, nuc_crds.dtype)
        # gradients are returned in the dtype of each given parameter
        p_dtypes = [leaf.dtype for leaf in jax.tree.leaves(params)]
        if cdt is not None:
            elec_crds = elec_crds.astype(cdt)
            nuc_crds = nuc_crds.astype(cdt)
            params = cast_floats(params, cdt)
        P = dict(zip(paths, jax.tree.leaves(params)))
        dt = elec_crds.dtype
        eps = jnp.finfo(dt).eps
        # OmegaQMC interleaves spins; the network groups them
        r = jnp.concatenate([elec_crds[::2], elec_crds[1::2]], axis=0)
        R = nuc_crds

        # ---------------- forward ----------------
        d = r[None, :, :] - R[:, None, :]              # (n_nuc, n_e, 3)
        rr = jnp.sqrt(eps + jnp.sum(d * d, axis=-1))   # (n_nuc, n_e)
        fac = (jnp.log1p(rr) / rr if ne_log_rescale
               else jnp.ones_like(rr))
        f_dist = jnp.stack([rr ** p * fac for p in ne_powers], axis=-1)
        f_diff = d * fac[..., None]
        feats = jnp.concatenate([f_dist, f_diff], axis=-1)
        feats = jnp.swapaxes(feats, 0, 1).reshape(n_e, -1)
        spins = jnp.concatenate([jnp.ones(n_up, dt),
                                 -jnp.ones(n_down, dt)])[:, None]
        feats = jnp.concatenate([feats, spins], axis=-1)
        h = feats @ P[emb_name + '/kernel']

        tape = []
        for i in range(n_layers):
            H = num_heads[i]
            emb = h.shape[-1]
            dh = emb // H
            q = (h @ P[attn_name(i, 'wq') + '/kernel']).reshape(
                n_e, H, dh)
            k = (h @ P[attn_name(i, 'wk') + '/kernel']).reshape(
                n_e, H, dh)
            v = (h @ P[attn_name(i, 'wv') + '/kernel']).reshape(
                n_e, H, dh)
            scale = 1.0 / jnp.sqrt(jnp.asarray(dh, dt))
            scores = jnp.einsum('ihd,jhd->hij', q, k) * scale
            A = jax.nn.softmax(scores, axis=-1)
            mixed = jnp.einsum('hij,jhd->ihd', A, v).reshape(n_e, emb)
            o = mixed @ P[attn_name(i, 'wo') + '/kernel']
            n1, n2 = attn_norm[i]
            c1 = 1.0 / jnp.sqrt(2.0) if n1 else 1.0
            att = (h + o) * c1
            x = att
            amlp = [x]
            for kk in range(n_attn_mlp[i]):
                x = jnp.tanh(x @ P[amlp_name(i, kk) + '/kernel']
                             + P[amlp_name(i, kk) + '/bias'])
                amlp.append(x)
            c2 = 1.0 / jnp.sqrt(2.0) if n2 else 1.0
            u = (att + x) * c2
            x = u
            sub = [x]
            for kk in range(n_subnet[i]):
                x = jnp.tanh(x @ P[sub_name(i, kk) + '/kernel']
                             + P[sub_name(i, kk) + '/bias'])
                sub.append(x)
            tape.append(dict(h=h, q=q, k=k, v=v, A=A, mixed=mixed,
                             scale=scale, c1=c1, c2=c2,
                             amlp=amlp, sub=sub))
            h = x

        # backflow (n_spin, n_det * n_e) -> (n_det, n_spin, n_e)
        def to_det(x, n_s):
            return x.reshape(n_s, n_det, n_e).transpose(1, 0, 2)

        def from_det(x, n_s):
            return x.transpose(1, 0, 2).reshape(n_s, n_det * n_e)

        blocks = {'up': (slice(None, n_up), n_up, '_up'),
                  'down': (slice(n_up, None), n_down, '_down')}
        rr_e = rr.T[:, center_idx]                       # (n_e, n_env)
        bf, env_t, orb = {}, {}, {}
        for s, (sl, n_s, suf) in blocks.items():
            t = jnp.tanh(0.25 * (h[sl] @ P[bf_names[s] + '/kernel']))
            act = to_det(1.0 + 2.0 * t, n_s)
            zeta = P['envelope/zetas' + suf]               # (n_orb, n_env)
            pi = P['envelope/pi' + suf]
            ex = jnp.exp(-jnp.abs(zeta)[None] * rr_e[sl][:, None, :])
            env = jnp.einsum('oc,ioc->io', pi, ex)         # (n_s, n_orb)
            bf[s], env_t[s] = (t, act), ex
            orb[s] = (to_det(env, n_s), act)
        S = jnp.concatenate([orb['up'][0] * orb['up'][1],
                             orb['down'][0] * orb['down'][1]], axis=1)

        S64 = S.astype(sdt)
        sign, logdet = jnp.linalg.slogdet(S64)
        shift = jnp.max(logdet)
        terms = sign * jnp.exp(logdet - shift)
        total = jnp.sum(terms)
        log_psi = jnp.log(jnp.abs(total)) + shift
        # d log|psi| / dS_d = (det_d / psi) S_d^-T
        w = terms / total
        gS = (w[:, None, None]
              * jnp.swapaxes(jnp.linalg.inv(S64), -1, -2)).astype(dt)

        grads = {}
        if cusp is not None:
            iu, ju = jnp.triu_indices(n_e, k=1)
            dd = r[iu] - r[ju]
            dist = jnp.sqrt(eps + jnp.sum(dd * dd, axis=-1))
            same = (ju < n_up) | (iu >= n_up)
            for kind, mask in (('same', same), ('anti', ~same)):
                name = f'cusp_electrons/{kind}_alpha'
                # a constant (cusp[...]) unless alpha is trainable
                a = P.get(name, jnp.asarray(cusp[f'{kind}_alpha'], dt))
                sc = cusp[f'{kind}_scale']
                m = mask.astype(dt)
                log_psi = log_psi - jnp.sum(
                    m * sc * a * a / (a + dist))
                grads[name] = -jnp.sum(
                    m * sc * a * (a + 2.0 * dist) / (a + dist) ** 2)

        # ---------------- backward ----------------
        caps = {}
        g_h = jnp.zeros_like(h)
        n_up_rows = gS[:, :n_up, :]
        n_dn_rows = gS[:, n_up:, :]
        for s, g_orb in (('up', n_up_rows), ('down', n_dn_rows)):
            sl, n_s, suf = blocks[s]
            env_det, act = orb[s]
            g_env = from_det(g_orb * act, n_s)             # (n_s, n_orb)
            g_act = g_orb * env_det
            t = bf[s][0]
            g_bf = from_det(g_act, n_s) * 0.5 * (1.0 - t * t)
            a_bf = h[sl]
            caps[bf_names[s]] = (a_bf, g_bf)
            grads[bf_names[s] + '/kernel'] = a_bf.T @ g_bf
            g_h = g_h.at[sl].add(g_bf @ P[bf_names[s] + '/kernel'].T)
            zeta = P['envelope/zetas' + suf]
            pi = P['envelope/pi' + suf]
            ex = env_t[s]
            grads['envelope/pi' + suf] = jnp.einsum('io,ioc->oc',
                                                    g_env, ex)
            grads['envelope/zetas' + suf] = -jnp.sign(zeta) * pi * (
                jnp.einsum('io,ioc,ic->oc', g_env, ex, rr_e[sl]))

        for i in reversed(range(n_layers)):
            T = tape[i]
            n_s_ = n_subnet[i]
            for kk in reversed(range(n_s_)):
                y = T['sub'][kk + 1]
                a = T['sub'][kk]
                gz = _tanh_bwd(g_h, y)
                caps[sub_name(i, kk)] = (a, gz)
                grads[sub_name(i, kk) + '/kernel'] = a.T @ gz
                grads[sub_name(i, kk) + '/bias'] = gz.sum(axis=0)
                g_h = gz @ P[sub_name(i, kk) + '/kernel'].T
            g_u = g_h * T['c2']
            g_x = g_u
            g_att = g_u
            for kk in reversed(range(n_attn_mlp[i])):
                y = T['amlp'][kk + 1]
                a = T['amlp'][kk]
                gz = _tanh_bwd(g_x, y)
                caps[amlp_name(i, kk)] = (a, gz)
                grads[amlp_name(i, kk) + '/kernel'] = a.T @ gz
                grads[amlp_name(i, kk) + '/bias'] = gz.sum(axis=0)
                g_x = gz @ P[amlp_name(i, kk) + '/kernel'].T
            g_att = (g_att + g_x) * T['c1']
            hin = T['h']
            # o = mixed @ Wo; att = (h + o) c1
            caps[attn_name(i, 'wo')] = (T['mixed'], g_att)
            grads[attn_name(i, 'wo') + '/kernel'] = T['mixed'].T @ g_att
            g_mixed = (g_att @ P[attn_name(i, 'wo') + '/kernel'].T
                       ).reshape(T['q'].shape)
            A, q, k, v = T['A'], T['q'], T['k'], T['v']
            g_A = jnp.einsum('ihd,jhd->hij', g_mixed, v)
            g_v = jnp.einsum('hij,ihd->jhd', A, g_mixed)
            g_sc = A * (g_A - jnp.sum(A * g_A, axis=-1, keepdims=True))
            g_sc = g_sc * T['scale']
            g_q = jnp.einsum('hij,jhd->ihd', g_sc, k)
            g_k = jnp.einsum('hij,ihd->jhd', g_sc, q)
            g_hin = g_att
            for nm, gg in (('wq', g_q), ('wk', g_k), ('wv', g_v)):
                gg = gg.reshape(n_e, -1)
                caps[attn_name(i, nm)] = (hin, gg)
                grads[attn_name(i, nm) + '/kernel'] = hin.T @ gg
                g_hin = g_hin + gg @ P[attn_name(i, nm) + '/kernel'].T
            g_h = g_hin

        caps[emb_name] = (feats, g_h)
        grads[emb_name + '/kernel'] = feats.T @ g_h

        grad_leaves = []
        for pth, leaf, odt in zip(paths, jax.tree.leaves(params),
                                  p_dtypes):
            gl = grads.get(pth)
            if gl is None:
                gl = jnp.zeros_like(leaf)
            grad_leaves.append(gl.astype(odt))
        grad_tree = jax.tree_util.tree_unflatten(treedef, grad_leaves)
        return log_psi.astype(out_dt), grad_tree, caps

    return backward, layer_names
