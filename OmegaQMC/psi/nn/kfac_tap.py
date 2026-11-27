"""Exact per-sample statistics of every ``nnx.Linear`` for KFAC.

kfac-jax (as used by DeepMind's FermiNet, see
``ferminet/curvature_tags_and_blocks.py``) builds the Kronecker factors
of a dense layer ``y = x W + b`` that is applied to many rows of one
walker (one per electron, per electron pair, ...) from those rows
themselves:

    A = E_{w,r}[x~ x~^T],     G = E_{w,r}[g g^T],

with ``x~ = [x, 1]`` the input row extended by a one for the bias and
``g = d log|psi_w| / d y_{w,r}`` the output cotangent of the same row.
The per-walker parameter gradient alone does not determine these: it
is ``sum_r x~_r g_r^T``, from which ``g`` can only be recovered by a
solve (the ``capture_activations`` path of
:mod:`OmegaQMC.vmcopt_nn_heg_kfac`).

This module gets ``x`` and ``g`` exactly, for any ansatz built from
``nnx.Linear``, without a model-specific backward pass: the model is
rebuilt with every Linear replaced by a :class:`TapLinear`, which
passes its input to a *tap* function and lets that function add a
perturbation ``eps`` to its output.  With ``eps = 0`` the network is
unchanged, its input rows are recorded, and ``d log|psi| / d eps`` is
the output cotangent.  Differentiating with respect to the
perturbations (and the few non-Linear leaves) only, with the kernels
held constant, costs one backward pass and never forms the per-walker
``(in, out)`` kernel gradients.

The taps are switched on by :func:`tapping` only inside the traced
function, so the model behaves exactly like the plain one elsewhere.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

_TAP = threading.local()


class TapLinear(nnx.Linear):
    """``nnx.Linear`` whose output can be observed and perturbed.

    ``kfac_name`` (the module's path in the model, set by
    :func:`build_tapped_model`) identifies the layer to the active tap.
    Without an active tap this is an ordinary ``nnx.Linear``.
    """

    kfac_name: str = ''

    def __call__(self, inputs, *args, **kwargs):
        y = super().__call__(inputs, *args, **kwargs)
        tap = getattr(_TAP, 'fn', None)
        if tap is not None:
            y = tap(self.kfac_name, inputs, y)
        return y


@contextlib.contextmanager
def use_tap_linears() -> Iterator[None]:
    """Inside this context ``nnx.Linear(...)`` builds a :class:`TapLinear`."""
    orig = nnx.Linear
    nnx.Linear = TapLinear
    try:
        yield
    finally:
        nnx.Linear = orig


@contextlib.contextmanager
def tapping(fn: Callable) -> Iterator[None]:
    """Route every :class:`TapLinear` call through ``fn(name, x, y)``.

    ``fn`` returns the layer output to use (``y`` plus a perturbation).
    Active only for the Python code (i.e. the tracing) inside the
    ``with`` block.
    """
    prev = getattr(_TAP, 'fn', None)
    _TAP.fn = fn
    try:
        yield
    finally:
        _TAP.fn = prev


def leaf_path_str(path) -> str:
    """``'a/b/0/kernel'`` for a parameter-State key path.

    A trailing ``.value`` attribute (NNX ``Variable`` leaves) is dropped,
    so the result names the variable.
    """
    keys = []
    for p in path:
        for attr in ('key', 'idx', 'name'):
            if hasattr(p, attr):
                keys.append(str(getattr(p, attr)))
                break
        else:
            keys.append(str(p))
    if keys and keys[-1] == 'value':
        keys = keys[:-1]
    return '/'.join(keys)


def build_tapped_model(build: Callable[[], nnx.Module]) -> nnx.Module:
    """Call ``build()`` with :class:`TapLinear` s and name them by path."""
    with use_tap_linears():
        model = build()
    for path, node in nnx.iter_graph(model):
        if isinstance(node, TapLinear):
            node.kfac_name = '/'.join(str(p) for p in path)
    return model


class LinearTaps:
    """Per-walker Linear inputs, output cotangents and other gradients.

    Args:
        apply: ``apply(params, elec) -> log|psi|`` for one walker that
            evaluates a model built by :func:`build_tapped_model`
            (``params`` is that model's ``nnx.Param`` State).
        params: A parameter State of the model (for shapes).
        elec: One walker, for discovering the call pattern.

    Attributes:
        layers: ``{name: info}`` for every Linear the forward pass
            calls, in call order.  ``info`` has ``kernel`` and ``bias``
            (indices into the flattened params; ``bias`` is ``None``
            without one), ``in_dim``, ``out_dim``, ``calls`` (the tap
            keys of its calls) and ``n_rep`` (rows per walker, summed
            over calls).
        other_idx: Indices of the flattened param leaves that belong to
            no called Linear (envelope exponents and weights, unused
            Linears, ...).
    """

    def __init__(self, apply, params, elec):
        self._apply = apply
        flat, self._treedef = jax.tree_util.tree_flatten_with_path(params)
        names = [leaf_path_str(p) for p, _ in flat]
        index = {n: i for i, n in enumerate(names)}

        calls: List[Tuple[str, Tuple[int, ...], Tuple[int, ...], object]] = []

        def discover(p, e):
            def tap(name, x, y):
                calls.append((name, x.shape, y.shape, y.dtype))
                return y
            with tapping(tap):
                return apply(p, e)

        jax.eval_shape(discover, params, elec)

        self.layers: Dict[str, dict] = {}
        self._pert_shapes: Dict[str, Tuple[Tuple[int, ...], object]] = {}
        for name, xs, ys, ydt in calls:
            if not name:
                raise RuntimeError("an unnamed TapLinear was called")
            info = self.layers.get(name)
            if info is None:
                kname, bname = f"{name}/kernel", f"{name}/bias"
                if kname not in index:
                    raise RuntimeError(f"no kernel leaf for Linear {name}")
                info = dict(kernel=index[kname],
                            bias=index.get(bname),
                            in_dim=int(xs[-1]), out_dim=int(ys[-1]),
                            calls=[], n_rep=0)
                self.layers[name] = info
            key = f"{name}#{len(info['calls'])}"
            info['calls'].append(key)
            reps = 1
            for d in xs[:-1]:
                reps *= int(d)
            info['n_rep'] += reps
            self._pert_shapes[key] = (tuple(ys), ydt)
        used = set()
        for info in self.layers.values():
            used.add(info['kernel'])
            if info['bias'] is not None:
                used.add(info['bias'])
        self.other_idx = [i for i in range(len(flat)) if i not in used]
        self.leaf_names = names

    def zero_perturbations(self) -> Dict[str, jax.Array]:
        return {k: jnp.zeros(s, dt) for k, (s, dt) in self._pert_shapes.items()}

    def per_walker(self, params, elec):
        """Statistics of one walker (``vmap`` this over walkers).

        Returns:
            ``(xs, gs, other_grads)``: ``xs[name]`` ``(n_rep, in)`` and
            ``gs[name]`` ``(n_rep, out)``, the input rows and output
            cotangents of each called Linear (calls concatenated), and
            the list of ``d log|psi| / d leaf`` for the leaves in
            ``other_idx``.
        """
        leaves = self._treedef.flatten_up_to(params)
        other = [leaves[i] for i in self.other_idx]

        def f(perts, other_leaves):
            lv = list(leaves)
            for i, v in zip(self.other_idx, other_leaves):
                lv[i] = v
            p = self._treedef.unflatten(lv)
            rec = {}
            counts: Dict[str, int] = {}

            def tap(name, x, y):
                i = counts.get(name, 0)
                counts[name] = i + 1
                key = f"{name}#{i}"
                rec[key] = x
                return y + perts[key].astype(y.dtype)

            with tapping(tap):
                lp = self._apply(p, elec)
            return lp, rec

        (g_pert, g_other), rec = jax.grad(
            f, argnums=(0, 1), has_aux=True,
        )(self.zero_perturbations(), other)
        xs, gs = {}, {}
        for name, info in self.layers.items():
            xs[name] = jnp.concatenate(
                [rec[k].reshape(-1, info['in_dim']) for k in info['calls']])
            gs[name] = jnp.concatenate(
                [g_pert[k].reshape(-1, info['out_dim'])
                 for k in info['calls']])
        return xs, gs, g_other
