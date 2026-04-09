"""Reusable neural network layers (Flax NNX).

Rewritten from ``deepqmc/hkext.py``. Provides MLP,
residual connections, GLU, and related building blocks.
"""

import math
from collections.abc import Callable, Sequence
from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn import sigmoid, softplus


def ssp(x: jax.Array) -> jax.Array:
    r"""Shifted softplus activation.

    .. math::
        \mathrm{ssp}(x) = \ln(1 + e^x) + \ln\tfrac12
    """
    return softplus(x) + jnp.log(0.5)


# ---- Activation registry ----
_ACTIVATIONS = {
    'tanh': jnp.tanh,
    'ssp': ssp,
    'silu': jax.nn.silu,
    None: None,
}


def get_activation(name):
    """Look up activation function by name or pass through."""
    if callable(name):
        return name
    return _ACTIVATIONS[name]


# ---- Weight initializers matching DeepQMC conventions ----
def _w_init(init_name: str):
    """Kernel initializer by name."""
    return {
        'deeperwin': nnx.initializers.variance_scaling(
            1.0, 'fan_avg', 'uniform',
        ),
        'default': nnx.initializers.variance_scaling(
            1.0, 'fan_in', 'truncated_normal',
        ),
        'ferminet': nnx.initializers.variance_scaling(
            1.0, 'fan_in', 'normal',
        ),
    }[init_name]


def _ferminet_bias_init(key, shape, dtype=jnp.float32):
    """FermiNet-style bias init (variance-scaled normal).

    Falls back to zeros for 1-D shapes (pure bias
    vectors) where fan computation is undefined.
    """
    if len(shape) < 2:
        return jnp.zeros(shape, dtype=dtype)
    return nnx.initializers.variance_scaling(
        1.0, 'fan_out', 'normal',
    )(key, shape, dtype)


def _b_init(init_name: str):
    """Bias initializer by name."""
    return {
        'deeperwin': nnx.initializers.zeros_init(),
        'default': nnx.initializers.zeros_init(),
        'ferminet': _ferminet_bias_init,
    }[init_name]


class MLP(nnx.Module):
    r"""Multilayer perceptron.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        hidden_layers: Either ``('log', N)`` for *N*
            layers with logarithmically spaced widths,
            or a tuple of ints for explicit widths.
        bias: Which layers get bias terms. ``True`` =
            all, ``False`` = none, ``'not_last'`` = all
            but the output layer.
        last_linear: If True, skip activation on the
            output layer.
        activation: Activation function (name or
            callable).
        init: Weight initializer name (``'default'``,
            ``'ferminet'``, or ``'deeperwin'``).
        rngs: Flax NNX RNG state.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_layers: Sequence[Union[int, str]],
        bias: Union[bool, str] = True,
        last_linear: bool = False,
        activation: Union[str, Callable, None] = 'tanh',
        init: str = 'default',
        rngs: nnx.Rngs,
    ):
        assert bias in (True, False, 'not_last')
        self.activation = get_activation(activation)
        self.last_linear = last_linear

        # Compute layer dimensions
        if (
            len(hidden_layers) == 2
            and hidden_layers[0] == 'log'
        ):
            n_hidden = int(hidden_layers[1])
            qs = [
                k / n_hidden
                for k in range(1, n_hidden + 1)
            ]
            dims = [
                round(
                    in_dim ** (1 - q) * out_dim ** q
                )
                for q in qs
            ]
        else:
            dims = [*hidden_layers, out_dim]

        n_layers = len(dims)
        w_i = _w_init(init)
        b_i = _b_init(init)

        layers = []
        d_in = in_dim
        for idx, d_out in enumerate(dims):
            with_bias = bias is True or (
                bias == 'not_last' and idx < n_layers - 1
            )
            layers.append(
                nnx.Linear(
                    d_in, d_out,
                    use_bias=with_bias,
                    kernel_init=w_i,
                    bias_init=b_i,
                    rngs=rngs,
                )
            )
            d_in = d_out
        self.layers = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through all linear layers.

        Args:
            x: Input array of shape ``(..., in_dim)``.

        Returns:
            Output array of shape ``(..., out_dim)``.
        """
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < n - 1 or not self.last_linear:
                if self.activation is not None:
                    x = self.activation(x)
        return x


class ResidualConnection:
    """Residual connection between pytrees.

    If shapes match, ``out = (inp + update) / sqrt(2)``
    when *normalize* is True, else ``out = inp + update``.
    If shapes differ, ``out = update`` (projection-free).

    Args:
        normalize: Divide the sum by ``sqrt(2)``.
    """

    def __init__(self, *, normalize: bool):
        self.normalize = normalize

    def __call__(self, inp, update):
        """Apply residual connection leaf-by-leaf.

        Args:
            inp: Original pytree (pre-update).
            update: Updated pytree (same structure).

        Returns:
            Pytree of the same structure as *inp*; each
            leaf is ``(x + y) / sqrt(2)`` if shapes match
            and *normalize* is ``True``, ``x + y`` if
            *normalize* is ``False``, or ``y`` when shapes
            differ.
        """
        def leaf_residual(x, y):
            if x.shape != y.shape:
                return y
            z = x + y
            if self.normalize:
                return z / jnp.sqrt(2.0)
            return z

        return jax.tree.map(leaf_residual, inp, update)


class SumPool:
    """Global sum pooling (output dim must be 1).

    Args:
        out_dim: Must be 1.
        name: Unused; accepted for API compatibility.
    """

    def __init__(self, out_dim, name=None):
        assert out_dim == 1

    def __call__(self, x):
        return jax.tree.map(
            lambda leaf: leaf.sum(
                axis=-1, keepdims=True,
            ),
            x,
        )


class Identity:
    """Identity (pass-through) module."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


class GLU(nnx.Module):
    """Gated Linear Unit.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        bias: Include bias in linear layers.
        layer_norm_before: Apply LayerNorm before gating.
        activation: Gating activation (default sigmoid).
        rngs: Flax NNX RNG state.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        layer_norm_before: bool = True,
        activation: Callable = sigmoid,
        rngs: nnx.Rngs,
    ):
        self.activated_linear = nnx.Linear(
            in_dim, out_dim, use_bias=bias, rngs=rngs,
        )
        self.linear = nnx.Linear(
            in_dim, out_dim, use_bias=bias, rngs=rngs,
        )
        self.activation = activation
        self.layer_norm_before = layer_norm_before
        if layer_norm_before:
            self.ln_x = nnx.LayerNorm(
                in_dim,
                use_bias=False,
                use_scale=False,
                rngs=rngs,
            )
            self.ln_y = nnx.LayerNorm(
                in_dim,
                use_bias=False,
                use_scale=False,
                rngs=rngs,
            )

    def __call__(self, x, y):
        """Compute gated linear output.

        Args:
            x: Input for the gating branch, shape
                ``(..., in_dim)``.
            y: Input for the value branch, shape
                ``(..., in_dim)``.

        Returns:
            ``activation(W_x · x) ⊙ (W_y · y)``,
            shape ``(..., out_dim)``.
        """
        if self.layer_norm_before:
            x = self.ln_x(x)
            y = self.ln_y(y)
        return (
            self.activation(self.activated_linear(x))
            * self.linear(y)
        )
