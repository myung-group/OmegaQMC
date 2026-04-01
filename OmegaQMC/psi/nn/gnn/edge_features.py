"""Edge feature classes for the GNN.

Rewritten from ``deepqmc/gnn/edge_features.py``. These are
stateless callables — no neural network parameters.
"""

from typing import Optional

import jax
import jax.numpy as jnp

from ..utils import norm


class DifferenceEdgeFeature:
    """Use the 3-D difference vector as edge features.

    Args:
        log_rescale: Rescale by ``log(1+d)/d``.
    """

    def __init__(self, *, log_rescale=False):
        self.log_rescale = log_rescale

    def __call__(self, d: jax.Array) -> jax.Array:
        """Compute difference-based edge features.

        Args:
            d: Pairwise difference vectors
                ``(..., 3)``.

        Returns:
            Array ``(..., 3)``, optionally log-rescaled.
        """
        if self.log_rescale:
            r = norm(d, safe=True)
            d = d * (jnp.log1p(r) / r)[..., None]
        return d

    def __len__(self) -> int:
        return 3


class DistancePowerEdgeFeature:
    """Powers of the interparticle distance.

    Args:
        powers: List of exponents (e.g. ``[1]``).
        eps: Regulariser for negative powers.
        log_rescale: Rescale by ``log(1+d)/d``.
    """

    def __init__(
        self, *,
        powers: list,
        eps: Optional[float] = None,
        log_rescale: bool = False,
    ):
        if any(p < 0 for p in powers):
            assert eps is not None
        self.powers = jnp.asarray(powers)
        self.eps = eps or 0.0
        self.log_rescale = log_rescale

    def __call__(self, d: jax.Array) -> jax.Array:
        """Compute distance-power edge features.

        Args:
            d: Pairwise difference vectors
                ``(..., 3)``.

        Returns:
            Array ``(..., len(powers))``.
        """
        r = norm(d, safe=True)
        pw = jnp.where(
            self.powers > 0,
            r[..., None] ** self.powers,
            1.0 / (
                r[..., None] ** (-self.powers)
                + self.eps
            ),
        )
        if self.log_rescale:
            pw = pw * (
                jnp.log1p(r) / r
            )[..., None]
        return pw

    def __len__(self) -> int:
        return len(self.powers)


class GaussianEdgeFeature:
    """Gaussian basis expansion of the distance.

    Args:
        n_gaussian: Number of Gaussians.
        radius: Cutoff radius.
        offset: Offset the first Gaussian from zero.
    """

    def __init__(
        self, *,
        n_gaussian: int,
        radius: float,
        offset: bool,
    ):
        delta = 1 / (2 * n_gaussian) if offset else 0
        qs = jnp.linspace(delta, 1 - delta, n_gaussian)
        self.mus = radius * qs**2
        self.sigmas = (1 + radius * qs) / 7

    def __call__(self, d: jax.Array) -> jax.Array:
        """Compute Gaussian-basis edge features.

        Args:
            d: Pairwise difference vectors
                ``(..., 3)``.

        Returns:
            Array ``(..., n_gaussian)``.
        """
        r = norm(d, safe=True)
        return jnp.exp(
            -((r[..., None] - self.mus) ** 2)
            / self.sigmas**2
        )

    def __len__(self) -> int:
        return len(self.mus)


class CombinedEdgeFeature:
    """Concatenation of multiple edge features.

    Args:
        features: List of edge feature objects.
    """

    def __init__(self, *, features):
        self.features = features

    def __call__(self, d: jax.Array) -> jax.Array:
        """Concatenate all constituent feature outputs.

        Args:
            d: Pairwise difference vectors
                ``(..., 3)``.

        Returns:
            Concatenated array ``(..., sum_of_dims)``.
        """
        return jnp.concatenate(
            [f(d) for f in self.features], axis=-1,
        )

    def __len__(self) -> int:
        return sum(map(len, self.features))
