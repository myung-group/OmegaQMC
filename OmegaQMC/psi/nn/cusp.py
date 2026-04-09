"""Electronic and nuclear cusp corrections (Flax NNX).

Rewritten from ``deepqmc/wf/cusp.py``.
"""

import jax.numpy as jnp
from flax import nnx

from .compat import param_value


class DeepQMCCusp:
    r"""DeepQMC cusp factor.

    .. math::
        -\frac{\mathrm{scale}}
        {\sum_{i<j} \alpha\,(1 + \alpha\,r_{ij})}
    """

    def __call__(self, scale, alpha, dist):
        return -(
            scale / (alpha * (1 + alpha * dist))
        ).sum()


class PsiformerCusp:
    r"""PsiFormer cusp factor.

    .. math::
        -\frac{\mathrm{scale}\,\alpha^2}
        {\sum_{i<j}(\alpha + r_{ij})}
    """

    def __call__(self, scale, alpha, dist):
        return -(
            (scale * alpha**2) / (alpha + dist)
        ).sum()


class CuspAsymptotic(nnx.Module):
    """Base class for cusp corrections.

    Args:
        cusp_function: One of :class:`DeepQMCCusp`
            or :class:`PsiformerCusp`.
        trainable_alpha: Make alpha a trainable param.
    """

    def __init__(
        self, *, cusp_function, trainable_alpha,
    ):
        self.trainable_alpha = trainable_alpha
        self.cusp_function = cusp_function

    def _get_alpha(self, value, name):
        if self.trainable_alpha:
            attr = f'{name}_alpha'
            if not hasattr(self, attr):
                setattr(
                    self, attr,
                    nnx.Param(jnp.array(value)),
                )
            return param_value(getattr(self, attr))
        return jnp.array(value, dtype=float)


class ElectronicCuspAsymptotic(CuspAsymptotic):
    """Electronic cusp correction.

    Args:
        same_scale: Scale for same-spin cusps.
        anti_scale: Scale for anti-spin cusps.
        alpha: Initial alpha value.
        cusp_function: Cusp function instance.
        trainable_alpha: Make alpha trainable.
    """

    def __init__(
        self, *,
        same_scale,
        anti_scale,
        alpha=1.0,
        cusp_function,
        trainable_alpha=False,
    ):
        super().__init__(
            cusp_function=cusp_function,
            trainable_alpha=trainable_alpha,
        )
        self.same_scale = same_scale
        self.anti_scale = anti_scale
        self.initial_alpha = alpha
        if trainable_alpha:
            self.same_alpha = nnx.Param(
                jnp.array(alpha),
            )
            self.anti_alpha = nnx.Param(
                jnp.array(alpha),
            )

    def __call__(self, same_dists, anti_dists):
        cusp = jnp.array(0.0)
        if same_dists.size > 0:
            alpha_s = (
                param_value(self.same_alpha)
                if self.trainable_alpha
                else jnp.array(
                    self.initial_alpha, dtype=float,
                )
            )
            cusp = cusp + self.cusp_function(
                self.same_scale, alpha_s, same_dists,
            )
        if anti_dists.size > 0:
            alpha_a = (
                param_value(self.anti_alpha)
                if self.trainable_alpha
                else jnp.array(
                    self.initial_alpha, dtype=float,
                )
            )
            cusp = cusp + self.cusp_function(
                self.anti_scale, alpha_a, anti_dists,
            )
        return cusp


class NuclearCuspAsymptotic(CuspAsymptotic):
    """Nuclear cusp correction.

    Args:
        nuclear_charges: Array of nuclear charges.
        alpha: Initial alpha value.
        cusp_function: Cusp function instance.
        trainable_alpha: Make alpha trainable.
    """

    def __init__(
        self,
        nuclear_charges, *,
        alpha=1.0,
        cusp_function,
        trainable_alpha=False,
    ):
        super().__init__(
            cusp_function=cusp_function,
            trainable_alpha=trainable_alpha,
        )
        self.nuclear_charges = nuclear_charges[None]
        if trainable_alpha:
            self.nuc_alpha = nnx.Param(
                jnp.array(alpha),
            )
        self._alpha_init = alpha

    def __call__(self, dists):
        alpha = (
            param_value(self.nuc_alpha)
            if self.trainable_alpha
            else jnp.array(
                self._alpha_init, dtype=float,
            )
        )
        return self.cusp_function(
            self.nuclear_charges, alpha, dists,
        )
