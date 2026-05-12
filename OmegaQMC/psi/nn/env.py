"""Exponential orbital envelopes (Flax NNX).

Rewritten from ``deepqmc/wf/env.py``. The
``kfac_jax.register_scale_and_shift`` call is removed.
"""

import jax.numpy as jnp
from flax import nnx
from jax.nn import softplus

from .compat import param_value
from .physics import pairwise_diffs
from .utils import norm, unflatten


class ExponentialEnvelopes(nnx.Module):
    """Exponential envelopes centred on the nuclei.

    Args:
        mol_info: Molecule information dataclass.
        n_determinants: Number of determinants.
        isotropic: Use isotropic exponents.
        per_shell: One envelope per atomic shell.
        per_orbital_exponent: Per-orbital zeta params.
        spin_restricted: Shared params for up/down.
        init_to_ones: Initialise pi/zeta to ones.
        softplus_zeta: Apply softplus to zeta.
    """

    def __init__(
        self,
        mol_info,
        n_determinants,
        *,
        isotropic,
        per_shell,
        per_orbital_exponent,
        spin_restricted,
        init_to_ones,
        softplus_zeta,
    ):
        n_up = mol_info.n_up
        n_down = mol_info.n_down
        charges = mol_info.charges

        shells = []
        for i, (z, n_sh, n_ecp) in enumerate(zip(
            charges,
            mol_info.mol_shells,
            mol_info.mol_ecp_shells,
        )):
            end = n_sh if per_shell else n_ecp + 1
            for k in range(n_ecp, end):
                shells.append((i, z / (k + 1)))

        center_idx, zetas_init = map(
            jnp.array, zip(*shells),
        )
        self.center_idx = center_idx
        n_env = len(zetas_init)
        n_orb = n_determinants * (n_up + n_down)

        self.isotropic = isotropic
        self.per_orbital_exponent = per_orbital_exponent
        self.spin_restricted = spin_restricted
        self.n_up = n_up
        self.n_det = n_determinants
        self.softplus_zeta = softplus_zeta
        self.init_to_ones = init_to_ones

        # --- pi parameters ---
        pi_names = (
            ['pi'] if spin_restricted
            else ['pi_up', 'pi_down']
        )
        pi_shape = (n_orb, n_env)
        for name in pi_names:
            if init_to_ones:
                val = jnp.ones(pi_shape)
            else:
                val = jnp.ones(pi_shape) + 0.01 * (
                    jnp.arange(pi_shape[0])[:, None]
                    * 0.1
                )
            setattr(self, name, nnx.Param(val))
        self._pi_names = pi_names

        # --- zeta parameters ---
        if per_orbital_exponent:
            zetas_init = jnp.tile(
                zetas_init[None], (n_orb, 1),
            )
        if not isotropic:
            zetas_init = (
                zetas_init[..., None, None]
                * jnp.eye(3)
            )
        zeta_names = (
            ['zetas'] if spin_restricted
            else ['zetas_up', 'zetas_down']
        )
        for name in zeta_names:
            if init_to_ones:
                val = jnp.ones(zetas_init.shape)
            else:
                val = jnp.copy(zetas_init)
            setattr(self, name, nnx.Param(val))
        self._zeta_names = zeta_names

    def _call_one_spin(self, zeta, pi, diffs):
        d = diffs[..., self.center_idx, :-1]
        if self.isotropic:
            d = norm(d, safe=True)
            if self.per_orbital_exponent:
                d = d[:, None]
            if self.softplus_zeta:
                exponent = softplus(zeta) * d
            else:
                exponent = jnp.abs(zeta * d)
        else:
            exponent = norm(
                jnp.einsum(
                    '...ers,ies->i...er', zeta, d,
                ),
                safe=True,
            )
        if not self.per_orbital_exponent:
            exponent = exponent[:, None]
        orbs = (pi * jnp.exp(-exponent)).sum(axis=-1)
        return unflatten(
            orbs, -1, (self.n_det, -1),
        ).swapaxes(-2, -3)

    def __call__(self, phys_conf, nuc_params):
        diffs = pairwise_diffs(phys_conf.r, phys_conf.R)
        pis = [
            param_value(getattr(self, n))
            for n in self._pi_names
        ]
        zetas = [
            param_value(getattr(self, n))
            for n in self._zeta_names
        ]
        if self.spin_restricted:
            return self._call_one_spin(
                zetas[0], pis[0], diffs,
            )
        # Spin-unrestricted. For fully-spin-polarized cases
        # (n_up=0 or n_down=0), the empty spin block hits an
        # `unflatten` ZeroDivisionError. Skip the empty side and
        # pad with a zero array of the matching (n_det, 0, n_orb)
        # shape using the shape from the non-empty side.
        splits = jnp.split(diffs, (self.n_up,))
        n_per = [splits[0].shape[0], splits[1].shape[0]]
        orbs_partial = []
        ref_shape = None
        for z, p, d, n_spin in zip(zetas, pis, splits, n_per):
            if n_spin > 0:
                o = self._call_one_spin(z, p, d)
                orbs_partial.append(o)
                ref_shape = o.shape
            else:
                orbs_partial.append(None)
        if ref_shape is None:
            raise ValueError(
                "envelope: both spin blocks empty"
            )
        orbs = []
        for o in orbs_partial:
            if o is None:
                orbs.append(jnp.zeros(
                    (ref_shape[0], 0, ref_shape[2]),
                    dtype=orbs_partial[0].dtype
                    if orbs_partial[0] is not None
                    else orbs_partial[1].dtype,
                ))
            else:
                orbs.append(o)
        return jnp.concatenate(orbs, axis=-2)
