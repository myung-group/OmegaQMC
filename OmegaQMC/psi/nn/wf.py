"""Neural network wavefunction (Flax NNX).

Rewritten from ``deepqmc/wf/nn_wave_function.py``.
Top-level module combining envelopes, GNN, backflow,
cusp corrections, and Slater determinant(s).
"""

from dataclasses import dataclass, field
from itertools import count
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from .physics import (
    pairwise_diffs,
    pairwise_self_distance,
)
from .types import Psi
from .utils import flatten, triu_flat


# ----------------------------------------------------------------
# Lightweight molecule info (replaces DeepQMC Hamiltonian)
# ----------------------------------------------------------------

def get_shell(z):
    """Number of (partially) occupied shells for *z*
    electrons. Ported from ``deepqmc/hamil.py``.

    Args:
        z: Number of electrons (integer).

    Returns:
        Shell count ``n`` such that the first *n* shells
        can hold at least *z* electrons.
    """
    max_elec = 0
    for n in count():
        if z <= max_elec:
            break
        max_elec += 2 * (1 + n) ** 2
    return n


@dataclass
class MoleculeInfo:
    """Lightweight molecular specification.

    Provides the subset of ``MolecularHamiltonian``
    attributes that the NN wavefunction needs.
    """

    charges: jnp.ndarray
    coords: jnp.ndarray
    n_up: int
    n_down: int
    spin: int = 0
    charge: int = 0
    mol_shells: list = field(default_factory=list)
    mol_ecp_shells: list = field(default_factory=list)

    def __post_init__(self):
        if not self.mol_shells:
            self.mol_shells = [
                get_shell(int(z))
                for z in self.charges
            ]
        if not self.mol_ecp_shells:
            self.mol_ecp_shells = [0] * len(
                self.charges
            )


# ----------------------------------------------------------------
# Helper: log Slater determinant
# ----------------------------------------------------------------

def eval_log_slater(xs):
    """Evaluate ``log|det(xs)|`` with sign.

    Args:
        xs: Orbital matrix ``(n_det, n_elec, n_orb)``.

    Returns:
        ``(sign, log_abs_det)`` each of shape
        ``(n_det,)``.
    """
    if xs.shape[-1] == 0:
        return (
            jnp.ones(xs.shape[:-2]),
            jnp.zeros(xs.shape[:-2]),
        )
    return jnp.linalg.slogdet(xs)


# ----------------------------------------------------------------
# Backflow operator
# ----------------------------------------------------------------

class BackflowOp(nnx.Module):
    """Apply backflow transformations to orbitals.

    Args:
        mult_act: Multiplicative activation.
        add_act: Additive activation.
        with_envelope: Scale additive term by envelope.
    """

    def __init__(
        self,
        mult_act=None,
        add_act=None,
        with_envelope=True,
    ):
        self.mult_act = mult_act or (
            lambda x: 1 + 2 * jnp.tanh(x / 4)
        )
        self.add_act = add_act or (
            lambda x: 0.1 * jnp.tanh(x / 4)
        )
        self.with_envelope = with_envelope

    def __call__(self, xs, fs_mult, fs_add, dists_nuc):
        """Apply backflow transformation to orbital matrix.

        Args:
            xs: Orbital matrix ``(n_det, n_elec, n_orb)``.
            fs_mult: Multiplicative correction tensor,
                or ``None``.
            fs_add: Additive correction tensor, or
                ``None``.
            dists_nuc: Electron-nucleus distances
                ``(n_elec, n_nuc)``.

        Returns:
            Transformed orbital matrix, same shape as
            *xs*.
        """
        if self.with_envelope:
            envel = jnp.sqrt(
                (xs**2).sum(
                    axis=(-1, -3), keepdims=True,
                )
            )
        else:
            envel = 1
        if fs_mult is not None:
            xs = xs * self.mult_act(fs_mult)
        if fs_add is not None:
            R = dists_nuc.min(axis=-1) / 0.5
            cutoff = jnp.where(
                R < 1,
                R**2 * (6 - 8 * R + 3 * R**2),
                jnp.ones_like(R),
            )
            xs = xs + (
                cutoff[None, :, None]
                * envel
                * self.add_act(fs_add)
            )
        return xs


# ----------------------------------------------------------------
# Main wavefunction
# ----------------------------------------------------------------

class NeuralNetworkWaveFunction(nnx.Module):
    """Neural network wavefunction ansatz.

    Combines exponential envelopes, electron GNN
    (OmniNet), backflow, cusp corrections, and Slater
    determinant(s) into a single differentiable model.

    Args:
        mol_info: :class:`MoleculeInfo` instance.
        omni: :class:`OmniNet` instance (or None).
        envelope: Envelope module.
        backflow_op: :class:`BackflowOp` (or None).
        n_determinants: Number of determinants.
        full_determinant: If False, factorise into
            spin-up and spin-down.
        cusp_electrons: Electronic cusp module or None.
        cusp_nuclei: Nuclear cusp module or None.
        backflow_transform: ``'mult'``, ``'add'``, or
            ``'both'``.
        conf_coeff: Module combining determinants
            (e.g. ``nnx.Linear`` or ``SumPool``).
    """

    def __init__(
        self,
        mol_info,
        *,
        omni,
        envelope,
        backflow_op,
        n_determinants,
        full_determinant,
        cusp_electrons,
        cusp_nuclei,
        backflow_transform: Literal[
            'mult', 'add', 'both',
        ],
        conf_coeff,
    ):
        self.n_up = mol_info.n_up
        self.n_down = mol_info.n_down
        self.charges = mol_info.charges
        self.n_det = n_determinants
        self.full_determinant = full_determinant
        self.backflow_transform = backflow_transform

        self.envelope = envelope
        self.conf_coeff = conf_coeff
        self.omni = omni
        self.backflow_op = backflow_op
        self.cusp_electrons = cusp_electrons
        self.cusp_nuclei = cusp_nuclei

    @property
    def spin_slices(self):
        return (
            slice(None, self.n_up),
            slice(self.n_up, None),
        )

    def _apply_backflow(self, xs, fs, dists_nuc):
        assert self.backflow_op is not None
        bt = self.backflow_transform
        if bt == 'mult':
            fs_mult, fs_add = fs, None
        elif bt == 'add':
            fs_mult, fs_add = None, fs
        elif bt == 'both':
            fs_mult, fs_add = jnp.split(
                fs, 2, axis=0,
            )
        else:
            fs_mult, fs_add = None, None
        if fs_add is not None:
            fs_add = fs_add.squeeze(axis=0)
        if fs_mult is not None:
            fs_mult = fs_mult.squeeze(axis=0)
        return self.backflow_op(
            xs, fs_mult, fs_add, dists_nuc,
        )

    def __call__(self, phys_conf, return_mos=False):
        """Evaluate the wavefunction.

        Args:
            phys_conf: :class:`~.types.PhysicalConfiguration`
                with nuclear coordinates *R* and electron
                coordinates *r*.
            return_mos: If ``True``, return the raw
                spin-up and spin-down orbital matrices
                instead of ``Psi``.

        Returns:
            :class:`~.types.Psi` ``(sign, log)`` when
            *return_mos* is ``False``, otherwise a tuple
            ``(orb_up, orb_down)`` of orbital matrices.
        """
        diffs_nuc = pairwise_diffs(
            phys_conf.r, phys_conf.R,
        )
        dists_nuc = jnp.sqrt(diffs_nuc[..., -1])
        dists_elec = pairwise_self_distance(
            phys_conf.r, full=True,
        )
        jastrow, fs, nuc_params = (
            self.omni(phys_conf)
            if self.omni is not None
            else (None, None, None)
        )
        orb = self.envelope(phys_conf, nuc_params)
        if self.full_determinant:
            orb_up, orb_down = orb, orb
        else:
            orb_up, orb_down = jnp.split(
                orb, [self.n_up], axis=-1,
            )
        orb_up = orb_up[:, :self.n_up]
        orb_down = orb_down[:, self.n_up:]
        if fs is not None:
            orb_up = self._apply_backflow(
                orb_up, fs[0],
                dists_nuc[:self.n_up],
            )
            orb_down = self._apply_backflow(
                orb_down, fs[1],
                dists_nuc[self.n_up:],
            )
        if return_mos:
            return orb_up, orb_down

        if self.full_determinant:
            sign, xs = eval_log_slater(
                jnp.concatenate(
                    [orb_up, orb_down], axis=-2,
                ),
            )
        else:
            sign_u, det_u = eval_log_slater(orb_up)
            sign_d, det_d = eval_log_slater(orb_down)
            sign = sign_u * sign_d
            xs = det_u + det_d

        xs_shift = xs.max()
        xs_shift = jnp.where(
            ~jnp.isinf(xs_shift),
            xs_shift,
            jnp.zeros_like(xs_shift),
        )
        xs = sign * jnp.exp(xs - xs_shift)
        psi = self.conf_coeff(xs).squeeze()
        log_psi = (
            jnp.log(jnp.abs(psi)) + xs_shift
        )
        sign_psi = jax.lax.stop_gradient(
            jnp.sign(psi),
        )
        if self.cusp_electrons is not None:
            same_dists = jnp.concatenate([
                triu_flat(dists_elec[idx, idx])
                for idx in self.spin_slices
            ], axis=-1)
            anti_dists = flatten(
                dists_elec[:self.n_up, self.n_up:],
            )
            log_psi = log_psi + self.cusp_electrons(
                same_dists, anti_dists,
            )
        if self.cusp_nuclei is not None:
            log_psi = log_psi + self.cusp_nuclei(
                dists_nuc,
            )
        if jastrow is not None:
            log_psi = log_psi + jastrow

        return Psi(sign_psi, log_psi)
