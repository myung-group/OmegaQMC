"""Hartree-Fock pretraining of molecular NN wavefunctions.

Supervised pretraining of the network's orbital matrices against
unrestricted Hartree-Fock (UHF) orbitals, as in the JAX FermiNet
(``ferminet/pretrain.py`` and ``ferminet/train.py`` at commit 4cfdf1e,
2022-06-29, the version whose defaults were used by Qian, Fu, Ren and
Chen, J. Chem. Phys. 157, 164104 (2022)).  FermiNet's defaults are
this module's defaults:

* **Target.**  UHF in the ``sto-6g`` basis (``pyscf.scf.UHF`` with
  PySCF's default settings), molecular orbitals evaluated with
  ``GTOval_sph``.  Spin-up electrons take the lowest ``n_up`` alpha
  MOs and spin-down electrons the lowest ``n_down`` beta MOs.  For a
  full determinant the target is the block-diagonal
  ``(n_e, n_e)`` matrix ``[[HF_up, 0], [0, HF_down]]``; otherwise
  each spin block is compared with its own orbital matrix.
* **Loss.**  The mean squared difference between the target and the
  network's orbital matrices (envelopes included, before the
  determinant), averaged over walkers, determinants and matrix
  elements, broadcasting the target over the determinants; for
  separate spin blocks the two means are added.  No energy term.
* **Optimiser.**  ``optax.adam(3e-4)``, 1000 iterations.
* **Sampling.**  The walkers sample the network's own ``|psi|^2``
  with one all-electron Gaussian Metropolis move of width 0.02 per
  iteration, after each parameter update.  The target is evaluated
  on the current walkers every iteration.  The pretrained walkers
  are returned so that training can continue from them.
* **Initial walkers.**  :func:`init_electrons`, FermiNet's
  ``init_electrons``: each atom gets the (alpha, beta) occupation of
  its neutral ground state, atoms' spins are flipped at random until
  the totals match ``(n_up, n_down)``, and every electron is placed at
  its atom with Gaussian noise of width 1 bohr.

One deliberate difference: FermiNet's Metropolis acceptance in
pretraining compares the proposal against a ``log|psi|`` cached from
the previous parameters; here it is re-evaluated with the current
parameters, which costs one network evaluation per iteration.

The Hartree-Fock orbitals are evaluated on the host with PySCF, as
in FermiNet; at ``sto-6g`` this is a small fraction of an iteration.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .spin_layout import layout_permutation, to_grouped

# FermiNet defaults (ferminet/base_config.py, pretrain.py at 4cfdf1e).
PRETRAIN_ITERATIONS = 1000
PRETRAIN_BASIS = 'sto-6g'
PRETRAIN_LR = 3.0e-4
PRETRAIN_MCMC_WIDTH = 0.02
INIT_WIDTH = 1.0


# ---------------------------------------------------------------------
# Initial electron positions
# ---------------------------------------------------------------------
def atomic_spin_config(z: int) -> tuple:
    """``(n_alpha, n_beta)`` of the neutral ground-state atom.

    The number of unpaired electrons follows Hund's rule from
    PySCF's ground-state configuration table
    (:data:`pyscf.data.elements.CONFIGURATION`, electrons per
    angular momentum): an open ``l`` shell holding ``n`` electrons
    contributes ``min(n, 2 (2l + 1) - n)``.
    """
    from pyscf.data.elements import CONFIGURATION
    z = int(z)
    conf = CONFIGURATION[z]
    unpaired = 0
    for l_ang, n in enumerate(conf):
        cap = 2 * (2 * l_ang + 1)
        rem = int(n) % cap
        if rem:
            unpaired += min(rem, cap - rem)
    n_alpha = (z + unpaired) // 2
    return n_alpha, z - n_alpha


def _assign_electrons(charges, n_up, n_down, rng):
    """Atom index of every spin-up and spin-down electron.

    FermiNet's scheme: neutral-atom occupations, then random spin
    flips of single atoms until the spin totals match.  If the
    molecule is charged, electrons are removed from (or added to)
    the atoms round-robin to reach ``(n_up, n_down)``.
    """
    occ = [list(atomic_spin_config(z)) for z in charges]
    n_at = len(occ)
    for _ in range(100 * max(n_at, 1)):
        n_a = sum(o[0] for o in occ)
        if n_a == n_up:
            break
        want_flip = [
            i for i in range(n_at)
            if (occ[i][0] > occ[i][1] if n_a > n_up
                else occ[i][0] < occ[i][1])
        ]
        if not want_flip:
            break
        i = int(rng.choice(want_flip))
        occ[i] = [occ[i][1], occ[i][0]]
    up = [i for i in range(n_at) for _ in range(occ[i][0])]
    dn = [i for i in range(n_at) for _ in range(occ[i][1])]

    def _fit(lst, n):
        lst = list(lst[:n])
        k = 0
        while len(lst) < n:
            lst.append(k % n_at)
            k += 1
        return lst
    return _fit(up, n_up), _fit(dn, n_down)


def init_electrons(mol_info, num_walkers, rng_key, *,
                   init_width: float = INIT_WIDTH, seed: int = 0):
    """FermiNet's initial walkers in OmegaQMC's electron layout.

    Returns ``(num_walkers, n_e, 3)`` float64 coordinates in
    OmegaQMC's electron layout (:mod:`OmegaQMC.spin_layout`), so that
    their spin-grouped view (what the network sees, see
    :func:`~OmegaQMC.psi.nn.adapter.make_nn_log_psi`) has the spin-up
    electrons of :func:`_assign_electrons` first.  *seed* drives its
    random spin flips.
    """
    charges = [int(round(float(z))) for z in mol_info.charges]
    n_up, n_down = int(mol_info.n_up), int(mol_info.n_down)
    n_e = n_up + n_down
    up, dn = _assign_electrons(
        charges, n_up, n_down, np.random.default_rng(seed),
    )
    coords = np.asarray(mol_info.coords, dtype=np.float64)
    grouped = coords[np.asarray(up + dn, dtype=int)]
    centres = grouped[layout_permutation(n_up, n_down)]
    noise = init_width * jax.random.normal(
        rng_key, (num_walkers, n_e, 3), dtype=jnp.float64,
    )
    return jnp.asarray(centres)[None] + noise


# ---------------------------------------------------------------------
# Hartree-Fock target
# ---------------------------------------------------------------------
class HartreeFockTarget:
    """UHF orbitals of a molecule evaluated at electron positions."""

    def __init__(self, mol_info, basis: str = PRETRAIN_BASIS,
                 verbose: int = 0):
        from pyscf import gto, scf

        charges = [int(round(float(z))) for z in mol_info.charges]
        coords = np.asarray(mol_info.coords, dtype=np.float64)
        self.n_up = int(mol_info.n_up)
        self.n_down = int(mol_info.n_down)
        n_e = self.n_up + self.n_down
        self.mol = gto.M(
            atom=[(z, tuple(xyz)) for z, xyz in zip(charges, coords)],
            unit='Bohr', basis=basis,
            spin=self.n_up - self.n_down,
            charge=sum(charges) - n_e,
            verbose=verbose,
        )
        self.mf = scf.UHF(self.mol)
        self.energy = float(self.mf.kernel())
        self.converged = bool(self.mf.converged)
        mo_a, mo_b = self.mf.mo_coeff
        self.c_up = np.asarray(mo_a[:, :self.n_up])
        self.c_down = np.asarray(mo_b[:, :self.n_down])

    def __call__(self, r_grouped):
        """Target blocks for spin-grouped walkers.

        Args:
            r_grouped: ``(W, n_e, 3)``, spin-up electrons first.

        Returns:
            ``(t_up, t_down)`` of shapes ``(W, n_up, n_up)`` and
            ``(W, n_down, n_down)``: rows are electrons, columns the
            occupied MOs.
        """
        r = np.asarray(r_grouped, dtype=np.float64)
        w, n_e, _ = r.shape
        nu = self.n_up
        ao_up = self.mol.eval_gto(
            'GTOval_sph', r[:, :nu].reshape(-1, 3),
        )
        ao_dn = self.mol.eval_gto(
            'GTOval_sph', r[:, nu:].reshape(-1, 3),
        )
        t_up = (ao_up @ self.c_up).reshape(w, nu, nu)
        t_dn = (ao_dn @ self.c_down).reshape(
            w, n_e - nu, self.n_down,
        )
        return t_up, t_dn


def _target_matrix(t_up, t_dn, full_det):
    """Block-diagonal full-determinant target (FermiNet's)."""
    if not full_det:
        return t_up, t_dn
    w, nu, _ = t_up.shape
    nd = t_dn.shape[1]
    top = jnp.concatenate(
        [t_up, jnp.zeros((w, nu, nd), t_up.dtype)], axis=-1,
    )
    bot = jnp.concatenate(
        [jnp.zeros((w, nd, nu), t_dn.dtype), t_dn], axis=-1,
    )
    return jnp.concatenate([top, bot], axis=-2)


# ---------------------------------------------------------------------
# Pretraining loop
# ---------------------------------------------------------------------
def pretrain_hf(
    log_psi,
    mol_info,
    params,
    rng_key,
    *,
    iterations: int = PRETRAIN_ITERATIONS,
    num_walkers: int = 4096,
    basis: str = PRETRAIN_BASIS,
    lr: float = PRETRAIN_LR,
    mcmc_width: float = PRETRAIN_MCMC_WIDTH,
    init_width: float = INIT_WIDTH,
    walkers=None,
    log_every: int = 100,
    fout=None,
    verbose: int = 1,
):
    """Pretrain an NN wavefunction's orbitals against UHF.

    Args:
        log_psi: ``log|psi|`` callable from
            :func:`~OmegaQMC.psi.nn.adapter.make_nn_log_psi`; must
            carry the ``.orbitals`` and ``.full_determinant``
            attributes set there.
        mol_info: :class:`~OmegaQMC.utils.Mole_custom` (coordinates
            in bohr).
        params: Initial parameters.
        rng_key: JAX PRNG key.
        iterations: Adam steps (FermiNet: 1000).
        num_walkers: Walkers (FermiNet: the training batch, 4096).
        basis: Hartree-Fock basis (FermiNet: ``'sto-6g'``).
        lr: Adam learning rate (FermiNet: 3e-4).
        mcmc_width: Gaussian Metropolis step per iteration (bohr).
        init_width: Width of :func:`init_electrons`.
        walkers: Optional starting walkers; default
            :func:`init_electrons`.
        log_every: Iterations between log lines.
        fout: Text stream for the log (default stdout).
        verbose: 0 silences the log.

    Returns:
        ``(params, walkers, info)``: the pretrained parameters (same
        dtypes as *params*), the final walkers ``(W, n_e, 3)``, and a
        dict with ``'loss'`` (per-iteration array), ``'acceptance'``
        (mean acceptance), ``'e_hf'`` and ``'hf_converged'``.
    """
    import optax

    if not hasattr(log_psi, 'orbitals'):
        raise ValueError(
            "log_psi has no .orbitals; build it with make_nn_log_psi"
        )
    if fout is None:
        fout = sys.stdout
    if isinstance(rng_key, int):
        rng_key = jax.random.key(rng_key)
    full_det = bool(log_psi.full_determinant)
    orbitals = log_psi.orbitals
    nuc = jnp.asarray(mol_info.coords, dtype=jnp.float64)

    t0 = time.perf_counter()
    target = HartreeFockTarget(mol_info, basis=basis)
    if verbose:
        print(f"# HF pretraining: UHF/{basis} E = {target.energy:.8f} Ha"
              f" (converged: {target.converged}), {iterations}"
              f" iterations, {num_walkers} walkers, Adam lr {lr:g},"
              f" MCMC width {mcmc_width:g} bohr,"
              f" full_det={full_det}", file=fout)

    rng_key, k_init = jax.random.split(rng_key)
    if walkers is None:
        walkers = init_electrons(
            mol_info, num_walkers, k_init, init_width=init_width,
        )
    walkers = jnp.asarray(walkers, dtype=jnp.float64)

    batch_orbitals = jax.vmap(orbitals, in_axes=(0, None, None))
    batch_lp = jax.vmap(log_psi, in_axes=(0, None, None))

    def loss_fn(p, x, t_up, t_dn):
        o_up, o_dn, _ = batch_orbitals(x, nuc, p)
        if full_det:
            tgt = _target_matrix(t_up, t_dn, True)
            orb = jnp.concatenate([o_up, o_dn], axis=-2)
            return jnp.mean((tgt[:, None] - orb) ** 2)
        loss = 0.0
        for t, o in ((t_up, o_up), (t_dn, o_dn)):
            if t.size:
                loss = loss + jnp.mean((t[:, None] - o) ** 2)
        return loss

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p, state, x, t_up, t_dn, key):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, t_up, t_dn)
        updates, state = optimizer.update(grads, state, p)
        p = jax.tree.map(
            lambda a, u: (a + u).astype(a.dtype), p, updates,
        )
        # One all-electron Metropolis move under the updated |psi|^2.
        k_prop, k_acc = jax.random.split(key)
        prop = x + mcmc_width * jax.random.normal(
            k_prop, x.shape, dtype=x.dtype,
        )
        lp_old = batch_lp(x, nuc, p)
        lp_new = batch_lp(prop, nuc, p)
        acc = jnp.log(jax.random.uniform(
            k_acc, (x.shape[0],), dtype=x.dtype,
        )) < 2.0 * (lp_new - lp_old)
        x = jnp.where(acc[:, None, None], prop, x)
        return p, state, x, loss, jnp.mean(acc)

    n_up, n_down = int(mol_info.n_up), int(mol_info.n_down)

    def grouped(x):
        return np.asarray(to_grouped(x, n_up, n_down))

    losses = np.empty(iterations)
    accs = np.empty(iterations)
    for it in range(iterations):
        t_up, t_dn = target(grouped(walkers))
        rng_key, k = jax.random.split(rng_key)
        params, opt_state, walkers, loss, acc = step(
            params, opt_state, walkers,
            jnp.asarray(t_up), jnp.asarray(t_dn), k,
        )
        losses[it] = float(loss)
        accs[it] = float(acc)
        if verbose and (it % log_every == 0 or it == iterations - 1):
            print(f"# pretrain {it:6d}  loss {losses[it]:.6e}"
                  f"  acc {np.mean(accs[max(0, it - log_every + 1):it + 1]):.3f}"
                  f"  {time.perf_counter() - t0:8.1f} s", file=fout)

    if not np.all(np.isfinite(losses)):
        raise FloatingPointError("non-finite pretraining loss")
    return params, walkers, {
        'loss': losses,
        'acceptance': float(np.mean(accs)) if iterations else float('nan'),
        'e_hf': target.energy,
        'hf_converged': target.converged,
    }
