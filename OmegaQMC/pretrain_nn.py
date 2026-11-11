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

Optional, off by default: **HF-density sampling** (``hf_fraction``),
the pretraining of Pfau, Spencer, Matthews and Foulkes, Phys. Rev.
Research 2, 033429 (2020), whose TensorFlow FermiNet drew "half the
samples in the batch from |psi|^2 and half from the product of
Hartree-Fock orbitals using MCMC" (``hf_fraction=0.5``).  That share
of the walkers is a second, persistent Metropolis chain on
``p_HF(R) ~ |det phi_up(r_up)|^2 |det phi_down(r_down)|^2`` built from
the same UHF orbitals as the target; the loss is the mean over both
sets.  With ``hf_fraction=0`` (the default) pretraining is the
FermiNet 4cfdf1e scheme above, bit for bit.

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
# Metropolis chain on the Hartree-Fock density (Pfau et al. 2020)
# ---------------------------------------------------------------------
def _hf_log_density(t_up, t_dn):
    """``log p_HF`` (unnormalised) from the target blocks.

    ``2 log|det t_up| + 2 log|det t_dn|`` per walker; an empty spin
    block contributes 0.
    """
    out = np.zeros(t_up.shape[0])
    for t in (t_up, t_dn):
        if t.shape[-1]:
            out = out + 2.0 * np.linalg.slogdet(t)[1]
    return out


def hf_density_state(target, walkers):
    """Target blocks and ``log p_HF`` of layout-ordered walkers.

    Returns ``(t_up, t_dn, log_p)`` (host arrays), the cache that
    :func:`hf_metropolis_step` carries from move to move.
    """
    x = np.asarray(walkers, dtype=np.float64)
    t_up, t_dn = target(to_grouped(x, target.n_up, target.n_down))
    return t_up, t_dn, _hf_log_density(t_up, t_dn)


def hf_metropolis_step(target, walkers, state, rng_key, width):
    """One all-electron Gaussian Metropolis move on ``p_HF``.

    Args:
        target: :class:`HartreeFockTarget`.
        walkers: ``(W, n_e, 3)`` host array in OmegaQMC's layout.
        state: ``(t_up, t_dn, log_p)`` of *walkers* from
            :func:`hf_density_state` (or a previous call).
        rng_key: JAX PRNG key.
        width: Standard deviation (bohr) of the Gaussian proposal.

    Returns:
        ``(walkers, state, acceptance)``: the moved walkers, their
        cached target blocks and ``log p_HF``, and the accepted
        fraction.  One target evaluation (at the proposals) per call.
    """
    t_up, t_dn, log_p = state
    k_prop, k_acc = jax.random.split(rng_key)
    x = np.asarray(walkers, dtype=np.float64)
    prop = x + width * np.asarray(jax.random.normal(
        k_prop, x.shape, dtype=jnp.float64,
    ))
    p_up, p_dn, log_p_new = hf_density_state(target, prop)
    u = np.asarray(jax.random.uniform(
        k_acc, (x.shape[0],), dtype=jnp.float64,
    ))
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.log(u) < log_p_new - log_p
    x = np.where(acc[:, None, None], prop, x)
    state = (
        np.where(acc[:, None, None], p_up, t_up),
        np.where(acc[:, None, None], p_dn, t_dn),
        np.where(acc, log_p_new, log_p),
    )
    return x, state, float(np.mean(acc))


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
    hf_fraction: float = 0.0,
    log_every: int = 100,
    fout=None,
    verbose: int = 1,
):
    """Pretrain an NN wavefunction's orbitals against UHF.

    With ``hf_fraction > 0`` (Pfau et al. 2020 use 0.5),
    ``n_hf = round(hf_fraction * num_walkers)`` of the walkers sample
    the Hartree-Fock density ``|det phi_up|^2 |det phi_down|^2`` of
    the UHF target instead of ``|psi|^2``: a separate persistent chain
    with one Gaussian Metropolis move of width *mcmc_width* per
    iteration (:func:`hf_metropolis_step`; the target of its accepted
    positions is cached, so it costs one target evaluation of
    ``n_hf`` walkers per iteration).  Both sets start from the same
    initial walkers (the first ``num_walkers - n_hf`` sample
    ``|psi|^2``, the last ``n_hf`` the HF density), the loss is the
    mean over all of them, and both move after each parameter update.
    ``hf_fraction=0`` is the FermiNet scheme, bit for bit.

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
        hf_fraction: Share of the walkers that sample the HF density
            (0 to 1; default 0, Pfau et al. 2020: 0.5).
        log_every: Iterations between log lines.
        fout: Text stream for the log (default stdout).
        verbose: 0 silences the log.

    Returns:
        ``(params, walkers, info)``: the pretrained parameters (same
        dtypes as *params*), the final walkers ``(W, n_e, 3)`` (the
        ``|psi|^2`` set first, the HF-density set last), and a dict
        with ``'loss'`` (per-iteration array), ``'acceptance'`` (mean
        acceptance of the ``|psi|^2`` set; NaN if it is empty),
        ``'acceptance_hf'`` (of the HF-density set; NaN if empty),
        ``'n_hf'`` (walkers in the HF-density set), ``'e_hf'`` and
        ``'hf_converged'``.
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
    hf_fraction = float(hf_fraction)
    if not 0.0 <= hf_fraction <= 1.0:
        raise ValueError(f"hf_fraction must be in [0, 1], got {hf_fraction}")
    full_det = bool(log_psi.full_determinant)
    orbitals = log_psi.orbitals
    nuc = jnp.asarray(mol_info.coords, dtype=jnp.float64)

    t0 = time.perf_counter()
    target = HartreeFockTarget(mol_info, basis=basis)

    rng_key, k_init = jax.random.split(rng_key)
    if walkers is None:
        walkers = init_electrons(
            mol_info, num_walkers, k_init, init_width=init_width,
        )
    walkers = jnp.asarray(walkers, dtype=jnp.float64)
    num_walkers = walkers.shape[0]
    n_hf = int(round(hf_fraction * num_walkers))
    n_psi = num_walkers - n_hf

    if verbose:
        print(f"# HF pretraining: UHF/{basis} E = {target.energy:.8f} Ha"
              f" (converged: {target.converged}), {iterations}"
              f" iterations, {num_walkers} walkers, Adam lr {lr:g},"
              f" MCMC width {mcmc_width:g} bohr,"
              f" full_det={full_det}", file=fout)
        if n_hf:
            print(f"# HF pretraining: HF-density sampling (Pfau 2020) on"
                  f" {n_hf} of {num_walkers} walkers"
                  f" (hf_fraction {hf_fraction:g}), |psi|^2 on {n_psi}",
                  file=fout)

    # The HF-density chain (the last n_hf walkers) lives on the host,
    # next to the PySCF target; its random numbers come from a key
    # folded off rng_key, so the main key stream is that of
    # hf_fraction=0.
    if n_hf:
        hf_key = jax.random.fold_in(rng_key, 0x48460)
        walkers_hf = np.asarray(walkers[n_psi:])
        hf_state = hf_density_state(target, walkers_hf)
        walkers = walkers[:n_psi] if n_psi else None
    else:
        walkers_hf = None

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
    def step(p, state, x, x_hf, t_up, t_dn, key):
        # x: the |psi|^2 walkers, x_hf: the HF-density walkers (either
        # may be None); t_up, t_dn: targets of both, in that order.
        if x_hf is None:
            x_all = x
        elif x is None:
            x_all = x_hf
        else:
            x_all = jnp.concatenate([x, x_hf], axis=0)
        loss, grads = jax.value_and_grad(loss_fn)(p, x_all, t_up, t_dn)
        updates, state = optimizer.update(grads, state, p)
        p = jax.tree.map(
            lambda a, u: (a + u).astype(a.dtype), p, updates,
        )
        if x is None:
            return p, state, x, loss, jnp.nan
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
    accs_hf = np.full(iterations, np.nan)
    for it in range(iterations):
        if walkers is not None:
            t_up, t_dn = target(grouped(walkers))
        if n_hf:
            if walkers is None:
                t_up, t_dn = hf_state[0], hf_state[1]
            else:
                t_up = np.concatenate([t_up, hf_state[0]], axis=0)
                t_dn = np.concatenate([t_dn, hf_state[1]], axis=0)
        rng_key, k = jax.random.split(rng_key)
        params, opt_state, walkers_new, loss, acc = step(
            params, opt_state, walkers,
            None if walkers_hf is None else jnp.asarray(walkers_hf),
            jnp.asarray(t_up), jnp.asarray(t_dn), k,
        )
        if n_hf:
            # One Metropolis move of the HF-density set (host, PySCF);
            # it does not depend on the parameters.
            walkers_hf, hf_state, accs_hf[it] = hf_metropolis_step(
                target, walkers_hf, hf_state,
                jax.random.fold_in(hf_key, it), mcmc_width,
            )
        walkers = walkers_new
        losses[it] = float(loss)
        accs[it] = float(acc)
        if verbose and (it % log_every == 0 or it == iterations - 1):
            window = slice(max(0, it - log_every + 1), it + 1)
            line = (f"# pretrain {it:6d}  loss {losses[it]:.6e}"
                    f"  acc {np.mean(accs[window]):.3f}")
            if n_hf:
                line += (f"  acc_hf {np.mean(accs_hf[window]):.3f}"
                         f"  (hf {n_hf}/{num_walkers})")
            print(f"{line}  {time.perf_counter() - t0:8.1f} s", file=fout)

    if not np.all(np.isfinite(losses)):
        raise FloatingPointError("non-finite pretraining loss")
    if n_hf:
        walkers_hf = jnp.asarray(walkers_hf)
        walkers = (walkers_hf if walkers is None
                   else jnp.concatenate([walkers, walkers_hf], axis=0))
    return params, walkers, {
        'loss': losses,
        'acceptance': (float(np.mean(accs)) if iterations and n_psi
                       else float('nan')),
        'acceptance_hf': (float(np.mean(accs_hf)) if iterations and n_hf
                          else float('nan')),
        'n_hf': n_hf,
        'e_hf': target.energy,
        'hf_converged': target.converged,
    }
