#!/usr/bin/env python3
"""Reference data from DeepMind's FermiNet + kfac-jax for OmegaQMC's tests.

usage: JAX_PLATFORMS=cpu python tools/ferminet_kfac_reference.py OUT.npz

Needs ``ferminet`` (github.com/google-deepmind/ferminet) and ``kfac_jax``;
written with ferminet e9d5a60, kfac-jax 0.0.8, jax 0.9.2.  Runs on the
CPU in float64.  For a small LiH-like FermiNet (DeepMind's network with
hidden_dims 3 x (16, 4), 2 full determinants, isotropic envelope) it
records, for each case below, the initial parameters, log|psi| and E_L
on 16 fixed walkers, and the parameters after a few kfac-jax steps on
those walkers, with the optimiser configured as ferminet/train.py does
(estimation_mode 'fisher_exact', curvature EMA 0.95, damping 1e-3,
lr 0.05 / (1 + t / 1e4), E_L clipped at 5 mean absolute deviations
about the mean).  tests/test_vmcopt_nn_kfac_jax.py transplants these
parameters into OmegaQMC's ``ferminet_deepmind`` ansatz and checks
log|psi|, E_L and ``kfac_style='kfac_jax'`` against them.

Cases: 'a' (2, 2) electrons, norm constraint 1e-3, 2 steps;
       'b' (3, 2) electrons (LiH anion), no norm constraint, 3 steps.
"""
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ['JAX_ENABLE_X64'] = '1'

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import kfac_jax  # noqa: E402
import numpy as np  # noqa: E402

from ferminet import constants, envelopes, hamiltonian, networks  # noqa
from ferminet import loss as qmc_loss  # noqa: E402
from ferminet import curvature_tags_and_blocks  # noqa: E402

HIDDEN = ((16, 4), (16, 4), (16, 4))
NDET = 2
ATOMS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
CHARGES = np.array([3.0, 1.0])
B = 16
CASES = {'a': ((2, 2), 1.0e-3, 2), 'b': ((3, 2), None, 3)}


def flat_params(params, prefix):
  out = {}
  for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
    key = '/'.join(str(getattr(p, 'key', getattr(p, 'idx', p)))
                   for p in path)
    out[f'{prefix}/{key}'] = np.asarray(leaf)
  return out


def run_case(name, nspins, norm_constraint, n_steps):
  nelec = sum(nspins)
  atoms = jnp.asarray(ATOMS)
  charges = jnp.asarray(CHARGES)
  network = networks.make_fermi_net(
      nspins, charges, ndim=3, determinants=NDET,
      envelope=envelopes.make_isotropic_envelope(),
      feature_layer=networks.make_ferminet_features(
          natoms=len(CHARGES), nspins=nspins, ndim=3),
      jastrow='default', bias_orbitals=False, full_det=True,
      rescale_inputs=False, complex_output=False, hidden_dims=HIDDEN,
      use_last_layer=False, separate_spin_channels=False)
  key = jax.random.PRNGKey(7)
  key, sub = jax.random.split(key)
  params = network.init(sub)
  # move the envelope away from pi = sigma = 1 so that the transplant
  # test can tell its parameters apart
  params['envelope'] = [
      {k: v * (1.0 + 0.1 * jax.random.normal(
          jax.random.fold_in(sub, 7 * i + j), v.shape))
       for j, (k, v) in enumerate(sorted(p.items()))}
      for i, p in enumerate(params['envelope'])]
  logabs = lambda *a, **k: network.apply(*a, **k)[1]  # noqa: E731

  key, sub = jax.random.split(key)
  pos = jax.random.normal(sub, (B, nelec * 3)) * 1.2
  pos = pos.at[:, 2::3].add(1.0)
  spins = jnp.tile(jnp.array([1.0] * nspins[0] + [-1.0] * nspins[1]),
                   (B, 1))
  b_atoms = jnp.tile(atoms[None], (B, 1, 1))
  b_charges = jnp.tile(charges[None], (B, 1))
  logpsi = jax.vmap(logabs, in_axes=(None, 0, 0, 0, 0))(
      params, pos, spins, b_atoms, b_charges)

  local_energy = hamiltonian.local_energy(
      f=network.apply, charges=charges, nspins=nspins, use_scan=False,
      complex_output=False, laplacian_method='default')
  data = networks.FermiNetData(positions=pos, spins=spins, atoms=b_atoms,
                               charges=b_charges)
  e_l, _ = jax.vmap(local_energy, in_axes=(
      None, None, networks.FermiNetData(positions=0, spins=0, atoms=0,
                                        charges=0)))(params, None, data)

  evaluate_loss = qmc_loss.make_loss(
      logabs, local_energy, clip_local_energy=5.0, clip_from_median=False,
      center_at_clipped_energy=True, complex_output=False)
  optimizer = kfac_jax.Optimizer(
      jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True),
      l2_reg=0.0, norm_constraint=norm_constraint,
      value_func_has_aux=True, value_func_has_rng=True,
      learning_rate_schedule=lambda t: 0.05 / (1.0 + t / 1.0e4),
      curvature_ema=0.95, inverse_update_period=1, min_damping=1e-4,
      num_burnin_steps=0, register_only_generic=False,
      estimation_mode='fisher_exact', multi_device=True,
      pmap_axis_name=constants.PMAP_AXIS_NAME,
      auto_register_kwargs=dict(
          graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS))
  rep = lambda x: kfac_jax.utils.replicate_all_local_devices(  # noqa
      x, axis_name=constants.PMAP_AXIS_NAME)
  p_data = jax.tree_util.tree_map(lambda x: x[None], data)
  p_params = rep(params)
  sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(
      jax.random.PRNGKey(0))
  sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
  opt_state = optimizer.init(p_params, subkeys, p_data)
  for blk in optimizer._estimator.blocks:  # pylint: disable=protected-access
    print(name, type(blk).__name__, blk.name,
          [tuple(s) for s in blk.parameters_shapes])

  out = flat_params(params, f'{name}/traj0')
  for it in range(n_steps):
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    p_params, opt_state, stats = optimizer.step(
        params=p_params, state=opt_state, rng=subkeys, batch=p_data,
        momentum=rep(jnp.zeros([])), damping=rep(jnp.asarray(1e-3)))
    print(name, 'step', it, 'loss', float(stats['loss'][0]))
    out.update(flat_params(jax.tree_util.tree_map(lambda x: x[0], p_params),
                           f'{name}/traj{it + 1}'))
  out[f'{name}/pos'] = np.asarray(pos)
  out[f'{name}/logpsi'] = np.asarray(logpsi)
  out[f'{name}/e_l'] = np.asarray(e_l)
  out[f'{name}/nspins'] = np.asarray(nspins)
  out[f'{name}/norm_constraint'] = np.asarray(
      np.inf if norm_constraint is None else norm_constraint)
  out[f'{name}/n_steps'] = np.asarray(n_steps)
  return out


def main():
  if len(sys.argv) != 2:
    sys.exit(__doc__)
  if jax.local_device_count() != 1:
    sys.exit('run on one device (e.g. JAX_PLATFORMS=cpu)')
  out = dict(atoms=ATOMS, charges=CHARGES, hidden=np.asarray(HIDDEN),
             ndet=np.asarray(NDET))
  for name, (nspins, nc, n_steps) in CASES.items():
    out.update(run_case(name, nspins, nc, n_steps))
  np.savez_compressed(sys.argv[1], **out)
  print('saved', sys.argv[1])


if __name__ == '__main__':
  main()
