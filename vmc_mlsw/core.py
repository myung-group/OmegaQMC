import jax
import jax.numpy as jnp
from .psi import get_psi_fun
"""Core functionality for MLSW."""


def run():
    print("Hello World!")


def vqmc_energy(mf,
                nuc_crds,
                params_vmc,
                stacked_samples):

    log_trial_wavefunction, local_energy, get_psi_mo =\
        get_psi_fun(mf)

    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke = local_energy

    enr_ee_samples = jax.vmap(local_energy_ee)(stacked_samples)
    enr_nn = local_energy_nn(nuc_crds)
    enr_en_samples = jax.vmap(local_energy_en, in_axes=(0, None))(
        stacked_samples, nuc_crds
    )
    num_samples = stacked_samples.shape[0]
    num_batches = 10000
    enr_ke_samples = jnp.zeros((num_samples))
    for ist in range(0, num_samples, num_batches):
        ied = min(ist+num_batches, num_samples)
        enr_ke_samples = enr_ke_samples.at[ist:ied].set(
            jax.vmap(local_energy_ke, in_axes=(0, None, None))(
                stacked_samples[ist:ied], nuc_crds, params_vmc
            )
        )

    print('enr_ke_samples', enr_ke_samples.mean())
    print('enr_ee_samples', enr_ee_samples.mean())
    print('enr_en_samples', enr_en_samples.mean())
    print("enr_nn", enr_nn)

    enr_samples = enr_ee_samples+enr_en_samples+enr_ke_samples
    print("enr_samples", enr_samples.mean())
    return enr_samples, enr_nn


def vqmc_gradient(mf,
                  nuc_crds,
                  params_vmc,
                  stacked_samples,
                  enr_samples):

    log_trial_wavefunction, local_energy, get_psi_mo =\
        get_psi_fun(mf)

    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke = local_energy

    @jax.jit
    def redistribute_samples_ver1(elec_crds):
        mo_val = get_psi_mo(elec_crds, nuc_crds)
        rho_val = jnp.einsum('neo,neo->en', mo_val, mo_val)
        rho_val_sum = rho_val.sum(axis=-1)
        rho_val = jnp.einsum('en,e->en', rho_val, 1.0/rho_val_sum)

        return rho_val

    @jax.jit
    def redistribute_samples_ver2(elec_crds):
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1)+1e-10)
        weight = dist**(-4.0)
        return weight/(jnp.sum(weight, axis=-1, keepdims=True)+1.e-10)

    # --- Redistribute Gradient Samples ---
    jac_rescale_fn = jax.jacobian(redistribute_samples_ver1, argnums=0)

    rescale = jax.vmap(redistribute_samples_ver1)(stacked_samples)

    @jax.jit
    def single_sample_grad_elocal_en(e_pos_):
        return jax.grad(local_energy_en, argnums=(0, 1))(e_pos_, nuc_crds)

    @jax.jit
    def single_sample_grad_elocal_ee(e_pos_):
        return jax.grad(local_energy_ee)(e_pos_)

    @jax.jit
    def single_sample_grad_elocal_ke(e_pos_):
        return jax.grad(local_energy_ke, argnums=(0, 1))(e_pos_, nuc_crds, params_vmc)

    @jax.jit
    def single_sample_grad_logpsi(e_pos_):
        return jax.grad(log_trial_wavefunction, argnums=(0, 1))(e_pos_,
                                                                nuc_crds,
                                                                params_vmc)
    # --- Nuclear Gradient ---
    grad_eloc_nn_nuc = jax.grad(local_energy_nn)(nuc_crds)
    grd_nn = grad_eloc_nn_nuc
    print('grd_nn\n', grd_nn)
    # --- Electron Gradient ---
    grad_eloc_ee_elc = jax.vmap(single_sample_grad_elocal_ee)(stacked_samples)
    grad_eloc_ee_nuc = jnp.einsum('seK,sen->snK', grad_eloc_ee_elc, rescale)
    grd_ee = grad_eloc_ee_nuc.mean(axis=0)
    print('grd_ee\n', grd_ee)
    # print('grd_nn+grd_ee\n', grd_nn+grd_ee)
    # --- Electron-Nuclear Gradient ---
    grad_eloc_en_elc, grad_eloc_en_nuc = \
        jax.vmap(single_sample_grad_elocal_en)(stacked_samples)
    grad_eloc_en_nuc = grad_eloc_en_nuc + \
        jnp.einsum('seK,sen->snK', grad_eloc_en_elc, rescale)

    grd_en = grad_eloc_en_nuc.mean(axis=0)
    print('grd_en\n', grd_en)
    # print('grd_nn+grd_ee+grd_en\n', grd_nn+grd_ee+grd_en)
    # --- Electron-Kinetic Gradient ---
    num_samples = stacked_samples.shape[0]
    num_electrons = stacked_samples.shape[1]
    num_nuc = nuc_crds.shape[0]
    num_batches = 10000
    grad_eloc_ke_elc = jnp.zeros((num_samples, num_electrons, 3))
    grad_eloc_ke_nuc = jnp.zeros((num_samples, num_nuc, 3))

    for ist in range(0, num_samples, num_batches):
        ied = min(ist+num_batches, num_samples)
        grad_ke_elc, grad_ke_nuc = \
            jax.vmap(single_sample_grad_elocal_ke)(stacked_samples[ist:ied])
        grad_eloc_ke_elc = grad_eloc_ke_elc.at[ist:ied].set(grad_ke_elc)
        grad_eloc_ke_nuc = grad_eloc_ke_nuc.at[ist:ied].set(grad_ke_nuc)

    grad_eloc_ke_nuc = grad_eloc_ke_nuc + \
        jnp.einsum('seK,sen->snK', grad_eloc_ke_elc, rescale)
    grd_ke = grad_eloc_ke_nuc.mean(axis=0)
    print('grd_ke\n', grd_ke)
    # print('grd_nn+grd_ee+grd_en+grd_ke\n',
    #       grd_nn+grd_ee+grd_en+grd_ke)

    # --- Electron-LogPsi Gradient ---
    grad_logpsi_elc = jnp.zeros((num_samples, num_electrons, 3))
    grad_logpsi_nuc = jnp.zeros((num_samples, num_nuc, 3))
    for ist in range(0, num_samples, num_batches):
        ied = min(ist+num_batches, num_samples)
        grad_elc, grad_nuc = \
            jax.vmap(single_sample_grad_logpsi)(stacked_samples[ist:ied])
        grad_logpsi_elc = grad_logpsi_elc.at[ist:ied].set(grad_elc)
        grad_logpsi_nuc = grad_logpsi_nuc.at[ist:ied].set(grad_nuc)

    jac_rescale_elec = jax.vmap(jac_rescale_fn)(stacked_samples)
    novel_correction = 0.5*jnp.einsum('senek->snk', jac_rescale_elec)

    print('before logpsi_nuc\n', grad_logpsi_nuc.mean(axis=0))
    grad_logpsi_nuc = grad_logpsi_nuc + \
        jnp.einsum('seK,sen->snK', grad_logpsi_elc, rescale) + \
        novel_correction

    print('after logpsi_nuc\n', grad_logpsi_nuc.mean(axis=0))

    # --- Pulay Gradient ---
    enr_mean = enr_samples.mean()
    d_enr = enr_samples - enr_mean
    print('d_enr', d_enr.mean(axis=0))
    pulay_terms = \
        2.0*jnp.einsum('s,snK->snK',
                       d_enr,
                       grad_logpsi_nuc).mean(axis=0)
    print('grd_pulay\n', pulay_terms)
    # print('grd_nn+grd_ee+grd_en+grd_ke+grd_pulay\n',
    #       grd_nn+grd_ee+grd_en+grd_ke+pulay_terms)
    total_grad = grd_nn+grd_ee+grd_en+grd_ke+pulay_terms
    return total_grad
