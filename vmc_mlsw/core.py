import jax
import jax.numpy as jnp
from .psi import get_psi_fun
import h5py
"""Core functionality for MLSW."""



def vmc_run(mf,
             rng_key,
             nuc_crds,
             params_vmc,
             num_steps=5000,
             num_equilibration=1000,
             step_size=0.25,
             chkfile='vmc_chk.hdf5'):

    log_trial_wavefunction, local_energy, get_psi_mo = \
          get_psi_fun(mf)
    nelec = mf.mol.tot_electrons()
    Z_charges = mf.mol.atom_charges()
    species = list(set (Z_charges))
    unique_species = [i+1 for i, s in enumerate(species)]

    @jax.jit
    def metropolis_step(rng_key,
                        elec_crds,
                        _step_size):
        
        rng_key, key_prop, key_accept = \
            jax.random.split(rng_key, 3)
        proposed_crds = elec_crds + _step_size*jax.random.normal(
            key_prop, elec_crds.shape)
        log_psi_old_sq = 2*log_trial_wavefunction(
            elec_crds, nuc_crds, params_vmc)
        log_psi_new_sq = 2*log_trial_wavefunction(
            proposed_crds, nuc_crds, params_vmc)
        acceptance_log_ratio = jnp.minimum(0, log_psi_new_sq - log_psi_old_sq)
        accept = jnp.log(jax.random.uniform(key_accept)) < acceptance_log_ratio
        new_crds = jnp.where(accept, proposed_crds, elec_crds)

        return rng_key, new_crds, accept

    rng_key, rng = jax.random.split(rng_key)
    idx_cnt = []
    for ia, iz in enumerate(Z_charges):
        idx_cnt = idx_cnt + [ia]*iz

    if mf.mol.charge < 0:
        # add electrons to the system
        idx_cnt = idx_cnt + [0]* abs (mf.mol.charge)
    elif mf.mol.charge > 0:
        # remove electrons from the system
        for _ in range (mf.mol.charge):
            idx_cnt.pop(-1)
    

    idx_cnt = jnp.array(idx_cnt)
    centers = nuc_crds[idx_cnt]
    elec_crds = centers + 0.05*jax.random.normal(rng, (nelec, 3))

    accepts_eq = 0
    ratio = 0.0

    for step in range(num_equilibration):
        rng_key, elec_crds, accepted = \
            metropolis_step(rng_key, elec_crds, step_size)

        accepts_eq += accepted.sum()
        if (step+1)%5000 == 0:
            ratio = accepts_eq / 5000 
            step_size = step_size * (0.5+ratio)
            accepts_eq = 0

    if num_equilibration > 0:
        print("Equilibration Acceptance Rate:"
              f"{ratio:.2f}")
        print("step_size", step_size)

    samples_list = []
    accepts_main = 0
    for step in range(num_steps):
        rng_key, elec_crds, accepted = \
            metropolis_step(rng_key, elec_crds, step_size)

        accepts_main += accepted.sum()
        if (step+1)%50000 == 0:
            ratio = accepts_main / (50000)
            step_size = step_size * (0.5 + ratio)
            accepts_main = 0

        if (step+1) % 10 == 0:
            samples_list.append(elec_crds)

    if num_steps > 0:
        print("Main Sampling Acceptance Rate:"
              f"{ratio:.2f}")
        print("step_size", step_size)

    if not samples_list:
        print("Warning: No samples collected in vmc_run.")
        return
    
    stacked_samples = jnp.stack(samples_list)

    with h5py.File(chkfile, 'w') as f:
        f.create_dataset('species', data=unique_species)
        f.create_dataset('nuc_crds', data=nuc_crds)
        f.create_dataset('params_vmc', data=params_vmc)
        f.create_dataset('stacked_samples', data=stacked_samples)
        f.create_dataset('atomic_masses', data=mf.mol.atom_mass_list())
       

def vmc_energy(mf,
               params_vmc,
               chkfile='vmc_chk.hdf5'):

    nuc_crds = None
    stacked_samples = None 
    # Load samples from checkpoint file
    with h5py.File(chkfile, 'r') as f:
        stacked_samples = jnp.array(f['stacked_samples'][:])
        nuc_crds = jnp.array(f['nuc_crds'][:])

    log_trial_wavefunction, local_energy, get_psi_mo \
        = get_psi_fun(mf)

    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke \
        = local_energy

    ener_ee_samples = jax.vmap(local_energy_ee)(stacked_samples)
    ener_nn = local_energy_nn(nuc_crds)
    ener_en_samples = jax.vmap(local_energy_en, in_axes=(0, None))(
        stacked_samples, nuc_crds
    )
    num_samples = stacked_samples.shape[0]
    num_batches = 10000
    ener_ke_samples = jnp.zeros((num_samples))
    for ist in range(0, num_samples, num_batches):
        ied = min(ist+num_batches, num_samples)
        ener_ke_samples = ener_ke_samples.at[ist:ied].set(
            jax.vmap(local_energy_ke, in_axes=(0, None, None))(
                stacked_samples[ist:ied], nuc_crds, params_vmc
            )
        )

    print('ener_ke_samples', ener_ke_samples.mean())
    print('ener_ee_samples', ener_ee_samples.mean())
    print('ener_en_samples', ener_en_samples.mean())
    print("ener_nn", ener_nn)

    ener_samples = ener_ee_samples+ener_en_samples+ener_ke_samples
    print("enr_samples", ener_samples.mean())
    print("enr_total[Ha]", ener_samples.mean()+ener_nn)

    with h5py.File(chkfile, 'a') as f:
        f.create_dataset('ener_nn', data=ener_nn)
        f.create_dataset('ener_ee_samples', data=ener_ee_samples)
        f.create_dataset('ener_en_samples', data=ener_en_samples)
        f.create_dataset('ener_ke_samples', data=ener_ke_samples)

    return 


def vmc_gradient_prep(mf,
                      params_vmc,
                      chkfile='vmc_chk.hdf5'):

    log_trial_wavefunction, local_energy, get_psi_mo =\
        get_psi_fun(mf)

    local_energy_ee, local_energy_nn, local_energy_en, local_energy_ke \
        = local_energy

    stacked_samples = None 
    nuc_crds = None  
    with h5py.File(chkfile, 'r') as f:
        stacked_samples = jnp.array(f['stacked_samples'][:])
        nuc_crds = jnp.array(f['nuc_crds'][:])

    num_samples, num_electrons, num_nuc = \
        stacked_samples.shape[0], stacked_samples.shape[1], nuc_crds.shape[0]

    # --- JIT-compiled gradient functions ---
    @jax.jit
    def single_sample_grad_elocal_en(e_pos_): 
        return jax.grad(local_energy_en, argnums=(0,1))(e_pos_, nuc_crds)
    @jax.jit
    def single_sample_grad_elocal_ee(e_pos_): 
        return jax.grad(local_energy_ee)(e_pos_)
    @jax.jit
    def single_sample_grad_elocal_ke(e_pos_): 
        return jax.grad(local_energy_ke, argnums=(0,1))(e_pos_, nuc_crds, params_vmc)
    @jax.jit
    def single_sample_grad_logpsi(e_pos_): 
        return jax.grad(log_trial_wavefunction, argnums=(0,1))(e_pos_, nuc_crds, params_vmc)

    # --- STEP 1: Pre-compute all gradient components that DON'T depend on the redistribution ---
    grad_eloc_nn_nuc = jax.grad(local_energy_nn)(nuc_crds)
    grad_eloc_ee_elc = jax.vmap(single_sample_grad_elocal_ee)(stacked_samples)
    grad_eloc_en_elc, grad_eloc_en_nuc_base = jax.vmap(single_sample_grad_elocal_en)(stacked_samples)
    
    num_batches = 2000
    grad_eloc_ke_elc = jnp.zeros((num_samples, num_electrons, 3))
    grad_eloc_ke_nuc_base = jnp.zeros((num_samples, num_nuc, 3))
    grad_logpsi_elc = jnp.zeros((num_samples, num_electrons, 3))
    grad_logpsi_nuc_base = jnp.zeros((num_samples, num_nuc, 3))

    for ist in range(0, num_samples, num_batches):
        ied = min(ist + num_batches, num_samples)
        batch = stacked_samples[ist:ied]
        grad_ke_elc_b, grad_ke_nuc_b = jax.vmap(single_sample_grad_elocal_ke)(batch)
        grad_eloc_ke_elc = grad_eloc_ke_elc.at[ist:ied].set(grad_ke_elc_b)
        grad_eloc_ke_nuc_base = grad_eloc_ke_nuc_base.at[ist:ied].set(grad_ke_nuc_b)
        grad_elc_b, grad_nuc_b = jax.vmap(single_sample_grad_logpsi)(batch)
        grad_logpsi_elc = grad_logpsi_elc.at[ist:ied].set(grad_elc_b)
        grad_logpsi_nuc_base = grad_logpsi_nuc_base.at[ist:ied].set(grad_nuc_b)

    with h5py.File(chkfile, 'a') as f:
        f.create_dataset('grad_eloc_nn_nuc', data=grad_eloc_nn_nuc)
        f.create_dataset('grad_eloc_ee_elc', data=grad_eloc_ee_elc)
        f.create_dataset('grad_eloc_en_elc', data=grad_eloc_en_elc)
        f.create_dataset('grad_eloc_en_nuc_base', data=grad_eloc_en_nuc_base)
        f.create_dataset('grad_eloc_ke_elc', data=grad_eloc_ke_elc)
        f.create_dataset('grad_eloc_ke_nuc_base', data=grad_eloc_ke_nuc_base)
        f.create_dataset('grad_logpsi_elc', data=grad_logpsi_elc)
        f.create_dataset('grad_logpsi_nuc_base', data=grad_logpsi_nuc_base)

    return 


def vmc_gradient_with_space_warping(mf,
                                   chkfile='vmc_chk.hdf5',
                                   scheme='scheme1'):

    log_trial_wavefunction, local_energy, get_psi_mo =\
        get_psi_fun(mf)
    
    @jax.jit
    def redistribute_samples_scheme1(elec_crds, nuc_crds):
        mo_val = get_psi_mo(elec_crds, nuc_crds)
        rho_val = jnp.einsum('neo,neo->en', mo_val, mo_val)
        rho_val_sum = rho_val.sum(axis=-1)
        rho_val = jnp.einsum('en,e->en', rho_val, 1.0/rho_val_sum)

        return rho_val

    @jax.jit
    def redistribute_samples_scheme2(elec_crds, nuc_crds):
        diff = elec_crds[:, None, :] - nuc_crds[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1)+1e-10)
        weight = dist**(-4.0)
        return weight/(jnp.sum(weight, axis=-1, keepdims=True)+1.e-10)

    # --- Redistribute Gradient Samples ---
    if scheme in ['scheme1']:
        rescale_fn = redistribute_samples_scheme1 
    elif scheme in ['scheme2']:
        rescale_fn = redistribute_samples_scheme2
    else:
        # default redistribute scheme
        rescale_fn = redistribute_samples_scheme1 

    jac_rescale_fn = jax.jacobian(rescale_fn, argnums=0)

    with h5py.File(chkfile,'r') as f:
        stacked_samples = jnp.array(f['stacked_samples'][:])
        nuc_crds = jnp.array(f['nuc_crds'][:])

        atomic_masses = jnp.array(f['atomic_masses'][:])
        mass_center = jnp.einsum('i,ij->j', atomic_masses, nuc_crds)/atomic_masses.sum()
        relative_nuc_pos = nuc_crds - mass_center 

        #
        num_batches = 2000
        num_samples = stacked_samples.shape[0]
        num_elc = stacked_samples.shape[1]
        num_nuc = nuc_crds.shape[0]
        
        rescale = jnp.zeros ( (num_samples, num_elc, num_nuc))
        for ist in range (0, num_samples, num_batches):
            ied = min (ist+num_batches, num_samples)
            val = jax.vmap (rescale_fn, in_axes=(0,None)) (
                stacked_samples[ist:ied], nuc_crds
            )
            rescale = rescale.at[ist:ied].set (val)

        # --- Nuclear Gradient ---
        grd_nn = jnp.array(f['grad_eloc_nn_nuc'][:])
        # --- Electron Gradient ---
        grad_eloc_ee_elc = jnp.array(f['grad_eloc_ee_elc'][:])
        grd_ee = jnp.einsum('seK,sen->snK', 
                            grad_eloc_ee_elc, 
                            rescale)
        # --- Electron-Nuclear Gradient ---
        grad_eloc_en_elc = jnp.array(f['grad_eloc_en_elc'][:])
        grad_eloc_en_nuc_base = jnp.array(f['grad_eloc_en_nuc_base'][:])
        grd_en = (grad_eloc_en_nuc_base+
                  jnp.einsum('seK,sen->snK',
                             grad_eloc_en_elc, 
                             rescale))
        #    --- Electron-Kinetic Gradient ---
        grad_eloc_ke_elc = jnp.array(f['grad_eloc_ke_elc'][:])
        grad_eloc_ke_nuc_base = jnp.array(f['grad_eloc_ke_nuc_base'][:])
        grd_ke = (grad_eloc_ke_nuc_base+
                  jnp.einsum('seK,sen->snK',
                             grad_eloc_ke_elc, 
                             rescale))
        # --- Electron-LogPsi Gradient ---
        grad_logpsi_elc = jnp.array(f['grad_logpsi_elc'][:])
        grad_logpsi_nuc_base = jnp.array(f['grad_logpsi_nuc_base'][:])

        enr_ke_samples = jnp.array(f['ener_ke_samples'][:])
        enr_ee_samples = jnp.array(f['ener_ee_samples'][:])
        enr_en_samples = jnp.array(f['ener_en_samples'][:])
        
        enr_samples = enr_ke_samples+enr_ee_samples+enr_en_samples
        d_enr = enr_samples - enr_samples.mean()
        novel_correction = jnp.zeros ( (num_samples, num_nuc, 3))
        for ist in range (0, num_samples, num_batches):
            ied = min (ist+num_batches, num_samples)
            jac_rescale_elec = jax.vmap (jac_rescale_fn, in_axes=(0,None)) (
                    stacked_samples[ist:ied], nuc_crds
            )
            val = 0.5*jnp.einsum('seneK->snK', jac_rescale_elec)
            novel_correction = novel_correction.at[ist:ied].set (val)
        
        grad_logpsi_nuc = grad_logpsi_nuc_base + \
            jnp.einsum('seK,sen->snK', 
                       grad_logpsi_elc, 
                       rescale) + \
            novel_correction
        pulay_terms = 2.0 * jnp.einsum('s,snK->snK', 
                                       d_enr, 
                                       grad_logpsi_nuc)
        
        print('grd_nn\n', grd_nn)
        print('grd_ee\n', grd_ee.mean(axis=0))
        print('grd_en\n', grd_en.mean(axis=0))
        print('grd_ke\n', grd_ke.mean(axis=0))
        print('grd_pulay\n', pulay_terms.mean(axis=0))
        
        total_grad = grd_nn[None,...] + \
            grd_ee + grd_en + grd_ke + pulay_terms
        
        loss_variance = jnp.mean(jnp.var (total_grad, axis=0))
        torques_per_nucleus = jnp.cross (relative_nuc_pos,
                                        total_grad)
        total_torque_per_sample = jnp.sum (torques_per_nucleus, axis=1)
        loss_torque = jnp.mean (jnp.sum(total_torque_per_sample**2, axis=-1))
        print ('loss_variance', loss_variance)
        print ('loss_torque', loss_torque)
        print ('loss', loss_variance+loss_torque)

        return total_grad.mean(axis=0)
    
    return None

