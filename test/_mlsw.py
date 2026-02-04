import jax 
import jax.numpy as jnp
import flax.linen as nn
import optax 
from flax.training.train_state import TrainState
#from functools import partial 
import e3nn_jax as e3nn 
import h5py 
import pickle
import sys 

def checkpoint_save (fname, ckpt):
    with open (fname, 'wb') as fp:
        pickle.dump (ckpt, fp)

def checkpoint_load (fname):
    with open (fname, 'rb') as fp:
        return pickle.load (fp)
    
def default_radial_basis (r, n:int):
    """
    Radial Basis Function
    """
    return e3nn.bessel (r, n) * e3nn.poly_envelope (1,2) (r) [:, None]

# --- Neural Network Definition ---
class GNN (nn.Module):
    R_nuclei: jax.Array
    species : jax.Array # [z_A]
    senders : jax.Array
    receivers:jax.Array
    cutoff  : float 
    num_elc : int
    num_nuc : int
    num_species: int
    max_ell : int = 2
    num_basis_func: int = 8
    hidden_irreps : e3nn.Irreps = e3nn.Irreps ("16x0e + 8x1o + 1x2e")
    nlayers: int = 2
    features_dim: int = 32 
    
    @nn.compact 
    def __call__ (self, e_pos):
        # e_pos [num_elec, 3]
        
        feats = nn.Embed (num_embeddings=self.num_species,
                          features=self.features_dim) (self.species)
        
        edges_feats = feats[self.receivers]
        edges_feats = edges_feats.reshape (self.num_elc,
                                           self.num_nuc,
                                           -1)
        
        edges_feats = e3nn.IrrepsArray(f"{edges_feats.shape[-1]}x0e", edges_feats)
        
        # [n_elec*n_nuc, 3]
        Rij = e_pos[:, None, :] - self.R_nuclei[None, :, :]
        lengths = jnp.linalg.norm (Rij, axis=-1)/self.cutoff
        
        Rij = e3nn.IrrepsArray ("1o", Rij)
        radial_embedding = default_radial_basis (lengths.reshape(-1), 
                                                 self.num_basis_func)
        radial_embedding = radial_embedding.reshape(self.num_elc,
                                                    self.num_nuc,
                                                    -1)
        
        radial_embedding = e3nn.IrrepsArray (f"{self.num_basis_func}x0e",
                                             radial_embedding)
       
        interaction_irreps = e3nn.Irreps.spherical_harmonics (self.max_ell)

        edges_attr = e3nn.spherical_harmonics (interaction_irreps,
                                               Rij,
                                               normalize=True,
                                               normalization="component")
        
        for _ in range (self.nlayers):
            edges_feats = e3nn.flax.Linear (self.hidden_irreps) (edges_feats)
            hidden_irreps = e3nn.Irreps(self.hidden_irreps).regroup()
            edges_feats = e3nn.concatenate (
                [
                    edges_feats.filter (hidden_irreps+"0e"),
                    e3nn.tensor_product (
                        edges_feats,
                        edges_attr,
                        filter_ir_out=hidden_irreps+"0e",
                    ),
                ]
            ).regroup()

            # Radial Part
            wgt_irreps = e3nn.Irreps (f"{edges_feats.irreps.num_irreps}x0e")
            wgt = e3nn.flax.Linear (wgt_irreps,
                                        force_irreps_out=True) (
                                            radial_embedding
                                        )
            edges_feats = edges_feats*wgt.array 
            edges_feats = e3nn.flax.Linear (self.hidden_irreps) (edges_feats)
            
        # edges_feats [num_elc*num_nuc,1]
        edges_feats = e3nn.flax.Linear (e3nn.Irreps("1x0e")) (edges_feats)
        edges_feats = edges_feats.array.reshape (self.num_elc, self.num_nuc)
        edges_feats = nn.softmax (edges_feats,axis=-1)

        return edges_feats

# --- Neural Network Definition ---
class SimpleMLP_NN (nn.Module):
    """A simple MLP to learn the redistribution weights."""
    R_nuclei : jax.Array
    num_nuc : int

    @nn.compact
    def __call__(self, e_pos):
        num_elc = e_pos.shape[0]
        #num_nuc = self.R_nuclei[0]
        epsilon = 1e-8

        # --- 1. Electron-Nucleus Features (Generally Safe) ---
        en_vectors = e_pos[:, None, :] - self.R_nuclei[None, :, :]
        en_distances = jnp.linalg.norm(en_vectors, axis=-1)
    
        # Feature 1: Inverse e-n distances
        inv_en_distances = 1.0 / (en_distances + epsilon)
        # Feature 2: Normalized e-n direction vectors
        norm_en_vectors = en_vectors / (en_distances[..., None] + epsilon)

        features = jnp.concatenate(
            [inv_en_distances.reshape(num_elc,-1),
             norm_en_vectors.reshape(num_elc,-1)],
            axis=-1)

        x = nn.Dense(features=64)(features) # Increased capacity for rich features
        x = nn.silu(x)
        x = nn.Dense(features=128)(x)
        x = nn.silu(x)
        x = nn.Dense(features=64) (x)
        x = nn.silu(x)
        x = nn.Dense(features=32) (x)
        x = nn.silu(x)
        x = nn.Dense(features=16)(x)
        x = nn.silu(x)
        x = nn.Dense(features=self.num_nuc)(x)
        
        return nn.softmax(x, axis=-1)
    

def mlsw_trainer (chkfile, num_batches=1000):

    f = h5py.File (chkfile, 'r')
    
    stacked_samples = jnp.array(f['stacked_samples'][:])
    nuc_crds = jnp.array(f['nuc_crds'][:])

    nuc_species = list(f['species'][:])
    atomic_masses = jnp.array(f['atomic_masses'][:])
    mass_center = jnp.einsum('i,ij->j', atomic_masses, nuc_crds)/atomic_masses.sum()
    relative_nuc_pos = nuc_crds - mass_center 

    enr_ke_samples = jnp.array(f['ener_ke_samples'][:])
    enr_ee_samples = jnp.array(f['ener_ee_samples'][:])
    enr_en_samples = jnp.array(f['ener_en_samples'][:])

    enr_samples = enr_ke_samples + enr_ee_samples + enr_en_samples
    d_enr = enr_samples - enr_samples.mean()

    grd_nn = jnp.array(f['grad_eloc_nn_nuc'][:])
    grad_eloc_ee_elc = jnp.array(f['grad_eloc_ee_elc'][:])
    grad_eloc_en_elc = jnp.array(f['grad_eloc_en_elc'][:])
    grad_eloc_en_nuc_base = jnp.array(f['grad_eloc_en_nuc_base'][:])
    grad_eloc_ke_elc = jnp.array(f['grad_eloc_ke_elc'][:])
    grad_eloc_ke_nuc_base = jnp.array(f['grad_eloc_ke_nuc_base'][:])
    #grd_eloc_elc = grad_eloc_ee_elc + grad_eloc_en_elc + grad_eloc_ke_elc
    #grd_eloc_nuc = grad_eloc_en_nuc_base + grad_eloc_ke_nuc_base

    grd_logpsi_elc = jnp.array(f['grad_logpsi_elc'][:])
    grd_logpsi_nuc_base = jnp.array(f['grad_logpsi_nuc_base'][:])

    
    f.close()
    
    
    num_samples = stacked_samples.shape[0]
    
    def calculate_forces_samples (params, 
                                  model_apply_fn, 
                                  ist, ied):
        
        
        def rescale_fn (e_pos):
            return model_apply_fn ({'params': params}, e_pos)

        rescale = jax.vmap(rescale_fn) (stacked_samples[ist:ied])
        
        
        grd_HF = grd_nn[None,...] + \
                    jnp.einsum('seK,sen->snK', 
                               grad_eloc_ee_elc[ist:ied], 
                               rescale) + \
                    (grad_eloc_en_nuc_base[ist:ied] + 
                     jnp.einsum('seK,sen->snK', 
                                grad_eloc_en_elc[ist:ied], 
                                rescale)) + \
                    (grad_eloc_ke_nuc_base[ist:ied] + 
                     jnp.einsum('seK,sen->snK',
                                 grad_eloc_ke_elc[ist:ied], 
                                 rescale))
        '''
        grd_ee = jnp.einsum('seK,sen->snK', 
                               grad_eloc_ee_elc[ist:ied], 
                               rescale)
        grd_en = (grad_eloc_en_nuc_base[ist:ied] + 
                     jnp.einsum('seK,sen->snK', 
                                grad_eloc_en_elc[ist:ied], 
                                rescale))
        grd_ke = (grad_eloc_ke_nuc_base[ist:ied] + 
                     jnp.einsum('seK,sen->snK',
                                 grad_eloc_ke_elc[ist:ied], 
                                 rescale))
        '''
        jac_rescale_fn = jax.jacobian (rescale_fn, 
                                       argnums=0)
        jac_rescale_elc = jax.vmap(jac_rescale_fn) (
                            stacked_samples[ist:ied])
        novel_correction = 0.5*jnp.einsum('seneK->snK', 
                                          jac_rescale_elc)
        grd_logpsi_nuc = grd_logpsi_nuc_base[ist:ied] + \
                              jnp.einsum('seK,sen->snK', 
                                         grd_logpsi_elc[ist:ied], 
                                         rescale) + \
                              novel_correction
        
        pulay_term = 2.0*jnp.einsum('s,snK->snK', 
                                    d_enr[ist:ied], 
                                    grd_logpsi_nuc)
        
        return -(grd_HF + pulay_term)
        #return grd_ee, grd_en, grd_ke, pulay_term


    def loss_fn (params, model_apply_fn, ist, ied):
        # (n_sample, n_nuc, 3)
        forces_samples = calculate_forces_samples (params, 
                                                   model_apply_fn,
                                                   ist, ied)

        
        #print ('forces_samples', forces_samples.shape)
        # 1. Variance loss 
        loss_variance = jnp.mean(jnp.var (forces_samples, axis=0))
        #loss_ee = jnp.mean(jnp.var (grd_ee, axis=0))
        #loss_en = jnp.mean(jnp.var (grd_en, axis=0))
        #loss_ke = jnp.mean(jnp.var (grd_ke, axis=0))
        #loss_pulay = jnp.mean(jnp.var (pulay_term, axis=0))
        # 2. Total torque is constrained to zero
        # (num_nuc, 3) x (num_sample, num_nuc, 3)
        torques_per_nucleus = jnp.cross (relative_nuc_pos,
                                        forces_samples)
        #print ('torques_per_nucleus', torques_per_nucleus.shape)
        total_torque_per_sample = jnp.sum (torques_per_nucleus, axis=1) # sum over nuclei
        #print ('total_torque_per_sample', total_torque_per_sample.shape)

        loss_torque = jnp.mean (jnp.sum(total_torque_per_sample**2, axis=-1))
        #print ('loss', loss_variance, loss_torque)
        return (loss_variance+loss_torque)
    

    #@jax.jit
    def train_step (state, ist, ied):

        loss, grads = jax.value_and_grad (loss_fn) (state.params, 
                                                    state.apply_fn,
                                                    ist, ied)
        state = state.apply_gradients (grads=grads)
        
        return state, loss 


    def _rescale_fn (e_pos, state):
        return state.apply_fn ({'params':state.params}, e_pos)
    
    jac_rescale_fn = jax.jacobian (_rescale_fn, argnums=0)

    def trainer (rng_seed, fname_log, fname_pkl, l_restart,
                 nn_learning_rate, num_epoch,
                 l_NN_simple=True):
        rng = jax.random.key(rng_seed)
        fout = open (fname_log, 'w', 1)
        
        num_nuc = nuc_crds.shape[0]
        num_elc = stacked_samples.shape[1]
        
        
        # iatoms = elec
        if l_NN_simple:
            nn_model = SimpleMLP_NN (R_nuclei=nuc_crds,
                                     num_nuc=num_nuc)
        else:
            cutoff = 4.0
            species = jnp.array (nuc_species)
            # iatoms = electrons
            senders = jnp.array ([[i]*num_nuc for i in range(num_elc)], dtype=jnp.int32).reshape(-1)
            # jatoms = nuclei
            receivers = jnp.array ([j for j in range(num_nuc)]*num_elc, dtype=jnp.int32).reshape(-1)
            #print ('senders', senders)
            #print ('receivers', receivers)
            num_species = int(species.max()) + 1
        
            nn_model = GNN (R_nuclei=nuc_crds,
                        species=species,
                        cutoff=cutoff,
                        num_elc=num_elc,
                        num_nuc=num_nuc,
                        num_species=num_species,
                        senders=senders,
                        receivers=receivers
                        )
        
        rng, key = jax.random.split (rng)
        params = nn_model.init (key, stacked_samples[0])['params']
    
        opt_method = optax.chain (
            optax.clip_by_global_norm (0.5),
            optax.clip (0.5),
            optax.adam (learning_rate = nn_learning_rate)
        )
        nn_state = TrainState.create (
            apply_fn = nn_model.apply,
            params = params,
            tx = opt_method
        )

        if l_restart:
            ckpt = checkpoint_load (fname_pkl)
            opt_state = ckpt['opt_state']
            #opt_state = optax.tree_utils.tree_set (opt_state, learning_rate=nn_learning_rate)
            nn_state = nn_state.replace (params=ckpt['params'],
                                         opt_state=opt_state)
            
        for epoch in range (num_epoch):
            
            loss_list = []
            for ist in range (0, num_samples, num_batches):
                ied = min (ist+num_batches, num_samples)
                
                nn_state, loss_val = train_step (nn_state, ist, ied)
                loss_list.append (loss_val)
                print ('ist, loss', ist, loss_val, file=fout)
            
            
            if (epoch+1)%1 == 0 or (epoch==0):
                loss_val = jnp.array(loss_list).mean()
                print(f"  NN Opt Step {epoch+1:03d}, Force Variance Loss: {loss_val:.6f}", 
                      file=fout)
        
                ckpt = {
                    'params' : nn_state.params,
                    'opt_state' : nn_state.opt_state
                }
                checkpoint_save (fname_pkl, ckpt)

            novel_correction = jnp.zeros ((num_samples, num_nuc, 3))
            rescale = jnp.zeros ((num_samples, num_elc, num_nuc))
            for ist in range (0, num_samples, num_batches):
                ied = min (ist+num_batches, num_samples)

                _rescale = jax.vmap (_rescale_fn, 
                                    in_axes=(0,None)) (
                                        stacked_samples[ist:ied],
                                        nn_state) 
                rescale = rescale.at[ist:ied].set (_rescale)

                jac_rescale_elc = jax.vmap(jac_rescale_fn, 
                                           in_axes=(0,None)) (
                                               stacked_samples[ist:ied],
                                                nn_state)
                n_corr = 0.5*jnp.einsum('seneK->snK', jac_rescale_elc)
                novel_correction = novel_correction.at[ist:ied].set (n_corr)

                                                               
            grd_ee = jnp.einsum('seK,sen->snK', grad_eloc_ee_elc, rescale).mean(axis=0)
            print ('grd_ee\n', 
                   grd_ee, file=fout)
            grd_en = (grad_eloc_en_nuc_base + 
                     jnp.einsum('seK,sen->snK', grad_eloc_en_elc, rescale)).mean(axis=0)
            print ('grd_en\n', 
                   grd_en, file=fout)
            grd_ke = (grad_eloc_ke_nuc_base + 
                     jnp.einsum('seK,sen->snK', grad_eloc_ke_elc, rescale)).mean(axis=0)
            print ('grd_ke\n', 
                   grd_ke,file=fout)

            
            grd_logpsi_nuc = grd_logpsi_nuc_base + \
                              jnp.einsum('seK,sen->snK', grd_logpsi_elc, rescale) + \
                              novel_correction
        
            pulay_term = 2.0*jnp.einsum('s,snK->snK', d_enr, grd_logpsi_nuc).mean(axis=0)
            print ('pulay_term\n', 
                   pulay_term, 
                   file=fout)
            print ('grd_tot\n', 
                   grd_nn+grd_ee+grd_en+grd_ke+pulay_term,
                   file=fout)
        
        #return nn_model
    
    return trainer