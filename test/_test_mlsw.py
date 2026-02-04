#import jax
#import jax.numpy as jnp
#from pyscf import gto, scf
from mlsw import mlsw_trainer

rng_seed = 7
fname_log = 'mlsw_simple.log'
learning_rate = 0.01
num_epoch = 10
chkfile = 'LiH_vmc.hdf5'
fname_pkl = 'LiH_model_simple.pkl'
l_restart = False
num_batches = 5000
trainer = mlsw_trainer (chkfile, num_batches=num_batches)
trainer (rng_seed, fname_log, fname_pkl, l_restart,
         learning_rate, num_epoch,
         l_NN_simple=True)

