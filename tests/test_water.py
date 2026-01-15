import jax.numpy as jnp
import jax
# from functools import partial

from pyscf import gto, scf
from vmc_mlsw import get_vmc_func
# from vmc_mlsw.vmc_gto_symm import process_symmetric_water_molecule

rng_key = jax.random.key(777)
bset_name = "cc-pVDZ"
# other choices: 6-31G, cc-pVDZ

# No optimizable Jastrow parameters:
params_vmc_no_jastrow = {
    "J1_params": jnp.array([]),
    "J2_params": jnp.array([])
}

mol = gto.M(
              atom='''
O                0.   0.   0.
H                0.   1.52610182  1.12172672
H                0.  -1.51745721  1.11537270
''',
              basis=bset_name,
              unit='Ang'
          )
# O        0.000000  0.000000  0.000000
# H        0.000000  0.000000  0.957800
# H        0.927385  0.000000 -0.239451
mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
grad = mf_grad.kernel()

# nuc_crds = mol.atom_coords(unit='Bohr')
# print('nuc_crds(Bohr)\n', nuc_crds)

chkfile_prefix = 'H2O_vmc_{}'.format(bset_name)

vmc_run, vmc_grad = get_vmc_func(mf, params_vmc_no_jastrow,
                                 cusp_scheme='Quady2025',
                                 gr_scheme='scheme1',
                                 chkfile_prefix=chkfile_prefix,
                                 symmop_list=['I', 'x', 'y', 'C2'])

l_grad = True
vmc_run(rng_key,
        num_walkers=100,
        num_steps_per_block=100,  # MC steps per each walker
        num_blocks=10,
        num_blocks_equil=5,       # MC blocks for equilibration
        mc_timestep=0.1,    # Brownian time; will be auto-adjusted
        l_grad=l_grad)

if l_grad:
    grd = vmc_grad()
