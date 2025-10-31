import jax
import jax.numpy as jnp
from pyscf import gto, scf
# , cc
from vmc_mlsw import get_vmc_func

rng_key = jax.random.key(888)

L = 3.015
bset_name = "aug-cc-pVTZ"

# No optimizable Jastrow parameters:
params_jastrow = {
    "J1_params": jnp.array([48.89, 8.93]),
    "J2_params": jnp.array([1.78,  0.917])
}

# LiH molecule
mol = gto.M(atom='''
Li       0.000000    0.00    0.00
H        0.000000    0.00    {:.6f}
'''.format(L),
            basis=bset_name,
            unit='Bohr')

mol.build()
mf = scf.RHF(mol)
mf.kernel()

# mf_grad = mf.nuc_grad_method()
# grad = mf_grad.kernel()

# postmf = cc.CCSD(mf).run()
# cc_grad = postmf.nuc_grad_method()
# cc_grad.kernel()

chkfile_prefix = 'LiH_vmc_aVTZ'

reflection_op_list = ['I', 'x', 'y', 'xy']
# 'xy' here means negate both x and y coordinates

vmc_run, vmc_grad \
    = get_vmc_func(mf, params_jastrow,
                   cusp_scheme='Quady2025',
                   gr_scheme='scheme1',
                   chkfile_prefix=chkfile_prefix,
                   reflection_op_list=reflection_op_list)

l_grad = True
vmc_run(rng_key,
        num_walkers=100,
        num_steps_per_block=100,   # MC steps per block (per walker)
        num_blocks=10,            # MC blocks
        num_blocks_equil=5,       # MC blocks for equilibration
        mc_timestep=0.1,    # Brownian time; will be auto-adjusted
        l_grad=l_grad)

if l_grad:
    forces = vmc_grad()
