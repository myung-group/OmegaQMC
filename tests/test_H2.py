import jax
import jax.numpy as jnp
from pyscf import gto, scf
from vmc_mlsw.vmc_gto import get_vmc_func
# from vmc_mlsw.vmc_gto_symm import process_symmetric_diatomic_molecule
from pytest import approx

jax.config.update("jax_enable_x64", True)
rng_key = jax.random.key(888)

L = 1.4010
bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_params": jnp.array([4.2, 4.2]),  # H_1, H_2
    "J2_params": jnp.array([0.6046799, 0.6046799])
}

# H2 molecule
mol = gto.M(atom='''
H       0.0000    0.0000    {:.4f}
H       0.0000    0.0000    {:.4f}
'''.format(-L/2, L/2),
            basis=bset_name,
            unit='Bohr')

mol.build()
mf = scf.RHF(mol)
mf.kernel()
# mf_grad = mf.nuc_grad_method()
# grad = mf_grad.kernel()

# nuc_crds = jnp.array(mol.atom_coords(unit='Bohr'))
# print('nuc_crds(Bohr)\n', nuc_crds)


chkfile_prefix = 'H2_vmc_{}'.format(bset_name)

reflection_op_list = ['I', 'x', 'y', 'xy']

vmc_run, vmc_grad \
    = get_vmc_func(mf, params_jastrow,
                   cusp_scheme='Quady2025',
                   gr_scheme='scheme1',
                   chkfile_prefix=chkfile_prefix,
                   symmop_list=reflection_op_list)

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


def test_force():
    assert forces[0, 2] == approx(-0.000819990074, abs=1e-6)
