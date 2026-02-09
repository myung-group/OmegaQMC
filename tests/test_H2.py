import jax
import jax.numpy as jnp
from vmc_mlsw import generate_molecular_orbitals, get_vmc_func
from vmc_mlsw.utils import vmc_forces_with_space_warping as vmc_forces
from vmc_mlsw.utils import format_basis_name
# from vmc_mlsw.vmc_gto_symm import process_symmetric_diatomic_molecule
from pytest import approx

rng_key = jax.random.key(888)

L = 1.4010
bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_params": jnp.array([4.2, 4.2]),  # H_1, H_2
    "J2_params": jnp.array([0.6046799, 0.6046799])
}

atoms_string = '''
H       0.0000    0.0000    {:.4f}
H       0.0000    0.0000    {:.4f}
'''.format(-L/2, L/2)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

chkfile_prefix = 'H2_vmc_{}'.format(format_basis_name(bset_name))

# symmetry_ops = ['E', 'x', 'y', 'Rz180']
symmetry_ops = ['E', 'Rz180']
# symmetry_ops = ['E', 'i']
# symmetry_ops = ['E', 'Rz180', 'Rz90', 'Rz270']
# symmetry_ops = ['Rz90', 'Rz270']
# symmetry_ops = ['E']

vmc_run = get_vmc_func(modrv, params_jastrow,
                       cusp_scheme='Quady2025',
                       gr_scheme='scheme1',
                       prefix=chkfile_prefix)
# symmop_list=symmetry_ops

l_grad = True
vmc_run(rng_key,
        num_walkers=100,
        num_steps_per_block=100,   # MC steps per block (per walker)
        num_blocks=10,            # MC blocks
        num_blocks_equil=5,       # MC blocks for equilibration
        mc_timestep=0.1,    # Brownian time; will be auto-adjusted
        compute_gradients=l_grad)

if l_grad:
    forces, std_forces = vmc_forces(prefix=chkfile_prefix)


def test_force():
    assert forces[0, 2] == approx(-0.000344851447, abs=1e-6)
