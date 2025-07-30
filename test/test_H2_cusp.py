
import jax 
import jax.numpy as jnp 
from pyscf import gto, scf 
from vmc_mlsw.vmc_gto import get_vmc_func 

rng_key = jax.random.key(888)

# No optimizable Jastrow parameters:
params_vmc_no_jastrow = jnp.array([])

# H2 molecule
mol = gto.M(
              atom='''
H       0.000000    0.00    0.25
H       0.000000    0.00   -0.25
''',
              basis='6-31g*',
              #basis='cc-pvdz',
              #basis="ccECP_cc-pVDZ", ecp="ccecp",
              unit='Ang'
          )
"""
# H2O molecule
mol = gto.M(
              atom='''
O        0.000000    0.000000    0.117307
H       -0.000000    0.757216   -0.469229
H       -0.000000   -0.757216   -0.469229
''',
              basis='6-31g*',
              #basis='cc-pvdz',
              #basis="ccECP_cc-pVDZ", ecp="ccecp",
              unit='Ang'
          )
"""

mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
grad = mf_grad.kernel()

nuc_crds = jnp.array(mol.atom_coords(unit='Bohr'))
print('nuc_crds(Bohr)\n', nuc_crds)

chkfile_mc =  'H2_vmc_cusp_631gd_mc.hdf5'
chkfile_enr = 'H2_vmc_cusp_631gd_enr.hdf5'
chkfile_grd = 'H2_vmc_cusp_631gd_grd.hdf5'

cgto_coeff = {
    1: jnp.array ([1, 1.0431879, -0.02914878, 0.78355617,
    -2.95081286, 5.43507108, -5.08491324, 1.94265234
    ]),
    8: jnp.array ([1, 12.45593615, -2.38348643, 30.46159315,
    -125.8242091, 252.61904634, -239.5024989, 86.70950789])
}

vmc_run, vmc_energy, vmc_gradient_prep, vmc_grad =\
        get_vmc_func (mf,
                      params_vmc_no_jastrow,
                      chkfile_mc=chkfile_mc,
                      chkfile_enr=chkfile_enr,
                      chkfile_grd=chkfile_grd,
                      cgto_coeff=cgto_coeff)

# (1) Sample electrons

vmc_run(rng_key,
        num_steps=500000,
        num_equilibration=50000,
        step_size=0.10
        )

# (2) Estimate VMC energy
vmc_energy()

sym_op_list = ['E', 'Sx', 'Sy', 'Sxy']
# (3) Calculate the gradients acting on electrons and nuclei 
# based on sampled electrons

vmc_gradient_prep(sym_op_list)


# (4) Calculate the total VMC gradients
print ("\n *** Scheme 1 ***\n")
grd = vmc_grad (scheme='scheme1', sym_op_list=['E'])
with jnp.printoptions (precision=5, suppress=True):
    print ('\nScheme1(nomark):grd\n', grd, '\n')


# 
print ("\n *** Scheme 1 (with clipping) ***\n")
grd = vmc_grad (scheme='scheme1',
                mark_std=3.0,
                sym_op_list=['E']) # 

with jnp.printoptions (precision=5, suppress=True):
    print ('\nScheme1(mark with std_3):grd\n', grd, '\n')

print ("\n *** Scheme 1 (with symmetry) ***\n")
grd = vmc_grad (scheme='scheme1',
                sym_op_list=sym_op_list)

with jnp.printoptions (precision=5, suppress=True):
    print ('\nwith_symmetry:Scheme1(nomark):grd\n', grd, '\n')


# 
print ("\n *** Scheme 1 (with symmetry and clipping) ***\n")
grd = vmc_grad (scheme='scheme1',
                mark_std=3.0,
                sym_op_list=sym_op_list) # 

with jnp.printoptions (precision=5, suppress=True):
    print ('\nwith_symmetry:Scheme1(mark with std_3):grd\n', grd, '\n')

