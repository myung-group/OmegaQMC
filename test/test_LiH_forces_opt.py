import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
from pyscf import gto, scf
from vmc_mlsw.vmcopt_gto_forces import get_vmc_func
#from vmc_mlsw.vmc_gto_symm import process_symmetric_diatomic_molecule
import sys
import re
import json
from importlib import resources
from vmc_mlsw.basis import extract_basis_block


def extract_numbers(s):
    match = re.search(r"params:\s*\[([^\]]+)\]", s)
    if match:
        values_str = match.group(1).strip() 
        values = values_str.split()         
        values = [float(v) for v in values]  
    return values


#print("distance:", sys.argv[1])
#print("jastrow file:", sys.argv[2])
#d = float(sys.argv[1])
d = 3.015

init_jastrow_file = open("init_jastrow_LiH.log")
init_jastrows_raw = init_jastrow_file.readlines()
for l in range(int(len(init_jastrows_raw)/3)):
    line = init_jastrows_raw[3*l]
    if float(line.strip()) == d:
        line1 = init_jastrows_raw[3*l+1].split()
        line2 = init_jastrows_raw[3*l+2].split()
        jastrow_1 = float(line1[1])
        jastrow_2 = float(line2[1])
        break


rng_key = jax.random.key(888)

# No optimizable Jastrow parameters:
#params_vmc_no_jastrow = jnp.array([6.503499e-01])
#params_vmc_no_jastrow = jnp.array(jastrow)
params_vmc_no_jastrow= {
    "J1_params": jnp.array([23., 3.3]) , 
    "J2_params":  jnp.array([jastrow_1, jastrow_2]) 
}
print("jastrow:", params_vmc_no_jastrow)
# H2 molecule
mol = gto.M(
            atom=f'''
    Li          0.000000    0.000000    0.000000
    H           0.000000    0.000000    {d}
    ''',
            basis={
            'Li':  '6-31g',
            'H' :  "6-31g"
            },
            unit='Bohr'
)

mol.build()
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
grad = mf_grad.kernel()

#nuc_crds = jnp.array(mol.atom_coords(unit='Bohr'))
#print('nuc_crds(Bohr)\n', nuc_crds)


chkfile_grd = 'LiH_vmc_grd.hdf5'


with resources.open_text('vmc_mlsw.basis', 'cusp_coeff_631g.json') as f:
    coeff_data = json.load(f)
cgto_coeff = {
    int(Z): {
        'q0': v['q0'],
        'coeff': jnp.array(v['coeff'])
    } for Z, v in coeff_data.items()
}
import pprint
pprint.pprint(cgto_coeff)

#reflection_op_list = ["I", "x", "y", "xy"]
reflection_op_list = ["I"]

l_cusp = True
if l_cusp:
    vmc_run, vmc_gradient_calc, vmc_force_train =\
                get_vmc_func(mf,
                     params_vmc_no_jastrow,
                     cgto_coeff=cgto_coeff)
else:
    vmc_run, vmc_gradient_calc, vmc_force_train =\
                get_vmc_func(mf,
                     params_vmc_no_jastrow,
                     cgto_coeff=None)
"""
l_grad = False
vmc_run(rng_key=rng_key,
        nwalkers=1000,
        num_steps=1000, # MC steps per each walker
        num_epochs=10000,
        num_equilibration=1000,
        step_size=0.10, # electrons movement distance
        optimizer='sgd',
        batch_size=100)

if l_grad:
    grd = vmc_grad ()
"""

trained_params = vmc_force_train(
    rng_key=rng_key,
    nwalkers=100,
    num_mc_steps=1000,
    step_size=0.25,
    sample_every=10,
    num_epochs=1000,
    batch_size_force=50,
    lr=0.02,
    optimizer="adam",
    grad_clip=None,   # 0.1
    fname_log="vmc_force_train.log",
    resample_every=10,
    verbose=True,
)

print("trained J1 norm:", jnp.linalg.norm(trained_params["J1_params"]))
print("J2 changed?", jnp.linalg.norm(trained_params["J2_params"] - J2_init))