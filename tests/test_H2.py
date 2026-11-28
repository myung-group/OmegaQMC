import jax
import jax.numpy as jnp
import numpy as np
from OmegaQMC import generate_molecular_orbitals, get_vmc_gto_func
from OmegaQMC.observables.force import postproc_h5_pgcs as vmc_forces
from OmegaQMC.utils import format_basis_name
# from OmegaQMC.vmc_gto_symm import process_symmetric_diatomic_molecule
from pytest import approx

rng_key = jax.random.key(888)

L = 1.4010
bset_name = "6-31G"

# optimizable Jastrow parameters:
params_jastrow = {
    "J1_pade": {"H": jnp.array([-0.05574627,  0.08272289])},
    "J2_pade": {"like": jnp.array([0.25, 0.6046799]),
                "unlike": jnp.array([0.5, 0.38077791])}
}

atoms_string = '''
H       0.0000    0.0000    {:.6f}      1
H       0.0000    0.0000    {:.6f}      1
'''.format(-L/2, L/2)

modrv = generate_molecular_orbitals(atoms_string, units="Bohr",
                                    basis=bset_name)

chkfile_prefix = 'H2_vmc_{}'.format(format_basis_name(bset_name))

# symmetry_ops = ['E', 'x', 'y', 'Rz180']
symmetry_ops = ['E', 'C2z']
# symmetry_ops = ['E', 'i']
# symmetry_ops = ['E', 'Rz180', 'Rz90', 'Rz270']
# symmetry_ops = ['Rz90', 'Rz270']
# symmetry_ops = ['E']

vmc_run = get_vmc_gto_func(modrv, params_jastrow,
                           cusp_scheme='Quady2025',
                           gr_scheme='scheme1',
                           force_estimator="simple",
                           prefix=chkfile_prefix,
                           symmop_list=symmetry_ops)

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
    # H2 is placed at its equilibrium bond length (1.401 bohr),
    # so the z-force on each atom is physically ~0.  This short
    # 100-walker / 5-block smoke test recovers it only to within
    # its (large) statistical error (~6e-3 Ha/bohr), so assert
    # consistency with zero at a generous multiple of that rather
    # than pinning an exact Monte Carlo trajectory.
    assert forces[0, 2] == approx(0.0, abs=3.0e-2)


def _selection(text):
    return [ln for ln in text.splitlines()
            if ln.startswith(('PGCS state selection', 'BLUE:'))
            or 'ESS/N = ' in ln and ln.startswith('  ')]


def test_pgcs_logs_gto(tmp_path):
    # GTO (Slater-Jastrow) PGCS post-processing end to end: C2z is an
    # exact symmetry of the RHF trial state, so the secondary state is
    # kept by both energy and force paths, the averaged forces are
    # projected, and the energy log ends with the reported energy.
    from OmegaQMC.observables.energy import postproc_h5_pgcs as vmc_enr
    flog, elog = str(tmp_path / 'f.log'), str(tmp_path / 'e.log')
    vmc_forces(prefix=chkfile_prefix, logfile=flog)
    enr, err = vmc_enr(prefix=chkfile_prefix, logfile=elog)
    with open(flog) as fh:
        ftext = fh.read()
    with open(elog) as fh:
        etext = fh.read()
    assert _selection(ftext) == _selection(etext)
    assert 'BLUE: no states dropped (ESS/N >= 0.1)' in etext
    assert ('Sum-rule projection (translational only): weighting = '
            in ftext)
    last = etext.rstrip().splitlines()[-1]
    assert last == (f"Reported E = {enr:.8f} ± {err:.8f}"
                    "  (BLUE over 2 states)")

    # force_estimator 'simple' (SWCT) makes sum_I F_I = 0 sample by
    # sample, so each state's net force has a zero error once it is
    # taken from its own series; the projected average is 0 +- 0.
    # The torque is not constrained (frozen MO coefficients): each
    # state's has a finite error about x and y, while the C2z average
    # cancels it sample by sample (Rz180 flips tau_x and tau_y).
    def net_lines(header):
        out = []
        for chunk in ftext.split(header + '\n')[1:]:
            nums = chunk.splitlines()[0].replace('±', ' ').split()
            out.append([float(x) for x in nums])
        return np.array(out)

    per_state = net_lines('Total force on system')
    assert per_state.shape == (2, 6)
    assert np.all(np.abs(per_state) <= 1e-8)
    assert np.all(
        np.abs(net_lines('Total force on system (averaged)')) <= 1e-8
    )
    tau = net_lines('Total torque (about center of mass)')
    assert tau.shape == (2, 6) and np.all(tau[:, 1:4:2] > 0)
    np.testing.assert_allclose(tau[0, ::2], -tau[1, ::2], atol=1e-12)
    assert np.all(np.abs(net_lines(
        'Total torque (about center of mass, averaged)')) <= 1e-8)
