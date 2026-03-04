"""Test script to analyze walker initialization and fragment membership.

For each randomly initialized walker, this script prints which molecular
fragment's centroid is closest to each electron, and whether the electron
is within the fragment's inradius.
"""
import jax
import jax.numpy as jnp
from vmc_pgcs import generate_molecular_orbitals

# Import _initialize_walkers from vmc_gto (it's a module-level function)
from vmc_pgcs.vmc_gto import _initialize_walkers

rng_key = jax.random.key(42)
bset_name = '6-31G'

# Water dimer geometry from test_water2.py
myUnits = "ang"
atoms_string = '''
O       8.707172158e-01  -1.720661566e+00  -4.814067179e-01     1
H       1.689789528e+00  -1.221338203e+00  -3.052747557e-01     1
H       1.142438797e+00  -2.587969761e+00  -7.799456175e-01     1
O      -7.398283056e-01   4.040418183e-01  -1.654300203e+00     2
H      -2.723133426e-01  -4.319081553e-01  -1.528862134e+00     2
H      -1.614078540e+00   2.476812916e-01  -1.263515900e+00     2
'''

# Generate molecular orbitals (runs HF calculation)
print("Setting up water dimer molecule...")
mf = generate_molecular_orbitals(atoms_string, units=myUnits, basis=bset_name,
                                 ignore_hydrogen_mass=True)
mol = mf.mol

# Extract molecular properties
nuc_crds = jnp.array(mol.atom_coords(unit='Bohr'))
nelec = mol.tot_electrons()
Z_charges = mol.atom_charges()
mol_charge = mol.charge
num_walkers = 5

print("\nMolecular properties:")
print(f"  Number of electrons: {nelec}")
print(f"  Number of atoms: {mol.natm}")
print(f"  Nuclear charges: {Z_charges}")
print(f"  Molecular charge: {mol_charge}")

# Print fragment information
print("\nFragment information:")
print(f"  map_nuc_frag (atom -> fragment): {mol.map_nuc_frag}")
for frag_id, centroid in mol.map_frag_ctr.items():
    inradius = mol.inradii[frag_id]
    print(f"  Fragment {frag_id}: centroid = {centroid} Bohr, "
          f"inradius = {inradius:.4f} Bohr")
    # print(f"  Fragment {frag_id}: centroid = {centroid * 0.529} Angstroms, "
    #       f"inradius = {inradius * 0.529:.4f} Angstroms")

# Initialize walkers
print(f"\nInitializing {num_walkers} walkers...")
walkers = _initialize_walkers(rng_key, num_walkers, nelec,
                              Z_charges, nuc_crds, mol_charge)
print(f"  Walker array shape: {walkers.shape}")

# Convert fragment centroids to JAX array for distance calculations
frag_ids = list(mol.map_frag_ctr.keys())
frag_centroids = jnp.array([mol.map_frag_ctr[fid] for fid in frag_ids])
frag_inradii = jnp.array([mol.inradii[fid] for fid in frag_ids])

print("\nAnalyzing electron-fragment assignments for each walker:\n")
print("=" * 80)

for w_idx in range(num_walkers):
    print(f"\nWalker {w_idx + 1}:")
    print("-" * 40)

    walker_coords = walkers[w_idx]  # shape: (nelec, 3)

    # Count electrons in each fragment
    frag_counts = {fid: {'within': 0, 'outside': 0} for fid in frag_ids}

    for e_idx in range(nelec):
        elec_pos = walker_coords[e_idx]

        # Calculate distance to each fragment centroid
        distances = jnp.linalg.norm(frag_centroids - elec_pos, axis=1)

        # Find closest fragment
        closest_idx = jnp.argmin(distances)
        closest_frag_id = frag_ids[closest_idx]
        closest_dist = distances[closest_idx]
        closest_inradius = frag_inradii[closest_idx]

        # Check if within inradius
        within_inradius = closest_dist <= closest_inradius
        status = "WITHIN" if within_inradius else "OUTSIDE"

        if within_inradius:
            frag_counts[closest_frag_id]['within'] += 1
        else:
            frag_counts[closest_frag_id]['outside'] += 1

        # Print details for first few electrons of each walker
        if e_idx < 5 or e_idx >= nelec - 2:
            print(f"  Electron {e_idx:2d}: closest to frag {closest_frag_id}, "
                  f"dist = {closest_dist:.4f} Bohr, "
                  f"inradius = {closest_inradius:.4f} Bohr "
                  f"-> {status}")
        elif e_idx == 5:
            print(f"  ... (electrons 5-{nelec-3} omitted for brevity) ...")

    # Summary for this walker
    print(f"\n  Summary for Walker {w_idx + 1}:")
    for fid in frag_ids:
        total = frag_counts[fid]['within'] + frag_counts[fid]['outside']
        within = frag_counts[fid]['within']
        outside = frag_counts[fid]['outside']
        print(f"    Fragment {fid}: {within} within inradius, "
              f"{outside} outside ({total} total closest)")

print("\n" + "=" * 80)
print("\nDone!")
