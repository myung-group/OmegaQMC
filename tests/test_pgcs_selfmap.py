"""Nuclear self-map diagnostic of the PGCS fragment frames.

:func:`OmegaQMC.symm.fragments.build_frag_transform_data` prints, for
every requested non-identity operation, how far it moves the nuclei of
its fragment off themselves, and warns (never raises) when the
operation is only an approximate symmetry.
"""
import warnings

import jax.numpy as jnp
import pytest

from OmegaQMC.symm.fragments import (
    SELFMAP_WARN_BOHR,
    build_frag_symmops,
    build_frag_transform_data,
)
from OmegaQMC.symm.operations import populate_fragment_symmops
from OmegaQMC.utils import Mole_custom, parse_molecular_inspheres

WATER = [('O', 0.0, 0.0, 0.1173),
         ('H', 0.0, 0.7572, -0.4692),
         ('H', 0.0, -0.7572, -0.4692)]


def _frames(tmp_path, atoms, ops):
    xyz = tmp_path / 'water.xyz'
    xyz.write_text(
        f"{len(atoms)}\nwater\n" + "".join(
            f"{s} {x:.6f} {y:.6f} {z:.6f} 0\n" for s, x, y, z in atoms
        )
    )
    mol = Mole_custom()
    mol.build(atom=str(xyz), unit='Angstrom', basis='sto-3g',
              spin=0, charge=0, verbose=0)
    mol.ignore_hydrogen_mass = False
    parse_molecular_inspheres(mol)
    populate_fragment_symmops(mol)
    fids = sorted(mol.map_frag_ctr.keys())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fs = build_frag_symmops(mol, ops, fids)
        build_frag_transform_data(
            mol, jnp.asarray(mol.coords), frag_symmops=fs,
        )
    return [
        str(w.message) for w in caught
        if 'symmetry' in str(w.message) and 'bohr' in str(w.message)
    ]


@pytest.mark.parametrize('ops', [['E', 'Rz180'], ['E', 'sx', 'sy']])
def test_symmetric_water_no_warning(tmp_path, capsys, ops):
    assert _frames(tmp_path, WATER, ops) == []
    out = capsys.readouterr().out
    assert out.count('maps the nuclei onto themselves') == len(ops) - 1


def test_distorted_water_warns_without_raising(tmp_path, capsys):
    atoms = list(WATER)
    atoms[1] = ('H', 0.0, 0.7572 + 0.05, -0.4692)   # stretch one O-H
    msgs = _frames(tmp_path, atoms, ['E', 'Rz180'])
    assert len(msgs) == 1 and 'approximate symmetry' in msgs[0]
    out = capsys.readouterr().out
    err = float(out.split('maps the nuclei onto themselves to ')[1]
                .split()[0])
    assert SELFMAP_WARN_BOHR < err < 0.5
