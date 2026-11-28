"""PGCS fragment frames: rotation axes and nuclear self-maps.

The fragment symmetry operations must map distorted molecules onto
themselves (otherwise the PGCS weights collapse, ESS/N ~ 0.005), and
the self-map diagnostic warns, without raising, when they do not.
"""
import tempfile
import numpy as np
import pytest
import pyscf.gto as _pgto
import warnings
import jax.numpy as jnp

from OmegaQMC.symm.fragments import (  # noqa: E402
    SELFMAP_WARN_BOHR,
    build_frag_symmops,
    build_frag_transform_data,
    _geometric_symmetry_frame,
)
from OmegaQMC.symm.operations import (  # noqa: E402
    populate_fragment_symmops,
    symmetry_operations_map,
    POINT_GROUP_OPS,
)
from OmegaQMC.utils import (  # noqa: E402
    Mole_custom,
    parse_molecular_inspheres,
)


# ---------------------------------------------------------------------------
# From test_pgcs_axis_frame.py
#
# Regression test for the PGCS fragment-rotation axis.
#
# A fragment ``Rz180`` (correlated-sampling C2) must rotate
# about the fragment's true C2 axis -- the H-O-H bisector for
# water -- so that it maps the fragment onto itself and leaves
# the importance weight ``|psi(C2.R)/psi(R)|^2`` near unity.
#
# A prior bug took the rotation frame from PySCF
# ``detect_symm``: the moment a water fell below the exact-C2v
# tolerance it was classified ``Cs`` and local-z switched
# discontinuously from the C2 axis to the *molecular-plane
# normal*.  ``Rz180`` then rotated about the wrong axis,
# displacing the affected electrons by ~1 A instead of
# self-mapping, which collapsed the PGCS effective sample size
# from ~100% to ~0.1% for *any* non-exactly-symmetric
# fragment (i.e. every realistic, H-bonded, or relaxed water).
#
# The same flaw affects every frame-dependent operation, not
# just ``Rz180``: ``Rz90`` / ``Rz270`` / ``S4`` / ``S4_3`` need
# local-z on the C_n / S_n axis, and the reflections / secondary
# C2 rotations need local-x / y on the sigma_v / C2' elements.
# ``build_frag_transform_data`` now builds the whole frame
# geometrically (continuously in the distortion) for all
# supported groups (C2v / C2h / Cs / D2h / C4v / D4h, including
# the linear Coov/Dooh mappings).
#
# These tests pin that behaviour so the axis bug cannot silently
# return -- note it was invisible at exact symmetry, hence the
# deliberately *distorted* / multi-axis geometries below.
# ---------------------------------------------------------------------------


if not hasattr(_pgto, "bse_predefined_ecp"):
    _pgto.bse_predefined_ecp = lambda *a, **k: ([], [])
from OmegaQMC import generate_molecular_orbitals  # noqa: E402

BOHR_TO_ANG = 0.52917721
REQ = 0.9572                          # equilibrium O-H (Angstrom)
HALF = np.radians(104.52 / 2.0)       # half of equilibrium angle
_D1 = np.array([np.sin(HALF), 0.0, np.cos(HALF)])
_D2 = np.array([-np.sin(HALF), 0.0, np.cos(HALF)])
RZ180 = np.diag([-1.0, -1.0, 1.0])    # 180 deg about local-z


def _water_string(stretch):
    """Water with one O-H scaled by ``stretch`` (fragment 1)."""
    H1 = stretch * REQ * _D1
    H2 = REQ * _D2
    rows = [("O", np.zeros(3)), ("H", H1), ("H", H2)]
    return "\n".join(
        f"{s} {r[0]:.8f} {r[1]:.8f} {r[2]:.8f} 1"
        for s, r in rows
    )


def _rz180_frame(stretch):
    """Production Rz180 frame for a distorted water.

    Returns ``(nuc, centroid, Vh, charges)`` with coordinates
    in Bohr, exactly as the VMC driver obtains them.
    """
    mf = generate_molecular_orbitals(
        _water_string(stretch), units="ang", basis="6-31G",
    )
    mol = mf.mol
    nuc = np.asarray(mol.atom_coords())
    charges = np.asarray(mol.atom_charges())
    # The driver passes the *requested* ops; a distorted
    # water is detected as Cs, so this is what exposes the
    # bug if the frame were taken from the detected group.
    frag_symmops = {k: ["E", "Rz180"] for k in mol.map_frag_ctr}
    fc, _, Vh, _ = build_frag_transform_data(
        mol, nuc, frag_symmops=frag_symmops,
    )
    return nuc, np.asarray(fc[0]), np.asarray(Vh[0]), charges


def _selfmap_displacement(nuc, centroid, Vh, charges):
    """Max nuclear displacement under frag-Rz180, in Angstrom."""
    op = (nuc - centroid) @ Vh.T @ RZ180 @ Vh + centroid
    disp = 0.0
    for i in range(len(nuc)):
        same = [j for j in range(len(nuc))
                if charges[j] == charges[i]]
        d = min(np.linalg.norm(op[i] - nuc[j]) for j in same)
        disp = max(disp, d)
    return disp * BOHR_TO_ANG


@pytest.mark.parametrize(
    "stretch", [1.000, 0.999, 0.995, 0.990, 1.005, 1.010],
)
def test_rz180_selfmaps_distorted_water(stretch):
    """Realistic distortions (<=1%) self-map to < 0.05 A."""
    nuc, centroid, Vh, charges = _rz180_frame(stretch)
    disp = _selfmap_displacement(nuc, centroid, Vh, charges)
    assert disp < 0.05, (
        f"frag-Rz180 displaced nuclei by {disp:.4f} A at "
        f"stretch={stretch}; axis is not the C2 bisector"
    )


@pytest.mark.parametrize("stretch", [0.95, 1.05])
def test_rz180_never_uses_wrong_axis(stretch):
    """Even a 5% distortion stays far from the ~1.04 A
    plane-normal (wrong-axis) bug signature."""
    nuc, centroid, Vh, charges = _rz180_frame(stretch)
    disp = _selfmap_displacement(nuc, centroid, Vh, charges)
    assert disp < 0.20, (
        f"frag-Rz180 displaced nuclei by {disp:.4f} A at "
        f"stretch={stretch}; looks like the plane-normal bug"
    )


@pytest.mark.parametrize("stretch", [0.999, 0.990, 0.950])
def test_rz180_axis_is_in_plane(stretch):
    """Local-z is the in-plane C2 axis, not the plane normal."""
    nuc, _, Vh, _ = _rz180_frame(stretch)
    plane_normal = np.cross(nuc[1] - nuc[0], nuc[2] - nuc[0])
    plane_normal /= np.linalg.norm(plane_normal)
    cos_zn = abs(float(np.dot(Vh[2], plane_normal)))
    assert cos_zn < 0.1, (
        "local-z aligned with the molecular-plane normal "
        f"(|z.n|={cos_zn:.3f}) -- the detect_symm Cs bug"
    )


def _frag_weight_ess(stretch):
    """Run a short VMC and return ESS/n of the Rz180 weights."""
    import h5py
    import jax
    from OmegaQMC import get_vmc_gto_func

    with tempfile.TemporaryDirectory() as tmp:
        prefix = f"{tmp}/pgcs_ess"
        mf = generate_molecular_orbitals(
            _water_string(stretch), units="ang", basis="6-31G",
        )
        vmc = get_vmc_gto_func(
            mf, None, prefix=prefix,
            symmop_list={1: ["E", "Rz180"]},
            cluster_idx=[[0, 1, 2]],
        )
        vmc(jax.random.key(0), num_walkers=400,
            num_steps_per_block=20, num_blocks=6,
            num_blocks_equil=2, mc_timestep=0.05,
            compute_gradients=True)
        with h5py.File(prefix + ".grd.h5", "r") as f:
            fw = f["fragment_weights"]
            lab = [k for k in fw
                   if isinstance(fw[k], h5py.Group)][0]
            blocks = sorted(int(k) for k in fw[lab]
                            if k.isdigit())
            w = np.concatenate([
                np.asarray(fw[lab][str(b)][()]).ravel()
                for b in blocks
            ]).astype(np.float64)
    return (w.sum() ** 2) / np.square(w).sum() / w.size


def test_rz180_ess_high_for_distorted_water():
    """End-to-end: a 1%-distorted water keeps a usable ESS.

    Slower than the deterministic checks above: needs JAX
    (GPU recommended) and runs a short VMC.  Deselect with
    ``-k "not ess"``.  With the bug, ESS/n was ~0.005; the
    geometric axis restores it to ~0.99, so the 0.5 threshold
    is a wide, robust separator.
    """
    ess = _frag_weight_ess(0.990)
    assert ess > 0.5, (
        f"ESS/n={ess:.4f} for a 1%-distorted water; the "
        "fragment-Rz180 axis has regressed to the wrong axis"
    )


# ---------------------------------------------------------------
# Multi-axis groups (C4v / D4h / D2h, incl. linear mappings).
# These exercise ``_geometric_symmetry_frame`` directly (pure
# geometry, no SCF), checking that *every* requested operation
# self-maps the nuclei -- i.e. local-z lands on the true C_n/S_n
# axis (long axis for linear, plane normal for square-planar)
# and local-x/y align with the sigma_v / C2' elements.
# ---------------------------------------------------------------
def _op_matrix(op):
    """Local 3x3 matrix of ``op`` (op(v) = M @ v)."""
    return np.asarray(symmetry_operations_map[op](jnp.eye(3))).T


def _worst_op_selfmap(nuc, charges, frame_ops, check_ops):
    """Build the frame from ``frame_ops``; return the largest
    self-map displacement over ``check_ops`` and the frame."""
    nuc = np.asarray(nuc, dtype=float)
    charges = np.asarray(charges)
    centroid = np.average(nuc, axis=0, weights=charges)
    Vh = np.asarray(
        _geometric_symmetry_frame(
            nuc, charges, centroid, frame_ops,
        )
    )
    Xc = nuc - centroid
    worst = 0.0
    for op in check_ops:
        if op in ("E", "i"):
            continue
        m_lab = Vh.T @ _op_matrix(op) @ Vh
        Xr = Xc @ m_lab.T
        for i in range(len(Xc)):
            same = [j for j in range(len(Xc))
                    if charges[j] == charges[i]]
            worst = max(
                worst,
                min(np.linalg.norm(Xr[i] - Xc[j])
                    for j in same),
            )
    return worst, Vh


def test_d4h_linear_all_ops_selfmap():
    """Linear CO2 (Dooh→D4h): all ops self-map; z = mol axis."""
    nuc = [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]]
    q = [6, 8, 8]
    ops = POINT_GROUP_OPS["D4h"]
    worst, Vh = _worst_op_selfmap(nuc, q, ops, ops)
    assert worst < 0.02, f"worst D4h self-map {worst:.4f} A"
    assert abs(float(Vh[2] @ np.array([0.0, 0.0, 1.0]))) > 0.99


def test_d4h_square_planar_all_ops_selfmap():
    """Square-planar D4h: all ops self-map; z = plane normal."""
    a = 1.5
    nuc = [[a, 0, 0], [-a, 0, 0], [0, a, 0], [0, -a, 0]]
    q = [1, 1, 1, 1]
    ops = POINT_GROUP_OPS["D4h"]
    worst, Vh = _worst_op_selfmap(nuc, q, ops, ops)
    assert worst < 0.02, f"worst D4h self-map {worst:.4f} A"
    # C4 axis is the plane normal, not an in-plane axis.
    assert abs(float(Vh[2] @ np.array([0.0, 0.0, 1.0]))) > 0.99


def test_d2h_rectangle_all_ops_selfmap():
    """Rectangular D2h: all three C2 axes / mirrors self-map."""
    a, b = 1.6, 1.0
    nuc = [[a, b, 0], [a, -b, 0], [-a, b, 0], [-a, -b, 0]]
    q = [1, 1, 1, 1]
    ops = POINT_GROUP_OPS["D2h"]
    worst, _ = _worst_op_selfmap(nuc, q, ops, ops)
    assert worst < 0.02, f"worst D2h self-map {worst:.4f} A"


@pytest.mark.parametrize("bend", [0.06, 0.12])
def test_bent_linear_rz90_not_catastrophic(bend):
    """A near-linear (bent) CO2 keeps z on the molecular axis,
    so Rz90 stays continuous in the bend (~0.1 A) instead of
    the ~1.6 A wrong-axis (plane-normal) bug."""
    nuc = [[0, 0, 0], [bend, 0, 1.16], [bend, 0, -1.16]]
    q = [6, 8, 8]
    ops = POINT_GROUP_OPS["D4h"]
    worst, Vh = _worst_op_selfmap(nuc, q, ops, ["Rz90"])
    assert worst < 0.25, f"Rz90 self-map {worst:.4f} A"
    long_axis = np.linalg.svd(
        np.asarray(nuc, float) - np.mean(nuc, axis=0)
    )[2][0]
    assert abs(float(Vh[2] @ long_axis)) > 0.95


# ---------------------------------------------------------------------------
# From test_pgcs_selfmap.py
#
# Nuclear self-map diagnostic of the PGCS fragment frames.
#
# :func:`OmegaQMC.symm.fragments.build_frag_transform_data` prints, for
# every requested non-identity operation, how far it moves the nuclei of
# its fragment off themselves, and warns (never raises) when the
# operation is only an approximate symmetry.
# ---------------------------------------------------------------------------


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
