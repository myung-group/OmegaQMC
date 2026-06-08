#!/usr/bin/env python3
"""Regression test for the PGCS fragment-rotation axis.

A fragment ``Rz180`` (correlated-sampling C2) must rotate
about the fragment's true C2 axis -- the H-O-H bisector for
water -- so that it maps the fragment onto itself and leaves
the importance weight ``|psi(C2.R)/psi(R)|^2`` near unity.

A prior bug took the rotation frame from PySCF
``detect_symm``: the moment a water fell below the exact-C2v
tolerance it was classified ``Cs`` and local-z switched
discontinuously from the C2 axis to the *molecular-plane
normal*.  ``Rz180`` then rotated about the wrong axis,
displacing the affected electrons by ~1 A instead of
self-mapping, which collapsed the PGCS effective sample size
from ~100% to ~0.1% for *any* non-exactly-symmetric
fragment (i.e. every realistic, H-bonded, or relaxed water).

``build_frag_transform_data`` now builds the axis geometrically
(continuously in the distortion).  These tests pin that
behaviour so the axis bug cannot silently return -- note it
was invisible at exact symmetry, hence the deliberately
*distorted* geometries below.
"""
import tempfile

import numpy as np
import pytest

# Compatibility shim: some PySCF builds lack
# ``gto.bse_predefined_ecp``, referenced by
# ``Mole_custom.check_sanity``.  It only advises about ECPs,
# which are irrelevant for the all-electron 6-31G water here.
import pyscf.gto as _pgto
if not hasattr(_pgto, "bse_predefined_ecp"):
    _pgto.bse_predefined_ecp = lambda *a, **k: ([], [])

from OmegaQMC import generate_molecular_orbitals
from OmegaQMC.symm.fragments import build_frag_transform_data

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


if __name__ == "__main__":
    for s in (1.000, 0.999, 0.995, 0.990, 1.005, 1.010):
        nuc, c, Vh, q = _rz180_frame(s)
        d = _selfmap_displacement(nuc, c, Vh, q)
        print(f"stretch={s:.3f}  self-map displacement="
              f"{d:.4f} A  (must be < 0.05)")
        assert d < 0.05
    for s in (0.95, 1.05):
        nuc, c, Vh, q = _rz180_frame(s)
        d = _selfmap_displacement(nuc, c, Vh, q)
        print(f"stretch={s:.3f}  self-map displacement="
              f"{d:.4f} A  (must be < 0.20)")
        assert d < 0.20
    print("deterministic axis checks passed; running VMC ESS...")
    ess = _frag_weight_ess(0.990)
    print(f"ESS/n at 1% distortion = {ess:.4f}  (must be > 0.5)")
    assert ess > 0.5
    print("ALL PASSED")
