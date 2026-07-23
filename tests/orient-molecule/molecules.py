"""Test molecule geometries for reverse-engineering g16 standard
orientation.

Each geometry is given in Angstrom in some *arbitrary* input frame
(deliberately translated/rotated off any nice axis) so that the g16
centering + reorientation is fully exercised.  Coordinates need not be
perfectly symmetric; g16 will report the point group it detects.
"""
import numpy as np

# A fixed "random" rotation + translation applied to clean geometries so
# the input orientation is generic.  Built from a fixed quaternion so it
# is reproducible without depending on RNG.
def _fixed_rot():
    # axis (1,2,3) normalized, angle 37 deg
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    th = np.deg2rad(37.0)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


R_FIXED = _fixed_rot()
T_FIXED = np.array([0.31, -0.72, 0.55])


def _place(coords, scramble=True):
    coords = np.asarray(coords, float)
    if scramble:
        coords = coords @ R_FIXED.T + T_FIXED
    return coords


def water():
    # clean C2v water in yz-plane
    b, a = 0.9578, np.deg2rad(104.4776)
    c = [[0, 0, 0],
         [0, b * np.sin(a / 2), b * np.cos(a / 2)],
         [0, -b * np.sin(a / 2), b * np.cos(a / 2)]]
    return ['O', 'H', 'H'], _place(c)


def ammonia():
    # C3v NH3; solve bond half-angle so HNH angle is 106.7 deg.
    from scipy.optimize import brentq
    nh = 1.012
    ang = np.deg2rad(106.7)  # HNH angle

    def f(theta):  # theta = angle of each N-H bond from the C3 axis
        h1 = np.array([np.sin(theta), 0, np.cos(theta)])
        h2 = np.array([np.sin(theta) * np.cos(np.deg2rad(120)),
                       np.sin(theta) * np.sin(np.deg2rad(120)),
                       np.cos(theta)])
        return np.dot(h1, h2) - np.cos(ang)
    theta = brentq(f, 0.1, np.pi / 2)
    hs = []
    for k in range(3):
        phi = np.deg2rad(120 * k)
        hs.append(nh * np.array([np.sin(theta) * np.cos(phi),
                                 np.sin(theta) * np.sin(phi),
                                 np.cos(theta)]))
    c = [[0, 0, 0]] + hs
    return ['N', 'H', 'H', 'H'], _place(c)


def methane():
    ch = 1.087
    # tetrahedral directions
    d = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                 float)
    d = d / np.linalg.norm(d[0])
    c = [[0, 0, 0]] + list(ch * d)
    return ['C', 'H', 'H', 'H', 'H'], _place(c)


def hcn():
    # linear Coov
    c = [[0, 0, 0], [0, 0, 1.064], [0, 0, -1.156]]  # H, C, N
    return ['H', 'C', 'N'], _place(c)


def co2():
    c = [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]]
    return ['C', 'O', 'O'], _place(c)


def h2o2():
    # C2 hydrogen peroxide (skew)
    doo, doh = 1.475, 0.95
    ooh = np.deg2rad(94.8)
    dih = np.deg2rad(111.5)
    # O1 at -doo/2 z, O2 at +doo/2 z along... build standard
    o1 = np.array([0, 0, 0])
    o2 = np.array([0, 0, doo])
    # H on O1
    h1 = o1 + doh * np.array([np.sin(ooh), 0, -np.cos(ooh)])
    # H on O2, rotated by dihedral about O-O (z) axis
    h2local = doh * np.array([np.sin(ooh), 0, np.cos(ooh)])
    cz, sz = np.cos(dih), np.sin(dih)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    h2 = o2 + Rz @ h2local
    c = [o1, o2, h1, h2]
    return ['O', 'O', 'H', 'H'], _place(c)


def hocl():
    # Cs planar HOCl
    doh, docl = 0.964, 1.69
    ang = np.deg2rad(102.5)
    o = np.array([0, 0, 0])
    h = o + doh * np.array([1, 0, 0])
    cl = o + docl * np.array([np.cos(ang), np.sin(ang), 0])
    return ['O', 'H', 'Cl'], _place([o, h, cl])


def chfclbr():
    # C1 fully asymmetric top
    c = np.array([0, 0, 0], float)
    # roughly tetrahedral bonds, different lengths
    d = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                 float)
    d = d / np.linalg.norm(d[0])
    h = c + 1.09 * d[0]
    f = c + 1.35 * d[1]
    cl = c + 1.77 * d[2]
    br = c + 1.94 * d[3]
    return ['C', 'H', 'F', 'Cl', 'Br'], _place([c, h, f, cl, br])


def ethylene():
    # D2h C2H4 in a plane
    dcc, dch = 1.339, 1.086
    ang = np.deg2rad(121.3)
    c1 = np.array([0, 0, dcc / 2])
    c2 = np.array([0, 0, -dcc / 2])
    # H's on c1
    h1 = c1 + dch * np.array([0, np.sin(ang - np.pi / 2 + np.pi / 2),
                              np.cos(ang)])
    # simpler explicit planar coordinates in yz-plane
    hy = dch * np.sin(ang - np.pi / 2)
    hz = dch * np.cos(np.pi - ang)
    c1 = np.array([0, 0, dcc / 2])
    c2 = np.array([0, 0, -dcc / 2])
    h1 = np.array([0, hy, dcc / 2 + hz])
    h2 = np.array([0, -hy, dcc / 2 + hz])
    h3 = np.array([0, hy, -dcc / 2 - hz])
    h4 = np.array([0, -hy, -dcc / 2 - hz])
    return (['C', 'C', 'H', 'H', 'H', 'H'],
            _place([c1, c2, h1, h2, h3, h4]))


def chfclbr2():
    # Another C1 top: distinct bond lengths/directions so the inertia
    # ordering differs from chfclbr.
    c = np.array([0, 0, 0], float)
    h = c + np.array([1.05, 0.10, 0.20])
    f = c + np.array([-0.30, 1.30, 0.40])
    cl = c + np.array([0.50, -0.80, 1.60])
    br = c + np.array([-1.10, -1.20, -1.00])
    return ['C', 'H', 'F', 'Cl', 'Br'], _place([c, h, f, cl, br])


def sfclbr():
    # SFClBrH-ish C1 cluster with heavy distinct atoms.
    s = np.array([0, 0, 0], float)
    f = s + np.array([1.56, 0.0, 0.0])
    cl = s + np.array([-0.40, 1.90, 0.10])
    br = s + np.array([0.20, -0.30, 2.10])
    h = s + np.array([-0.90, -0.90, -0.60])
    return ['S', 'F', 'Cl', 'Br', 'H'], _place([s, f, cl, br, h])


def fclethylene():
    # 1-fluoro-2-chloro substituted planar (Cs, plane = molecular plane)
    # mass and charge give different in-plane inertia axes.
    dcc = 1.34
    c1 = np.array([0, 0, dcc / 2])
    c2 = np.array([0, 0, -dcc / 2])
    f = c1 + np.array([0, 1.30, 0.50])
    h1 = c1 + np.array([0, -1.00, 0.40])
    cl = c2 + np.array([0, -1.70, -0.60])
    h2 = c2 + np.array([0, 1.05, -0.35])
    return (['C', 'C', 'F', 'H', 'Cl', 'H'],
            _place([c1, c2, f, h1, cl, h2]))


MOLECULES = {
    'water': water,
    'ammonia': ammonia,
    'methane': methane,
    'hcn': hcn,
    'co2': co2,
    'h2o2': h2o2,
    'hocl': hocl,
    'chfclbr': chfclbr,
    'ethylene': ethylene,
    'chfclbr2': chfclbr2,
    'sfclbr': sfclbr,
    'fclethylene': fclethylene,
}
