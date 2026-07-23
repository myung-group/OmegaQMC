"""Generate g16 inputs and parse input/standard orientations + masses."""
import os
import re
import subprocess
import numpy as np

G16 = "/opt/g16-c.01/g16"


def write_com(name, syms, coords_ang, workdir="."):
    """Write a minimal ROHF/cc-pVDZ single-point with explicit Cartesians."""
    path = os.path.join(workdir, name + ".com")
    lines = [f"%Chk={name}", "%Mem=2GB", "%Nprocshared=4",
             "#P ROHF/cc-pVDZ",
             "# Units(Ang,Deg)", "", f"{name} standard orientation test",
             ""]
    # charge/multiplicity (all closed-shell singlets here)
    lines.append("0 1")
    for s, c in zip(syms, coords_ang):
        lines.append(f"{s:2s} {c[0]:20.12f} {c[1]:20.12f} {c[2]:20.12f}")
    lines.append("")
    lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def run_g16(name, workdir="."):
    com = name + ".com"
    log = name + ".log"
    subprocess.run([G16, com], cwd=workdir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return os.path.join(workdir, log)


def _parse_orientation(text, header):
    """Return Nx3 array following an 'orientation:' header block."""
    idx = text.find(header)
    if idx < 0:
        return None
    tail = text[idx:]
    rows = []
    # data lines look like: int int int  x y z  (6 numeric cols after label)
    for line in tail.splitlines()[5:]:
        if set(line.strip()) <= set("- "):
            break
        parts = line.split()
        if len(parts) == 6 and parts[0].isdigit():
            rows.append([float(parts[3]), float(parts[4]),
                         float(parts[5])])
        elif rows:
            break
    return np.array(rows) if rows else None


def parse_log(path):
    with open(path) as fh:
        text = fh.read()
    out = {}
    out['input'] = _parse_orientation(text, "Input orientation:")
    std = _parse_orientation(text, "Standard orientation:")
    if std is None:
        std = _parse_orientation(text, "Z-Matrix orientation:")
    out['standard'] = std
    # atomic numbers from input block
    m = re.search(r"AtmWgt=\s*(.*)", text)
    # AtmWgt lines can wrap; collect the isotope-property block.
    wl = re.findall(r"AtmWgt=\s*([-\d.\s]+)", text)
    weights = []
    for chunk in wl:
        weights += [float(x) for x in chunk.split()]
    out['masses'] = np.array(weights) if weights else None
    znuc = re.findall(r"AtZNuc=\s*([-\d.\s]+)", text)
    zs = []
    for chunk in znuc:
        zs += [float(x) for x in chunk.split()]
    out['charges'] = np.array(zs) if zs else None
    # framework / point group
    fg = re.search(r"Framework group\s+(\S+)", text)
    out['framework'] = fg.group(1) if fg else None
    pg = re.search(r"Full point group\s+(\S+)", text)
    out['pointgroup'] = pg.group(1) if pg else None
    return out
