"""Post-process the raw Wick einsum output (eqs/*.py.txt) into the residual
kernels of OmegaQMC/addons/qed_ccsd_new.py.

Transformations applied to every generated term:

* slice ``f`` / ``d`` / ``B`` by occ/vir blocks (``B`` keeps its aux index;
  the ``d`` coupling is ``-dip`` exactly as in qed_ccsd.py);
* drop all boson (photon-mode) indices — a single cavity mode has dimension
  one, so summing over it equals taking the single element; scalar photon
  tensors (``s1``, ``s2``, ``w``) become scalar prefactors;
* rename amplitudes to the qed_ccsd.py convention
  (t1→t1_10, t2→t2_20, s1→t1_01, s2→t2_02, u11→t2_11, u21→t2_21,
  u12→t2_12, u22→t2_22);
* guard every term that references an optional photonic amplitude with an
  ``in active`` check so disabled flavours cost neither memory nor flops;
* particle-ladder terms — two ``B_vv`` factors contracted with a 4-index
  amplitude — are routed through the batched helper ``_vvvv_ladder`` so no
  ``nvir^4`` or ``naux*nvir^2*nocc^2`` intermediate is ever formed.
"""
import os
import re
import sys

OCC = set("ijklmnop")
VIR = set("abcdefgh")
BOS = set("IJKLMNOP")

AMP = {  # tensor name -> (array name, is_scalar)
    "t1": ("t1_10", False),
    "t2": ("t2_20", False),
    "s1": ("t1_01", True),
    "s2": ("t2_02", True),
    "u11": ("t2_11", False),
    "u21": ("t2_21", False),
    "u12": ("t2_12", False),
    "u22": ("t2_22", False),
}

OPTIONAL = {"t1_01", "t2_02", "t2_11", "t2_21", "t2_12", "t2_22"}

LINE_RE = re.compile(
    r"^res_(\w+) \+= (-?[\d.]+)\*einsum\('([^']*)->([^']*)',\s*([^)]+?),"
    r"\s*optimize=True\)\s*$")


def cls(ch):
    if ch in OCC:
        return "o"
    if ch in VIR:
        return "v"
    raise ValueError(ch)


def strip_boson(sub):
    return "".join(c for c in sub if c not in BOS)


def convert_term(coef, subs, out, names, used_slices):
    """Return (rhs_expression, guard_amps, uses_ladder) for one term."""
    out_f = strip_boson(out)
    scalars = []
    arrays = []  # (arrname, fermion_sub_with_aux)
    guards = set()
    for name, sub in zip(names, subs):
        if name == "f":
            sl = f"f_{cls(sub[0])}{cls(sub[1])}"
            used_slices.add(sl)
            arrays.append((sl, sub))
        elif name == "d":
            fs = strip_boson(sub)
            sl = f"d_{cls(fs[0])}{cls(fs[1])}"
            used_slices.add(sl)
            arrays.append((sl, fs))
        elif name == "B":
            fs = sub[1:]
            sl = f"B_{cls(fs[0])}{cls(fs[1])}"
            used_slices.add(sl)
            arrays.append((sl, sub))
        elif name == "w":
            scalars.append("w")
        elif name in AMP:
            arr, is_scalar = AMP[name]
            if arr in OPTIONAL:
                guards.add(arr)
            if is_scalar:
                scalars.append(arr)
            else:
                arrays.append((arr, strip_boson(sub)))
        else:
            raise ValueError(f"unknown tensor {name}")

    pre = coef
    for s in scalars:
        pre += f" * {s}"

    if not arrays:
        return f"{pre}", guards, False

    # --- particle-ladder detection: two B_vv factors whose summed virtual
    # indices both land on one 4-index amplitude ---
    bvv = [a for a in arrays if a[0] == "B_vv"]
    amp4 = [a for a in arrays
            if a[0] in ("t2_20", "t2_21", "t2_22") and len(a[1]) == 4]
    if len(bvv) == 2 and len(amp4) == 1 and len(arrays) == 3 \
            and len(out_f) == 4:
        e1, e2 = out_f[0], out_f[1]
        occ_out = out_f[2:]
        s1 = next((a[1] for a in bvv if e1 in a[1]), None)
        s2 = next((a[1] for a in bvv if e2 in a[1]), None)
        ampname, samp = amp4[0]
        if s1 is not None and s2 is not None and s1 != s2:
            c1 = [c for c in s1[1:] if c != e1][0]
            d1 = [c for c in s2[1:] if c != e2][0]
            perm = c1 + d1 + occ_out
            if set(perm) == set(samp) and c1 != d1:
                rhs = (f"{pre} * _vvvv_ladder(B_vv, "
                       f"np.ascontiguousarray(np.einsum("
                       f"'{samp}->{perm}', {ampname})))")
                return rhs, guards, True
        raise RuntimeError(f"unhandled ladder pattern: {subs} -> {out}")

    spec = ",".join(sub for _, sub in arrays) + "->" + out_f
    ops = ", ".join(arr for arr, _ in arrays)
    return f"{pre} * contract('{spec}', {ops})", guards, False


DENOM_2IDX = {
    "t1_10": "eps_occ[None, :] - eps_vir[:, None]",
    "t2_11": "eps_occ[None, :] - eps_vir[:, None] - w",
    "t2_12": "eps_occ[None, :] - eps_vir[:, None] - 2.0 * w",
}
W_SHIFT = {"t2_20": "0.0", "t2_21": "w", "t2_22": "2.0 * w"}


def gen_function(target, lines):
    used = set()
    body = []
    n_ladder = 0
    for ln in lines:
        m = LINE_RE.match(ln)
        if not m:
            if ln.strip():
                raise ValueError(f"unparsed line: {ln}")
            continue
        tgt, coef, subs, out, names = m.groups()
        assert tgt == target
        names = [n.strip() for n in names.split(",")]
        subs = subs.split(",")
        assert len(names) == len(subs), ln
        rhs, guards, is_ladder = convert_term(coef, subs, out, names, used)
        n_ladder += is_ladder
        if guards:
            cond = " and ".join(f"'{g}' in active" for g in sorted(guards))
            body.append(f"    if {cond}:")
            body.append(f"        res += {rhs}")
        else:
            body.append(f"    res += {rhs}")

    scalar_res = target in ("t1_01", "t2_02", "energy")
    fname = "ccsd_energy" if target == "energy" else f"ccsd_{target}"
    hdr = []
    hdr.append(f"def {fname}(f_so, B_so, dip, G, w, t1_10, t1_01,"
               " t2_20, t2_02,")
    hdr.append(f"{' ' * (len(fname) + 5)}t2_11, t2_21, t2_12, t2_22,"
               " active=_ALL_AMPS):")
    if target == "energy":
        hdr.append('    """QED-CCSD correlation energy'
                   ' (Wick-derived, DF integrals)."""')
    hdr.append("    nocc = t2_20.shape[2]")
    hdr.append("    nvir = f_so.shape[0] - nocc")
    hdr.append("    o = slice(None, nocc)")
    hdr.append("    v = slice(nocc, None)")
    if not scalar_res:
        hdr.append("    eps = f_so.diagonal()")
        hdr.append("    eps_occ = eps[o]")
        hdr.append("    eps_vir = eps[v]")

    slice_defs = {
        "f_oo": "    f_oo = f_so[o, o]",
        "f_ov": "    f_ov = f_so[o, v]",
        "f_vo": "    f_vo = f_so[v, o]",
        "f_vv": "    f_vv = f_so[v, v]",
        "d_oo": "    d_oo = -dip[o, o]",
        "d_ov": "    d_ov = -dip[o, v]",
        "d_vo": "    d_vo = -dip[v, o]",
        "d_vv": "    d_vv = -dip[v, v]",
        "B_oo": "    B_oo = B_so[:, o, o]",
        "B_ov": "    B_ov = B_so[:, o, v]",
        "B_vo": "    B_vo = B_so[:, v, o]",
        "B_vv": "    B_vv = B_so[:, v, v]",
    }
    for sl in sorted(used):
        hdr.append(slice_defs[sl])
    if scalar_res:
        hdr.append("    res = 0.0")
    elif target in DENOM_2IDX:
        hdr.append("    res = np.zeros((nvir, nocc))")
    else:
        hdr.append("    res = np.zeros((nvir, nvir, nocc, nocc))")
    hdr.append("")

    tail = [""]
    if target == "energy":
        tail.append("    return float(res)")
    elif target == "t1_01":
        tail.append("    if w == 0:")
        tail.append("        return 0.0")
        tail.append("    return t1_01 - res / w")
    elif target == "t2_02":
        tail.append("    if w == 0:")
        tail.append("        return 0.0")
        tail.append("    return t2_02 - res / (2.0 * w)")
    elif target in DENOM_2IDX:
        tail.append(f"    e_denom = 1.0 / ({DENOM_2IDX[target]})")
        tail.append(f"    return {target} + res * e_denom")
    else:  # 4-index doubles
        shift = W_SHIFT[target]
        tail.append("    # chunked in-place denominator: never materialises"
                    " the o^2 v^2 tensor")
        tail.append("    for i in range(nocc):")
        tail.append("        res[:, :, i, :] *= 1.0 / ("
                    "eps_occ[i] + eps_occ[None, None, :]")
        tail.append("                                  "
                    "- eps_vir[:, None, None]")
        tail.append("                                  "
                    f"- eps_vir[None, :, None] - ({shift}))")
        tail.append(f"    res += {target}")
        tail.append("    return res")
    tail.append("")
    tail.append("")

    return "\n".join(hdr + body + tail), n_ladder


def main():
    eqdir = sys.argv[1] if len(sys.argv) > 1 else "eqs"
    outdir = "gen_out"
    os.makedirs(outdir, exist_ok=True)
    targets = ["energy", "t1_10", "t1_01", "t2_20", "t2_02",
               "t2_11", "t2_21", "t2_12", "t2_22"]
    for tgt in targets:
        path = os.path.join(eqdir, f"{tgt}.py.txt")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"{tgt}: MISSING, skipped")
            continue
        with open(path) as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        code, n_ladder = gen_function(tgt, lines)
        with open(os.path.join(outdir, f"{tgt}.py"), "w") as fh:
            fh.write(code)
        print(f"{tgt}: {len(lines)} terms, {n_ladder} ladder terms")


if __name__ == "__main__":
    main()
