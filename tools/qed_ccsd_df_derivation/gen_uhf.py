"""Emit the alpha/beta-blocked (UHF) QED-CCSD kernels from the traced
terms of spin_trace_uhf.py. One kernel per residual spin block (18 total);
kernels read amplitudes from an `amps` namespace and integrals from an
`ints` namespace whose 4-index chemist blocks are built once per run."""
import os
import sys
from collections import defaultdict

import spin_trace_uhf as stu
from spin_trace import cls

OPTIONAL = {"t1_01", "t2_02", "t2_11", "t2_21", "t2_12", "t2_22"}
AMP4 = {"t2_20", "t2_21", "t2_22"}
SCALARS = {"t1_01", "t2_02"}

TARGETS = {
    "energy": 0, "t1_10": 2, "t1_01": 0, "t2_20": 4, "t2_02": 0,
    "t2_11": 2, "t2_21": 4, "t2_12": 2, "t2_22": 4,
}
SHIFT = {"t1_10": "0.0", "t2_11": "w", "t2_12": "2.0 * w",
         "t2_20": "0.0", "t2_21": "w", "t2_22": "2.0 * w"}


def fmt_coef(c):
    return repr(round(c, 10))


def base_amp(name):
    """t2_20_ab -> t2_20; t2_11_a -> t2_11; t1_01 -> t1_01"""
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("a", "b", "aa", "ab", "bb") \
            and parts[0] in (AMP4 | {"t1_10", "t2_11", "t2_12"}):
        return parts[0]
    return name


def convert(coef, ops, out_f, block, used_v):
    scal = []
    arrays = []
    guards = set()
    for name, sub, sp in ops:
        if name in SCALARS:
            scal.append(f"amps.{name}")
            guards.add(name)
        elif name == "w":
            scal.append("w")
        elif name.startswith("f_") or name.startswith("d_"):
            arrays.append((f"ints.{name}_{cls(sub[0])}{cls(sub[1])}", sub))
        elif name.startswith("B_"):
            arrays.append((f"ints.B_vv_{sp}", "x" + sub))
        elif name.startswith("v_"):
            used_v.add(name)
            arrays.append((f"ints.{name}", sub))
        else:
            b = base_amp(name)
            if b in OPTIONAL:
                guards.add(b)
            arrays.append((f"amps.{name}", sub))

    pre = fmt_coef(coef)
    for s in scal:
        pre += f" * {s}"

    bvv = [a for a in arrays if a[0].startswith("ints.B_vv_")]
    amp4 = [a for a in arrays if a[0].startswith("amps.t2_2")
            and len(a[1]) == 4]
    if len(bvv) == 2 and len(amp4) == 1 and len(arrays) == 3 \
            and len(out_f) == 4:
        e1, e2 = out_f[0], out_f[1]
        occ_out = out_f[2:]
        b1 = next((a for a in bvv if e1 in a[1]), None)
        b2 = next((a for a in bvv if e2 in a[1]), None)
        ampname, samp = amp4[0]
        if b1 and b2 and b1[1] != b2[1]:
            c1 = [c for c in b1[1][1:] if c != e1][0]
            d1 = [c for c in b2[1][1:] if c != e2][0]
            perm = c1 + d1 + occ_out
            if set(perm) == set(samp) and c1 != d1:
                if perm == samp:
                    w_expr = ampname
                else:
                    w_expr = (f"np.ascontiguousarray(np.einsum("
                              f"'{samp}->{perm}', {ampname}))")
                stmt = (f"_vvvv_ladder2({b1[0]}, {b2[0]}, {w_expr}, "
                        f"out=res, alpha={pre})")
                return stmt, frozenset(guards)
        raise RuntimeError(f"unhandled ladder {ops} -> {out_f}")

    spec = ",".join(s for _, s in arrays) + "->" + out_f
    names = ", ".join(a for a, _ in arrays)
    if arrays:
        stmt = f"res += {pre} * contract('{spec}', {names})"
    else:
        stmt = f"res += {pre}"
    return stmt, frozenset(guards)


def gen_kernel(target, block, terms, used_v):
    out_len = TARGETS[target]
    out_f = "" if out_len == 0 else ("ai" if out_len == 2 else "abij")
    body = []
    for coef, ops in sorted(terms, key=lambda t: [t[1], t[0]]):
        stmt, guards = convert(coef, ops, out_f, block, used_v)
        if guards:
            cond = " and ".join(f"'{g}' in active" for g in sorted(guards))
            body.append(f"    if {cond}:")
            body.append(f"        {stmt}")
        else:
            body.append(f"    {stmt}")

    suffix = f"_{block}" if block else ""
    fname = ("ccsd_energy" if target == "energy"
             else f"ccsd_{target}{suffix}")
    L = [f"def {fname}(ints, w, amps, active=_ALL_AMPS):"]
    if out_len == 0:
        L.append("    res = 0.0")
    elif out_len == 2:
        s = block
        L.append(f"    res = np.zeros((ints.eps_vir_{s}.size,"
                 f" ints.eps_occ_{s}.size))")
    else:
        sA, sB = block[0], block[1]
        L.append(f"    res = np.zeros((ints.eps_vir_{sA}.size,"
                 f" ints.eps_vir_{sB}.size,")
        L.append(f"                    ints.eps_occ_{sA}.size,"
                 f" ints.eps_occ_{sB}.size))")
    L.append("")
    L.extend(body)
    L.append("")
    amp_name = f"{target}{suffix}"
    if target == "energy":
        L.append("    return float(res)")
    elif target == "t1_01":
        L.append("    if w == 0:")
        L.append("        return 0.0")
        L.append("    return amps.t1_01 - res / w")
    elif target == "t2_02":
        L.append("    if w == 0:")
        L.append("        return 0.0")
        L.append("    return amps.t2_02 - res / (2.0 * w)")
    elif out_len == 2:
        s = block
        L.append(f"    e_denom = 1.0 / (ints.eps_occ_{s}[None, :]"
                 f" - ints.eps_vir_{s}[:, None] - ({SHIFT[target]}))")
        L.append(f"    return amps.{amp_name} + res * e_denom")
    else:
        sA, sB = block[0], block[1]
        if sA == sB:
            L.append("    # exact antisymmetry of the same-spin residual;"
                     " enforce against rounding drift")
            L.append("    res = 0.5 * (res - res.transpose(1, 0, 2, 3))")
            L.append("    res = 0.5 * (res - res.transpose(0, 1, 3, 2))")
        L.append(f"    nocc_j = ints.eps_occ_{sB}.size")
        L.append("    for j in range(nocc_j):")
        L.append(f"        res[:, :, :, j] *= 1.0 / ("
                 f"ints.eps_occ_{sA}[None, None, :]"
                 f" + ints.eps_occ_{sB}[j]")
        L.append(f"                                  "
                 f"- ints.eps_vir_{sA}[:, None, None]")
        L.append(f"                                  "
                 f"- ints.eps_vir_{sB}[None, :, None]"
                 f" - ({SHIFT[target]}))")
        L.append(f"    res += amps.{amp_name}")
        L.append("    return res")
    L.append("")
    L.append("")
    return "\n".join(L)


def main():
    eqdir = sys.argv[1] if len(sys.argv) > 1 else "eqs"
    outdir = "gen_uhf_out"
    os.makedirs(outdir, exist_ok=True)
    used_v = set()
    for target, out_len in TARGETS.items():
        blocks = stu.trace_file_blocks(
            os.path.join(eqdir, f"{target}.py.txt"), target, out_len)
        code = []
        for block, terms in blocks.items():
            code.append(gen_kernel(target, block, terms, used_v))
            print(f"{target}{'_' + block if block else '':10s}"
                  f" {len(terms)} terms")
        with open(os.path.join(outdir, f"{target}.py"), "w") as fh:
            fh.write("".join(code))
    with open(os.path.join(outdir, "_needed.py"), "w") as fh:
        fh.write(f"_NEEDED_V = {sorted(used_v)!r}\n\n\n")
    print("V blocks:", sorted(used_v))


if __name__ == "__main__":
    main()
