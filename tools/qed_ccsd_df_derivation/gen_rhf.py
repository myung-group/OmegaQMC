"""Emit the closed-shell (spin-adapted) QED-CCSD kernels from the traced
spatial terms produced by spin_trace.py.

Intermediate reuse: every non-(vv|vv) B.B pair was substituted by a
precomputed 4-index chemist block v_XXXX (built once per run in the
driver); the remaining all-virtual pairs go through the batched
_vvvv_ladder (grouped so each kernel reconstructs the (vv|vv) slabs once
per guard group, not once per term).
"""
import os
import sys
from collections import defaultdict

import spin_trace as st

OPTIONAL = {"t1_01", "t2_02", "t2_11", "t2_21", "t2_12", "t2_22"}
AMP_ARRAYS = {"t1_10", "t2_20", "t2_11", "t2_21", "t2_12", "t2_22"}
SCALAR_AMPS = {"t1_01", "t2_02"}

DENOM_2IDX = {
    "t1_10": "ints.eps_occ[None, :] - ints.eps_vir[:, None]",
    "t2_11": "ints.eps_occ[None, :] - ints.eps_vir[:, None] - w",
    "t2_12": "ints.eps_occ[None, :] - ints.eps_vir[:, None] - 2.0 * w",
}
W_SHIFT = {"t2_20": "0.0", "t2_21": "w", "t2_22": "2.0 * w"}


def fmt_coef(c):
    return repr(round(c, 10))


def convert(term, out_f, used_v, used_slices):
    """Return (kind, payload) where kind is 'plain' (stmt string) or
    'ladder' (coef_expr, amp, samp, guards)."""
    coef, ops = term
    scal = []
    arrays = []
    guards = set()
    for arr, sub in ops:
        if arr in SCALAR_AMPS:
            scal.append(arr)
            guards.add(arr)
        elif arr == "w":
            scal.append("w")
        elif arr == "f" or arr == "d":
            sl = f"{arr}_{st.cls(sub[0])}{st.cls(sub[1])}"
            used_slices.add(sl)
            arrays.append((f"ints.{sl}", sub))
        elif arr == "B":
            used_slices.add("B_vv")
            arrays.append(("ints.B_vv", "x" + sub))
        elif arr.startswith("v_"):
            used_v.add(arr)
            arrays.append((f"ints.{arr}", sub))
        elif arr in AMP_ARRAYS:
            if arr in OPTIONAL:
                guards.add(arr)
            arrays.append((arr, sub))
        else:
            raise ValueError(arr)

    pre = fmt_coef(coef)
    for s in scal:
        pre += f" * {s}"

    # ladder: two B_vv + one 4-index doubles amplitude
    bvv = [a for a in arrays if a[0] == "ints.B_vv"]
    amp4 = [a for a in arrays if a[0] in ("t2_20", "t2_21", "t2_22")
            and len(a[1]) == 4]
    if len(bvv) == 2 and len(amp4) == 1 and len(arrays) == 3 \
            and len(out_f) == 4:
        e1, e2 = out_f[0], out_f[1]
        occ_out = out_f[2:]
        s1 = next((a[1][1:] for a in bvv if e1 in a[1]), None)
        s2 = next((a[1][1:] for a in bvv if e2 in a[1]), None)
        ampname, samp = amp4[0]
        if s1 and s2 and s1 != s2:
            c1 = [c for c in s1 if c != e1][0]
            d1 = [c for c in s2 if c != e2][0]
            perm = c1 + d1 + occ_out
            if set(perm) == set(samp) and c1 != d1:
                return "ladder", (pre, ampname, samp, perm,
                                  frozenset(guards))
        raise RuntimeError(f"unhandled ladder pattern {ops} -> {out_f}")

    spec = ",".join(s for _, s in arrays) + "->" + out_f
    names = ", ".join(a for a, _ in arrays)
    if arrays:
        stmt = f"res += {pre} * contract('{spec}', {names})"
    else:
        stmt = f"res += {pre}"
    return "plain", (stmt, frozenset(guards))


def gen_kernel(target, terms, out_f, used_v):
    used_slices = set()
    plain = []
    ladders = defaultdict(list)   # guards -> [(pre, amp, samp, perm)]
    for term in sorted(terms, key=lambda t: [t[1], t[0]]):
        kind, payload = convert(term, out_f, used_v, used_slices)
        if kind == "plain":
            plain.append(payload)
        else:
            pre, amp, samp, perm, guards = payload
            ladders[guards].append((pre, amp, samp, perm))

    fname = "ccsd_energy" if target == "energy" else f"ccsd_{target}"
    L = []
    L.append(f"def {fname}(ints, w, t1_10, t1_01, t2_20, t2_02,")
    L.append(f"{' ' * (len(fname) + 5)}t2_11, t2_21, t2_12, t2_22,"
             " active=_ALL_AMPS):")
    L.append("    nocc = t2_20.shape[2]")
    L.append("    nvir = t2_20.shape[0]")
    scalar_res = target in ("t1_01", "t2_02", "energy")
    if scalar_res:
        L.append("    res = 0.0")
    elif target in DENOM_2IDX:
        L.append("    res = np.zeros((nvir, nocc))")
    else:
        L.append("    res = np.zeros((nvir, nvir, nocc, nocc))")
    L.append("")
    for stmt, guards in plain:
        if guards:
            cond = " and ".join(f"'{g}' in active" for g in sorted(guards))
            L.append(f"    if {cond}:")
            L.append(f"        {stmt}")
        else:
            L.append(f"    {stmt}")
    for guards, items in sorted(ladders.items(), key=lambda kv: sorted(kv[0])):
        body = []
        first = True
        for pre, amp, samp, perm in items:
            op = "=" if first else "+="
            if perm == samp:
                body.append(f"        _W {op} {pre} * {amp}")
            else:
                body.append(f"        _W {op} {pre} * "
                            f"np.einsum('{samp}->{perm}', {amp})")
            first = False
        body.append("        _vvvv_ladder(ints.B_vv, "
                    "np.ascontiguousarray(_W), out=res, alpha=1.0)")
        if guards:
            cond = " and ".join(f"'{g}' in active" for g in sorted(guards))
            L.append(f"    if {cond}:")
            L.extend(body)
        else:
            L.extend(ln[4:] for ln in body)
    L.append("")
    if target == "energy":
        L.append("    return float(res)")
    elif target == "t1_01":
        L.append("    if w == 0:")
        L.append("        return 0.0")
        L.append("    return t1_01 - res / w")
    elif target == "t2_02":
        L.append("    if w == 0:")
        L.append("        return 0.0")
        L.append("    return t2_02 - res / (2.0 * w)")
    elif target in DENOM_2IDX:
        L.append(f"    e_denom = 1.0 / ({DENOM_2IDX[target]})")
        L.append(f"    return {target} + res * e_denom")
    else:
        shift = W_SHIFT[target]
        L.append("    # exact (ab)(ij) pair symmetry of the mixed-spin"
                 " block; enforce against rounding drift")
        L.append("    res = 0.5 * (res + res.transpose(1, 0, 3, 2))")
        L.append("    for i in range(nocc):")
        L.append("        res[:, :, i, :] *= 1.0 / ("
                 "ints.eps_occ[i] + ints.eps_occ[None, None, :]")
        L.append("                                  "
                 "- ints.eps_vir[:, None, None]")
        L.append("                                  "
                 f"- ints.eps_vir[None, :, None] - ({shift}))")
        L.append(f"    res += {target}")
        L.append("    return res")
    L.append("")
    L.append("")
    return "\n".join(L), used_slices


def main():
    eqdir = sys.argv[1] if len(sys.argv) > 1 else "eqs"
    outdir = "gen_rhf_out"
    os.makedirs(outdir, exist_ok=True)
    targets = ["energy", "t1_10", "t1_01", "t2_20", "t2_02",
               "t2_11", "t2_21", "t2_12", "t2_22"]
    used_v = set()
    all_slices = set()
    for tgt in targets:
        terms, out_f, _ = st.trace_file(os.path.join(eqdir, f"{tgt}.py.txt"),
                                        tgt)
        code, used_slices = gen_kernel(tgt, terms, out_f, used_v)
        all_slices |= used_slices
        with open(os.path.join(outdir, f"{tgt}.py"), "w") as fh:
            fh.write(code)
        print(f"{tgt}: {len(terms)} spatial terms")
    with open(os.path.join(outdir, "_needed.py"), "w") as fh:
        fh.write(f"_NEEDED_V = {sorted(used_v)!r}\n")
        fh.write(f"_NEEDED_SLICES = {sorted(all_slices)!r}\n\n\n")
    print("V blocks needed:", sorted(used_v))
    print("slices needed:", sorted(all_slices))


if __name__ == "__main__":
    main()
