"""Spin-trace the spin-orbital QED-CCSD equations (eqs/*.py.txt) to
alpha/beta-blocked (UHF) spatial form.

Unlike the closed-shell tracer (spin_trace.py), the two spins keep
independent orbitals and amplitudes:

    t1_10_a[a,i], t1_10_b        (alpha / beta singles-type blocks)
    t2_20_aa[a,b,i,j]            antisymmetric same-spin doubles
    t2_20_bb[a,b,i,j]
    t2_20_ab[a,b,i,j]            mixed block: a,i alpha; b,j beta
    (analogously u21/u22; u11/u12 like t1; scalars unchanged)

Every spin-orbital term is expanded by explicit enumeration of spin
assignments per delta-constraint group (spin-diagonal f/d/B/singles fix
sigma_p = sigma_q; doubles-type operands require equal spin multisets on
their particle/hole pairs). Mixed doubles operands are brought to the
canonical (alpha-particle, beta-particle, alpha-hole, beta-hole) slot
order with the antisymmetry sign. The residual for each output block is
the literal spin-orbital residual block, so denominators/updates carry
over per block.
"""
from collections import defaultdict
import itertools

from spin_trace import LINE_RE, OCC, VIR, cls, strip_boson, _UF

# tensor name -> (base spatial name, kind)
TENSORS = {
    "f": ("f", "diag2"),
    "d": ("d", "diag2"),
    "B": ("B", "B"),
    "w": ("w", "scalar"),
    "t1": ("t1_10", "diag2amp"),
    "s1": ("t1_01", "scalar"),
    "t2": ("t2_20", "pair4"),
    "s2": ("t2_02", "scalar"),
    "u11": ("t2_11", "diag2amp"),
    "u21": ("t2_21", "pair4"),
    "u12": ("t2_12", "diag2amp"),
    "u22": ("t2_22", "pair4"),
}

# output spin patterns per residual block (letters of the printed output)
BLOCKS = {
    "2": [("a", {0: 'a', 1: 'a'}), ("b", {0: 'b', 1: 'b'})],
    "4": [("aa", {0: 'a', 1: 'a', 2: 'a', 3: 'a'}),
          ("ab", {0: 'a', 1: 'b', 2: 'a', 3: 'b'}),
          ("bb", {0: 'b', 1: 'b', 2: 'b', 3: 'b'})],
}


def trace_term_blocks(coef, subs, out, names, out_spin):
    """Yield (coefficient, ops) blocked spatial terms for one so-term and
    one output spin assignment (out_spin: letter -> 'a'/'b')."""
    out_f = strip_boson(out)

    base_ops = []
    for name, sub in zip(names, subs):
        base, kind = TENSORS[name]
        if kind == "scalar":
            base_ops.append((base, "", kind))
        elif kind == "B":
            base_ops.append((base, sub[1:], kind))
        else:
            base_ops.append((base, strip_boson(sub), kind))

    letters = sorted({c for _, s, _ in base_ops for c in s})
    uf = _UF()
    for base, s, kind in base_ops:
        if kind in ("diag2", "diag2amp", "B"):
            uf.union(s[0], s[1])

    roots = sorted({uf.find(c) for c in letters})
    fixed = {}
    ok = True
    for ch, sp in out_spin.items():
        r = uf.find(ch)
        if fixed.get(r, sp) != sp:
            ok = False
        fixed[r] = sp
    if not ok:
        return
    free = [r for r in roots if r not in fixed]

    for combo in itertools.product('ab', repeat=len(free)):
        spin_of_root = dict(fixed)
        spin_of_root.update(zip(free, combo))
        spin = {c: spin_of_root[uf.find(c)] for c in letters}

        sign = 1.0
        ops = []
        drop = False
        for base, s, kind in base_ops:
            if kind == "scalar":
                ops.append((base, s, kind, ""))
            elif kind in ("diag2", "diag2amp", "B"):
                ops.append((base, s, kind, spin[s[0]]))
            else:  # pair4
                s0, s1, s2, s3 = s
                ps = sorted((spin[s0], spin[s1]))
                hs = sorted((spin[s2], spin[s3]))
                if ps != hs:
                    drop = True
                    break
                if ps[0] == ps[1]:
                    ops.append((base, s, kind, ps[0] * 2))
                else:
                    # canonical mixed slot order (alpha, beta | alpha, beta)
                    if spin[s0] == 'a':
                        p_sub = s0 + s1
                    else:
                        p_sub = s1 + s0
                        sign = -sign
                    if spin[s2] == 'a':
                        h_sub = s2 + s3
                    else:
                        h_sub = s3 + s2
                        sign = -sign
                    ops.append((base, p_sub + h_sub, kind, "ab"))
        if drop:
            continue
        yield coef * sign, ops


def substitute_V(ops):
    """Replace the B pair by a spin-labelled 4-index chemist block unless
    both pairs are all-virtual."""
    bs = [(i, op) for i, op in enumerate(ops) if op[0] == "B"]
    if not bs:
        return ops
    assert len(bs) == 2
    (i1, (_, s1, _, sp1)), (i2, (_, s2, _, sp2)) = bs
    c1 = cls(s1[0]) + cls(s1[1])
    c2 = cls(s2[0]) + cls(s2[1])
    if c1 == "vv" and c2 == "vv":
        return ops
    if c1 == "vo":
        s1, c1 = s1[::-1], "ov"
    if c2 == "vo":
        s2, c2 = s2[::-1], "ov"
    if (c1, sp1) > (c2, sp2):
        s1, s2, c1, c2, sp1, sp2 = s2, s1, c2, c1, sp2, sp1
    vname = f"v_{c1}{c2}_{sp1}{sp2}"
    out = [op for i, op in enumerate(ops) if i not in (i1, i2)]
    out.append((vname, s1 + s2, "V", sp1 + sp2))
    return out


def canonicalise(ops, out_f):
    """Signed canonical key: exploit same-spin doubles antisymmetry, V/B
    pair symmetries and f/d symmetry; deterministic dummy relabelling.
    Returns (key, sign)."""
    ext = set(out_f)

    def variants(op):
        base, sub, kind, sp = op
        outs = {(sub, 1)}
        if kind == "pair4" and sp in ("aa", "bb"):
            outs.add((sub[1] + sub[0] + sub[2:], -1))
            outs.add((sub[:2] + sub[3] + sub[2], -1))
            outs.add((sub[1] + sub[0] + sub[3] + sub[2], 1))
        elif kind == "V":
            c1, c2 = base[2:4], base[4:6]
            sp1, sp2 = sp[0], sp[1]
            if (c1, sp1) == (c2, sp2):
                outs.add((sub[2:] + sub[:2], 1))
            extra = set()
            for s, sg in outs:
                if c1 in ("oo", "vv"):
                    extra.add((s[1] + s[0] + s[2:], sg))
                if c2 in ("oo", "vv"):
                    extra.add((s[:2] + s[3] + s[2], sg))
                if c1 in ("oo", "vv") and c2 in ("oo", "vv"):
                    extra.add((s[1] + s[0] + s[3] + s[2], sg))
            outs |= extra
        elif base in ("f", "d") or kind == "B":
            outs.add((sub[::-1], 1))
        name = base if kind in ("scalar", "V") else f"{base}_{sp}" \
            if sp else base
        return [((name, s, kind, sp), sg) for s, sg in sorted(outs)]

    def relabel(ops_l):
        omap, vmap = {}, {}
        opool = [c for c in "klmnopij" if c not in ext]
        vpool = [c for c in "cdefghab" if c not in ext]
        for (_, sub, _, _) in ops_l:
            for ch in sub:
                if ch in ext:
                    continue
                if ch in OCC and ch not in omap:
                    omap[ch] = opool[len(omap)]
                elif ch in VIR and ch not in vmap:
                    vmap[ch] = vpool[len(vmap)]

        def m(ch):
            if ch in ext:
                return ch
            return omap.get(ch) or vmap.get(ch) or ch
        return [(n, "".join(m(c) for c in s), k, sp)
                for n, s, k, sp in ops_l]

    var_lists = [variants(op) for op in ops]
    n_comb = 1
    for v in var_lists:
        n_comb *= len(v)
    best = None
    best_sign = 1
    if n_comb > 256:
        var_lists = [[vl[0]] for vl in var_lists]
    for combo in itertools.product(*var_lists):
        sign = 1
        ops_l = []
        for op, sg in combo:
            sign *= sg
            ops_l.append(op)
        ops_l = sorted(ops_l)
        for _ in range(3):
            ops_l = sorted(relabel(ops_l))
        key = tuple((n, s, sp) for n, s, k, sp in ops_l)
        if best is None or key < best:
            best = key
            best_sign = sign
    return best, best_sign


def trace_file_blocks(path, target, out_len):
    """Return {block_suffix: [(coef, [(name, sub, kind, spinpat)...])]}."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    parsed = []
    for ln in lines:
        m = LINE_RE.match(ln)
        if not m:
            raise ValueError(f"unparsed: {ln}")
        tgt, coef, subs, out, names = m.groups()
        assert tgt == target
        parsed.append((float(coef), subs.split(","), out,
                       [n.strip() for n in names.split(",")]))

    result = {}
    blocks = [("", {})] if out_len == 0 else BLOCKS[str(out_len)]
    for suffix, spin_by_pos in blocks:
        merged = defaultdict(float)
        opmeta = {}
        for coef, subs, out, names in parsed:
            out_f = strip_boson(out)
            out_spin = {out_f[pos]: sp for pos, sp in spin_by_pos.items()}
            for c, ops in trace_term_blocks(coef, subs, out, names,
                                            out_spin):
                ops2 = substitute_V(ops)
                key, sign = canonicalise(ops2, out_f)
                merged[key] += c * sign
                for (n, s, sp) in key:
                    opmeta[n] = sp
        terms = [(c, list(k)) for k, c in merged.items()
                 if abs(c) > 1e-12]
        result[suffix] = terms
    return result
