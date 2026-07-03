"""Spin-trace the spin-orbital QED-CCSD equations (eqs/*.py.txt) to
closed-shell spatial-orbital form.

Closed-shell parametrization (RHF reference, interleaved even=alpha spin
ordering of qed_ccsd_new):

* 2-index amplitudes/tensors are spin-diagonal:
      X_so[p sigma, q tau] = X[P, Q] * delta(sigma, tau)
* doubles-type amplitudes use the mixed-spin block as the spatial tensor,
      T[a, b, i, j] := t2_so[a alpha, b beta, i alpha, j beta],
  with the exact expansion (pair slots (0,2) and (1,3)):
      t2_so[A s, B t, I m, J n] = T[A,B,I,J] d(s,m) d(t,n)
                                 - T[A,B,J,I] d(s,n) d(t,m)
  which requires (and preserves) T[a,b,i,j] = T[b,a,j,i].

Each spin-orbital term is expanded over the doubles pieces, the spin
delta-constraints are resolved with union-find, inconsistent pieces drop,
and every free spin group contributes a factor 2. The residuals produced
are exactly the alpha (singles) / alpha-beta (doubles) blocks of the
spin-orbital residuals, so the Jacobi/DIIS update and denominators carry
over unchanged.

Sanity anchor: tracing the energy term 0.5*B(xia)B(xjb)t2(baji) yields
2*(ia|jb)T[a,b,i,j] - (ia|jb)T[a,b,j,i] — the textbook closed-shell pair
energy.
"""
import re
from collections import defaultdict

OCC = set("ijklmnop")
VIR = set("abcdefgh")
BOS = set("IJKLMNOP")

# tensor name -> (spatial array name, boson index count, kind)
# kind: 'diag2' (spin-diagonal 2-index), 'pair4' (doubles-type),
#       'scalar', 'B'
TENSORS = {
    "f": ("f", 0, "diag2"),
    "d": ("d", 1, "diag2"),
    "B": ("B", 0, "B"),
    "w": ("w", 2, "scalar"),
    "t1": ("t1_10", 0, "diag2"),
    "s1": ("t1_01", 1, "scalar"),
    "t2": ("t2_20", 0, "pair4"),
    "s2": ("t2_02", 2, "scalar"),
    "u11": ("t2_11", 1, "diag2"),
    "u21": ("t2_21", 1, "pair4"),
    "u12": ("t2_12", 2, "diag2"),
    "u22": ("t2_22", 2, "pair4"),
}

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


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def trace_term(coef, subs, out, names):
    """Expand one spin-orbital term into spatial terms.

    Returns a list of (coefficient, [(array_name, sub), ...]) with the
    boson indices dropped and every doubles-type operand replaced by its
    mixed-spin spatial block."""
    out_f = strip_boson(out)

    # collect operands: (array, fermion_sub, kind, aux_char_or_None)
    base_ops = []
    for name, sub in zip(names, subs):
        arr, _nb, kind = TENSORS[name]
        if kind == "scalar":
            base_ops.append((arr, "", "scalar", None))
        elif kind == "B":
            base_ops.append((arr, sub[1:], "B", sub[0]))
        else:
            base_ops.append((arr, strip_boson(sub), kind, None))

    # expand doubles-type operands into their two spin pieces
    pieces = [(1.0, [], [])]   # (sign, ops, deltas)
    for arr, fsub, kind, aux in base_ops:
        new = []
        for sign, ops, deltas in pieces:
            if kind == "pair4":
                s0, s1, s2, s3 = fsub
                new.append((sign,
                            ops + [(arr, fsub, kind, aux)],
                            deltas + [(s0, s2), (s1, s3)]))
                new.append((-sign,
                            ops + [(arr, s0 + s1 + s3 + s2, kind, aux)],
                            deltas + [(s0, s3), (s1, s2)]))
            elif kind == "diag2":
                new.append((sign, ops + [(arr, fsub, kind, aux)],
                            deltas + ([(fsub[0], fsub[1])] if fsub else [])))
            elif kind == "B":
                new.append((sign, ops + [(arr, fsub, kind, aux)],
                            deltas + [(fsub[0], fsub[1])]))
            else:
                new.append((sign, ops + [(arr, fsub, kind, aux)], deltas))
        pieces = new

    spatial_terms = []
    for sign, ops, deltas in pieces:
        uf = _UF()
        letters = set()
        for _, fsub, _, _ in ops:
            letters.update(fsub)
        for a, b in deltas:
            uf.union(a, b)
        # external spin assignment
        spin = {}   # root -> 'a'/'b'
        ok = True
        if len(out_f) == 2:            # singles block: both alpha
            uf.union(out_f[0], out_f[1])
            spin[uf.find(out_f[0])] = 'a'
        elif len(out_f) == 4:          # doubles mixed block: a,i alpha; b,j beta
            uf.union(out_f[0], out_f[2])
            uf.union(out_f[1], out_f[3])
            ra = uf.find(out_f[0])
            rb = uf.find(out_f[1])
            if ra == rb:
                ok = False
            else:
                spin[ra] = 'a'
                spin[rb] = 'b'
        if not ok:
            continue
        # recheck: unions after assignment can't change roots now (all
        # unions already applied); count free groups
        roots = {uf.find(x) for x in letters}
        nfree = sum(1 for r in roots if r not in spin)
        c = coef * sign * (2.0 ** nfree)
        spatial_terms.append((c, [(arr, fsub, kind) for arr, fsub, kind, _
                                  in ops]))
    return spatial_terms, out_f


def substitute_V(term_ops):
    """Replace a pair of B operands by a precomputed 4-index chemist block
    v_c1c2[(p q | r s)] unless both pairs are all-virtual (those stay in DF
    form for the ladder / cheap x-paths)."""
    bs = [(i, op) for i, op in enumerate(term_ops) if op[0] == "B"]
    if not bs:
        return term_ops, None
    assert len(bs) == 2
    (i1, (_, s1, _)), (i2, (_, s2, _)) = bs
    c1 = cls(s1[0]) + cls(s1[1])
    c2 = cls(s2[0]) + cls(s2[1])
    if c1 == "vv" and c2 == "vv":
        return term_ops, None
    # canonicalise: within-pair 'vo' -> 'ov' (B symmetric), order pairs
    if c1 == "vo":
        s1, c1 = s1[::-1], "ov"
    if c2 == "vo":
        s2, c2 = s2[::-1], "ov"
    if c1 > c2:
        s1, s2, c1, c2 = s2, s1, c2, c1
    vname = f"v_{c1}{c2}"
    ops = [op for i, op in enumerate(term_ops) if i not in (i1, i2)]
    ops.append((vname, s1 + s2, "V"))
    return ops, vname


def canonicalise(coef, ops, out_f):
    """Canonical key for merging equivalent spatial terms: exploit the
    pair-swap symmetry of doubles amplitudes, the pair symmetries of the
    V blocks and f/d symmetry, and relabel dummy indices deterministically."""
    ext = set(out_f)

    def variants(op):
        arr, sub, kind = op
        outs = {(arr, sub)}
        if kind == "pair4":
            outs.add((arr, sub[1] + sub[0] + sub[3] + sub[2]))
        elif arr.startswith("v_"):
            c1, c2 = arr[2:4], arr[4:6]
            if c1 == c2:
                outs.add((arr, sub[2:] + sub[:2]))            # pair swap
            extra = set()
            for a, s in outs:
                if c1 in ("oo", "vv"):
                    extra.add((a, s[1] + s[0] + s[2:]))        # (pq|=(qp|
                if c2 in ("oo", "vv"):
                    extra.add((a, s[:2] + s[3] + s[2]))
                if c1 in ("oo", "vv") and c2 in ("oo", "vv"):
                    extra.add((a, s[1] + s[0] + s[3] + s[2]))
            outs |= extra
        elif arr in ("f", "d") and kind == "diag2":
            outs.add((arr, sub[::-1]))                        # symmetric
        elif kind == "B":
            outs.add((arr, sub[::-1]))                        # B_xpq = B_xqp
        return [(arr, s) for a, s in sorted(outs)]

    def relabel(ops_l):
        omap, vmap = {}, {}
        opool = [c for c in "klmnopij" if c not in ext]
        vpool = [c for c in "cdefghab" if c not in ext]
        for _, sub in ops_l:
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
        return [(a, "".join(m(c) for c in s)) for a, s in ops_l]

    # greedy canonical: pick per-operand variant minimising the whole key,
    # then iterate relabel+sort to a fixed point (bounded)
    cur = [(a, s) for a, s, _ in ops]
    kinds = {a: k for a, s, k in ops}
    best = None
    import itertools
    var_lists = [variants(op) for op in ops]
    n_comb = 1
    for v in var_lists:
        n_comb *= len(v)
    if n_comb <= 64:
        combos = itertools.product(*var_lists)
    else:   # too many symmetry variants — just use the original
        combos = [tuple((a, s) for a, s, _ in ops)]
    for combo in combos:
        ops_l = sorted(combo)
        for _ in range(3):
            ops_l = sorted(relabel(ops_l))
        key = tuple(ops_l)
        if best is None or key < best:
            best = key
    return best, kinds


def trace_file(path, target):
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    merged = defaultdict(float)
    kindmap = {}
    out_sub = None
    for ln in lines:
        m = LINE_RE.match(ln)
        if not m:
            raise ValueError(f"unparsed: {ln}")
        tgt, coef, subs, out, names = m.groups()
        assert tgt == target
        names = [n.strip() for n in names.split(",")]
        subs = subs.split(",")
        spatial, out_f = trace_term(float(coef), subs, out, names)
        out_sub = out_f
        for c, ops in spatial:
            ops2, _ = substitute_V(ops)
            key, kinds = canonicalise(c, ops2, out_f)
            merged[key] += c
            kindmap.update(kinds)
    terms = [(c, list(k)) for k, c in merged.items() if abs(c) > 1e-12]
    return terms, out_sub, kindmap
