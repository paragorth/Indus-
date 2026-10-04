"""R1 families (a) random sign partitions and (c) random reading transforms.

Both score held-out predictability of a document stream ('|' + line + '|' + line ... '|')
with bigram models trained on split A and scored on split B (stage 1), then trained on A+B and
scored on C (stage 2). Units of the z statistic are documents (pages/tablets/inscriptions).
"""
import math, random
import numpy as np
from collections import Counter
from scipy.stats import norm
from r1_lib import FIX, sign_counts

ALPHA = 0.05


# ---------------- encoding ----------------
class Enc:
    def __init__(self, corpus, min_part=5):
        sc = sign_counts(corpus)
        self.part = sorted([s for s, n in sc.items() if n >= min_part], key=lambda s: -sc[s])
        fixed = ['|', '_', '#', '?', 'RARE']
        self.toks = fixed + self.part
        self.idx = {t: i for i, t in enumerate(self.toks)}
        self.nfix = len(fixed)
        self.V = len(self.toks)
        self.freq = np.array([sc.get(t, 0) for t in self.toks], float)

    def code(self, t):
        return self.idx.get(t, 4)


def doc_stream(doc_lines, enc):
    s = [0]
    for l in doc_lines:
        s.extend(enc.code(t) for t in l)
        s.append(0)
    return s


def pair_arrays(corpus, enc, splits, transform=None):
    """Return (doc_index, prev, next) arrays for all bigrams in docs of the given splits."""
    D, P, N = [], [], []
    di = 0
    for d in corpus['docs']:
        if d['split'] not in splits:
            continue
        lines = d['lines'] if transform is None else transform(d['lines'])
        s = doc_stream(lines, enc)
        a = np.asarray(s, np.int32)
        P.append(a[:-1]); N.append(a[1:]); D.append(np.full(len(a) - 1, di, np.int32))
        di += 1
    if not P:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32), 0
    return np.concatenate(D), np.concatenate(P), np.concatenate(N), di


def z_of(e):
    e = np.asarray(e, float)
    if len(e) < 3 or e.std(ddof=1) == 0:
        return 0.0
    return float(e.mean() / (e.std(ddof=1) / math.sqrt(len(e))))


def z_jack(e_full, e_drops):
    """z for the mean per-document excess, with variance = test-document variance
    + delete-one-group jackknife variance over training documents (training noise is a
    fixed effect shared by all test documents and is otherwise invisible to the page SE)."""
    e_full = np.asarray(e_full, float)
    P = len(e_full)
    if P < 3:
        return 0.0
    m = e_full.mean()
    v_test = e_full.var(ddof=1) / P
    md = np.array([np.mean(e) for e in e_drops])
    G = len(md)
    v_tr = (G - 1) / G * np.sum((md - md.mean()) ** 2)
    v = v_test + v_tr
    return float(m / math.sqrt(v)) if v > 0 else 0.0


def bonf_z(n):
    return float(norm.isf(ALPHA / max(n, 1)))


# ---------------- family (a): partitions ----------------
class PartScorer:
    """Class-bigram held-out gain (bits) over class-unigram, per document."""

    def __init__(self, corpus, enc, train, test, G=8):
        self.enc = enc
        self.G = G
        d, p, n, _ = pair_arrays(corpus, enc, train)
        key = (d.astype(np.int64) % G) * enc.V * enc.V + p.astype(np.int64) * enc.V + n
        u, c = np.unique(key, return_counts=True)
        self.tr_g = (u // (enc.V * enc.V)).astype(np.int32)
        r = u % (enc.V * enc.V)
        self.tr_p, self.tr_n, self.tr_c = (r // enc.V).astype(np.int32), (r % enc.V).astype(np.int32), c.astype(float)
        d, p, n, nd = pair_arrays(corpus, enc, test)
        key = d.astype(np.int64) * enc.V * enc.V + p.astype(np.int64) * enc.V + n
        u, c = np.unique(key, return_counts=True)
        self.te_d = (u // (enc.V * enc.V)).astype(np.int32)
        r = u % (enc.V * enc.V)
        self.te_p, self.te_n, self.te_c = (r // enc.V).astype(np.int32), (r % enc.V).astype(np.int32), c.astype(float)
        self.nd = nd
        self.ntok = np.bincount(self.te_d, weights=self.te_c, minlength=nd)

    def score(self, lab, K, drop=None):
        """lab: class id for each token index (len V); K = number of classes.
        drop: training-document group left out (jackknife over training noise)."""
        wt = self.tr_c if drop is None else self.tr_c * (self.tr_g != drop)
        CA = np.bincount(lab[self.tr_p] * K + lab[self.tr_n], weights=wt, minlength=K * K).reshape(K, K)
        a = 0.5
        cond = (CA + a) / (CA.sum(1, keepdims=True) + a * K)
        uni = (CA.sum(0) + a) / (CA.sum() + a * K)
        W = np.log2(cond) - np.log2(uni)[None, :]
        w = W[lab[self.te_p], lab[self.te_n]] * self.te_c
        return np.bincount(self.te_d, weights=w, minlength=self.nd)


def make_lab(enc, assign, k):
    """assign: array over partitionable signs -> class 0..k-1. Fixed tokens get classes k..k+4."""
    lab = np.empty(enc.V, np.int64)
    lab[:enc.nfix] = k + np.arange(enc.nfix)
    lab[enc.nfix:] = assign
    return lab, k + enc.nfix


def matched_null(assign, rng, bin_size=6):
    """Permute class labels among signs of similar frequency (signs are frequency-sorted)."""
    a = assign.copy()
    for s in range(0, len(a), bin_size):
        seg = a[s:s + bin_size].copy()
        rng.shuffle(seg)
        a[s:s + bin_size] = seg
    return a


def run_partitions(corpus, N, seed, M=12, kmax=30, top2=200, log=None):
    rng = np.random.default_rng(seed)
    enc = Enc(corpus)
    q = enc.V - enc.nfix
    s1 = PartScorer(corpus, enc, ('A',), ('B',))
    hyps = []
    for h in range(N):
        k = int(rng.integers(2, min(kmax, q) + 1))
        assign = rng.integers(0, k, q)
        lab, K = make_lab(enc, assign, k)
        sh = s1.score(lab, K)
        nulls = np.mean([s1.score(make_lab(enc, matched_null(assign, rng), k)[0], K) for _ in range(M)], 0)
        e = (sh - nulls)
        z = z_of(e)
        hyps.append((z, float(e.sum() / s1.ntok.sum()), k, assign.astype(np.int8)))
        if log and h % 2000 == 0:
            log(f'  part {corpus["name"]} {h}/{N}')
    thr1 = bonf_z(N)
    surv = sorted([h for h in hyps if h[0] > thr1], key=lambda x: -x[0])
    s2 = PartScorer(corpus, enc, ('B',), ('C',))
    tested = surv[:top2]
    thr2 = bonf_z(len(tested))
    rep = []
    for z1, g1, k, assign in tested:
        assign = assign.astype(np.int64)
        lab, K = make_lab(enc, assign, k)
        nlabs = [make_lab(enc, matched_null(assign, rng), k)[0] for _ in range(M)]
        def ex(drop):
            return s2.score(lab, K, drop) - np.mean([s2.score(nl, K, drop) for nl in nlabs], 0)
        e = ex(None)
        z2 = z_jack(e, [ex(g) for g in range(s2.G)])
        rep.append({'z1': z1, 'gain1': g1, 'k': k, 'z2': z2, 'z2_naive': z_of(e), 'gain2': float(e.sum() / s2.ntok.sum()),
                    'replicated': bool(z2 > thr2), 'assign': assign.tolist()})
    zs = np.array([h[0] for h in hyps])
    return {'N': N, 'thr1': thr1, 'n_stage1': len(surv), 'n_tested2': len(tested), 'thr2': thr2,
            'n_replicated': sum(r['replicated'] for r in rep), 'z_quantiles': np.quantile(zs, [.5, .9, .99, 1]).tolist(),
            'signs': enc.part, 'stage2': rep}


def planted_partition_corpus(corpus, rng, kstar=3, rho=0.5):
    """Shuffled-skeleton corpus with one hidden rule: signs fall in kstar hidden classes and,
    after a sign of class c, the next sign is of class (c+1) mod kstar with prob rho."""
    enc = Enc(corpus)
    sc = sign_counts(corpus)
    signs = list(sc)
    hidden = {s: int(rng.integers(0, kstar)) for s in signs}
    w = np.array([sc[s] for s in signs], float)
    by_cls = {c: [i for i, s in enumerate(signs) if hidden[s] == c] for c in range(kstar)}
    p_all = w / w.sum()
    p_cls = {c: w[ix] / w[ix].sum() for c, ix in by_cls.items()}
    cdf_all = np.cumsum(p_all)
    cdf_cls = {c: np.cumsum(v) for c, v in p_cls.items()}
    docs = []
    for d in corpus['docs']:
        nl = []
        for l in d['lines']:
            out, prev = [], None
            for t in l:
                if t in FIX:
                    out.append(t); prev = None if t in ('|',) else prev
                    continue
                if prev is not None and rng.random() < rho:
                    c = (hidden[prev] + 1) % kstar
                    i = min(int(np.searchsorted(cdf_cls[c], rng.random() * cdf_cls[c][-1])), len(by_cls[c]) - 1)
                    s = signs[by_cls[c][i]]
                else:
                    s = signs[min(int(np.searchsorted(cdf_all, rng.random() * cdf_all[-1])), len(signs) - 1)]
                out.append(s); prev = s
            nl.append(out)
        nd = dict(d); nd['lines'] = nl
        docs.append(nd)
    return {'name': corpus['name'], 'docs': docs}, hidden


# ---------------- family (c): reading transforms ----------------
def _deint(n, k):
    return [i for r in range(k) for i in range(r, n, k)]


def _inv(perm):
    inv = [0] * len(perm)
    for j, i in enumerate(perm):
        inv[i] = j
    return inv


def op_apply(op, lines):
    kind = op[0]
    if kind == 'revL':
        return [l[::-1] for l in lines]
    if kind == 'revW':
        out = []
        for l in lines:
            o, w = [], []
            for t in l + ['_']:
                if t == '_':
                    o.extend(w[::-1]); o.append('_'); w = []
                else:
                    w.append(t)
            out.append(o[:-1])
        return out
    if kind == 'revWO':
        out = []
        for l in lines:
            ws, w = [], []
            for t in l:
                if t == '_':
                    ws.append(w); w = []
                else:
                    w.append(t)
            ws.append(w)
            o = []
            for i, w in enumerate(ws[::-1]):
                if i:
                    o.append('_')
                o.extend(w)
            out.append(o)
        return out
    if kind in ('deint', 'int'):
        k = op[1]
        out = []
        for l in lines:
            p = _deint(len(l), k)
            if kind == 'int':
                p = _inv(p)
            out.append([l[i] for i in p])
        return out
    if kind == 'rot':
        return [l[op[1] % len(l):] + l[:op[1] % len(l)] if l else l for l in lines]
    if kind == 'swap':
        x, y = op[1], op[2]
        out = []
        for l in lines:
            o, i = list(l), 0
            while i < len(o) - 1:
                if o[i] == x and o[i + 1] == y:
                    o[i], o[i + 1] = y, x; i += 2
                else:
                    i += 1
            out.append(o)
        return out
    if kind in ('ilv', 'dilv'):
        out = []
        for j in range(0, len(lines), 2):
            if j + 1 >= len(lines):
                out.append(lines[j]); continue
            a, b = lines[j], lines[j + 1]
            n1, n2 = len(a), len(b)
            # interleave permutation over the concatenation a+b
            order, ia, ib = [], 0, n1
            while ia < n1 or ib < n1 + n2:
                if ia < n1:
                    order.append(ia); ia += 1
                if ib < n1 + n2:
                    order.append(ib); ib += 1
            cat = a + b
            if kind == 'dilv':
                order = _inv(order)
            m = [cat[i] for i in order]
            out.extend([m[:n1], m[n1:]])
        return out
    if kind == 'revD':
        return lines[::-1]
    raise ValueError(op)


def op_apply_skel(op, lines):
    """Apply op to the sign tokens only; the skeleton of fixed tokens ('_', '#', '?') of every
    line stays where it was (so word lengths and numeral slots never move). revD moves whole lines."""
    if op[0] == 'revD':
        return op_apply(op, lines)
    if op[0] in ('revW', 'revWO'):
        g = [[t for t in l if t not in FIX] for l in op_apply(op, lines)]
    else:
        g = op_apply(op, [[t for t in l if t not in FIX] for l in lines])
    out = []
    for l, gl in zip(lines, g):
        it = iter(gl)
        out.append([t if t in FIX else next(it) for t in l])
    return out


def compose(ops):
    def f(lines):
        for op in ops:
            lines = op_apply_skel(op, lines)
        return lines
    return f


def random_op(rng, top_signs):
    r = rng.random()
    if r < 0.08:
        return ('revL',)
    if r < 0.16:
        return ('revW',)
    if r < 0.22:
        return ('revWO',)
    if r < 0.38:
        return ('deint', int(rng.integers(2, 6)))
    if r < 0.54:
        return ('int', int(rng.integers(2, 6)))
    if r < 0.62:
        return ('rot', int(rng.integers(1, 8)))
    if r < 0.86:
        a, b = rng.choice(len(top_signs), 2, replace=False)
        return ('swap', top_signs[a], top_signs[b])
    if r < 0.93:
        return ('ilv',) if rng.random() < .5 else ('dilv',)
    return ('revD',)


class BigramLL:
    def __init__(self, enc):
        self.enc = enc

    def per_doc(self, corpus, transform, train, test, a=0.1):
        V = self.enc.V
        _, p, n, _ = pair_arrays(corpus, self.enc, train, transform)
        C = np.bincount(p.astype(np.int64) * V + n, minlength=V * V).reshape(V, V).astype(float)
        L = np.log2((C + a) / (C.sum(1, keepdims=True) + a * V))
        d, p, n, nd = pair_arrays(corpus, self.enc, test, transform)
        return np.bincount(d, weights=L[p, n], minlength=nd), len(p)

    def per_doc_jack(self, corpus, transform, train, test, G=8, a=0.1):
        """Per-test-document log-lik for the full training set and for each left-out training group."""
        V = self.enc.V
        dt, p, n, _ = pair_arrays(corpus, self.enc, train, transform)
        key = p.astype(np.int64) * V + n
        d, pp, nn, nd = pair_arrays(corpus, self.enc, test, transform)
        outs = []
        for drop in [None] + list(range(G)):
            m = np.ones(len(key), bool) if drop is None else (dt % G != drop)
            C = np.bincount(key[m], minlength=V * V).reshape(V, V).astype(float)
            L = np.log2((C + a) / (C.sum(1, keepdims=True) + a * V))
            outs.append(np.bincount(d, weights=L[pp, nn], minlength=nd))
        return outs, len(pp)


def run_transforms(corpus, N, seed, top2=200, max_ops=3, log=None):
    rng = np.random.default_rng(seed)
    enc = Enc(corpus)
    top = enc.part[:min(40, len(enc.part))]
    bl = BigramLL(enc)
    base1, ntok1 = bl.per_doc(corpus, None, ('A',), ('B',))
    seen, hyps = {}, []
    tries = 0
    while len(hyps) < N and tries < N * 20:
        tries += 1
        ops = tuple(random_op(rng, top) for _ in range(int(rng.integers(1, max_ops + 1))))
        if ops in seen:
            continue
        seen[ops] = 1
        ll, _ = bl.per_doc(corpus, compose(ops), ('A',), ('B',))
        e = ll - base1
        hyps.append((z_of(e), float(e.sum() / ntok1), ops))
        if log and len(hyps) % 2000 == 0:
            log(f'  tr {corpus["name"]} {len(hyps)}/{N}')
    n = len(hyps)
    thr1 = bonf_z(n)
    surv = sorted([h for h in hyps if h[0] > thr1], key=lambda x: -x[0])
    tested = surv[:top2]
    thr2 = bonf_z(len(tested))
    base2, ntok2 = bl.per_doc_jack(corpus, None, ('B',), ('C',))
    rep = []
    for z1, g1, ops in tested:
        ll, _ = bl.per_doc_jack(corpus, compose(ops), ('B',), ('C',))
        es = [a - b for a, b in zip(ll, base2)]
        z2 = z_jack(es[0], es[1:])
        rep.append({'ops': [list(o) for o in ops], 'z1': z1, 'gain1': g1, 'z2': z2, 'z2_naive': z_of(es[0]),
                    'gain2': float(es[0].sum() / ntok2), 'replicated': bool(z2 > thr2)})
    zs = np.array([h[0] for h in hyps])
    return {'N': n, 'tries': tries, 'thr1': thr1, 'n_stage1': len(surv), 'n_tested2': len(tested), 'thr2': thr2,
            'n_replicated': sum(r['replicated'] for r in rep),
            'z_quantiles': np.quantile(zs, [.5, .9, .99, 1]).tolist(), 'stage2': rep}


def markov_corpus(corpus, rng):
    """Synthetic corpus: same docs/line lengths, tokens drawn from the real token bigram chain."""
    enc = Enc(corpus, min_part=1)
    V = enc.V
    _, p, n, _ = pair_arrays(corpus, enc, ('A', 'B', 'C'))
    C = np.bincount(p.astype(np.int64) * V + n, minlength=V * V).reshape(V, V).astype(float)
    C[:, 0] = 0  # never emit line break inside a line
    rows = C / np.maximum(C.sum(1, keepdims=True), 1)
    cdf = np.cumsum(rows, 1)
    docs = []
    for d in corpus['docs']:
        nl = []
        for l in d['lines']:
            prev, out = 0, []
            for _ in range(len(l)):
                r = rng.random() * cdf[prev, -1]
                if cdf[prev, -1] == 0:
                    nxt = int(rng.integers(enc.nfix, V))
                else:
                    nxt = int(np.searchsorted(cdf[prev], r, side='right'))
                    nxt = min(nxt, V - 1)
                out.append(enc.toks[nxt]); prev = nxt
            nl.append(out)
        nd = dict(d); nd['lines'] = nl
        docs.append(nd)
    return {'name': corpus['name'], 'docs': docs}


PLANT_OPS = ['deint2', 'deint3', 'int2', 'ilv', 'swap']


def planted_transform_corpus(corpus, rng):
    syn = markov_corpus(corpus, rng)
    enc = Enc(syn)
    kind = PLANT_OPS[int(rng.integers(0, len(PLANT_OPS)))]
    if kind == 'deint2':
        op = ('deint', 2)
    elif kind == 'deint3':
        op = ('deint', 3)
    elif kind == 'int2':
        op = ('int', 2)
    elif kind == 'ilv':
        op = ('ilv',)
    else:
        top = enc.part[:10]
        a, b = rng.choice(len(top), 2, replace=False)
        op = ('swap', top[a], top[b])
    docs = []
    for d in syn['docs']:
        nd = dict(d); nd['lines'] = op_apply_skel(op, d['lines']); nd['orig'] = d['lines']
        docs.append(nd)
    return {'name': corpus['name'], 'docs': docs}, op


def restoration(corpus_scr, ops):
    f = compose([tuple(o) for o in ops])
    same = tot = 0
    for d in corpus_scr['docs']:
        r = f(d['lines'])
        for a, b in zip(r, d['orig']):
            tot += len(b)
            same += sum(x == y for x, y in zip(a, b))
    return same / max(tot, 1)


# ---------------- fast numpy transform engine (same semantics as op_apply_skel) ----------------
_PERM = {}


def _perm(kind, a, b=0):
    key = (kind, a, b)
    p = _PERM.get(key)
    if p is None:
        if kind == 'deint':
            p = np.array(_deint(a, b), np.int64) if a else np.zeros(0, np.int64)
        elif kind == 'int':
            p = np.array(_inv(_deint(a, b)), np.int64) if a else np.zeros(0, np.int64)
        else:  # ilv over n1=a, n2=b
            order, ia, ib = [], 0, a
            while ia < a or ib < a + b:
                if ia < a:
                    order.append(ia); ia += 1
                if ib < a + b:
                    order.append(ib); ib += 1
            if kind == 'dilv':
                order = _inv(order)
            p = np.array(order, np.int64)
        _PERM[key] = p
    return p


class FastDocs:
    """Encoded docs: each line = (full codes, glyph positions, word id per glyph)."""

    def __init__(self, corpus, enc, splits):
        self.docs = []
        fixed_codes = set(range(enc.nfix))
        for d in corpus['docs']:
            if d['split'] not in splits:
                continue
            lines = []
            for l in d['lines']:
                full = np.array([enc.code(t) for t in l], np.int32)
                isg = np.array([t not in FIX for t in l], bool)
                wid = np.cumsum(np.array([t == '_' for t in l], int))[isg]
                lines.append((full, np.nonzero(isg)[0], wid))
            self.docs.append(lines)

    def stream_pairs(self, ops):
        D, P, N = [], [], []
        for di, lines in enumerate(self.docs):
            cur = [(f, gp, w, f[gp]) for f, gp, w in lines]
            for op in ops:
                cur = _fast_op(op, cur)
            parts = [np.zeros(1, np.int32)]
            for f, gp, w, g in cur:
                o = f.copy()
                o[gp] = g
                parts.append(o); parts.append(np.zeros(1, np.int32))
            s = np.concatenate(parts)
            P.append(s[:-1]); N.append(s[1:]); D.append(np.full(len(s) - 1, di, np.int32))
        return np.concatenate(D), np.concatenate(P), np.concatenate(N), len(self.docs)


def _fast_op(op, cur):
    k = op[0]
    if k == 'revD':
        return cur[::-1]
    out = []
    if k in ('ilv', 'dilv'):
        for j in range(0, len(cur), 2):
            if j + 1 >= len(cur):
                out.append(cur[j]); continue
            a, b = cur[j], cur[j + 1]
            n1, n2 = len(a[3]), len(b[3])
            m = np.concatenate([a[3], b[3]])[_perm(k, n1, n2)]
            out.append((a[0], a[1], a[2], m[:n1])); out.append((b[0], b[1], b[2], m[n1:]))
        return out
    for f, gp, w, g in cur:
        n = len(g)
        if n == 0:
            out.append((f, gp, w, g)); continue
        if k == 'revL':
            g2 = g[::-1]
        elif k in ('deint', 'int'):
            g2 = g[_perm(k, n, op[1])]
        elif k == 'rot':
            r = op[1] % n
            g2 = np.concatenate([g[r:], g[:r]])
        elif k == 'swap':
            x, y = op[1], op[2]
            g2 = g.copy()
            if n > 1:
                hit = np.nonzero((g[:-1] == x) & (g[1:] == y))[0]
                g2[hit] = y; g2[hit + 1] = x
        elif k == 'revW':
            # reverse glyph order inside each word (word ids are non-decreasing)
            order = np.lexsort((-np.arange(n), w))
            g2 = g[order]
        elif k == 'revWO':
            order = np.lexsort((np.arange(n), -w))
            g2 = g[order]
        else:
            raise ValueError(op)
        out.append((f, gp, w, g2))
    return out


def run_transforms_fast(corpus, N, seed, top2=200, max_ops=3, log=None, G=8, a=0.1):
    rng = np.random.default_rng(seed)
    enc = Enc(corpus)
    top_codes = [enc.idx[s] for s in enc.part[:min(40, len(enc.part))]]
    V = enc.V
    FA, FB, FC = FastDocs(corpus, enc, ('A',)), FastDocs(corpus, enc, ('B',)), FastDocs(corpus, enc, ('C',))

    def ll(train, test, ops, jack=False):
        dt, p, n, _ = train.stream_pairs(ops)
        key = p.astype(np.int64) * V + n
        d, pp, nn, nd = test.stream_pairs(ops)
        res = []
        for drop in ([None] + list(range(G))) if jack else [None]:
            m = slice(None) if drop is None else (dt % G != drop)
            C = np.bincount(key[m], minlength=V * V).reshape(V, V).astype(float)
            Lg = np.log2((C + a) / (C.sum(1, keepdims=True) + a * V))
            res.append(np.bincount(d, weights=Lg[pp, nn], minlength=nd))
        return res, len(pp)

    base1, ntok1 = ll(FA, FB, ())
    seen, hyps, tries = set(), [], 0
    while len(hyps) < N and tries < N * 20:
        tries += 1
        ops = []
        for _ in range(int(rng.integers(1, max_ops + 1))):
            op = random_op(rng, top_codes)
            if op[0] == 'swap':
                op = ('swap', int(op[1]), int(op[2]))
            ops.append(op)
        ops = tuple(ops)
        if ops in seen:
            continue
        seen.add(ops)
        r, _ = ll(FA, FB, ops)
        e = r[0] - base1[0]
        hyps.append((z_of(e), float(e.sum() / ntok1), ops))
        if log and len(hyps) % 2000 == 0:
            log(f'  trf {corpus["name"]} {len(hyps)}/{N}')
    n = len(hyps)
    thr1 = bonf_z(n)
    surv = sorted([h for h in hyps if h[0] > thr1], key=lambda x: -x[0])
    tested = surv[:top2]
    thr2 = bonf_z(len(tested))
    base2, ntok2 = ll(FB, FC, (), jack=True)
    rep = []
    for z1, g1, ops in tested:
        r, _ = ll(FB, FC, ops, jack=True)
        es = [x - y for x, y in zip(r, base2)]
        z2 = z_jack(es[0], es[1:])
        rep.append({'ops': [[o[0]] + [enc.toks[x] if o[0] == 'swap' else x for x in o[1:]] for o in ops],
                    'z1': z1, 'gain1': g1, 'z2': z2, 'z2_naive': z_of(es[0]),
                    'gain2': float(es[0].sum() / ntok2), 'replicated': bool(z2 > thr2)})
    zs = np.array([h[0] for h in hyps])
    return {'N': n, 'tries': tries, 'thr1': thr1, 'n_stage1': len(surv), 'n_tested2': len(tested), 'thr2': thr2,
            'n_replicated': sum(r['replicated'] for r in rep),
            'z_quantiles': np.quantile(zs, [.5, .9, .99, 1]).tolist(), 'stage2': rep}


def run_partitions_climb(corpus, R, seed, M=12, kmax=6, sweeps=8, log=None):
    """Random-restart hill climbing: each restart starts from a random partition (k in 2..kmax)
    and climbs on split A only (train on half of A, score the other half of A). The climbed
    partitions are then treated exactly like random hypotheses: stage 1 on B (Bonferroni over R),
    stage 2 on C (train B, jackknife SE)."""
    rng = np.random.default_rng(seed)
    enc = Enc(corpus)
    q = enc.V - enc.nfix
    inner = {'name': corpus['name'], 'docs': []}
    for i, d in enumerate(corpus['docs']):
        nd = dict(d)
        if d['split'] == 'A':
            nd['split'] = 'A1' if i % 2 == 0 else 'A2'
        inner['docs'].append(nd)
    sA = PartScorer(inner, enc, ('A1',), ('A2',))
    sA2 = PartScorer(inner, enc, ('A2',), ('A1',))
    s1 = PartScorer(corpus, enc, ('A',), ('B',))
    s2 = PartScorer(corpus, enc, ('B',), ('C',))
    res = []
    for r in range(R):
        k = int(rng.integers(2, kmax + 1))
        a = rng.integers(0, k, q)
        lab, K = make_lab(enc, a, k)
        best = sA.score(lab, K).sum() + sA2.score(lab, K).sum()
        for sw in range(sweeps):
            changed = 0
            for i in rng.permutation(q):
                cur = a[i]
                for c in range(k):
                    if c == cur:
                        continue
                    lab[enc.nfix + i] = c
                    v = sA.score(lab, K).sum() + sA2.score(lab, K).sum()
                    if v > best + 1e-9:
                        best, cur, changed = v, c, changed + 1
                lab[enc.nfix + i] = cur
                a[i] = cur
            if not changed:
                break
        def ex(sc, drop=None, nlabs=None):
            return sc.score(lab, K, drop) - np.mean([sc.score(nl, K, drop) for nl in nlabs], 0)
        nl1 = [make_lab(enc, matched_null(a, rng), k)[0] for _ in range(M)]
        e1 = ex(s1, None, nl1)
        res.append({'k': k, 'assign': a.tolist(), 'trainA_gain': float(best), 'z1': z_of(e1),
                    'gain1': float(e1.sum() / s1.ntok.sum())})
        if log and r % 20 == 0:
            log(f'  climb {corpus["name"]} {r}/{R}')
    thr1 = bonf_z(R)
    surv = [x for x in res if x['z1'] > thr1]
    thr2 = bonf_z(len(surv))
    for x in surv:
        a = np.array(x['assign']); lab, K = make_lab(enc, a, x['k'])
        nl2 = [make_lab(enc, matched_null(a, rng), x['k'])[0] for _ in range(M)]
        def ex2(drop):
            return s2.score(lab, K, drop) - np.mean([s2.score(nl, K, drop) for nl in nl2], 0)
        e = ex2(None)
        x['z2'] = z_jack(e, [ex2(g) for g in range(s2.G)])
        x['gain2'] = float(e.sum() / s2.ntok.sum())
        x['replicated'] = bool(x['z2'] > thr2)
    return {'N': R, 'thr1': thr1, 'n_stage1': len(surv), 'thr2': thr2,
            'n_replicated': sum(x.get('replicated', False) for x in res), 'signs': enc.part,
            'z_quantiles': np.quantile([x['z1'] for x in res], [.5, .9, .99, 1]).tolist(),
            'stage2': sorted(res, key=lambda x: -x['z1'])}
