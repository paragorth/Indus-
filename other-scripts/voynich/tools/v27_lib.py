"""v27 THE LINES WERE CARDS, AND THE DECK WAS SHUFFLED: shared library.

Every line is a card. A continuity scorer gives each ordered card pair (i -> j) a score from
  J  junction: log P(first word of j | last word of i) - log P(first word of j), a word bigram model trained
     ONLY on within-line adjacent word pairs (never on line-to-line pairs, so the written line order is not
     used to fit it), backed off to the last glyph of the previous word and the unigram;
  L  lexical: idf-weighted shared word types + half weight for near-identical types (edit distance 1, len>=4).
  S  = J + L.
Paragraph sequencing: the head line is fixed as start, the body lines are cards; the best path is exact
(Held-Karp) up to 14 body lines, simulated annealing beyond (C, tools/v27_seq.c).
"""
import os, sys, math, random, json, ctypes, subprocess
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
import v21_lib as V21

LOOPS = os.path.join(vlib.ROOT, 'loops')
CK = os.path.join(vlib.DATA, 'v27_ckpt'); os.makedirs(CK, exist_ok=True)
SO = os.path.join(CK, 'libv27.so')
if not os.path.exists(SO):
    subprocess.check_call(['gcc', '-O3', '-march=native', '-shared', '-fPIC', '-o', SO,
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v27_seq.c'), '-lm'])
_lib = ctypes.CDLL(SO)
_lib.solve_path.restype = ctypes.c_double
_lib.solve_path.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_void_p]
_lib.global_path.restype = ctypes.c_double
_lib.global_path.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_long, ctypes.c_double, ctypes.c_double,
                             ctypes.c_ulonglong, ctypes.c_void_p]


# ------------------------------------------------------------------ corpora
def voynich_pages(name='ZL3b', minw=40):
    """v21 pages plus a parallel 'ids' structure (folio, line number) for cross-transcription matching."""
    lines = vlib.load_voynich(name, drop_uncertain=True)
    pages, order, cur, curi = {}, [], None, None
    for L in lines:
        f = L['folio']
        if f not in pages:
            sec = (L.get('illus') or 'x') + (L.get('lang') or 'x')
            if sec not in V21.MAIN_SECS: sec = 'other'
            pages[f] = {'id': f, 'sec': sec, 'paras': [], 'ids': [], 'quire': L.get('quire')}
            order.append(f); cur = None
        pg = pages[f]
        ws = [V21.U(w) for w in L['words']]
        ws = [w for w in ws if w and '?' not in w]
        if not ws: continue
        if L['para_start'] or cur is None:
            cur = []; curi = []; pg['paras'].append(cur); pg['ids'].append(curi)
        cur.append(ws); curi.append((f, L['n']))
        if L.get('para_end'): cur = None
    out = [pages[f] for f in order]
    return [p for p in out if sum(len(l) for pa in p['paras'] for l in pa) >= minw]


def corpus(name):
    if name == 'ZL': return voynich_pages('ZL3b')
    if name == 'IT2a': return voynich_pages('IT2a')
    if name == 'LA': return V21.latin_herbal()
    if name == 'IT': return V21.italian_herbal()
    raise KeyError(name)


def word_shuffle(C, rng, keep_ends=False):
    out = []
    for p in C:
        q = dict(p); q['paras'] = []
        for pa in p['paras']:
            npa = []
            for l in pa:
                l = list(l)
                if keep_ends and len(l) > 3:
                    mid = l[1:-1]; rng.shuffle(mid); l = [l[0]] + mid + [l[-1]]
                elif not keep_ends:
                    rng.shuffle(l)
                npa.append(l)
            q['paras'].append(npa)
        out.append(q)
    return out


def forge(C, kind, seed):
    rng = random.Random(seed)
    if kind == 'F3': f = V21.Forger(C, scope='sec', pos=True, name='F3')
    elif kind == 'F5': f = V21.Forger(C, scope='sec', pos=True, lam=0.3, name='F5')
    else: raise KeyError(kind)
    return f.forge(C, rng), f


def plant(C, seed, rho_j=0.0, rho_l=0.0):
    """F3 resynthesis (independent lines) with a hidden line order planted: with prob rho_j the first word of
    line k>0 is redrawn from the WITHIN-LINE junction table keyed on the last unit of line k-1 (the line
    continues the previous one); with prob rho_l a word of line k-1 is copied into a random slot of line k."""
    G, f = forge(C, 'F3', seed)
    rng = random.Random(seed + 7)
    for p in G:
        s = p['sec']
        for pa in p['paras']:
            for k in range(1, len(pa)):
                if rho_j and rng.random() < rho_j:
                    key = pa[k - 1][-1][-1]
                    tab = f._get(('j', s, 'p1', key), ('j', s, 'any', key))
                    if tab: pa[k][0] = V21._draw(rng, tab)
                if rho_l and rng.random() < rho_l:
                    pa[k][rng.randrange(len(pa[k]))] = rng.choice(pa[k - 1])
    return G


def cross_page(C, seed):
    """NEGATIVE control / null: every paragraph keeps its head line and size, but each body line is replaced by a
    body line from a DIFFERENT page of the same section (drawn without replacement): no card follows another."""
    rng = random.Random(seed)
    pool = defaultdict(list)
    for pi, p in enumerate(C):
        for pa in p['paras']:
            for l in pa[1:]:
                pool[p['sec']].append((pi, l))
    for s in pool: rng.shuffle(pool[s])
    out = []
    for pi, p in enumerate(C):
        q = dict(p); q['paras'] = []
        for pa in p['paras']:
            npa = [pa[0]]
            for _ in pa[1:]:
                P = pool[p['sec']]
                for t in range(len(P) - 1, -1, -1):
                    if P[t][0] != pi: npa.append(P.pop(t)[1]); break
                else:
                    npa.append(P.pop()[1])
            q['paras'].append(npa)
        out.append(q)
    return out


def independent_lines(C, seed, plen=8):
    """NEGATIVE control: fake paragraphs whose lines come from different pages (no line follows another)."""
    rng = random.Random(seed)
    pool = []
    for pi, p in enumerate(C):
        for pa in p['paras']:
            for l in pa[1:-1]:
                pool.append((pi, l))
    rng.shuffle(pool)
    out, cur, used = [], [], set()
    pages = []
    for pi, l in pool:
        if pi in used: continue
        cur.append(l); used.add(pi)
        if len(cur) == plen:
            pages.append({'id': 'neg%d' % len(pages), 'sec': 'neg', 'paras': [cur]}); cur = []; used = set()
    return pages


# ------------------------------------------------------------------ scorer
def _variants(w):
    return {w[:i] + '*' + w[i + 1:] for i in range(len(w))} | {w[:i] + '+' + w[i:] for i in range(len(w) + 1)}


class Scorer:
    def __init__(self, C, kappa=2.0, kappa2=20.0):
        big = Counter(); ctx = Counter(); g = defaultdict(Counter); gc = Counter(); uni = Counter()
        df = Counter(); nl = 0
        for p in C:
            for pa in p['paras']:
                for l in pa:
                    nl += 1
                    for t in set(l): df[t] += 1
                    for a, b in zip(l, l[1:]):
                        big[(a, b)] += 1; ctx[a] += 1; g[a[-1]][b] += 1; gc[a[-1]] += 1; uni[b] += 1
        self.big, self.ctx, self.g, self.gc, self.uni = big, ctx, g, gc, uni
        self.V = len(set(uni) | set(df)) + 1
        self.Nu = sum(uni.values())
        self.k1, self.k2 = kappa, kappa2
        self.idf = {w: math.log(nl / c) for w, c in df.items()}
        self.nl = nl
        # near-identical type pairs (edit distance 1 through a shared single-edit variant), len >= 4
        vix = defaultdict(set)
        for w in df:
            if len(w) >= 4:
                for v in _variants(w): vix[v].add(w)
        near = defaultdict(set)
        for v, ws in vix.items():
            if 1 < len(ws) < 30:
                for a in ws:
                    near[a] |= ws - {a}
        self.near = near
        self._jc = {}

    def pu(self, w):
        return (self.uni.get(w, 0) + 0.5) / (self.Nu + 0.5 * self.V)

    def J(self, a, b):
        key = (a, b)
        if key in self._jc: return self._jc[key]
        pu = self.pu(b)
        ga = a[-1]
        pg = (self.g[ga].get(b, 0) + self.k2 * pu) / (self.gc.get(ga, 0) + self.k2)
        pb = (self.big.get((a, b), 0) + self.k1 * pg) / (self.ctx.get(a, 0) + self.k1)
        v = math.log(pb) - math.log(pu)
        self._jc[key] = v
        return v

    def L(self, la, lb):
        sa, sb = set(la), set(lb)
        v = sum(self.idf.get(w, 0) for w in sa & sb)
        for w in sa - sb:
            nb = self.near.get(w)
            if nb:
                for x in nb & sb:
                    v += 0.5 * 0.5 * (self.idf.get(w, 0) + self.idf.get(x, 0)); break
        return v

    def matrix(self, lines, kind='S'):
        n = len(lines)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j: continue
                v = 0.0
                if kind in ('J', 'S'): v += self.J(lines[i][-1], lines[j][0])
                if kind in ('L', 'S'): v += self.L(lines[i], lines[j])
                M[i, j] = v
        return M


# ------------------------------------------------------------------ sequencing
def best_path(M, seed=0, restarts=6, iters=60000):
    n = M.shape[0]
    M = np.ascontiguousarray(M, dtype=np.float64)
    out = np.zeros(n, dtype=np.int32)
    s = _lib.solve_path(n, M.ctypes.data, restarts, iters, seed, out.ctypes.data)
    return s, out.tolist()


def path_score(M, p):
    return float(sum(M[p[i], p[i + 1]] for i in range(len(p) - 1)))


def rand_stats(M, rng, nr=200):
    n = M.shape[0]
    sc = []; body = list(range(1, n))
    for _ in range(nr):
        rng.shuffle(body); p = [0] + body
        sc.append(path_score(M, p))
    return float(np.mean(sc)), float(np.var(sc))


def units(C, minlines=3):
    U = []
    for p in C:
        for qi, pa in enumerate(p['paras']):
            if len(pa) >= minlines:
                U.append((p['id'], qi, pa))
    return U


def sequence_corpus(C, sc, kind='S', seed=0, nr=200, restarts=6, iters=60000):
    """Per paragraph: written score, random mean/var, best score, adjacency recovery (best path vs written),
    random-path adjacency expectation, mutual-best rate."""
    rng = random.Random(seed)
    R = []
    for k, (pid, qi, pa) in enumerate(units(C)):
        M = sc.matrix(pa, kind)
        n = len(pa)
        w = path_score(M, list(range(n)))
        mu, var = rand_stats(M, rng, nr)
        b, p = best_path(M, seed + k, restarts, iters)
        we = {(i, i + 1) for i in range(n - 1)}
        be = {(p[i], p[i + 1]) for i in range(n - 1)}
        # body-only mutual best: i's best successor j (j != 0) has i as its best predecessor
        mb = 0; nb = 0
        Mm = M.copy(); np.fill_diagonal(Mm, -1e9); Mm[:, 0] = -1e9
        for i in range(n):
            j = int(np.argmax(Mm[i]))
            nb += 1
            if int(np.argmax(Mm[:, j])) == i: mb += 1
        ub = float(Mm[0].max() + sum(sorted([Mm[i].max() for i in range(1, n)])[1:]))  # path upper bound
        R.append(dict(pid=pid, qi=qi, n=n, w=w, mu=mu, var=var, best=b, path=p, ub=ub,
                      adj=len(we & be), mb=mb, nb=nb, wnext=sum(1 for i in range(n - 1) if int(np.argmax(Mm[i])) == i + 1)))
    return R


def summarise(R):
    n_edges = sum(r['n'] - 1 for r in R)
    # chance adjacency: written edge (0,1) present with prob 1/(n-1); each body edge (i,i+1) with prob 1/(n-1)
    chance = sum((r['n'] - 1) * (1.0 / (r['n'] - 1)) for r in R)
    Wz = sum(r['w'] - r['mu'] for r in R) / math.sqrt(sum(r['var'] for r in R) + 1e-12)
    Gz = sum(r['best'] - r['mu'] for r in R) / math.sqrt(sum(r['var'] for r in R) + 1e-12)
    gap = sum(r['best'] - r['w'] for r in R) / math.sqrt(sum(r['var'] for r in R) + 1e-12)
    wnext = sum(r['wnext'] for r in R); wnext_ch = sum(1.0 + 1.0 / (r['n'] - 1) for r in R)
    exact = sum(1 for r in R if r['path'] == list(range(r['n'])))
    exact_ch = sum(1.0 / math.factorial(r['n'] - 1) for r in R)
    pe = sum(r['best'] - r['mu'] for r in R) / (sum(r['ub'] - r['mu'] for r in R) + 1e-12)
    return dict(units=len(R), edges=n_edges, PE=round(pe, 4), adj=sum(r['adj'] for r in R), adj_chance=round(chance, 1),
                adj_rate=round(sum(r['adj'] for r in R) / n_edges, 4), adj_rate_chance=round(chance / n_edges, 4),
                Wz=round(Wz, 2), Gz=round(Gz, 2), gapz=round(gap, 2),
                mb_rate=round(sum(r['mb'] for r in R) / sum(r['nb'] for r in R), 4),
                argmax_next=wnext, argmax_next_chance=round(wnext_ch, 1), exact=exact, exact_chance=round(exact_ch, 2))


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def save(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'))


# ------------------------------------------------------------------ global (whole-book) sequencing
def cards(C, heads=False):
    """Body lines (and optionally head lines) as cards with metadata."""
    out = []
    for pi, p in enumerate(C):
        for qi, pa in enumerate(p['paras']):
            for li, l in enumerate(pa):
                if li == 0 and not heads: continue
                cid = p['ids'][qi][li] if 'ids' in p else (p['id'], qi, li)
                out.append(dict(words=l, page=p['id'], pi=pi, sec=p['sec'], qi=qi, li=li, cid=tuple(cid)))
    return out


def global_matrix(sc, K, kind='S'):
    """Dense float32 N x N continuity matrix (row i -> col j), diagonal set very low."""
    import scipy.sparse as sp
    N = len(K)
    M = np.zeros((N, N), dtype=np.float32)
    if kind in ('J', 'S'):
        lw = sorted({c['words'][-1] for c in K}); fw = sorted({c['words'][0] for c in K})
        li = {w: i for i, w in enumerate(lw)}; fi = {w: i for i, w in enumerate(fw)}
        JT = np.array([[sc.J(a, b) for b in fw] for a in lw], dtype=np.float32)
        r = np.array([li[c['words'][-1]] for c in K]); q = np.array([fi[c['words'][0]] for c in K])
        M += JT[r][:, q]
    if kind in ('L', 'S'):
        types = sorted({w for c in K for w in c['words']}); ti = {w: i for i, w in enumerate(types)}
        rows, cols = [], []
        for i, c in enumerate(K):
            for w in set(c['words']): rows.append(i); cols.append(ti[w])
        A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, len(types)))
        idf = np.array([sc.idf.get(w, 0.0) for w in types], dtype=np.float32)
        Lx = (A.multiply(idf[None, :]).tocsr() @ A.T).toarray()
        nr, nc, nv = [], [], []
        for w in types:
            for x in sc.near.get(w, ()):
                if x in ti:
                    nr.append(ti[w]); nc.append(ti[x]); nv.append(0.25 * (sc.idf.get(w, 0) + sc.idf.get(x, 0)))
        if nr:
            Nw = sp.csr_matrix((np.array(nv, dtype=np.float32), (nr, nc)), shape=(len(types), len(types)))
            Lx = Lx + (A @ Nw @ A.T).toarray()
        M += Lx.astype(np.float32)
    np.fill_diagonal(M, -1e4)
    return M


def global_solve(M, seed, iters=200_000_000, init=None):
    N = M.shape[0]
    M = np.ascontiguousarray(M, dtype=np.float32)
    off = M[~np.eye(N, dtype=bool)] if N < 2000 else M[np.random.RandomState(seed).randint(0, N, 200000),
                                                        np.random.RandomState(seed + 1).randint(0, N, 200000)]
    sd = float(np.std(off[off > -1e3]))
    p = np.array(init if init is not None else np.random.RandomState(seed).permutation(N), dtype=np.int32)
    s = _lib.global_path(N, M.ctypes.data, iters, 0.5 * sd, 0.002 * sd, seed + 1, p.ctypes.data)
    return s, p.tolist()
