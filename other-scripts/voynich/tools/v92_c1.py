"""v92 cycle 1 (K1): NAMES HAVE A SLOT.
Event = a token whose unit (word or frame) is rare in the corpus (count lo..hi); label y = the unit also occurs on
another line of the same page. Baseline logistic model: y ~ section/hand/language group + the token's own spelling
(first glyph, last glyph, length bin) + count bin + page-size bin. Hypothesis = a context definition (1-2 atoms:
neighbour at offset d with a property, word position, line role, line index; optionally crossed). Gain = held-out
bits per event of baseline+context over baseline. The kill is built in: the same hypothesis on within-page-shuffled
twins (page recurrence kept exactly, syntax destroyed). Train folios (leaf parity 0): 2-fold page CV, score = gain
minus twin gain. Top 20 re-tested once on held-out folios against 8 twins (z)."""
import os, sys, json, time, math, random
from collections import Counter, defaultdict
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v92_lib as L

NH = int(os.environ.get('V92_NH', '1500'))
TOP = 20
NTW = 8
PROPS = ['w20', 'w50', 'w150', 'f1', 'l1', 'len', 'rare', 'f1l1', 'fr50']
ATOMS = [('nb', d, pr) for d in (-2, -1, 1, 2) for pr in PROPS] + [('wpos',), ('lrole',), ('lidx',), ('llen',)]


def lenbin(n): return min(n, 7) if n >= 3 else 2


class Ev:
    """token table for one corpus (all tokens), atoms precomputed."""
    def __init__(self, pages):
        W, pg, ln, pos, nl, half, grp, prole, lidx = [], [], [], [], [], [], [], [], []
        gk = {}
        L_ = 0
        self.page_tok = []
        for pi, p in enumerate(pages):
            h = L.split_half(p['id']); g = gk.setdefault('%s|%s|%s' % (p['sec'], p.get('lang', '-'), p.get('hand', '-')), len(gk))
            ntok = 0
            for li, l in enumerate(p['lines']):
                n = len(l['w'])
                role = 0 if li == 0 else (1 if l['ps'] else 2)
                for i, w in enumerate(l['w']):
                    W.append(w); pg.append(pi); ln.append(L_); pos.append(i); nl.append(n); half.append(h); grp.append(g)
                    prole.append(role); lidx.append(min(li, 12)); ntok += 1
                L_ += 1
            self.page_tok.append(ntok)
        self.W = W; self.n = len(W)
        self.pg = np.array(pg); self.ln = np.array(ln); self.pos = np.array(pos); self.nl = np.array(nl)
        self.half = np.array(half); self.grp = np.array(grp); self.role = np.array(prole); self.lidx = np.array(lidx)
        pt = np.array(self.page_tok); q = np.quantile(pt, [1 / 3, 2 / 3])
        self.pbin = np.digitize(pt[self.pg], q)
        self.f1 = self._codes([w[1] if (w[0] == 'q' and len(w) > 1) else w[0] for w in W])
        self.l1 = self._codes([w[-1] for w in W])
        self.lb = np.array([lenbin(len(w)) for w in W])
        self.units = {}
        for un, fn in (('word', lambda w: w), ('frame', L.frame)):
            U = [fn(w) for w in W]
            c = Counter(U)
            uid = self._codes(U)
            cnt = np.array([c[u] for u in U])
            # y: unit on another line of the same page; dbl: neighbour +-1 in line has the same unit
            key = Counter(); seen = defaultdict(set)
            for t in range(self.n): seen[(self.pg[t], U[t])].add(self.ln[t])
            y = np.array([len(seen[(self.pg[t], U[t])] - {self.ln[t]}) > 0 for t in range(self.n)])
            dbl = np.zeros(self.n, bool)
            for t in range(self.n):
                for d in (-1, 1):
                    s = t + d
                    if 0 <= s < self.n and self.ln[s] == self.ln[t] and U[s] == U[t]: dbl[t] = True
            self.units[un] = dict(U=U, cnt=cnt, y=y.astype(int), dbl=dbl)
        # neighbour word index per offset (-1 = boundary)
        self.nb = {}
        for d in (-2, -1, 1, 2):
            idx = np.arange(self.n) + d
            ok = (idx >= 0) & (idx < self.n)
            idx2 = np.where(ok, idx, 0)
            ok &= self.ln[idx2] == self.ln
            self.nb[d] = np.where(ok, idx2, -1)
        wc = Counter(W); top = [w for w, _ in wc.most_common(150)]
        self.rank = {w: i for i, w in enumerate(top)}
        fc = Counter(L.frame(w) for w in W); self.frank = {f: i for i, f in enumerate([f for f, _ in fc.most_common(50)])}
        self.rk = np.array([self.rank.get(w, 150) for w in W])
        self.frk = np.array([self.frank.get(L.frame(w), 50) for w in W])
        self.atom_cache = {}

    @staticmethod
    def _codes(xs):
        d = {}
        return np.array([d.setdefault(x, len(d)) for x in xs])

    def atom(self, a, un, hi):
        key = (a, un, hi if a[0] == 'nb' and a[2] == 'rare' else None)
        if key in self.atom_cache: return self.atom_cache[key]
        if a[0] == 'wpos':
            v = np.where(self.pos == self.nl - 1, 9, np.minimum(self.pos, 3))
        elif a[0] == 'lrole':
            v = self.role.copy()
        elif a[0] == 'lidx':
            v = np.digitize(self.lidx, [1, 2, 4, 7, 10])
        elif a[0] == 'llen':
            v = np.digitize(self.nl, [5, 7, 9, 11])
        else:
            _, d, pr = a
            nbi = self.nb[d]
            if pr in ('w20', 'w50', 'w150'):
                K = int(pr[1:]); prop = np.minimum(self.rk, K)
            elif pr == 'f1': prop = self.f1
            elif pr == 'l1': prop = self.l1
            elif pr == 'len': prop = self.lb
            elif pr == 'rare': prop = (self.units[un]['cnt'] <= hi).astype(int)
            elif pr == 'f1l1': prop = self.f1 * 1000 + self.l1
            elif pr == 'fr50': prop = self.frk
            v = np.where(nbi >= 0, prop[np.maximum(nbi, 0)], -1)
        self.atom_cache[key] = v
        return v


def onehot(cols, rows_fit, rows_all, minc=5):
    """cols: list of int arrays over all tokens; returns csr over rows_all; categories rare in rows_fit merged."""
    mats = []
    for c in cols:
        cf = Counter(c[rows_fit].tolist())
        keep = {k for k, v in cf.items() if v >= minc}
        m = {k: i + 1 for i, k in enumerate(sorted(keep))}
        x = np.array([m.get(k, 0) for k in c[rows_all].tolist()])
        mats.append(sp.csr_matrix((np.ones(len(x)), (np.arange(len(x)), x)), shape=(len(x), len(m) + 1)))
    return sp.hstack(mats).tocsr()


def _base_fit(E, un, fi, ei, y, base):
    allr = np.concatenate([fi, ei]); nf = len(fi)
    X = onehot(base, allr[:nf], allr)
    m = LogisticRegression(C=0.05, max_iter=300)
    m.fit(X[:nf], y[fi])
    return m.decision_function(X[:nf]), m.decision_function(X[nf:])


def _offsets(cols, fi, ei, y, z_f, z_e, prior=20.0, passes=3):
    """backfitted one-step logistic offsets per context category (shrunk by prior pseudo-weight)."""
    zf = z_f.copy(); ze = z_e.copy()
    codes = []
    for c in cols:
        cf = c[fi]; ce = c[ei]
        u, inv = np.unique(np.concatenate([cf, ce]), return_inverse=True)
        codes.append((inv[:len(fi)], inv[len(fi):], len(u), np.zeros(len(u))))
    yf = y[fi].astype(float)
    for _ in range(passes):
        for k, (a, b, K, off) in enumerate(codes):
            p = 1 / (1 + np.exp(-zf))
            g = np.bincount(a, weights=yf - p, minlength=K)
            h = np.bincount(a, weights=p * (1 - p), minlength=K) + prior * 0.1
            d = g / h
            off += d; zf += d[a]; ze += d[b]
    return ze


def loglik_gain(E, un, lo, hi, ctx_cols, fit_mask, ev_mask, base_cache=None):
    U = E.units[un]
    sel = (U['cnt'] >= lo) & (U['cnt'] <= hi) & ~U['dbl']
    fi = np.where(sel & fit_mask)[0]; ei = np.where(sel & ev_mask)[0]
    if len(fi) < 200 or len(ei) < 100 or U['y'][fi].sum() < 20: return None
    y = U['y']
    cb = np.digitize(U['cnt'], [3, 5, 9, 16])
    base = [E.grp, E.f1, E.l1, E.lb, cb, E.pbin]
    k = (un, lo, hi, hash(fit_mask.tobytes()), hash(ev_mask.tobytes()))
    if base_cache is not None and k in base_cache: z_f, z_e = base_cache[k]
    else:
        z_f, z_e = _base_fit(E, un, fi, ei, y, base)
        if base_cache is not None: base_cache[k] = (z_f, z_e)
    ze = _offsets(ctx_cols, fi, ei, y, z_f, z_e)
    ye = y[ei]
    def ll(z):
        p = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
        return ye * np.log2(p) + (1 - ye) * np.log2(1 - p)
    return float((ll(ze) - ll(z_e)).mean()), len(ei), float(ye.mean())


def ctx_cols(E, h, ev_un):
    cols = [E.atom(a, h['unit'], h['hi']) for a in h['atoms']]
    if h['cross'] and len(cols) == 2:
        a, b = cols
        cols = [a * 100000 + b]
    return cols


def random_hyp(rng):
    k = 1 if rng.random() < 0.35 else 2
    at = [ATOMS[i] for i in rng.choice(len(ATOMS), size=k, replace=False)]
    return dict(unit=['word', 'frame'][int(rng.integers(2))], lo=int(rng.choice([2, 3])), hi=int(rng.choice([8, 15, 30])),
                atoms=at, cross=bool(k == 2 and rng.random() < 0.4))


def page_folds(E, half, seed=0):
    """2 folds of the pages of one half."""
    pages = sorted(set(E.pg[E.half == half].tolist()))
    r = random.Random(seed); r.shuffle(pages)
    A = set(pages[::2])
    m = np.isin(E.pg, list(A))
    return m & (E.half == half), (~m) & (E.half == half)


def train_score(E, h, bc):
    a, b = page_folds(E, 0)
    g = []
    for f, e in ((a, b), (b, a)):
        r = loglik_gain(E, h['unit'], h['lo'], h['hi'], ctx_cols(E, h, None), f, e, bc)
        if r is None: return None
        g.append(r[0])
    return float(np.mean(g))


def run(name):
    out_p = os.path.join(L.CK, 'c1_%s.json' % name)
    if os.path.exists(out_p): return name, 0.0
    t0 = time.time()
    pages = L.corpus(name)
    E = Ev(pages)
    T0 = Ev(L.within_page_shuffle(pages, 9300))
    bc, bt = {}, {}
    rng = np.random.default_rng(920)
    rows = []
    for hi_ in range(NH):
        h = random_hyp(rng)
        s = train_score(E, h, bc)
        if s is None: continue
        st = train_score(T0, h, bt)
        if st is None: continue
        rows.append(dict(h=h, g_tr=s, g_tr_twin=st, score=s - st))
    rows.sort(key=lambda r: -r['score'])
    # held-out test of the top 20: fit on train folios, evaluate on held-out folios; 8 fresh twins
    twins = [Ev(L.within_page_shuffle(pages, 9400 + k)) for k in range(NTW)]
    wls = Ev(L.within_line_shuffle(pages, 9500))
    fit = E.half == 0; ev = E.half == 1
    bE = {}; bT = [dict() for _ in twins]; bW = {}
    top = []
    for r in rows[:TOP]:
        h = r['h']
        g = loglik_gain(E, h['unit'], h['lo'], h['hi'], ctx_cols(E, h, None), fit, ev, bE)
        gt = [loglik_gain(T, h['unit'], h['lo'], h['hi'], ctx_cols(T, h, None), T.half == 0, T.half == 1, bt_) for T, bt_ in zip(twins, bT)]
        gt = np.array([x[0] for x in gt if x is not None])
        gw = loglik_gain(wls, h['unit'], h['lo'], h['hi'], ctx_cols(wls, h, None), wls.half == 0, wls.half == 1, bW)
        z = (g[0] - gt.mean()) / max(gt.std(ddof=1), 1e-4)
        top.append(dict(r, g_te=g[0], n_te=g[1], ybar=g[2], tw_mu=float(gt.mean()), tw_sd=float(gt.std(ddof=1)), z=float(z),
                        g_wls=gw[0] if gw else None))
    res = dict(name=name, nh=len(rows), top=top, score_dist=[r['score'] for r in rows], secs=time.time() - t0)
    json.dump(res, open(out_p, 'w'), default=str)
    return name, res['secs']


NAMES = ['P_KONRAD', 'ZL3b', 'P_APIC', 'G_LX', 'P_HYGIN', 'IT2a', 'G_SEED', 'P_CIRCA', 'GC2a', 'G_SELF', 'P_CULP', 'G_GM']

if __name__ == '__main__':
    from multiprocessing import Pool
    names = sys.argv[1:] or NAMES
    for n in names: L.corpus(n)
    with Pool(2) as P:
        for n, s in P.imap_unordered(run, names):
            print(n, '%.0fs' % s, flush=True)
