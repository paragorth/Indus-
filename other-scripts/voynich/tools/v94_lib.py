"""v94: v89's recurrence-structure source finder, scaled to a 300+ text pool and made tolerant of re-cut chapters.

Kept from v89 (frozen): the residual unit-sharing feature (binary type sets, cosine, double centring, lag means,
unit-size regression, top-2 components stripped), score = Pearson over ODD-unit pairs (A, selection) and EVEN-unit
pairs (B, held out), block-shuffled target orders as null (zB), other texts as decoys (zDecoy), pass rule
rank 1 + zB >= 4 + zDecoy >= 3.

New:
 * every source carries several segmentations (boundary sets): natural entries ('ent'), paragraphs ('par'),
   sentences ('sen'), fixed 25-token windows ('tok'). An alignment picks a scheme, an offset and span in that
   scheme's segment units and a map: 'unif' (equal shares of segments), 'prop' (target-length proportional),
   'sub' (a RANDOM subset of the scheme's boundaries = a random re-segmentation of the candidate), each snapped to
   the scheme's boundaries. Thousands of these random segmentations x maps are scored per pair.
 * recut(): a re-cut-tolerant refinement. Each unit's start/end may move up to K fine boundaries; a linearised
   fit (unit i's piece vs the other current pieces, compared with the target residual row restricted to ODD
   partner units) is maximised by dynamic programming over monotone boundaries; accepted only if score A rises.
   Never uses even-even pairs, so B stays held out; nulls get the identical refinement.
"""
import os, sys, re, json, pickle, math
import numpy as np
import scipy.sparse as sp
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
VD = os.path.dirname(HERE)
CK = os.path.join(VD, 'data', 'v94_ckpt')
POOL = os.path.join(CK, 'pool')
sys.path.insert(0, HERE)
import v89_lib as L

resid_from_sets = L.resid_from_sets
zvec = L.zvec
pair_index = L.pair_index
sets_matrix = L.sets_matrix
encode_text = L.encode_text
DROPS = (0, 30, 100)
SCHEMES = ('ent', 'par', 'sen', 'tok')
TOKW = 25
MAXTOK = int(os.environ.get('V94_MAXTOK', '200000'))


# ------------------------------------------------------------------ pool storage
def save_text(key, meta, paras, ent_starts=None, sent_bounds=None):
    """paras: list of token lists; ent_starts: para indices where entries begin; sent_bounds: token positions."""
    os.makedirs(POOL, exist_ok=True)
    toks = [t for p in paras for t in p]
    pb = np.cumsum([0] + [len(p) for p in paras])
    if len(toks) > MAXTOK:   # keep the first MAXTOK tokens at a paragraph boundary
        k = int(np.searchsorted(pb, MAXTOK, 'right')) - 1
        k = max(k, 1)
        paras = paras[:k]; toks = toks[:int(pb[k])]; pb = pb[:k + 1]
        if ent_starts is not None: ent_starts = [e for e in ent_starts if e < k]
        if sent_bounds is not None: sent_bounds = [s for s in sent_bounds if s <= pb[-1]]
    cnt = Counter(toks)
    vocab = [w for w, _ in cnt.most_common()]
    rk = {w: i for i, w in enumerate(vocab)}
    stream = np.array([rk[t] for t in toks], dtype=np.int32)
    par = np.unique(pb).astype(np.int64)
    ent = np.unique(np.concatenate([[0], pb[ent_starts] if ent_starts else pb, [pb[-1]]])).astype(np.int64)
    sen = np.unique(np.concatenate([par, np.array(sent_bounds or [], dtype=np.int64)])).astype(np.int64)
    d = {'meta': meta, 'stream': stream, 'vocab': vocab, 'b': {'ent': ent, 'par': par, 'sen': sen}}
    pickle.dump(d, open(os.path.join(POOL, key + '.pkl'), 'wb'))
    return len(toks), len(ent) - 1, len(par) - 1, len(sen) - 1


def pool_keys():
    return sorted(f[:-4] for f in os.listdir(POOL) if f.endswith('.pkl'))


def load_text(key):
    return pickle.load(open(os.path.join(POOL, key + '.pkl'), 'rb'))


# ------------------------------------------------------------------ source
class Src:
    def __init__(self, d=None, units_tok=None):
        if units_tok is not None:   # build from token lists (calibration / Voynich-as-source)
            toks = [t for u in units_tok for t in u]
            cnt = Counter(toks); vocab = [w for w, _ in cnt.most_common()]; rk = {w: i for i, w in enumerate(vocab)}
            stream = np.array([rk[t] for t in toks], dtype=np.int64)
            pb = np.cumsum([0] + [len(u) for u in units_tok]).astype(np.int64)
            b = {'ent': np.unique(pb), 'par': np.unique(pb), 'sen': np.unique(pb)}
        else:
            stream = d['stream'].astype(np.int64); b = dict(d['b'])
        self.T = T = len(stream)
        self.V = int(stream.max()) + 1 if T else 1
        b['tok'] = np.unique(np.concatenate([np.arange(0, T, TOKW), [T]])).astype(np.int64)
        self.b = {}
        for s in SCHEMES:
            x = np.unique(np.clip(b[s], 0, T)).astype(np.int64)
            if x[0] != 0: x = np.r_[0, x]
            if x[-1] != T: x = np.r_[x, T]
            self.b[s] = x
        self.M = {s: len(self.b[s]) - 1 for s in SCHEMES}
        self.kept = {}
        for dr in DROPS:
            pos = np.nonzero(stream >= dr)[0]
            self.kept[dr] = (pos, stream[pos])

    def schemes(self, n):
        """schemes with at least n/3 segments; drop duplicates (identical boundary sets)."""
        out, seen = [], []
        for s in SCHEMES:
            if self.M[s] < max(4, n / 3): continue
            if any(len(self.b[s]) == len(self.b[t]) and np.array_equal(self.b[s], self.b[t]) for t in seen): continue
            out.append(s); seen.append(s)
        return out

    def pieces(self, cuts, drop):
        pos, ids = self.kept[drop]
        a = np.searchsorted(pos, cuts)
        n = len(cuts) - 1
        cnts = np.diff(a)
        rows = np.repeat(np.arange(n, dtype=np.int64), cnts)
        cols = ids[a[0]:a[-1]]
        key = np.sort(rows * self.V + cols)
        if len(key): key = key[np.r_[True, key[1:] != key[:-1]]]
        r = key // self.V; c = key % self.V
        indptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=n))])
        return sp.csr_matrix((np.ones(len(c)), c, indptr), shape=(n, self.V))

    def spans(self, starts, ends, drop):
        """arbitrary (start, end) token spans -> binary csr."""
        pos, ids = self.kept[drop]
        a = np.searchsorted(pos, starts); e = np.searchsorted(pos, ends)
        cnt = np.maximum(e - a, 0)
        rows = np.repeat(np.arange(len(a), dtype=np.int64), cnt)
        idx = np.concatenate([np.arange(x, y) for x, y in zip(a, e)]) if len(a) else np.zeros(0, np.int64)
        cols = ids[idx.astype(np.int64)] if len(idx) else np.zeros(0, np.int64)
        key = np.unique(rows * self.V + cols)
        r = key // self.V; c = key % self.V
        indptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=len(a)))])
        return sp.csr_matrix((np.ones(len(c)), c, indptr), shape=(len(a), self.V))


# ------------------------------------------------------------------ target
class Target(L.Target):
    """v89 target (block-shuffle nulls) + per-variant residual matrices for recut()."""
    def __init__(self, units_tok, n_null=12, seed=7):
        super().__init__(units_tok, n_null=n_null, seed=seed, null_kind='block')
        X = sets_matrix(units_tok)
        self.R = [resid_from_sets(X[p]) for p in self.perms]


# ------------------------------------------------------------------ alignments
def seg_to_tok(src, s, u):
    b = src.b[s]; M = len(b) - 1
    u = np.clip(np.asarray(u, float), 0, M)
    i = np.minimum(np.floor(u).astype(int), M - 1); f = u - i
    return b[i] + f * (b[i + 1] - b[i])


def snap(src, s, x):
    b = src.b[s]
    j = np.clip(np.searchsorted(b, x), 1, len(b) - 1)
    return np.where(x - b[j - 1] < b[j] - x, b[j - 1], b[j])


def draw(rng, src, n, schemes, base=None, step=None):
    if base is None:
        s = schemes[int(rng.integers(len(schemes)))]
        M = src.M[s]
        # span in segments: target units may each be 1/3..4 segments of this scheme (finer schemes: more)
        lo = max(1.0, n / 3); hi = min(M * 0.98, n * (4 if s == 'ent' else 25))
        if hi <= lo: hi = lo * 1.01
        span = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        span = min(span, M * 0.98)
        off = float(rng.uniform(0, max(1e-9, M - span)))
        mp = ['unif'] * 5 + ['prop'] + ['sub'] * 4
        mp = mp[int(rng.integers(len(mp)))]
        return {'sch': s, 'off': off, 'span': span, 'map': mp, 'jit': float(rng.uniform(0, 0.4)),
                'drop': int(rng.choice(DROPS)), 'seed': int(rng.integers(1 << 30))}
    a = dict(base); M = src.M[a['sch']]
    a['span'] = float(min(M * 0.98, max(n / 6, a['span'] * np.exp(rng.normal(0, step)))))
    a['off'] = float(np.clip(a['off'] + rng.normal(0, step * a['span'] / 4), 0, max(0, M - a['span'])))
    if rng.random() < 0.3: a['jit'] = float(np.clip(a['jit'] + rng.normal(0, 0.1), 0, 0.6))
    if rng.random() < 0.3: a['seed'] = int(rng.integers(1 << 30))
    return a


def make_cuts(src, a, lens):
    n = len(lens); s = a['sch']
    t0 = float(seg_to_tok(src, s, a['off'])); t1 = float(seg_to_tok(src, s, a['off'] + a['span']))
    t0 = float(snap(src, s, np.array([t0]))[0]); t1 = float(snap(src, s, np.array([t1]))[0])
    if t1 <= t0: t1 = float(src.b[s][min(len(src.b[s]) - 1, np.searchsorted(src.b[s], t0) + 1)])
    rng = np.random.default_rng(a['seed'])
    b = src.b[s]
    inside = b[(b > t0) & (b < t1)]
    if a['map'] == 'sub' and len(inside) >= n - 1:
        inner = np.sort(rng.choice(inside, n - 1, replace=False)).astype(float)
    else:
        if a['map'] == 'prop':
            cl = np.cumsum(lens)[:-1] / lens.sum()
            inner = t0 + cl * (t1 - t0)
        else:
            us = a['off'] + np.arange(1, n) * a['span'] / n
            inner = seg_to_tok(src, s, us)
        inner = inner + rng.normal(0, a['jit'] * (t1 - t0) / n, n - 1)
        inner = np.clip(np.sort(inner), t0, t1)
        inner = snap(src, s, inner).astype(float)
    inner = np.clip(np.sort(inner), t0, t1)
    return np.concatenate([[t0], inner, [t1]]).astype(np.int64)


def score_cuts(src, tgt, cuts, drop, v):
    R = resid_from_sets(src.pieces(cuts, drop))
    za, zb = zvec(R, tgt.PA), zvec(R, tgt.PB)
    return float(za @ tgt.ZA[v]) / len(za), float(zb @ tgt.ZB[v]) / len(zb)


def score_align(src, tgt, a, variants=None):
    V = list(range(len(tgt.perms))) if variants is None else variants
    sa, sb = [], []
    cache = {}
    for v in V:
        key = v if a['map'] == 'prop' else 0
        if key not in cache:
            cuts = make_cuts(src, a, tgt.vlens[v])
            R = resid_from_sets(src.pieces(cuts, a['drop']))
            cache[key] = (zvec(R, tgt.PA), zvec(R, tgt.PB))
        za, zb = cache[key]
        sa.append(float(za @ tgt.ZA[v]) / len(za)); sb.append(float(zb @ tgt.ZB[v]) / len(zb))
    return np.array(sa), np.array(sb)


def grid(src, n, rng, schemes, max_off=150, max_off_prop=60, max_off_fine=60):
    """ordered maps: 'ent' unif (spans 0.75n..2n, <= max_off offsets each), 'ent' prop (target-length cuts,
    re-cut tolerant; spans 0.8n, 1.25n), finer schemes unif (spans 2n, 4n, 8n segments)."""
    out = []
    def offs(M, sp_, k):
        hi = M - sp_
        return np.arange(0, hi + 1e-9, max(1.0, hi / k)) if hi > 0 else np.array([0.0])
    for s in schemes:
        M = src.M[s]
        if s == 'ent': plan = [(f * n, 'unif', max_off) for f in (0.75, 1.0, 1.5, 2.0)] + [(f * n, 'prop', max_off_prop) for f in (1.0,)]
        else: plan = [(f * n, 'unif', max_off_fine) for f in (2, 4, 8)]
        for sp_, mp, k in plan:
            sp_ = min(sp_, M * 0.98)
            if sp_ < max(2, n / 3): continue
            for o in offs(M, sp_, k):
                out.append({'sch': s, 'off': float(o), 'span': float(sp_), 'map': mp, 'jit': 0.0,
                            'drop': int(rng.choice(DROPS)), 'seed': 0})
    return out


def search(src, tgt, n_rand=300, top=3, refine=12, seed=0, max_off=150, max_off_prop=60, max_off_fine=60):
    rng = np.random.default_rng(seed)
    sch = src.schemes(tgt.n)
    if not sch: return None
    A = grid(src, tgt.n, rng, sch, max_off, max_off_prop, max_off_fine) + [draw(rng, src, tgt.n, sch) for _ in range(n_rand)]
    nv = len(tgt.perms)
    SA = np.zeros((len(A), nv)); SB = np.zeros((len(A), nv))
    for i, a in enumerate(A):
        SA[i], SB[i] = score_align(src, tgt, a)
    out = []
    for v in range(nv):
        best = []
        for i in np.argsort(-SA[:, v])[:top]:
            a, sa, sb = A[i], SA[i, v], SB[i, v]
            for r in range(refine):
                b = draw(rng, src, tgt.n, sch, base=a, step=0.1 if r < refine // 2 else 0.03)
                x, y = score_align(src, tgt, b, variants=[v])
                if x[0] > sa: a, sa, sb = b, x[0], y[0]
            best.append((sa, sb, a))
        best.sort(key=lambda z: -z[0])
        out.append({'A': best[0][0], 'B': best[0][1], 'a': best[0][2], 'nA': len(A)})
    return out


def summarize(res):
    return L.summarize([dict(r, Btop=r['B']) for r in res])


# ------------------------------------------------------------------ re-cut-tolerant refinement (DP)
def recut(src, tgt, a, v, K=None, iters=6, fine='sen'):
    """start from alignment a (target variant v); move unit boundaries over the fine scheme by DP.
    Returns (A, B, cuts)."""
    cuts = make_cuts(src, a, tgt.vlens[v]).astype(np.int64)
    drop = a['drop']
    if src.M['sen'] < max(2 * tgt.n, 2 * src.M['ent']): fine = 'tok'
    fb = src.b[fine]
    if K is None:   # let a boundary move up to ~40% of a mean piece
        mean_piece = (cuts[-1] - cuts[0]) / tgt.n
        mean_fine = max(1.0, (fb[-1] - fb[0]) / max(1, len(fb) - 1))
        K = int(np.clip(round(0.4 * mean_piece / mean_fine), 1, 8))
    n = tgt.n
    Rt = tgt.R[v]
    odd = np.arange(n) % 2 == 1
    A0, B0 = score_cuts(src, tgt, cuts, drop, v)
    for it in range(iters):
        idx = np.clip(np.searchsorted(fb, cuts), 0, len(fb) - 1)
        X = src.pieces(cuts, drop)
        kx = np.asarray(X.sum(1)).ravel()
        offs = np.arange(-K, K + 1)
        W = len(offs)
        # candidate boundary positions per cut index (fine boundary indices)
        cand = np.clip(idx[:, None] + offs[None, :], 0, len(fb) - 1)   # (n+1, W)
        st = []; en = []; uid = []
        for i in range(n):
            s_ = fb[cand[i]]; e_ = fb[cand[i + 1]]
            S_, E_ = np.meshgrid(s_, e_, indexing='ij')
            st.append(S_.ravel()); en.append(E_.ravel()); uid.append(np.full(W * W, i))
        st = np.concatenate(st); en = np.concatenate(en); uid = np.concatenate(uid)
        P = src.spans(st, np.maximum(en, st), drop)
        kp = np.asarray(P.sum(1)).ravel()
        C = (P @ X.T).toarray()
        with np.errstate(divide='ignore', invalid='ignore'):
            C = C / np.sqrt(np.outer(kp, kx))
        C[~np.isfinite(C)] = 0
        fit = np.full(len(st), -1.0)
        for i in range(n):
            J = np.nonzero(odd & (np.abs(np.arange(n) - i) > 1))[0]
            if len(J) < 4: continue
            rt = Rt[i, J]; rt = rt - np.nanmean(rt); rt = np.nan_to_num(rt)
            sel = np.nonzero(uid == i)[0]
            c = C[np.ix_(sel, J)]; c = c - c.mean(1, keepdims=True)
            den = np.sqrt((c ** 2).sum(1) * (rt ** 2).sum()) + 1e-12
            fit[sel] = (c @ rt) / den
        fit[en <= st] = -5.0
        F = fit.reshape(n, W, W)
        # DP over boundary choices w_0..w_n (index into offs per cut)
        dp = np.zeros(W); bp = np.zeros((n, W), int)
        for i in range(n):
            tot = dp[:, None] + F[i]                       # (w_i, w_{i+1})
            # monotone: candidate positions must increase
            pi = fb[cand[i]][:, None]; pj = fb[cand[i + 1]][None, :]
            tot = np.where(pj > pi, tot, -1e9)
            bp[i] = np.argmax(tot, 0); dp = tot[bp[i], np.arange(W)]
        w = np.zeros(n + 1, int); w[n] = int(np.argmax(dp))
        for i in range(n - 1, -1, -1): w[i] = bp[i][w[i + 1]]
        new = fb[cand[np.arange(n + 1), w]].astype(np.int64)
        if np.any(np.diff(new) <= 0): break
        A1, B1 = score_cuts(src, tgt, new, drop, v)
        if A1 <= A0 + 1e-6: break
        cuts, A0, B0 = new, A1, B1
    return A0, B0, cuts


def recut_search(src, tgt, res, K=None, iters=6):
    """apply recut() to every variant's best alignment from search(); returns new per-variant A/B."""
    out = []
    for v, r in enumerate(res):
        A, B, cuts = recut(src, tgt, r['a'], v, K=K, iters=iters)
        if A < r['A']: A, B = r['A'], r['B']
        out.append({'A': A, 'B': B, 'a': r['a'], 'cuts': [int(cuts[0]), int(cuts[-1])]})
    return out


# ------------------------------------------------------------------ helpers
def recut_units(units_tok, rng, p_split=0.25, p_merge=0.25):
    """re-cut a list of units: split some at a random point, merge some with the next."""
    out = []
    i = 0
    while i < len(units_tok):
        u = list(units_tok[i])
        if rng.random() < p_merge and i + 1 < len(units_tok):
            u = u + list(units_tok[i + 1]); i += 1
        if rng.random() < p_split and len(u) > 20:
            k = int(rng.integers(len(u) // 4, 3 * len(u) // 4))
            out.append(u[:k]); out.append(u[k:])
        else:
            out.append(u)
        i += 1
    return out


def recut_greedy(src, tgt, a, v, steps=(-4, -2, -1, 1, 2, 4), sweeps=2, fine=None, cuts=None):
    """re-cut-tolerant refinement by coordinate ascent on score A (odd-odd pairs only): each inner cut moves over
    fine boundaries; B (even-even pairs) is never consulted. Returns (A, B, cuts)."""
    if cuts is None: cuts = make_cuts(src, a, tgt.vlens[v]).astype(np.int64)
    drop = a['drop']
    if fine is None:
        fine = 'sen' if src.M['sen'] >= max(2 * tgt.n, 2 * src.M['ent']) else 'tok'
    fb = src.b[fine]
    A0, B0 = score_cuts(src, tgt, cuts, drop, v)
    n = tgt.n
    for sw in range(sweeps):
        moved = 0
        for i in range(1, n):
            j0 = int(np.clip(np.searchsorted(fb, cuts[i]), 0, len(fb) - 1))
            best = None
            for st in steps:
                j = j0 + st
                if j < 0 or j >= len(fb): continue
                p = fb[j]
                if p <= cuts[i - 1] or p >= cuts[i + 1]: continue
                c2 = cuts.copy(); c2[i] = p
                A1, B1 = score_cuts(src, tgt, c2, drop, v)
                if A1 > A0 + 1e-9 and (best is None or A1 > best[0]): best = (A1, B1, c2)
            if best: A0, B0, cuts = best; moved += 1
        if not moved: break
    return A0, B0, cuts
