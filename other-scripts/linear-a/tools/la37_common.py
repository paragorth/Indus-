#!/usr/bin/env python3
"""LA-37 'find the doublets': shared helpers.

Score every sign pair for contextual interchangeability, with no sound values:
  cos  : cosine of PPMI context vectors (left neighbour, right neighbour, 2-back, 2-ahead; '#' = word edge)
  xI   : split-half interchangeability index (as Voynich v35): contexts from two document halves,
         I(a,b) = mean(cos(a1,b2), cos(b1,a2)) / sqrt(cos(a1,a2) cos(b1,b2)) (self-terms floored)
  swap : number of word-type pairs that differ only by a <-> b in one slot
Each statistic is converted to a z against a null that shuffles signs inside every word (bags kept), and the
pair score T = mean of the z's. Search correction: max-T over all pairs in each null replicate (FWER).
Linear B values are used only to label the Linear B control and as an outside check.
"""
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la32_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
LA = os.path.join(HERE, '..')
CK = os.path.join(LA, 'data', 'la37_ckpt')
LOOPS = os.path.join(LA, 'loops')
os.makedirs(CK, exist_ok=True)


def la_units(minlen=2):
    return [dict(r) for r in C.la_words() if len(r['w']) >= minlen]


def lb_units(minlen=2):
    return [dict(r) for r in C.lb_words() if len(r['w']) >= minlen]


def lb_draw(units, ntok, rng):
    """documents drawn at random until the sign-token count reaches ntok (LA size)."""
    by = collections.defaultdict(list)
    for r in units:
        by[r['doc']].append(r)
    docs = list(by); rng.shuffle(docs)
    out, n = [], 0
    for d in docs:
        out += by[d]; n += sum(len(r['w']) for r in by[d])
        if n >= ntok:
            break
    return out


# ---------------------------------------------------------------- Linear B truth labels (control only)
DOUBLET = {frozenset(p) for p in [('a', 'a2'), ('a', 'a3'), ('a2', 'a3'), ('ra', 'ra2'), ('ra', 'ra3'), ('ra2', 'ra3'),
                                  ('ro', 'ro2'), ('pu', 'pu2'), ('ta', 'ta2'), ('a', 'au'), ('o', 'wo'),
                                  ('te', 'twe'), ('de', 'dwe'), ('do', 'dwo'), ('na', 'nwa'), ('to', 'two'),
                                  ('pe', 'pte'), ('ri', 'ra2'), ('ri', 'ro2')]}
CORE = {frozenset(p) for p in [('a', 'a2'), ('a', 'a3'), ('ra', 'ra2'), ('ra', 'ra3'), ('ro', 'ro2'), ('pu', 'pu2'),
                               ('ta', 'ta2'), ('o', 'wo')]}


def lb_label(a, b):
    """'doublet', 'sameC', 'sameV', 'other' or None (unlabelled signs)."""
    if frozenset((a, b)) in DOUBLET:
        return 'doublet'
    ca, cb = C.lb_cv(a), C.lb_cv(b)
    if ca is None or cb is None:
        return None
    if ca[0] == cb[0]:
        return 'sameC'
    if ca[1] == cb[1]:
        return 'sameV'
    return 'other'


# ---------------------------------------------------------------- statistics
def alphabet(units, fmin):
    c = collections.Counter(s for r in units for s in r['w'])
    return sorted([s for s, n in c.items() if n >= fmin], key=lambda s: -c[s]), c


def ctx_counts(words, idx, nfeat_index):
    n = len(idx)
    M = np.zeros((n, len(nfeat_index)))
    for w in words:
        L = len(w)
        for p, s in enumerate(w):
            i = idx.get(s)
            if i is None:
                continue
            for tag, q in (('L', p - 1), ('R', p + 1), ('L2', p - 2), ('R2', p + 2)):
                if 0 <= q < L:
                    f = (tag, w[q])
                elif (tag in ('L', 'R')) and (q == -1 or q == L):
                    f = (tag, '#')
                else:
                    continue
                j = nfeat_index.get(f)
                if j is not None:
                    M[i, j] += 1
    return M


def feat_index(words, idx):
    fs = set()
    for w in words:
        for p, s in enumerate(w):
            for tag in ('L', 'R', 'L2', 'R2'):
                pass
    keys = list(idx) + ['#', '?']
    out = {}
    for tag in ('L', 'R', 'L2', 'R2'):
        for k in keys:
            out[(tag, k)] = len(out)
    return out


def ppmi_rows(M, alpha=0.75):
    tot = M.sum()
    if tot == 0:
        return M
    pr = M.sum(1, keepdims=True) / tot
    pc = M.sum(0) ** alpha; pc = pc / max(pc.sum(), 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.log((M / tot) / (pr * pc[None, :]))
    P = np.where(np.isfinite(P) & (P > 0), P, 0.0)
    nrm = np.linalg.norm(P, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return P / nrm


def map_rare(words, idx):
    return [tuple(s if s in idx else '?' for s in w) for w in words]


def swap_matrix(words, idx):
    n = len(idx)
    S = np.zeros((n, n))
    types = set(words)
    buckets = collections.defaultdict(set)
    for w in types:
        for p in range(len(w)):
            buckets[(len(w), p, w[:p] + w[p + 1:])].add(w[p])
    for k, sg in buckets.items():
        sg = [s for s in sg if s in idx]
        for i in range(len(sg)):
            for j in range(i + 1, len(sg)):
                a, b = idx[sg[i]], idx[sg[j]]
                S[a, b] += 1; S[b, a] += 1
    return S


def _ctx_fast(words, idx, nfi):
    """vectorised context counts: features (tag, neighbour) with neighbours indexed as in feat_index."""
    n = len(idx); K_ = len(idx) + 2   # keys: signs, '#', '?'
    code = dict(idx); code['#'] = n; code['?'] = n + 1
    lens = np.array([len(w) for w in words]); tot = lens.sum()
    s = np.fromiter((code[x] for w in words for x in w), int, tot)
    wid = np.repeat(np.arange(len(words)), lens)
    start = np.repeat(np.cumsum(lens) - lens, lens); pos = np.arange(tot) - start; L = lens[wid]
    M = np.zeros((n, 4 * K_))
    keep = s < n
    for t, off in enumerate((-1, 1, -2, 2)):
        q = pos + off
        inside = (q >= 0) & (q < L)
        nb = np.where(inside, s[np.clip(np.arange(tot) + off, 0, tot - 1)], n)
        use = keep & (inside | ((abs(off) == 1) & ~inside))
        np.add.at(M, (s[use], t * K_ + nb[use]), 1)
    return M


def stats(units, alph, halves=None):
    """dict of n x n matrices: cos, xI, swap."""
    idx = {s: i for i, s in enumerate(alph)}
    words = map_rare([r['w'] for r in units], idx)
    P = ppmi_rows(_ctx_fast(words, idx, None))
    cos = P @ P.T
    if halves is None:
        halves = np.array([hash(r['doc']) % 2 for r in units])
    w1 = [w for w, h in zip(words, halves) if h == 0]; w2 = [w for w, h in zip(words, halves) if h == 1]
    P1 = ppmi_rows(_ctx_fast(w1, idx, None)); P2 = ppmi_rows(_ctx_fast(w2, idx, None))
    X = P1 @ P2.T
    sd = np.maximum(np.diag(X), 0.15)
    xI = 0.5 * (X + X.T) / np.sqrt(np.outer(sd, sd))
    sw = swap_matrix([r for r in words], idx)
    return dict(cos=cos, xI=xI, swap=sw)


def shuffle_within(units, rng):
    out = []
    for r in units:
        w = list(r['w']); rng.shuffle(w)
        d = dict(r); d['w'] = tuple(w); out.append(d)
    return out


def doc_halves(units, seed=0):
    rng = np.random.default_rng(seed)
    docs = sorted({r['doc'] for r in units})
    h = dict(zip(docs, rng.integers(0, 2, len(docs))))
    return np.array([h[r['doc']] for r in units])


def score(units, alph, R=200, seed=0, keys=('cos', 'xI', 'swap'), return_null=False):
    """per-pair z vs within-word shuffle null; T = mean z; FWER from max-T over pairs per replicate."""
    rng = np.random.default_rng(seed)
    hv = doc_halves(units, seed)
    obs = stats(units, alph, hv)
    n = len(alph); iu = np.triu_indices(n, 1)
    nulls = {k: np.empty((R, len(iu[0]))) for k in keys}
    for r in range(R):
        st = stats(shuffle_within(units, rng), alph, hv)
        for k in keys:
            nulls[k][r] = st[k][iu]
    Z = {}; Znull = {}
    for k in keys:
        mu = nulls[k].mean(0); sd = nulls[k].std(0) + 1e-6
        if k == 'swap':
            sd = np.sqrt(sd ** 2 + 0.25)  # discrete counts: floor the spread
        Z[k] = (obs[k][iu] - mu) / sd
        Znull[k] = (nulls[k] - mu) / sd
    T = np.mean([Z[k] for k in keys], 0)
    Tn = np.mean([Znull[k] for k in keys], 0)
    maxT = Tn.max(1)
    fwer = np.array([(1 + (maxT >= t).sum()) / (R + 1) for t in T])
    praw = np.array([(1 + (Tn[:, j] >= T[j]).sum()) / (R + 1) for j in range(len(T))])
    out = dict(alph=alph, iu=iu, T=T, Z=Z, fwer=fwer, praw=praw, obs={k: obs[k][iu] for k in obs})
    if return_null:
        out['Tn'] = Tn
    return out


def freq_matched_pct(T, iu, alph, cnt, nb=4):
    """percentile of each pair's T among pairs whose two signs fall in the same frequency-bin pair."""
    f = np.array([cnt[s] for s in alph])
    q = np.quantile(f, np.linspace(0, 1, nb + 1)[1:-1])
    b = np.searchsorted(q, f)
    key = np.minimum(b[iu[0]], b[iu[1]]) * 10 + np.maximum(b[iu[0]], b[iu[1]])
    pct = np.empty(len(T))
    for k in np.unique(key):
        m = key == k
        r = T[m].argsort().argsort()
        pct[m] = (r + 0.5) / m.sum()
    return pct


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def lb_eval(res, label_fn=lb_label):
    alph, iu, T = res['alph'], res['iu'], res['T']
    labs = np.array([label_fn(alph[a], alph[b]) for a, b in zip(*iu)], dtype=object)
    oth = T[labs == 'other']
    out = {}
    for g in ('doublet', 'sameC', 'sameV'):
        out[g] = (int((labs == g).sum()), auc(T[labs == g], oth))
    order = np.argsort(-T)
    top = [labs[j] for j in order[:20]]
    out['top20'] = collections.Counter(top)
    out['top20_pairs'] = [(alph[iu[0][j]], alph[iu[1][j]], round(float(T[j]), 2), labs[j]) for j in order[:20]]
    base = collections.Counter(l for l in labs if l is not None)
    out['base'] = {k: round(v / max(1, sum(base.values())), 3) for k, v in base.items()}
    return out


def plant(units, host, rate, mode, rng, newname='X*'):
    """add a doublet X* of sign host. modes: token (free variation), type (lexical), group (by site/scribe), initial."""
    out = []
    types = {}
    groups = sorted({r['site'] for r in units})
    gset = set(rng.choice(groups, max(1, len(groups) // 2), replace=False)) if mode == 'group' else set()
    for r in units:
        w = list(r['w'])
        for p, s in enumerate(w):
            if s != host:
                continue
            if mode == 'token':
                if rng.random() < rate: w[p] = newname
            elif mode == 'type':
                k = (r['w'], p)
                if k not in types: types[k] = rng.random() < rate
                if types[k]: w[p] = newname
            elif mode == 'group':
                if r['site'] in gset and rng.random() < rate * 2: w[p] = newname
            elif mode == 'initial':
                if p == 0 and rng.random() < 0.9: w[p] = newname
        d = dict(r); d['w'] = tuple(w); out.append(d)
    return out


def log(fn, s):
    print(s, flush=True)
    with open(fn, 'a') as f:
        f.write(s + '\n')


# ---------------------------------------------------------------- frozen statistic (chosen on Linear B, cycle 1)
def _rk(v):
    return v.argsort().argsort() / len(v)


def all3(st, iu):
    """rank average of PPMI cosine, split-half index and degree-normalised swap count."""
    S = st['swap']; deg = S.sum(1); E = max(deg.sum(), 1)
    sn = S / (np.outer(deg, deg) / E + 0.5)
    return (_rk(st['cos'][iu]) + _rk(st['xI'][iu]) + _rk(sn[iu])) / 3


def score2(units, alph, cnt, R=100, seed=0):
    """T = all3; Null A frequency-matched percentile; Null B within-word shuffle (per-pair z, max-z FWER, BH)."""
    rng = np.random.default_rng(seed)
    hv = doc_halves(units, seed)
    n = len(alph); iu = np.triu_indices(n, 1)
    T = all3(stats(units, alph, hv), iu)
    Nn = np.empty((R, len(T)))
    for r in range(R):
        Nn[r] = all3(stats(shuffle_within(units, rng), alph, hv), iu)
    mu = Nn.mean(0); sd = Nn.std(0) + 0.02
    z = (T - mu) / sd
    zn = (Nn - mu) / sd
    maxz = zn.max(1)
    fwer = np.array([(1 + (maxz >= x).sum()) / (R + 1) for x in z])
    from scipy.stats import norm
    praw = norm.sf(z)  # normal approximation (empirical floor 1/(R+1) is too coarse for BH over all pairs)
    o = np.argsort(praw); m = len(praw)
    q = np.empty(m); q[o] = np.minimum.accumulate((praw[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    fm = freq_matched_pct(T, iu, alph, cnt)
    return dict(alph=alph, iu=iu, T=T, z=z, fwer=fwer, praw=praw, q=np.minimum(q, 1), fm=fm)


def flagged(res, qmax=0.1, fmmin=0.95):
    return (res['q'] <= qmax) & (res['fm'] >= fmmin)
