"""v46 COUNT THE PICTURES, FIND THE NUMBER WORDS: shared library.

Units = pages (or panels) with an object count n (nymphs, stars, plant fragments, leaves) and their running
text (ZL3b P/C/R lines; label lines excluded). Candidate 'number-word patterns' = sets of word types
(single words, prefixes, suffixes, infixes, random glyph-regex families). For pattern c_i = tokens matching
on page i, T_i = tokens on page i.
  S1 RATE   : pooled Spearman, ranks z-scored WITHIN section, between log((c+.5)/T) and n
  S2 EXACT  : share of pages with |c - n| <= max(1, 0.1 n)   (a word written n times for n figures)
  S3 SLOT   : share of variant A among A+B (two suffix variants of one stem) vs n (cycle 2)
Null: n permuted among pages of the same section and size tercile (tokens) ; family-wise = max over patterns.
Held-out: random half-splits inside each section; top-K on the fit half re-scored (signed) on the other half.
"""
import os, re, sys, json, random, math, html
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v46_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v46'

# Ptolemy, Almagest star catalogue counts (stars listed per constellation, informata excluded), mapped to
# Hyginus, De astronomia III chapters (I = both Bears; XIII = Ophiuchus + Serpens; XXXIX = Hydra + Crater + Corvus)
PTOL = [7 + 27, 31, 22, 8, 28, 10, 17, 13, 13, 23, 26, 14, 24 + 18, 5, 9, 10, 20, 4, 13, 33, 18, 9, 27, 26, 21, 31,
        28, 42, 34, 22, 34, 12, 38, 18, 2, 45, 37, 7, 25 + 7 + 7]


def voy_units(name='ZL3b', secs=None):
    cnt = json.load(open(os.path.join(DATA, 'derived', 'v46_page_counts.json')))
    recs = json.load(open(os.path.join(DATA, 'derived', name + '_lines.json')))
    txt = defaultdict(list)
    for r in recs:
        if r['ltype'] in ('P', 'C', 'R'):
            ws = [w for w, u in zip(r['words'], r.get('uncertain', [False] * len(r['words']))) if not u and '?' not in w]
            txt[r['folio']].append(ws)
    out = []
    for c in cnt:
        if secs and c['sec'] not in secs: continue
        L = [l for l in txt.get(c['id'], []) if l]
        toks = [w for l in L for w in l]
        if len(toks) < 20: continue
        u = dict(c); u['toks'] = toks; u['lines'] = L
        out.append(u)
    return out


def hyginus_units():
    t = open(os.path.join(SCR, 'hyg3.json')).read()
    ch = json.loads(t)
    out = []
    for (num, head, s), n in zip(ch, PTOL):
        s = s.replace('Hyginus The Miscellany The Latin Library The Classics Page', '')
        toks = [w.lower() for w in re.findall(r"[A-Za-z]+", head + ' ' + s)]
        toks = [w.replace('j', 'i').replace('v', 'u') if not re.fullmatch(r'[ivxl]+', w) else w for w in toks]
        out.append(dict(id='hyg' + num, sec='hyg', n=n, toks=toks, lines=[toks]))
    return out


# ------------------------------------------------------------------ matrices
def type_matrix(units, min_count=1):
    tot = Counter(w for u in units for w in u['toks'])
    types = [w for w, c in tot.most_common() if c >= min_count]
    ti = {w: i for i, w in enumerate(types)}
    M = np.zeros((len(units), len(types)), np.float32)
    for r, u in enumerate(units):
        for w in u['toks']:
            if w in ti: M[r, ti[w]] += 1
    return M, types, tot


def make_patterns(types, tot, n_random=4000, seed=0, min_tok=8, alphabet=None):
    """-> list of (name, bool vector over types). Words, prefixes, suffixes, infixes (1-4 chars) and random
    families (prefix x suffix x length window) with >= min_tok tokens."""
    rng = random.Random(seed)
    T = np.array([tot[w] for w in types])
    pats, seen = [], set()

    def add(name, v):
        if v.sum() == 0 or T[v].sum() < min_tok: return
        key = np.packbits(v).tobytes()
        if key in seen: return
        seen.add(key); pats.append((name, v))
    for i, w in enumerate(types):
        if tot[w] >= min_tok:
            v = np.zeros(len(types), bool); v[i] = True; add('w:' + w, v)
    for k in (1, 2, 3, 4):
        for kind in ('pre', 'suf', 'inf'):
            subs = Counter()
            for w in types:
                if len(w) < k: continue
                if kind == 'pre': subs[w[:k]] += tot[w]
                elif kind == 'suf': subs[w[-k:]] += tot[w]
                else:
                    for s in set(w[j:j + k] for j in range(len(w) - k + 1)): subs[s] += tot[w]
            for s, c in subs.items():
                if c < min_tok: continue
                if kind == 'pre': v = np.array([w.startswith(s) for w in types])
                elif kind == 'suf': v = np.array([w.endswith(s) for w in types])
                else: v = np.array([s in w for w in types])
                add('%s:%s' % (kind, s), v)
    # random families
    pre = [p for p in set(w[:k] for w in types for k in (1, 2, 3) if len(w) >= k)]
    suf = [p for p in set(w[-k:] for w in types for k in (1, 2, 3) if len(w) >= k)]
    lens = np.array([len(w) for w in types])
    tries = 0
    while len(pats) < len(seen) and tries < n_random * 20:
        break
    nr = 0
    while nr < n_random and tries < n_random * 30:
        tries += 1
        p = rng.choice(pre) if rng.random() < .7 else ''
        s = rng.choice(suf) if rng.random() < .7 else ''
        a = rng.randint(1, 6); b = a + rng.randint(0, 5)
        v = np.array([w.startswith(p) and w.endswith(s) for w in types]) & (lens >= a) & (lens <= b)
        before = len(pats)
        add('rnd:%s*%s[%d-%d]' % (p, s, a, b), v)
        nr += len(pats) > before
    return pats


def counts_for(M, pats):
    P = np.stack([v for _, v in pats], 1).astype(np.float32)
    return M @ P          # units x patterns


# ------------------------------------------------------------------ statistics
def strata(units, nbin=3):
    """section x size-tercile labels"""
    lab = []
    by = defaultdict(list)
    for i, u in enumerate(units): by[u['sec']].append(i)
    out = [None] * len(units)
    for s, idx in by.items():
        T = np.array([len(units[i]['toks']) for i in idx])
        r = rankdata(T) / (len(T) + 1)
        for i, q in zip(idx, r): out[i] = '%s|%d' % (s, min(nbin - 1, int(q * nbin)))
    return out


def zrank_within(X, secs):
    """X: units x k. rank within section, z-score within section."""
    X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    Z = np.zeros_like(X)
    secs = np.array(secs)
    for s in set(secs):
        m = secs == s
        if m.sum() < 3: continue
        R = np.apply_along_axis(rankdata, 0, X[m])
        R = R - R.mean(0)
        sd = R.std(0); sd[sd == 0] = np.inf
        Z[m] = R / sd
    return Z


def perms_within(strata_lab, nperm, seed=0):
    rng = np.random.default_rng(seed)
    idx = defaultdict(list)
    for i, s in enumerate(strata_lab): idx[s].append(i)
    out = np.zeros((nperm, len(strata_lab)), int)
    for p in range(nperm):
        perm = np.arange(len(strata_lab))
        for s, ii in idx.items():
            ii = np.array(ii); perm[ii] = ii[rng.permutation(len(ii))]
        out[p] = perm
    return out


def s1_matrix(C, T, n, secs):
    """pooled within-section Spearman of log rate vs n -> vector over patterns"""
    R = np.log((C + 0.5) / T[:, None])
    Zr = zrank_within(R, secs)
    zn = zrank_within(n, secs)[:, 0]
    return (Zr * zn[:, None]).mean(0), Zr, zn


def s2_vector(C, n):
    tol = np.maximum(1, 0.1 * n)[:, None]
    return (np.abs(C - n[:, None]) <= tol).mean(0)


def familywise(C, T, n, secs, strat, nperm=500, seed=0, stat='s1'):
    if stat == 's1':
        obs, Zr, zn = s1_matrix(C, T, n, secs)
        P = perms_within(strat, nperm, seed)
        null = np.stack([(Zr * zn[p][:, None]).mean(0) for p in P])     # nperm x patterns
        a_obs = np.abs(obs); a_null = np.abs(null)
    else:
        obs = s2_vector(C, n)
        P = perms_within(strat, nperm, seed)
        null = np.stack([s2_vector(C, n[p]) for p in P])
        a_obs, a_null = obs, null
    # per-pattern z vs its own null, then family-wise max
    mu, sd = a_null.mean(0), a_null.std(0) + 1e-9
    zo = (a_obs - mu) / sd
    zn_ = (a_null - mu) / sd
    mx_null = zn_.max(1)
    order = np.argsort(-zo)
    pfw = [(1 + (mx_null >= zo[j]).sum()) / (1 + nperm) for j in order[:30]]
    return dict(obs=obs, z=zo, order=order, pfw=pfw, maxnull=mx_null)


def heldout(C, T, n, secs, strat, nsplit=20, K=10, nperm=100, seed=0):
    """select top-K by |S1| on a random within-section half, score sign-matched S1 on the other half.
    Null: same with n permuted within strata. Returns real mean held score, null distribution."""
    rng = np.random.default_rng(seed)
    secs = np.array(secs)

    def run(nv, rs):
        scores = []
        for s in range(nsplit):
            fit = np.zeros(len(nv), bool)
            for sc in set(secs):
                ii = np.nonzero(secs == sc)[0]
                ii = ii[rs.permutation(len(ii))]
                fit[ii[:len(ii) // 2]] = True
            a, _, _ = s1_matrix(C[fit], T[fit], nv[fit], secs[fit])
            top = np.argsort(-np.abs(a))[:K]
            b, _, _ = s1_matrix(C[~fit][:, top], T[~fit], nv[~fit], secs[~fit])
            scores.append(np.mean(np.sign(a[top]) * b))
        return float(np.mean(scores))
    real = run(n, np.random.default_rng(seed))
    P = perms_within(strat, nperm, seed + 1)
    null = [run(n[p], np.random.default_rng(seed + 7 + k)) for k, p in enumerate(P)]
    return real, np.array(null)


def plant(units, word, alpha, seed=0, mode='rate'):
    """replace tokens so that `word` occurs ~ alpha * n times (mode rate, Poisson) or exactly n times with
    probability alpha (mode exact). Returns new unit list (copies)."""
    rng = np.random.default_rng(seed)
    out = []
    for u in units:
        t = list(u['toks'])
        t = [w for w in t]
        if mode == 'rate':
            k = rng.poisson(alpha * u['n'])
        else:
            k = u['n'] if rng.random() < alpha else rng.poisson(max(1, np.mean([x['n'] for x in units])) * 0.3)
        k = min(k, len(t) // 3)
        pos = rng.choice(len(t), size=k, replace=False) if k else []
        for p in pos: t[p] = word
        v = dict(u); v['toks'] = t; out.append(v)
    return out


def run_search(units, nperm=500, n_random=4000, seed=0, min_tok=8, stats=('s1', 's2')):
    M, types, tot = type_matrix(units)
    pats = make_patterns(types, tot, n_random=n_random, seed=seed, min_tok=min_tok)
    C = counts_for(M, pats)
    T = np.array([len(u['toks']) for u in units], float)
    n = np.array([u['n'] for u in units], float)
    secs = [u['sec'] for u in units]
    st = strata(units)
    res = {'npat': len(pats), 'names': [p[0] for p in pats]}
    for s in stats:
        res[s] = familywise(C, T, n, secs, st, nperm=nperm, seed=seed, stat=s)
    return res, (C, T, n, secs, st)


def top_rows(res, stat, k=8):
    r = res[stat]
    rows = []
    for j, pf in zip(r['order'][:k], r['pfw'][:k]):
        rows.append('%s %s=%.3f z=%.1f pFW=%.3f' % (res['names'][j], stat, r['obs'][j], r['z'][j], pf))
    return rows
