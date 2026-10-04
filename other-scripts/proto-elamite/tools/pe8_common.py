"""pe8: 'the tablets were filled-in forms'.  Shared code.

Every tablet is abstracted into a SKELETON: sign roles and number systems only, names and quantities blanked.
Skeleton features (all categorical; nothing here reads a sign's meaning):
  HDR   header type: none / M157 (general opener) / M327-family / other sign header
  HLEN  header length: 0 / 1 / 2+ signs
  SYS   obverse entry number-system profile: S (counted only) / C (capacity only) / CS (both) / F (counted with
        fractions) / B / O (other, N23, modified)               -- the target in test (1), masked there
  NENT  number of obverse entries: 1 / 2-3 / 4-6 / 7-11 / 12+
  ONE   share of one-sign entries: all / most (>=.5) / few (<.5) / none
  CLS   share of entries that end in a final-class sign (FINDINGS test a): none / some (<.5) / most
  CLSR  the class sign repeats on every class-ending entry: na / same / varied
  PRE   some entry starts with an initial-slot sign (FINDINGS test a): 0/1
  BARE  some obverse numeral line has no sign: 0/1
  SUB   obverse sign line without numeral after the first entry (subscript / note): 0/1
  REV   reverse: none(not recorded) / blank / broken / total (one numeric line) / entries (2+) / text (signs only)
  TOT   written total relation: none / same (same system as entries) / conv (other system) / bare (no sign)
  TOP   numeral on the top edge: 0/1
  COL   more than one column: 0/1
  SEAL  seal impression or scribal design noted: 0/1
Quality: frag = any lacuna or '$ ... broken' note on the obverse.
"""
import json, math, os, random, sys
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, '..', 'data')
from common import load, base, is_sign, system_of

FINAL = {'M288', 'M297', 'M263', 'M346', 'M264', 'M072', 'M003', 'M354', 'M371', 'M096', 'M376', 'M036'}
INITIAL = {'M387', 'M370', 'M124', 'M305', 'M038', 'M111'}
FEATS = ['HDR', 'HLEN', 'SYS', 'NENT', 'ONE', 'CLS', 'CLSR', 'PRE', 'BARE', 'SUB', 'REV', 'TOT', 'TOP', 'COL', 'SEAL']


def esys(nums):
    """Entry system in 3 classes for prediction: S counted, C capacity, O other."""
    s = system_of(nums)
    if s in ('C', 'C*'):
        return 'C'
    if s in ('SDB', 'S-frac'):
        return 'S'
    return 'O'


def sys_profile(L):
    raw = [system_of(l['numerals']) for l in L]
    st = set(raw)
    if not st:
        return 'none'
    if st & {'B'}:
        return 'B'
    if st & {'N23', 'mod*'}:
        return 'O'
    c = bool(st & {'C', 'C*'})
    s = bool(st & {'SDB', 'S-frac'})
    if c and s:
        return 'CS'
    if c:
        return 'C'
    if 'S-frac' in st:
        return 'F'
    return 'S'


def bsigns(l):
    return [base(s) for s in l['signs'] if is_sign(s)] + (['x'] * sum(1 for s in l['signs'] if s == 'x'))


def bin_nent(n):
    return '1' if n <= 1 else '2-3' if n <= 3 else '4-6' if n <= 6 else '7-11' if n <= 11 else '12+'


def share_cat(k, n):
    if n == 0:
        return 'na'
    r = k / n
    return 'all' if r == 1 else 'most' if r >= .5 else 'few' if r > 0 else 'none'


def abstract(t, meta):
    """Tablet -> abstract record (roles only).  Entry e = dict(len, ini, fin, sys, bare, lac)."""
    L = t['lines']
    m = meta.get(t['id'], {})
    hdr = None
    start = 0
    if L and not L[0]['numerals'] and bsigns(L[0]):
        hdr = bsigns(L[0]); start = 1
    obv = [l for l in L[start:] if l['surface'] == 'obverse']
    ent = [l for l in obv if l['numerals']]
    off = [l for l in L if l['surface'] == 'reverse' and l['numerals']]
    top = [l for l in L if l['surface'] in ('top', 'left') and l['numerals']]
    revtext = [l for l in L if l['surface'] == 'reverse' and not l['numerals'] and bsigns(l)]
    notes = m.get('notes', [])
    rnotes = ' '.join(n for s, n in notes if s == 'reverse')
    onotes = ' '.join(n for s, n in notes if s in ('obverse', None))
    surfaces = m.get('surfaces', [])
    A = {}
    if hdr is None:
        A['hdr'] = 'none'
    else:
        h0 = hdr[0]
        A['hdr'] = 'M157' if 'M157' in h0 else 'M327' if ('M327' in h0 or 'M342' in h0) else 'other'
    A['hlen'] = 0 if hdr is None else len(hdr)
    A['hdr_miss'] = bool(L and L[0]['lacuna']) or 'beginning broken' in onotes
    E = []
    for l in ent:
        sg = bsigns(l)
        E.append(dict(len=len(sg), ini=bool(len(sg) > 1 and sg[0] in INITIAL),
                      fin=(sg[-1] if sg and sg[-1] in FINAL else None), sraw=system_of(l['numerals']),
                      sys=esys(l['numerals']), bare=bool(not sg and not l['lacuna']), lac=bool(l['lacuna']),
                      final=(sg[-1] if sg else '<bare>')))
    A['ents'] = E
    A['nent_miss'] = any(l['lacuna'] for l in obv) or ('broken' in onotes) or ('missing' in onotes)
    seen = False; sub = 0
    for l in obv:
        if l['numerals']:
            seen = True
        elif seen and bsigns(l):
            sub = 1
    A['sub'] = sub
    if len(off) >= 2:
        A['rev'] = 'entries'
    elif len(off) == 1:
        A['rev'] = 'total'
    elif revtext:
        A['rev'] = 'text'
    elif 'reverse' not in surfaces:
        A['rev'] = 'none'
    elif 'broken' in rnotes or 'missing' in rnotes:
        A['rev'] = 'broken'
    else:
        A['rev'] = 'blank'
    A['tot'] = None
    if len(off) == 1:
        A['tot'] = dict(sraw=system_of(off[0]['numerals']), sign=bool(bsigns(off[0])))
    A['top'] = int(bool(top))
    A['col'] = int(any(c[1] not in ('1',) for c in m.get('cols', [])))
    A['seal'] = int(bool(m.get('seal_note') or m.get('sd')))
    A['frag'] = any(l['lacuna'] for l in L) or ('broken' in onotes) or ('missing' in onotes)
    return A


def _prof(sraws):
    st = {s for s in sraws if s}
    if not st:
        return 'none'
    if 'B' in st:
        return 'B'
    if st & {'N23', 'mod*'}:
        return 'O'
    c = bool(st & {'C', 'C*'}); s_ = bool(st & {'SDB', 'S-frac'})
    if c and s_:
        return 'CS'
    if c:
        return 'C'
    return 'F' if 'S-frac' in st else 'S'


def features(A):
    """Abstract record -> skeleton feature dict (None = unobservable)."""
    f = {}
    E = A['ents']
    f['HDR'] = A['hdr']
    f['HLEN'] = '0' if A['hdr'] == 'none' else '1' if A['hlen'] == 1 else '2+'
    f['SYS'] = _prof([e['sraw'] for e in E])
    f['NENT'] = bin_nent(len(E))
    sg = [e for e in E if e['len'] > 0]
    f['ONE'] = share_cat(sum(1 for e in sg if e['len'] == 1), len(sg))
    cl = [e['fin'] for e in sg if e['fin']]
    r = len(cl) / len(sg) if sg else 0
    f['CLS'] = 'none' if not cl else 'most' if r >= .5 else 'some'
    f['CLSR'] = 'na' if len(cl) < 2 else 'same' if len(set(cl)) == 1 else 'varied'
    f['PRE'] = str(int(any(e['ini'] for e in sg)))
    f['BARE'] = str(int(any(e['bare'] for e in E)))
    f['SUB'] = str(A['sub'])
    f['REV'] = A['rev']
    if A['tot'] is None:
        f['TOT'] = 'none'
    elif not A['tot']['sign']:
        f['TOT'] = 'bare'
    else:
        ts, es = _prof([A['tot']['sraw']]), f['SYS']
        same = ts == es or (es == 'CS' and ts in ('C', 'S')) or ({es, ts} == {'F', 'S'})
        f['TOT'] = 'same' if same else 'conv'
    f['TOP'] = str(A['top']); f['COL'] = str(A['col']); f['SEAL'] = str(A['seal'])
    if A['hdr_miss']:
        f['HDR'] = f['HLEN'] = None
    if A['nent_miss']:
        f['NENT'] = None
    if f['REV'] in ('broken', 'none'):
        f['REV'] = f['TOT'] = None
    return f


def load_abstract(min_ent=2, clean=False):
    T = load()
    meta = json.load(open(os.path.join(DATA, 'pe8_meta.json')))
    out = []
    for t in T:
        if t.get('object_type', 'tablet') != 'tablet':
            continue
        A = abstract(t, meta)
        if len(A['ents']) < min_ent or (clean and A['frag']):
            continue
        A['id'] = t['id']; A['meta'] = meta.get(t['id'], {}); A['prov'] = t['provenience']
        out.append(A)
    return out


def load_skeletons(min_ent=2, clean=False):
    out = []
    for A in load_abstract(min_ent, clean):
        out.append({'id': A['id'], 'f': features(A), 'frag': A['frag'], 'ents': A['ents'], 'meta': A['meta'],
                    'prov': A['prov'], 'A': A})
    return out


# ------------------------------------------------------------------------ generative controls (abstract level)
TAB_FIELDS = [('hdr', 'hlen', 'hdr_miss'), ('sub',), ('rev', 'tot'), ('top',), ('col',), ('seal',), ('nent_miss',)]


def null_corpus(Areal, n, seed):
    """'Layout generated independently per entry': number of entries, each entry, and each tablet-level field
    group are drawn independently from the real pools.  No tablet-level coupling at all."""
    rng = random.Random(seed)
    pool = [e for A in Areal for e in A['ents']]
    out = []
    for i in range(n):
        A = {}
        for grp in TAB_FIELDS:
            src = rng.choice(Areal)
            for k in grp:
                A[k] = src[k]
        ne = len(rng.choice(Areal)['ents'])
        A['ents'] = [rng.choice(pool) for _ in range(ne)]
        out.append(A)
    return out


def planted_corpus(Areal, n, seed, ntemp=6, fidelity=0.85, min_ham=3):
    """Corpus written from `ntemp` fixed forms.  A form fixes: header block, reverse/total block, top, column,
    seal, subscript use (each copied from one real tablet), an entry-count range, ONE system set for entries,
    an entry-length class (one-sign / multi-sign) and a class-sign rate.  Each tablet copies each form field
    with probability `fidelity`, else draws that field from the real pool (scribal noise); entries are drawn
    from real entries that satisfy the form's entry rules.  Damage (missingness) is copied from real tablets."""
    rng = random.Random(seed)
    pool = [e for A in Areal for e in A['ents']]
    systems = [{'S'}, {'C'}, {'S', 'C'}, {'O'}]
    good = [A for A in Areal if not A['hdr_miss'] and A['rev'] not in ('broken', 'none')]
    def make_form():
        F = {}
        for grp in TAB_FIELDS[:-1]:
            src = rng.choice(good)
            for k in grp:
                F[k] = src[k]
        F['ne'] = rng.choice([(2, 3), (4, 6), (7, 11), (2, 6)])
        F['sys'] = rng.choice(systems[:3]) if rng.random() < .9 else systems[3]
        F['one'] = rng.random() < .5
        F['clsrate'] = rng.choice([0.0, 0.2, 0.8])
        return F
    def gen(F, noisy=True):
        A = {}
        for grp in TAB_FIELDS[:-1]:
            src = F if (not noisy or rng.random() < fidelity) else rng.choice(good)
            for k in grp:
                A[k] = src[k]
        A['hdr_miss'] = False
        A['nent_miss'] = False
        if A['tot'] is not None and A['rev'] != 'total':
            A['tot'] = None
        ne = rng.randint(*F['ne'])
        sysset = F['sys'] if (not noisy or rng.random() < fidelity) else rng.choice(systems)
        one = F['one'] if (not noisy or rng.random() < fidelity) else (rng.random() < .5)
        E = []
        for _ in range(ne):
            for _try in range(200):
                e = rng.choice(pool)
                if e['sys'] not in sysset:
                    continue
                if one != (e['len'] == 1):
                    continue
                wantcls = rng.random() < F['clsrate']
                if (e['fin'] is not None) != wantcls and e['len'] > 0:
                    continue
                break
            E.append(e)
        A['ents'] = E
        return A
    forms, modes = [], []
    while len(forms) < ntemp:
        F = make_form()
        md = features(gen(F, noisy=False))
        if all(sum(md[k] != m2[k] for k in FEATS) >= min_ham for m2 in modes):
            forms.append(F); modes.append(md)
    w = [1.0 / (i + 1) for i in range(ntemp)]
    out, lab = [], []
    for i in range(n):
        k = rng.choices(range(ntemp), weights=w)[0]
        A = gen(forms[k])
        # damage: copy the unobservability pattern of a random real tablet
        src = rng.choice(Areal)
        A['hdr_miss'] = src['hdr_miss']; A['nent_miss'] = src['nent_miss']
        if src['rev'] in ('broken', 'none'):
            A['rev'] = src['rev']; A['tot'] = None
        out.append(A); lab.append(k)
    return out, lab, modes


# ------------------------------------------------------------------------------------------------- encoding
def encode(rows, feats=FEATS):
    levels = {k: sorted({r[k] for r in rows if r[k] is not None}) for k in feats}
    X = np.array([[levels[k].index(r[k]) if r[k] is not None else -1 for k in feats] for r in rows],
                 dtype=np.int64)
    return X, levels


# ------------------------------------------------------------------------- latent class model (finite, EM)
def _lt(th, x):
    """n x K log-prob of column x under each cluster; missing (-1) contributes 0."""
    v = np.log(th[:, np.maximum(x, 0)]).T
    v[x < 0] = 0.0
    return v


def lcm_em(X, card, K, rng, iters=300, tol=1e-6, prior=0.5, init=None):
    """Mixture of independent categoricals.  X: n x F ints, card: list of level counts."""
    n, F = X.shape
    if init is None:
        R = rng.dirichlet(np.ones(K), size=n)
    else:
        R = init
    ll_old = -np.inf
    for it in range(iters):
        pi = (R.sum(0) + 1e-3) / (n + K * 1e-3)
        th = []
        for j in range(F):
            M = np.zeros((K, card[j])) + prior
            for v in range(card[j]):
                M[:, v] += R[X[:, j] == v].sum(0)
            th.append(M / M.sum(1, keepdims=True))
        logp = np.log(pi)[None, :].repeat(n, 0)
        for j in range(F):
            logp = logp + _lt(th[j], X[:, j])
        mx = logp.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(1))
        ll = lse.sum()
        R = np.exp(logp - lse[:, None])
        if ll - ll_old < tol * abs(ll):
            break
        ll_old = ll
    return dict(pi=pi, th=th, R=R, ll=ll, K=K)


def lcm_loglik(model, X, mask=None):
    """Per-row log-likelihood; mask = list of feature indices to use (others marginalised)."""
    n, F = X.shape
    feats = range(F) if mask is None else mask
    logp = np.log(model['pi'])[None, :].repeat(n, 0)
    for j in feats:
        logp = logp + _lt(model['th'][j], X[:, j])
    mx = logp.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(1))
    return lse, np.exp(logp - lse[:, None])


def nparams(card, K):
    return (K - 1) + K * sum(c - 1 for c in card)


def best_em(X, card, K, restarts, seed):
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        m = lcm_em(X, card, K, rng)
        if best is None or m['ll'] > best['ll']:
            best = m
    return best


def cv_ll(X, card, K, restarts, seed, folds=5):
    """Held-out log-likelihood per tablet (5-fold)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    parts = np.array_split(idx, folds)
    tot = 0.0
    for f in range(folds):
        te = parts[f]; tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        m = best_em(X[tr], card, K, restarts, seed + 17 * f + 1)
        tot += lcm_loglik(m, X[te])[0].sum()
    return tot / len(X)


# ------------------------------------------------------------------ DP mixture, collapsed Gibbs (CRP)
def dp_gibbs(X, card, alpha=1.0, beta=0.5, sweeps=200, seed=0, burn=100, sample_alpha=True):
    """Dirichlet-process mixture of independent categoricals, collapsed Gibbs, alpha resampled (Escobar-West,
    Gamma(1,1) prior).  Returns posterior samples of number of clusters and of clusters with >=1% of tablets."""
    rng = np.random.default_rng(seed)
    n, F = X.shape
    obs = [[j for j in range(F) if X[i, j] >= 0] for i in range(n)]
    z = rng.integers(0, 3, n)
    counts = {}  # cluster -> [size, list of F arrays]
    def newc():
        return [0, [np.zeros(c) for c in card]]
    for i in range(n):
        c = int(z[i])
        if c not in counts:
            counts[c] = newc()
        counts[c][0] += 1
        for j in range(F):
            if X[i, j] >= 0:
                counts[c][1][j][X[i, j]] += 1
    Ks, Kbig, alphas = [], [], []
    bsum = [beta * c for c in card]
    for sw in range(sweeps):
        for i in rng.permutation(n):
            c = int(z[i])
            counts[c][0] -= 1
            for j in obs[i]:
                counts[c][1][j][X[i, j]] -= 1
            if counts[c][0] == 0:
                del counts[c]
            keys = list(counts.keys())
            lp = np.empty(len(keys) + 1)
            for a, k in enumerate(keys):
                sz, cf = counts[k]
                v = math.log(sz)
                for j in obs[i]:
                    v += math.log((cf[j][X[i, j]] + beta) / (cf[j].sum() + bsum[j]))
                lp[a] = v
            v = math.log(alpha)
            for j in obs[i]:
                v += math.log(1.0 / card[j])
            lp[-1] = v
            lp -= lp.max()
            p = np.exp(lp); p /= p.sum()
            a = rng.choice(len(p), p=p)
            if a == len(keys):
                k = max(counts.keys(), default=-1) + 1
                counts[k] = newc()
            else:
                k = keys[a]
            z[i] = k
            counts[k][0] += 1
            for j in obs[i]:
                counts[k][1][j][X[i, j]] += 1
        K = len(counts)
        if sample_alpha:
            eta = rng.beta(alpha + 1, n)
            a0, b0 = 1.0, 1.0
            odds = (a0 + K - 1) / (n * (b0 - math.log(eta)))
            pm = odds / (1 + odds)
            alpha = rng.gamma(a0 + K if rng.random() < pm else a0 + K - 1, 1.0 / (b0 - math.log(eta)))
        if sw >= burn:
            Ks.append(K)
            Kbig.append(sum(1 for v in counts.values() if v[0] >= max(2, 0.01 * n)))
            alphas.append(alpha)
    return dict(K=Ks, Kbig=Kbig, alpha=alphas, z=z.copy())


# ------------------------------------------------------------------------------------- synthetic corpora
def column_shuffle(X, seed):
    rng = np.random.default_rng(seed)
    Y = X.copy()
    for j in range(X.shape[1]):
        Y[:, j] = rng.permutation(Y[:, j])
    return Y


def chow_liu(X, card, prior=0.5):
    """Chow-Liu tree (best tree-structured graded dependence model, no discrete classes)."""
    n, F = X.shape
    MI = np.zeros((F, F))
    for a in range(F):
        for b in range(a + 1, F):
            J = np.zeros((card[a], card[b])) + prior / (card[a] * card[b])
            ok = (X[:, a] >= 0) & (X[:, b] >= 0)
            np.add.at(J, (X[ok, a], X[ok, b]), 1)
            J /= J.sum()
            pa = J.sum(1, keepdims=True); pb = J.sum(0, keepdims=True)
            MI[a, b] = MI[b, a] = (J * np.log(J / (pa * pb))).sum()
    # max spanning tree (Prim) rooted at 0
    intree = {0}; par = {0: None}
    while len(intree) < F:
        best = None
        for a in intree:
            for b in range(F):
                if b not in intree and (best is None or MI[a, b] > best[0]):
                    best = (MI[a, b], a, b)
        intree.add(best[2]); par[best[2]] = best[1]
    cpt = {}
    for b in range(F):
        a = par[b]
        if a is None:
            p = np.bincount(X[X[:, b] >= 0, b], minlength=card[b]) + prior
            cpt[b] = p / p.sum()
        else:
            M = np.zeros((card[a], card[b])) + prior
            ok = (X[:, a] >= 0) & (X[:, b] >= 0)
            np.add.at(M, (X[ok, a], X[ok, b]), 1)
            cpt[b] = M / M.sum(1, keepdims=True)
    return dict(par=par, cpt=cpt)


def chow_liu_ll(m, X):
    ll = np.zeros(len(X))
    for b, a in m['par'].items():
        ll += np.log(m['cpt'][b][X[:, b]]) if a is None else np.log(m['cpt'][b][X[:, a], X[:, b]])
    return ll


def indep_ll(Xtr, Xte, card, prior=0.5):
    ll = np.zeros(len(Xte))
    for j in range(Xtr.shape[1]):
        p = np.bincount(Xtr[Xtr[:, j] >= 0, j], minlength=card[j]) + prior
        p = p / p.sum()
        ll += np.log(p[Xte[:, j]])
    return ll
