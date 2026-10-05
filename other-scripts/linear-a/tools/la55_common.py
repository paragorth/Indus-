#!/usr/bin/env python3
"""LA-55 shared code: WORDS GROW LIKE CITIES.

Urban-scaling analysis of archive vocabularies. An "archive" (unit) is a deposit / find area / site-year.
For every word (or sign) type w and unit a: k_wa = number of documents in a containing w, N_a = documents
in a. Scaling fits log E[k_wa] = alpha_w + beta_w log N_a (Poisson, zeros included), a threshold model
(absent below T, proportional above) and occupancy (number of units holding w). Every statistic is
z-scored against a null in which documents are re-dealt to units (unit sizes kept): N1 = free shuffle,
N3 = shuffle within document genre (support type / series letter). N2 = bootstrap of documents inside units.

Corpora: Linear A (la51 deposits), Linear B (DAMOS find areas: KN, PY, TH, MY, TI, KH), Ur III
(site x year-name). Words are opaque ids; no sound values are used.
"""
import json, os, re, sys, collections, hashlib, math, random
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la55_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# =================================================================== corpora
# doc = dict(id, unit, genre, terms:set)

def la_units(level='deposit', signs=False):
    from la51_common import load_la
    C = {d['id']: d for d in json.load(open(os.path.join(D, 'corpus.json')))}
    docs = []
    for d in load_la():
        if level == 'deposit':
            u = d['deposit'] or ('SITE_' + d['site'])
        else:
            u = d['site'] or '?'
        terms = set(d['terms'])
        if signs:
            terms = set()
            for t in d['terms']:
                if t.startswith('w:') or t.startswith('s:'):
                    for s in t[2:].split('-'):
                        terms.add('g:' + s)
        docs.append(dict(id=d['id'], unit=u, genre=d['support'].lower().split(' (')[0], site=d['site'],
                         terms=terms))
    return docs


def lb_units(signs=False):
    fn = os.path.join(CK, 'lb_docs.json')
    if not os.path.exists(fn):
        from la51_common import load_lb
        B = load_lb()
        out = []
        for d in B:
            ser = re.match(r'[A-Z]{2}\s+([A-Z])', d['id'])
            out.append(dict(id=d['id'], unit=d['site'] + '|' + d['deposit'], site=d['site'],
                            genre=d['site'][:2] + ':' + (ser.group(1) if ser else '?'), terms=sorted(d['terms'])))
        json.dump(out, open(fn, 'w'))
    out = json.load(open(fn))
    for d in out:
        d['terms'] = set(d['terms'])
        if signs:
            d['terms'] = {'g:' + s for t in d['terms'] if t.startswith('w:') for s in t[2:].split('-')}
    return out


def lb_truth_classes():
    """Known LB classes (controls only). FUNC = totals, deficits, transaction verbs/terms (la45 lists);
    COMM = logograms; PLACE = la45 place list; PERSON = word that, in at least half of its occurrences, opens a
    document of a person-headed series (KN D*, Vc, Sc, As, Ai, Uf; PY Ea, Eb, En, Eo, Ep)."""
    from la45_common import lb_docs_all, LB_TRUTH
    B = lb_docs_all()
    PSER = re.compile(r'^(KN (D|Vc|Sc|As|Ai|Uf)|PY (Ea|Eb|En|Eo|Ep))')
    first = collections.Counter(); tot = collections.Counter()
    for d in B:
        ws = [x[1] for x in d['toks'] if x[0] == 'T']
        for w in ws:
            tot[w] += 1
        if ws and PSER.match(d['site'] + ' ' + d.get('series', '')) and not ws[0].startswith('L:'):
            first[ws[0]] += 1
    out = {}
    for w, c in tot.items():
        if w.startswith('L:'):
            out['L:' + w[2:].split('+')[0]] = 'COMM'
        elif w in LB_TRUTH:
            r = LB_TRUTH[w]
            out['w:' + w] = 'PLACE' if r == 'PLACE' else 'FUNC'
        elif first[w] >= max(1, 0.5 * c):
            out['w:' + w] = 'PERSON'
    return out


def ur_units(max_docs=None):
    fn = os.path.join(CK, 'ur_docs.json')
    if not os.path.exists(fn):
        src = json.load(open(os.path.join(HERE, '..', '..', 'proto-elamite', 'data', 'pe38_ckpt', 'ur3_docs.json')))
        out = []
        for d in src:
            year = None; terms = set(); toks = []
            for l in d['lines']:
                t = [x for x in l['toks'] if x]
                if not t:
                    continue
                if t[0] == 'mu' and year is None:
                    year = ' '.join(t[:3]); continue
                for x in t:
                    terms.add('w:' + x); toks.append(x)
            if year is None or not d['site'] or d['site'] == 'uncertain' or not terms:
                continue
            site = d['site'].replace('Irisaĝrig', 'Irisagrig')
            out.append(dict(id=d['id'], unit=site + '|' + year, site=site, genre=site, terms=sorted(terms),
                            toks=toks))
        json.dump(out, open(fn, 'w'))
    out = json.load(open(fn))
    for d in out:
        d['terms'] = set(d['terms'])
    if max_docs and len(out) > max_docs:
        rng = random.Random(seed('la55-ur-sub'))
        units = sorted(set(d['unit'] for d in out)); rng.shuffle(units)
        keep, n = set(), 0
        cnt = collections.Counter(d['unit'] for d in out)
        for u in units:
            if n >= max_docs: break
            keep.add(u); n += cnt[u]
        out = [d for d in out if d['unit'] in keep]
    return out


def ur_truth_classes(docs):
    from la45_common import ur_truth
    t = ur_truth([dict(toks=[('T', x) for x in d['toks']]) for d in docs])
    out = {}
    for w, c in t.items():
        if c in ('TOTAL', 'DEFICIT', 'VERB', 'UNIT', 'DATE', 'OFFICE'):
            out['w:' + w] = 'FUNC'
        elif c == 'COMMODITY':
            out['w:' + w] = 'COMM'
        elif c in ('PERSON', 'PLACE'):
            out['w:' + w] = c
    return out


def build_corpus(corpus):
    if corpus == 'LA': return la_units(), 3
    if corpus == 'LAG': return la_units(signs=True), 3
    if corpus == 'LB': return lb_units(), 3
    if corpus == 'LBS':
        docs = lb_units(); cnt = collections.Counter(d['unit'] for d in docs)
        us = sorted(u for u in cnt if cnt[u] >= 3); r = random.Random(seed('la55-lbs')); r.shuffle(us)
        keep, n = set(), 0
        for u in us:
            if n >= 1574: break
            keep.add(u); n += cnt[u]
        return [d for d in docs if d['unit'] in keep], 3
    if corpus == 'UR': return ur_units(max_docs=12000), 5



# =================================================================== matrices
class Data:
    def __init__(self, docs, min_unit=3, min_k=5):
        cnt = collections.Counter(d['unit'] for d in docs)
        docs = [d for d in docs if cnt[d['unit']] >= min_unit]
        tk = collections.Counter(t for d in docs for t in d['terms'])
        self.types = sorted(t for t, c in tk.items() if c >= min_k)
        ti = {t: i for i, t in enumerate(self.types)}
        self.units = sorted(set(d['unit'] for d in docs))
        ui = {u: i for i, u in enumerate(self.units)}
        self.docs = docs
        self.u = np.array([ui[d['unit']] for d in docs])
        gl = sorted(set(d['genre'] for d in docs)); gi = {g: i for i, g in enumerate(gl)}
        self.g = np.array([gi[d['genre']] for d in docs])
        ln = np.array([len(d['terms']) for d in docs], float)
        lb = np.zeros(len(docs), int)
        for g in np.unique(self.g):         # length quartiles inside each genre
            m = self.g == g
            if m.sum() >= 8:
                e = np.unique(np.quantile(ln[m], [0.25, 0.5, 0.75]))
                lb[m] = np.digitize(ln[m], e, right=True)
        self.gl = self.g * 10 + lb
        r, c = [], []
        for i, d in enumerate(docs):
            for t in d['terms']:
                if t in ti:
                    r.append(i); c.append(ti[t])
        self.X = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(docs), len(self.types)))
        self.A = len(self.units)
        self.Ktot = np.asarray(self.X.sum(0)).ravel()

    def counts(self, u=None):
        u = self.u if u is None else u
        U = sp.csr_matrix((np.ones(len(u)), (u, np.arange(len(u)))), shape=(self.A, len(u)))
        K = np.asarray((U @ self.X).todense()).T          # W x A
        N = np.bincount(u, minlength=self.A).astype(float)
        return K, N

    def shuffle(self, rng, strat=False):
        u = self.u.copy()
        if not strat:
            return rng.permutation(u)
        G = self.gl if strat == 'len' else self.g
        for g in np.unique(G):
            idx = np.where(G == g)[0]
            u[idx] = rng.permutation(u[idx])
        return u

    def boot(self, rng):
        """N2: resample documents with replacement inside units -> row index array."""
        idx = []
        for a in range(self.A):
            m = np.where(self.u == a)[0]
            idx.append(rng.choice(m, size=len(m), replace=True))
        return np.concatenate(idx)


# =================================================================== statistics
def fit_beta(K, N, iters=30):
    """Vectorised Poisson MLE of log mu = a + b log N for each row of K. Returns b (ridge 1e-3 toward 1)."""
    x = np.log(N)
    W = K.shape[0]
    tot = K.sum(1)
    b = np.ones(W)
    a = np.log(np.maximum(tot, 1e-9) / N.sum())
    lam = 1e-3
    for _ in range(iters):
        eta = np.clip(a[:, None] + b[:, None] * x[None, :], -50, 50)
        mu = np.exp(eta)
        g_a = (K - mu).sum(1)
        g_b = ((K - mu) * x).sum(1) - lam * (b - 1)
        h_aa = mu.sum(1) + 1e-9
        h_ab = (mu * x).sum(1)
        h_bb = (mu * x * x).sum(1) + lam
        det = h_aa * h_bb - h_ab ** 2 + 1e-12
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a += np.clip(da, -5, 5); b += np.clip(db, -5, 5)
        b = np.clip(b, -3, 6)
    return b


def thr_gain(K, N):
    """Threshold model: mu = r N [N >= T]; best T over unit sizes. Returns (gain LL vs proportional, T*)."""
    order = np.argsort(N)
    Ns = N[order]; Ks = K[:, order]
    tot = K.sum(1)
    # proportional LL (constant terms dropped): sum k log(r N) - r N, r = tot / sum N
    r0 = tot / N.sum()
    ll0 = (Ks * np.log(np.maximum(r0[:, None] * Ns[None, :], 1e-300))).sum(1) - tot
    best = ll0.copy(); bestT = np.full(K.shape[0], Ns[0])
    cumK_below = np.cumsum(Ks, 1)
    sufN = np.cumsum(Ns[::-1])[::-1]
    for j in range(1, len(Ns)):
        if Ns[j] == Ns[j - 1]:
            continue
        ok = cumK_below[:, j - 1] == 0
        r = tot / sufN[j]
        ll = (Ks[:, j:] * np.log(np.maximum(r[:, None] * Ns[None, j:], 1e-300))).sum(1) - tot
        better = ok & (ll > best)
        best[better] = ll[better]; bestT[better] = Ns[j]
    return best - ll0, bestT


def occupancy(K):
    return (K > 0).sum(1).astype(float)


def features(K, N):
    b = fit_beta(K, N)
    g, T = thr_gain(K, N)
    return dict(beta=b, thr=g, T=T, occ=occupancy(K))


def null_features(data, nrep, rng, strat=False):
    out = collections.defaultdict(list)
    for _ in range(nrep):
        K, N = data.counts(data.shuffle(rng, strat))
        f = features(K, N)
        for k in ('beta', 'thr', 'occ'):
            out[k].append(f[k])
    return {k: np.array(v) for k, v in out.items()}


def zscores(real, null):
    z = {}
    for k in ('beta', 'thr', 'occ'):
        m = null[k].mean(0); s = null[k].std(0) + 1e-9
        z[k] = (real[k] - m) / s
        z[k + '_p_hi'] = ((null[k] >= real[k][None, :]).sum(0) + 1) / (len(null[k]) + 1)
        z[k + '_p_lo'] = ((null[k] <= real[k][None, :]).sum(0) + 1) / (len(null[k]) + 1)
    return z


def classify(z, alpha=0.05):
    """Fixed rule (set before Linear A is read):
       THRESHOLD  thr gain above null (p_hi < alpha) and beta not below null
       SATURATING beta below null (p_lo < alpha) or occupancy above null (p_hi < alpha) with beta not above
       SUPERLINEAR beta above null (p_hi < alpha)
       LINEAR     otherwise."""
    W = len(z['beta'])
    cls = np.array(['LINEAR'] * W, dtype=object)
    sup = z['beta_p_hi'] < alpha
    sat = (z['beta_p_lo'] < alpha) | ((z['occ_p_hi'] < alpha) & ~sup)
    thr = (z['thr_p_hi'] < alpha) & ~sat
    cls[sup] = 'SUPERLINEAR'
    cls[thr] = 'THRESHOLD'
    cls[sat] = 'SATURATING'
    return cls


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)
    # average ties
    from scipy.stats import rankdata
    ranks = rankdata(allv)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# =================================================================== held-out prediction (leave one unit out)
def louo_gain(K, N, min_tot=5):
    """For each unit a: fit (alpha, beta) on the other units; predict k_wa. Return bits gained per word-unit
    over the proportional model (beta = 1) fitted on the same other units. Words with total >= min_tot."""
    keep = K.sum(1) >= min_tot
    K = K[keep]
    A = K.shape[1]
    gain = 0.0; n = 0
    x = np.log(N)
    for a in range(A):
        m = np.ones(A, bool); m[a] = False
        Ko, No = K[:, m], N[m]
        ok = Ko.sum(1) > 0
        b = fit_beta(Ko[ok], No)
        xo = np.log(No)
        # alpha given beta: sum k = e^a sum N^b
        al = np.log(Ko[ok].sum(1) / (np.exp(b[:, None] * xo[None, :]).sum(1)))
        mu1 = np.exp(al + b * x[a])
        mu0 = Ko[ok].sum(1) / No.sum() * N[a]
        k = K[ok, a]
        ll1 = k * np.log(mu1 + 1e-300) - mu1
        ll0 = k * np.log(mu0 + 1e-300) - mu0
        gain += (ll1 - ll0).sum() / math.log(2); n += ok.sum()
    return gain / max(n, 1), int(n)


def jdump(obj, name):
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        raise TypeError(type(o))
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=conv)
