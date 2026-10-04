#!/usr/bin/env python3
"""LA-20 shared code: THE PECKING ORDER.

Every pair of items written in the same list is a contest won by the one written first.
Global rankings (Bradley-Terry, Plackett-Luce, Elo, Bayesian BT by MCMC, mixtures of hidden
orders) are fitted to all lists at once and judged by held-out prediction of the order of item
pairs on unseen tablets, against a null that shuffles entry order within each tablet.

Lists: one per tablet (sides a/b joined, a first). Item types:
  W  all words (first occurrence), totals (KU-RO, KI-RO, PO-TO-KU-RO) removed
  E  entry words only (word directly followed by a number or logogram), totals removed
  L  logograms (base sign before '+'), first occurrence
  F  first signs of words (first occurrence)
Each item carries the quantity (integer part) written right after it, when there is one.
"""
import json, os, re, random, unicodedata
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la20_ckpt')
os.makedirs(CK, exist_ok=True)
SUPPORTS = {'Tablet', 'Lames (short thin tablet)', '3-sided bar', '4-sided bar'}
TOTALS = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}


def base_id(i):
    return re.sub(r'(?<=\d)[a-e]$', '', i)


def _items_from_tokens(toks):
    """toks: list of (kind, value) with kind in word/logo/num/nl/div. Returns dict of item lists."""
    out = {'W': [], 'E': [], 'L': [], 'F': [], 'T': []}
    seen = {k: set() for k in out}
    n = len(toks)
    for i, (k, v) in enumerate(toks):
        if k not in ('word', 'logo'): continue
        # quantity: first number after this token before the next word
        q = None; nxt = None
        for j in range(i + 1, n):
            kj, vj = toks[j]
            if kj == 'nl' or kj == 'div': continue
            if nxt is None: nxt = kj
            if kj == 'word': break
            if kj == 'num': q = vj; break
            if kj == 'logo' and k == 'logo': break
        if k == 'word':
            if v in TOTALS:
                if v not in seen['T']: seen['T'].add(v); out['T'].append((v, q))
                continue
            if v not in seen['W']: seen['W'].add(v); out['W'].append((v, q))
            if nxt in ('num', 'logo') and v not in seen['E']: seen['E'].add(v); out['E'].append((v, q))
            f = v.split('-')[0]
            if f not in seen['F']: seen['F'].add(f); out['F'].append((f, q))
        else:
            b = v.split('+')[0]
            if b not in seen['L']: seen['L'].add(b); out['L'].append((b, q))
    return out


def load_la():
    c = json.load(open(os.path.join(D, 'corpus.json')))
    groups = defaultdict(list)
    meta = {}
    for r in c:
        if r['support'] not in SUPPORTS: continue
        b = base_id(r['id'])
        for t in r['tokens']:
            if t['t'] == 'word': groups[b].append(('word', '-'.join(t['s'])))
            elif t['t'] == 'logo': groups[b].append(('logo', t['v']))
            elif t['t'] == 'num': groups[b].append(('num', t['v']))
            elif t['t'] == 'nl': groups[b].append(('nl', None))
            elif t['t'] == 'div': groups[b].append(('div', None))
        groups[b].append(('nl', None))
        meta[b] = r['site']
    docs = []
    for b, toks in groups.items():
        it = _items_from_tokens(toks)
        docs.append({'id': b, 'site': meta[b], 'items': it})
    return docs


def _clean_lb(t):
    t = unicodedata.normalize('NFD', t)
    t = ''.join(ch for ch in t if unicodedata.category(ch) != 'Mn')
    t = re.sub(r"[\[\]⟦⟧⌞⌟⸢⸣'\"?!<>{}]", '', t)
    return t.strip('-')


def load_lb(prefixes=('PY',)):
    docs = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        x = json.loads(line)
        if not x.get('content'): continue
        h = x['heading']
        if not h[:2] in prefixes: continue
        toks = []
        for ln in x['content'].split('\n'):
            ln = ln.strip()
            if not ln: continue
            for raw in ln.split():
                if raw.startswith('.') or raw in (',', '/', '//', ':'): continue
                t = _clean_lb(raw)
                if not t or '.' in t: continue
                if t.lower() in ('vacat', 'vac', 'lat', 'sup', 'inf', 'mut', 'vest', 'deest', 'fr', 'v'): continue
                if re.fullmatch(r'\d+', t): toks.append(('num', int(t))); continue
                if any(ch.islower() for ch in t):
                    if not re.fullmatch(r'[a-z0-9*\-]+', t): continue
                    s = [p for p in t.split('-') if p]
                    if s: toks.append(('word', '-'.join(s).upper()))
                    continue
                if re.fullmatch(r"[A-Z*0-9+]+", t) and len(t) > 1: toks.append(('logo', t))
            toks.append(('nl', None))
        it = _items_from_tokens(toks)
        docs.append({'id': h, 'site': h[:2], 'items': it})
    return docs


def orders(docs, typ, min_len=2):
    """list of (doc index, [item names]) with >= min_len distinct items"""
    out = []
    for i, d in enumerate(docs):
        o = [x for x, _ in d['items'][typ]]
        if len(o) >= min_len: out.append((i, o))
    return out


# ------------------------------------------------------------------ models

def pair_counts(ords, index):
    """wins[a,b] = times a written before b. returns sparse arrays (a, b, count)"""
    c = Counter()
    for o in ords:
        ids = [index[x] for x in o if x in index]
        for p in range(len(ids)):
            for q in range(p + 1, len(ids)):
                c[(ids[p], ids[q])] += 1
    if not c: return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    k = np.array(list(c.keys())); v = np.array(list(c.values()), float)
    return k[:, 0], k[:, 1], v


def fit_bt(ords, lam=1.0, weight_by_len=True):
    """MAP Bradley-Terry with N(0, 1/lam) prior. Pairs from one list are down-weighted by
    1/(len-1) so a long list does not dominate. Returns dict item -> score."""
    items = sorted({x for o in ords for x in o})
    index = {x: i for i, x in enumerate(items)}
    c = Counter()
    for o in ords:
        w = 1.0 / (len(o) - 1) if weight_by_len else 1.0
        for p in range(len(o)):
            for q in range(p + 1, len(o)):
                c[(index[o[p]], index[o[q]])] += w
    if not c: return {}
    k = np.array(list(c.keys())); v = np.array(list(c.values()))
    a, b = k[:, 0], k[:, 1]
    n = len(items)

    def f(s):
        d = s[a] - s[b]
        lp = -np.logaddexp(0, -d)
        val = -(v * lp).sum() + 0.5 * lam * (s ** 2).sum()
        g = 1 / (1 + np.exp(d))  # = 1 - sigmoid(d)
        gr = np.zeros(n)
        np.add.at(gr, a, -v * g); np.add.at(gr, b, v * g)
        return val, gr + lam * s
    r = minimize(f, np.zeros(n), jac=True, method='L-BFGS-B')
    return dict(zip(items, r.x))


def fit_pl(ords, lam=1.0, iters=200):
    """MAP Plackett-Luce (top-first choice) by gradient; returns item -> log-worth."""
    items = sorted({x for o in ords for x in o})
    index = {x: i for i, x in enumerate(items)}
    O = [np.array([index[x] for x in o]) for o in ords]
    n = len(items)

    def f(s):
        val = 0.5 * lam * (s ** 2).sum(); gr = lam * s.copy()
        for o in O:
            so = s[o]
            # log P = sum_k so[k] - logsumexp(so[k:])
            m = len(o)
            for k in range(m - 1):
                t = so[k:]; mx = t.max(); e = np.exp(t - mx); Z = e.sum()
                val -= so[k] - (mx + np.log(Z))
                gr[o[k]] -= 1
                np.add.at(gr, o[k:], e / Z)
        return val, gr
    r = minimize(f, np.zeros(n), jac=True, method='L-BFGS-B', options={'maxiter': iters})
    return dict(zip(items, r.x))


def fit_elo(ords, K=16.0, reps=20, seed=0):
    rng = random.Random(seed)
    acc = defaultdict(float)
    for rep in range(reps):
        R = defaultdict(float)
        L = list(ords); rng.shuffle(L)
        for o in L:
            for p in range(len(o)):
                for q in range(p + 1, len(o)):
                    a, b = o[p], o[q]
                    e = 1 / (1 + 10 ** ((R[b] - R[a]) / 400))
                    R[a] += K * (1 - e); R[b] -= K * (1 - e)
        for x, v in R.items(): acc[x] += v / reps
    return dict(acc)


def fit_meanpos(ords, prior=1.0):
    """baseline: shrunk mean relative position (0 first .. 1 last); score = -(mean)"""
    s = defaultdict(float); n = defaultdict(float)
    for o in ords:
        m = len(o)
        for k, x in enumerate(o):
            s[x] += k / (m - 1) - 0.5; n[x] += 1
    return {x: -s[x] / (n[x] + prior) for x in s}


FITTERS = {'BT': fit_bt, 'PL': fit_pl, 'ELO': fit_elo, 'POS': fit_meanpos}


def eval_pairs(score, test_ords, train_cooc=None):
    """returns (n_testable, n_correct, n_unmet, n_unmet_correct). unmet = pair never in the same
    training list. Pairs with equal scores count half."""
    n = c = nu = cu = 0.0
    for o in test_ords:
        for p in range(len(o)):
            for q in range(p + 1, len(o)):
                a, b = o[p], o[q]
                if a not in score or b not in score: continue
                d = score[a] - score[b]
                w = 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
                n += 1; c += w
                if train_cooc is not None and frozenset((a, b)) not in train_cooc:
                    nu += 1; cu += w
    return n, c, nu, cu


def cooc_set(ords):
    s = set()
    for o in ords:
        for p in range(len(o)):
            for q in range(p + 1, len(o)):
                s.add(frozenset((o[p], o[q])))
    return s


def cv_score(ords, fitter, folds=5, seed=0, groups=None):
    """K-fold over lists. Returns totals (n, correct, n_unmet, correct_unmet)."""
    rng = random.Random(seed)
    idx = list(range(len(ords))); rng.shuffle(idx)
    tot = np.zeros(4)
    for f in range(folds):
        te = set(idx[f::folds])
        tr = [ords[i] for i in range(len(ords)) if i not in te]
        ts = [ords[i] for i in te]
        sc = fitter(tr)
        tot += eval_pairs(sc, ts, cooc_set(tr))
    return tot


def shuffle_within(ords, rng):
    out = []
    for o in ords:
        o = list(o); rng.shuffle(o); out.append(o)
    return out


def acc(t):
    return (t[1] / t[0] if t[0] else float('nan'), t[3] / t[2] if t[2] else float('nan'))


def row(f, tag, method, result, verdict):
    line = '| LA-20.%s | %s | %s | %s |' % (tag, method, result, verdict)
    print(line)
    if f: f.write(line + '\n'); f.flush()


# ------------------------------------------------------------------ mixtures (cycle 2)

class PairTable:
    """all within-list pairs as flat arrays: list id, earlier item, later item, weight 1/(m-1)"""
    def __init__(self, ords, index):
        L, A, B, W = [], [], [], []
        for li, o in enumerate(ords):
            ids = [index.get(x, -1) for x in o]; m = len(o)
            for p in range(m):
                for q in range(p + 1, m):
                    L.append(li); A.append(ids[p]); B.append(ids[q]); W.append(1.0 / (m - 1))
        self.l = np.array(L, int); self.a = np.array(A, int); self.b = np.array(B, int)
        self.w = np.array(W); self.n = len(ords)

    def ll(self, s):
        """per-list composite log-lik; unknown items (-1) score 0"""
        s2 = np.append(s, 0.0)
        d = s2[self.a] - s2[self.b]
        return np.bincount(self.l, weights=-self.w * np.logaddexp(0, -d), minlength=self.n)


def fit_bt_pairs(pt, v, n, lam=1.0, x0=None):
    a, b = pt.a, pt.b
    def f(s):
        d = s[a] - s[b]
        val = (v * np.logaddexp(0, -d)).sum() + 0.5 * lam * (s ** 2).sum()
        g = v / (1 + np.exp(d))
        gr = np.bincount(b, weights=g, minlength=n) - np.bincount(a, weights=g, minlength=n)
        return val, gr + lam * s
    return minimize(f, np.zeros(n) if x0 is None else x0, jac=True, method='L-BFGS-B').x


def fit_mixture(ords, K, iters=30, seed=0, lam=1.0):
    items = sorted({x for o in ords for x in o}); index = {x: i for i, x in enumerate(items)}
    pt = PairTable(ords, index); n = len(items)
    rng = np.random.default_rng(seed)
    R = rng.dirichlet(np.ones(K) * 0.5, size=len(ords)); S = [None] * K
    for it in range(iters):
        S = [fit_bt_pairs(pt, pt.w * R[pt.l, k], n, lam, S[k]) for k in range(K)]
        pi = R.mean(0) + 1e-9
        LL = np.stack([pt.ll(S[k]) for k in range(K)], 1) + np.log(pi)
        mx = LL.max(1, keepdims=True); P = np.exp(LL - mx); R = P / P.sum(1, keepdims=True)
    return items, index, S, pi


def mixture_heldout(ords, K, folds=5, seed=0, restarts=1):
    """held-out composite log-lik gain (nats, summed over test lists) over the coin-flip order"""
    rng = random.Random(seed); idx = list(range(len(ords))); rng.shuffle(idx)
    gain = 0.0
    for f in range(folds):
        te = sorted(set(idx[f::folds]))
        tr = [ords[i] for i in range(len(ords)) if i not in set(te)]
        best = None
        for r in range(restarts):
            fitm = fit_mixture(tr, K, seed=seed * 101 + f * 7 + r)
            items, index, S, pi = fitm
            ptr = PairTable(tr, index)
            LL = np.stack([ptr.ll(S[k]) for k in range(K)], 1) + np.log(pi)
            mx = LL.max(1, keepdims=True); tll = (mx[:, 0] + np.log(np.exp(LL - mx).sum(1))).sum()
            if best is None or tll > best[0]: best = (tll, fitm)
        items, index, S, pi = best[1]
        tst = [ords[i] for i in te]; pt = PairTable(tst, index)
        LL = np.stack([pt.ll(S[k]) for k in range(K)], 1) + np.log(pi)
        mx = LL.max(1, keepdims=True); lmix = mx[:, 0] + np.log(np.exp(LL - mx).sum(1))
        base = np.array([-len(o) / 2 * np.log(2) for o in tst])
        gain += (lmix - base).sum()
    return gain
