"""pe13 shared code: memory-kernel cache models over ordered tablet lines.

Model for a token w in line j of a tablet (j >= 1), history = earlier lines k < j:
    P(w) = (1 - lam) * Pbase(w) + lam * sum_k K(d_jk, s_jk) m_k(w) / sum_k K(d_jk, s_jk) n_k
  m_k(w) = copies of w in line k, n_k = tokens in line k,
  d = distance in lines (clock 'ent': written lines; clock 'raw': ATF line numbers),
  s = 0 same surface and column, 1 same surface other column, 2 other surface.
Kernel K(d, s) = g_s * rho^[d == 1] * ((1 - w) * shape(d) + w)
  shape: flat 1 | exp exp(-(d-1)/tau) | pow d^-a | step [d <= k]
Distances are capped at D (the cap bin collects everything farther).
"""
import json, math, os, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, '..', 'data', 'pe13_ckpt')
D = 40
NS = 3
LAMS = np.concatenate([np.linspace(0.0, 0.3, 16), np.linspace(0.32, 0.95, 22)])


def corpora():
    return json.load(open(os.path.join(CK, 'corpora.json')))


def prep(tabs, clock='ent', field='toks', target_filter=None):
    """Return dict: w (target word ids), tab (tablet index), M, N (T x D*NS float32),
    vocab, and per-tablet token lists for the base model."""
    vocab = {}
    rows_w, rows_t, Ms, Ns = [], [], [], []
    for ti, t in enumerate(tabs):
        L = [l for l in t['lines'] if l[field]]
        ids = [[vocab.setdefault(x, len(vocab)) for x in l[field]] for l in L]
        pos = [i if clock == 'ent' else l['li'] for i, l in enumerate(L)]
        for j in range(1, len(L)):
            for wi, w in enumerate(ids[j]):
                if target_filter is not None and not target_filter(L[j], wi):
                    continue
                m = np.zeros(D * NS, np.float32)
                n = np.zeros(D * NS, np.float32)
                for k in range(j):
                    d = min(max(pos[j] - pos[k], 1), D)
                    s = 0 if (L[k]['surf'] == L[j]['surf'] and L[k]['col'] == L[j]['col']) else (1 if L[k]['surf'] == L[j]['surf'] else 2)
                    b = s * D + d - 1
                    n[b] += len(ids[k])
                    m[b] += ids[k].count(w)
                rows_w.append(w)
                rows_t.append(ti)
                Ms.append(m)
                Ns.append(n)
    toks = [[vocab[x] for l in t['lines'] for x in l[field] if x in vocab] for t in tabs]
    import scipy.sparse as sp
    Ms = sp.csr_matrix(np.array(Ms, dtype=np.float64)) if Ms else sp.csr_matrix((0, D * NS))
    Ns = sp.csr_matrix(np.array(Ns, dtype=np.float64)) if Ns else sp.csr_matrix((0, D * NS))
    return {'w': np.array(rows_w), 'tab': np.array(rows_t), 'M': Ms, 'N': Ns,
            'V': len(vocab), 'toks': toks, 'ntab': len(tabs)}


def base_probs(P, train_tabs):
    c = np.zeros(P['V'] + 1)
    for ti in train_tabs:
        for x in P['toks'][ti]:
            c[x] += 1
    return (c + 0.5) / (c.sum() + 0.5 * (P['V'] + 1))


def kernel(par):
    """par: dict(shape, tau, a, k, w, rho, g1, g2). Returns vector len D*NS."""
    d = np.arange(1, D + 1, dtype=float)
    sh = par.get('shape', 'flat')
    if sh == 'flat':
        f = np.ones(D)
    elif sh == 'exp':
        f = np.exp(-(d - 1) / par['tau'])
    elif sh == 'pow':
        f = d ** (-par['a'])
    elif sh == 'step':
        f = (d <= par['k']).astype(float)
    elif sh == 'free':
        f = np.asarray(par['f'], float)
    w = par.get('w', 0.0)
    f = (1 - w) * f + w
    f[0] *= par.get('rho', 1.0)
    g = [1.0, par.get('g1', 1.0), par.get('g2', 1.0)]
    return np.concatenate([gi * f for gi in g]) + 1e-12


def cache_prob(P, idx, K):
    num = P['M'] @ K
    den = P['N'] @ K
    cp = num / den
    return cp if idx is None else cp[idx]


def ll_grid(pb, cp):
    """log-lik (nats) for each lam in LAMS; pb, cp arrays over targets."""
    return np.array([np.log((1 - l) * pb + l * cp).sum() for l in LAMS])


def fit_eval(P, K, tr_idx, te_idx, pbtr, pbte):
    cp_tr = cache_prob(P, tr_idx, K)
    g = ll_grid(pbtr, cp_tr)
    li = int(np.argmax(g))
    cp_te = cache_prob(P, te_idx, K)
    l = LAMS[li]
    te = np.log((1 - l) * pbte + l * cp_te).sum()
    return g[li], te, l


def folds(ntab, k=5, seed=0):
    r = np.random.RandomState(seed)
    a = r.permutation(ntab)
    return [np.sort(a[i::k]) for i in range(k)]


def random_kernel(rng):
    sh = rng.choice(['flat', 'exp', 'exp', 'pow', 'pow', 'step'])
    par = {'shape': sh}
    if sh == 'exp':
        par['tau'] = math.exp(rng.uniform(math.log(0.3), math.log(60)))
    if sh == 'pow':
        par['a'] = rng.uniform(0.05, 3.0)
    if sh == 'step':
        par['k'] = rng.randint(1, 12)
    if sh != 'flat':
        par['w'] = 0.0 if rng.random() < 0.3 else rng.uniform(0, 1)
    par['rho'] = 1.0 if rng.random() < 0.4 else math.exp(rng.uniform(math.log(0.05), math.log(5)))
    par['g1'] = 1.0 if rng.random() < 0.4 else rng.uniform(0, 2)
    par['g2'] = 1.0 if rng.random() < 0.4 else rng.uniform(0, 2)
    return par


def shuffle_lines(tabs, rng):
    """Shuffle line contents within tablet (keeps surfaces/columns/positions fixed:
    order is killed, topic and layout kept)."""
    out = []
    for t in tabs:
        L = t['lines']
        perm = list(range(len(L)))
        rng.shuffle(perm)
        NL = []
        for i, l in enumerate(L):
            src = L[perm[i]]
            x = dict(l)
            for f in ('toks', 'mid', 'cls', 'ent', 'q'):
                if f in src:
                    x[f] = src[f]
            NL.append(x)
        out.append({'id': t['id'], 'lines': NL})
    return out


def gen_planted(skel, kind, seed, unigram, params=None):
    """Planted corpora on a skeleton (list of tablets; we keep line lengths, surfaces,
    columns, positions). kind in TOPIC, PRIME, PRIME_REF, RESET, COPY."""
    rng = random.Random(seed)
    keys = list(unigram)
    wts = np.array([unigram[k] for k in keys], float)
    cum = np.cumsum(wts / wts.sum())

    def draw_base():
        return keys[int(np.searchsorted(cum, rng.random()))]
    p = {'TOPIC': dict(shape='flat', lam=0.4),
         'PRIME': dict(shape='exp', tau=1.5, w=0.0, lam=0.4),
         'PRIME_REF': dict(shape='exp', tau=3.0, w=0.0, rho=0.15, lam=0.4),
         'RESET': dict(shape='flat', g2=0.0, lam=0.4),
         'MIX': dict(shape='exp', tau=1.5, w=0.3, lam=0.4)}.get(kind, dict(shape='flat', lam=0.4))
    if params:
        p.update(params)

    def gen_tab(L):
        K = kernel(p)
        out = []
        for j, l in enumerate(L):
            toks = []
            for _ in range(len(l['toks'])):
                if j > 0 and rng.random() < p['lam']:
                    wts_k, cand = [], []
                    for k in range(j):
                        d = min(max(j - k, 1), D)
                        s = 0 if (L[k]['surf'] == l['surf'] and L[k]['col'] == l['col']) else (1 if L[k]['surf'] == l['surf'] else 2)
                        kw = K[s * D + d - 1]
                        for x in out[k]:
                            wts_k.append(kw)
                            cand.append(x)
                    tot = sum(wts_k)
                    if tot > 1e-9:
                        r = rng.random() * tot
                        acc = 0
                        for kw, x in zip(wts_k, cand):
                            acc += kw
                            if acc >= r:
                                toks.append(x)
                                break
                        continue
                toks.append(draw_base())
            out.append(toks)
        return out
    res = []
    if kind == 'COPY':
        masters = []
        for m in range(30):
            ML = []
            for _ in range(80):
                tt = skel[rng.randrange(len(skel))]['lines']
                ML.append({'toks': [0] * len(tt[rng.randrange(len(tt))]['toks']), 'surf': 0, 'col': 1})
            masters.append(gen_tab(ML))
        for t in skel:
            L = t['lines']
            mm = masters[rng.randrange(30)]
            st = rng.randrange(0, 80 - len(L)) if len(L) < 80 else 0
            NL = []
            for i, l in enumerate(L):
                src = mm[(st + i) % 80]
                tk = [x if rng.random() > 0.05 else draw_base() for x in src]
                x = dict(l)
                x['toks'] = tk
                x['mid'] = tk
                NL.append(x)
            res.append({'id': t['id'], 'lines': NL})
        return res
    for t in skel:
        L = t['lines']
        toks = gen_tab(L)
        NL = []
        for l, tk in zip(L, toks):
            x = dict(l)
            x['toks'] = tk
            x['mid'] = tk
            NL.append(x)
        res.append({'id': t['id'], 'lines': NL})
    return res


def copy_stat(tabs, field='toks'):
    """Cross-tablet shared ORDERED adjacent line pairs: fraction of adjacent pairs
    (line j, line j+1) whose exact token tuples occur adjacent, same order, on another tablet."""
    from collections import defaultdict
    where = defaultdict(set)
    pairs = []
    for ti, t in enumerate(tabs):
        L = [tuple(l[field]) for l in t['lines'] if l[field]]
        for j in range(len(L) - 1):
            pr = (L[j], L[j + 1])
            where[pr].add(ti)
            pairs.append((ti, pr))
    if not pairs:
        return 0.0
    return sum(1 for ti, pr in pairs if len(where[pr]) > 1) / len(pairs)
