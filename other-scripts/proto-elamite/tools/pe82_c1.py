"""pe82 cycle 1: INK WHERE THE READER WOULD STUMBLE (uniform-information writer).

For each numeric line (not the tablet's first numeric line... all numeric lines are context; targets = every numeric
line), a reader model R (random 1-3 subset of 9 context features, naive Bayes, cross-fitted leave-one-batch-out)
gives surprisal of the line's target (system | first numeral code).  A marker hypothesis M (sign set) is 'present'
on a line if any sign of M is on it.  Score: Z = sum over strata (tablet, target, n-signs) of cov(present, surprisal)
standardised by its exact permutation variance.  Discovery / confirmation by publication batch.

usage: python3 pe82_c1.py CORPUS MODE [seed]
  CORPUS: PE | PC     MODE: real | shuf (strings shuffled within tablet x target x n-signs) | plantB (b in 0,0.5,1,2)
"""
import sys, os, json, math, random, itertools, collections
import numpy as np
import pe82_common as pc

FEATS = ['hdr', 'prev_sys', 'next_sys', 'prev_last', 'pos', 'surf', 'maj', 'ntab', 'prev2_sys']


def target(l):
    return l['sys'] + '|' + (l['nums'][0][1] if l['nums'] else '')


def build_rows(T):
    rows = []
    for ti, t in enumerate(T):
        L = t['lines']
        tg = [target(l) for l in L]
        cnt = collections.Counter(tg)
        for i, l in enumerate(L):
            c2 = cnt.copy(); c2[tg[i]] -= 1
            maj = max(c2.items(), key=lambda kv: (kv[1], kv[0]))[0] if sum(c2.values()) > 0 else 'NONE'
            f = dict(hdr=t['hdr'] or 'NONE',
                     prev_sys=tg[i - 1] if i else 'START',
                     next_sys=tg[i + 1] if i + 1 < len(L) else 'END',
                     prev_last=(L[i - 1]['sg'][-1] if L[i - 1]['sg'] else 'NUM') if i else 'START',
                     pos=str(min(i, 3) if i < 3 else (3 if i < 6 else 6)),
                     surf=l['surf'], maj=maj,
                     ntab=str(min(len(L), 8) // 2),
                     prev2_sys=tg[i - 2] if i > 1 else 'START')
            rows.append(dict(tab=ti, batch=t['batch'], y=tg[i], ns=min(len(l['sg']), 5), sg=l['sg'], f=f))
    return rows


def surprisal(rows, feats, alpha=0.5):
    """Naive Bayes P(y | feats) cross-fitted leave-one-batch-out."""
    batches = sorted({r['batch'] for r in rows})
    ys = sorted({r['y'] for r in rows})
    yi = {y: i for i, y in enumerate(ys)}
    out = np.zeros(len(rows))
    for b in batches:
        tr = [r for r in rows if r['batch'] != b]
        prior = collections.Counter(r['y'] for r in tr)
        cond = {f: collections.defaultdict(collections.Counter) for f in feats}
        vocab = {f: set() for f in feats}
        for r in tr:
            for f in feats:
                cond[f][r['f'][f]][r['y']] += 1
                vocab[f].add(r['f'][f])
        n = len(tr)
        lp0 = np.array([math.log((prior[y] + alpha) / (n + alpha * len(ys))) for y in ys])
        for k, r in enumerate(rows):
            if r['batch'] != b:
                continue
            lp = lp0.copy()
            for f in feats:
                v = r['f'][f]
                c = cond[f].get(v, {})
                V = len(vocab[f]) + 1
                lp += np.array([math.log((c.get(y, 0) + alpha) / (prior[y] + alpha * V)) for y in ys])
            lp -= lp.max()
            p = np.exp(lp); p /= p.sum()
            out[k] = -math.log(p[yi[r['y']]] if r['y'] in yi else 1e-6)
    return out


class Scorer:
    def __init__(self, rows, s, mask):
        """mask: rows used (a batch group).  Strata = (tab, y, length WITHOUT the marker signs), built per hypothesis,
        so an optional marker is compared with the same entry written without it."""
        idx = np.where(mask)[0]
        key = {}
        self.base = np.array([key.setdefault((rows[i]['tab'], rows[i]['y']), len(key)) for i in idx])
        self.L = np.array([len(rows[i]['sg']) for i in idx])
        self.idx = idx
        self.sv = s[idx]

    def z(self, x, cnt=None):
        x = x[self.idx].astype(float)
        c = self.L if cnt is None else self.L - cnt[self.idx]
        k = self.base * 8 + np.minimum(c, 7)
        _, st = np.unique(k, return_inverse=True)
        K = st.max() + 1
        n = np.bincount(st, minlength=K).astype(float)
        r = self.sv - (np.bincount(st, self.sv, K) / n)[st]
        xc = x - (np.bincount(st, x, K) / n)[st]
        T = (xc * r).sum()
        ok = n > 1
        V = (np.bincount(st, xc ** 2, K)[ok] * np.bincount(st, r ** 2, K)[ok] / (n[ok] - 1)).sum()
        return (T / math.sqrt(V) if V > 0 else 0.0), int(x.sum())


def shuffle_strings(T, rng):
    for t in T:
        groups = collections.defaultdict(list)
        for i, l in enumerate(t['lines']):
            groups[(target(l), min(len(l['sg']), 5))].append(i)
        for g in groups.values():
            sgs = [t['lines'][i]['sg'] for i in g]
            rng.shuffle(sgs)
            for i, s in zip(g, sgs):
                t['lines'][i]['sg'] = s
    return T


def plant(T, rows_s, b, rng, rate=0.05):
    """Insert sign PLANT into lines with odds ~ exp(b * standardized surprisal); rows_s aligned with flat lines."""
    s = np.array(rows_s)
    from scipy.stats import norm, rankdata
    z = norm.ppf((rankdata(s) - 0.5) / len(s))      # rank-normal surprisal (heavy tail tamed)
    a = math.log(rate / (1 - rate))
    for _ in range(60):                             # calibrate intercept so the mean rate is `rate`
        p = 1 / (1 + np.exp(-(a + b * z)))
        a += math.log(rate / p.mean()) * 0.8
    k = 0
    for t in T:
        for l in t['lines']:
            if rng.random() < p[k]:
                l['sg'] = l['sg'] + ('PLANT',)
            k += 1
    return T


SPLITS = {'PE': [({'MDP26', 'MDP26S', 'OTHER'}, {'MDP17', 'MDP06', 'TCL31'}),
                 ({'MDP17', 'MDP06', 'TCL31'}, {'MDP26', 'MDP26S', 'OTHER'})],
          'PC': [({'U3'}, {'U4', 'OTHER'}), ({'U4', 'OTHER'}, {'U3'})]}


def run(corpus, mode, seed=0, n_rand=2000, top_signs=150):
    rng = random.Random(seed)
    T = pc.load_pe() if corpus == 'PE' else pc.load_pc()
    if mode == 'shuf':
        T = shuffle_strings(T, rng)
    if mode.startswith('plant'):
        b = float(mode[5:])
        rows0 = build_rows(T)
        s0 = surprisal(rows0, ['prev_sys', 'maj'])
        T = plant(T, s0, b, rng)
    rows = build_rows(T)
    freq = collections.Counter(s for r in rows for s in set(r['sg']))
    top = [s for s, c in freq.most_common() if c >= 15][:top_signs]
    if mode.startswith('plant') and 'PLANT' not in top:
        top.append('PLANT')
    hyps = [(s,) for s in top]
    seen = set(hyps)
    while len(hyps) < len(top) + n_rand:
        h = tuple(sorted(rng.sample(top, rng.choice([2, 3]))))
        if h not in seen:
            seen.add(h); hyps.append(h)
    sgsets = [set(r['sg']) for r in rows]
    X = {h: np.array([bool(sg & set(h)) for sg in sgsets]) for h in hyps}
    CN = {h: np.array([sum(1 for x in r['sg'] if x in set(h)) for r in rows]) for h in hyps}
    nslen = np.array([r['ns'] for r in rows], float)
    models = [m for k in (1, 2, 3) for m in itertools.combinations(FEATS, k)]
    rng.shuffle(models)
    models = models[:60]
    batch = np.array([r['batch'] for r in rows])
    res = []
    for m in models:
        s = surprisal(rows, list(m))
        for si, (D, C) in enumerate(SPLITS[corpus]):
            sd = Scorer(rows, s, np.isin(batch, list(D)))
            sc = Scorer(rows, s, np.isin(batch, list(C)))
            # length-level UID: n-signs vs surprisal within (tab, y) -> strata ignore ns: use separate scorer
            for h in hyps:
                zd, nd = sd.z(X[h], CN[h]); zc, nc = sc.z(X[h], CN[h])
                res.append((m, si, h, round(zd, 3), round(zc, 3), nd, nc))
        # length level
    return rows, res, models


def length_level(rows, models, corpus):
    batch = np.array([r['batch'] for r in rows])
    out = []
    for m in models:
        s = surprisal(rows, list(m))
        for si, (D, C) in enumerate(SPLITS[corpus]):
            zz = []
            for G in (D, C):
                mask = np.isin(batch, list(G))
                idx = np.where(mask)[0]
                key = {}
                st = np.array([key.setdefault((rows[i]['tab'], rows[i]['y']), len(key)) for i in idx])
                K = len(key); n = np.bincount(st, minlength=K).astype(float)
                sv = s[idx]; r = sv - (np.bincount(st, sv, K) / np.maximum(n, 1))[st]
                x = np.array([rows[i]['ns'] for i in idx], float)
                xc = x - (np.bincount(st, x, K) / np.maximum(n, 1))[st]
                ok = n > 1
                V = (np.bincount(st, xc ** 2, K)[ok] * np.bincount(st, r ** 2, K)[ok] / (n[ok] - 1)).sum()
                zz.append((xc * r).sum() / math.sqrt(V) if V > 0 else 0)
            out.append((m, si, round(zz[0], 3), round(zz[1], 3)))
    return out


if __name__ == '__main__':
    corpus, mode = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rows, res, models = run(corpus, mode, seed)
    L = length_level(rows, models, corpus)
    fn = os.path.join(pc.CK, 'c1_%s_%s_%d.json' % (corpus, mode, seed))
    json.dump(dict(res=res, length=L, n_rows=len(rows)), open(fn, 'w'))
    surv = [r for r in res if r[3] >= 3.5 and r[4] >= 2]
    surv_neg = [r for r in res if r[3] <= -3.5 and r[4] <= -2]
    print(corpus, mode, seed, 'tests', len(res), 'disc>=3.5', sum(r[3] >= 3.5 for r in res),
          'survivors+', len(surv), 'survivors-', len(surv_neg))
