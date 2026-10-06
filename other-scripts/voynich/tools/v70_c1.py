"""v70 cycle 1: thousands of ink-derived alphabets per corpus, scored by the battery.

For each corpus and cut threshold theta: unit patches -> 192-px vector + width -> PCA 30,
centred per page (removes page-level ink/scan drift) -> k-means at K = 10..60 (2 seeds)
-> 0/4/8/12 ligature merges (most frequent adjacent label pair inside words).
Null alphabets: random Voronoi cells (K random units as centres, same merges), matched size.
Each alphabet re-transcribes the words; battery on page-half A and page-half B separately.
Writes data/v70_ckpt/c1_<corpus>.json (feature rows) and recovery metrics for synthetic corpora.
Usage: python3 v70_c1.py [corpus ...]
"""
import sys, os, json, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v70_lib import *
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score as NMI

ARR = os.path.join(SCR, 'v70', 'arr')
THETAS = [0, 2.5, 4]
KS = list(range(10, 61, 2))
MERGES = [0, 4, 8, 12]
SEEDS = tuple(int(x) for x in os.environ.get('V70_SEEDS', '0,1').split(','))


def embed(name, th):
    fn = os.path.join(ARR, f'{name}_t{th}_E.npy')
    D = json.load(open(os.path.join(ARR, name + '.json')))
    U = D['units'][str(th)]
    if os.path.exists(fn):
        return np.load(fn), D, U
    P = np.load(os.path.join(ARR, f'{name}_t{th}_P.npy')).astype(np.float32) / 255.0
    X = P.reshape(-1, PH // 2, 2, PW // 2, 2).mean((2, 4)).reshape(len(P), -1)
    wd = np.array([u[1] for u in U], np.float32)[:, None]
    X = np.concatenate([X, 3 * wd], 1)
    rng = np.random.default_rng(0)
    sub = X[rng.choice(len(X), min(30000, len(X)), replace=False)]
    E = PCA(30, random_state=0).fit(sub).transform(X)
    pg = np.array([D['recs'][u[0]]['pg'] for u in U])
    for p in np.unique(pg):
        E[pg == p] -= E[pg == p].mean(0)
    E /= E.std(0) + 1e-9
    E = E.astype(np.float32)
    np.save(fn, E)
    return E, D, U


def nearest(E, C):
    cc = (C ** 2).sum(1)
    return np.concatenate([np.argmin(cc[None] - 2 * E[i:i + 8192] @ C.T, 1) for i in range(0, len(E), 8192)])


def words_from(labels, U, nrec):
    W = [[] for _ in range(nrec)]
    for lab, u in zip(labels, U):
        W[u[0]].append((u[2], int(lab)))
    return [tuple(l for _, l in sorted(w)) for w in W]


def bpe(words, m):
    words = [list(w) for w in words]
    nxt = max((max(w) for w in words if w), default=0) + 1
    merges = []
    for _ in range(m):
        c = collections.Counter()
        for w in words:
            for a, b in zip(w, w[1:]):
                c[(a, b)] += 1
        if not c:
            break
        (a, b), _ = c.most_common(1)[0]
        merges.append((a, b, nxt))
        for i, w in enumerate(words):
            o, j = [], 0
            while j < len(w):
                if j + 1 < len(w) and w[j] == a and w[j + 1] == b:
                    o.append(nxt); j += 2
                else:
                    o.append(w[j]); j += 1
            words[i] = o
        nxt += 1
    return [tuple(w) for w in words], merges


def bpe_snaps(words, ms):
    out = {0: (words, [])}
    cur, merges, done = words, [], 0
    for m in sorted(x for x in ms if x):
        cur, mm = bpe(cur, m - done)
        merges = merges + mm; done = m
        out[m] = (cur, list(merges))
    return out


def halves(D, words):
    pg = np.array([r['pg'] for r in D['recs']])
    A = [w for w, p in zip(words, pg) if p % 2 == 0 and w]
    B = [w for w, p in zip(words, pg) if p % 2 == 1 and w]
    return A, B


def run(name):
    out = os.path.join(CK, f'c1_{name}.json')
    if os.path.exists(out):
        return
    rows = []
    for th in THETAS:
        E, D, U = embed(name, th)
        nrec = len(D['recs'])
        truth = [u[3] for u in U]
        has_truth = any(truth)
        for K in KS:
            for seed in SEEDS:
                for kind in ('km', 'rnd'):
                    if kind == 'km':
                        lab = MiniBatchKMeans(K, random_state=seed, n_init=1, batch_size=4096).fit_predict(E)
                    else:
                        rng = np.random.default_rng(1000 + seed * 100 + K)
                        C = E[rng.choice(len(E), K, replace=False)]
                        lab = nearest(E, C)
                    rec = {'theta': th, 'K': K, 'seed': seed, 'kind': kind}
                    if has_truth:
                        ok = [i for i, t in enumerate(truth) if t]
                        rec['nmi'] = float(NMI([truth[i] for i in ok], lab[ok]))
                        lig = [i for i in ok if truth[i].startswith('LIG:')]
                        if lig:
                            cl = collections.Counter(lab[i] for i in lig).most_common(1)[0][0]
                            incl = sum(1 for i in ok if lab[i] == cl)
                            rec['lig_recall'] = collections.Counter(lab[i] for i in lig)[cl] / len(lig)
                            rec['lig_prec'] = collections.Counter(lab[i] for i in lig)[cl] / max(1, incl)
                    W0 = words_from(lab, U, nrec)
                    snaps = bpe_snaps(W0, MERGES)
                    for m in MERGES:
                        W, merges = snaps[m]
                        A, B = halves(D, W)
                        r = dict(rec); r['m'] = m
                        r['A'] = battery(A); r['B'] = battery(B)
                        if has_truth and m:
                            # does a merge rebuild the planted split letter?
                            s1 = collections.Counter(lab[i] for i, t in enumerate(truth) if t.startswith('SPL1'))
                            s2 = collections.Counter(lab[i] for i, t in enumerate(truth) if t.startswith('SPL2'))
                            if s1 and s2:
                                pair = (s1.most_common(1)[0][0], s2.most_common(1)[0][0])
                                r['split_rebuilt'] = any((a, b) == pair for a, b, _ in merges[:m])
                        rows.append(r)
        print(name, th, len(rows), flush=True)
    # reference: battery of the true / EVA units on the same words
    D = json.load(open(os.path.join(ARR, name + '.json')))
    if name == 'V':
        tw = [tuple(vglyphs(r['word'].replace('?', ''))) for r in D['recs']]
    elif name == 'L':
        tw = [tuple(r['word']) for r in D['recs']]
    else:
        tw = [tuple(r['truth']) for r in D['recs']]
    A, B = halves(D, tw)
    ref = {'A': battery(A), 'B': battery(B)}
    json.dump({'rows': rows, 'ref': ref}, open(out, 'w'))


if __name__ == '__main__':
    names = sys.argv[1:]
    with ProcessPoolExecutor(2) as ex:
        list(ex.map(run, names))
