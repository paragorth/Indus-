"""Cycle 2 analysis: per-hypothesis counts, per-key p against noise replicates,
family-wise min-p correction, split-half re-test."""
import json, collections, random, sys
import numpy as np
from pe42_common import CK, os

d = json.load(open(os.path.join(CK, 'c2_keys.json')))


def analyse(name, nsplit=200):
    D = d[name]
    real = D['-1']
    nulls = [D[k] for k in D if k != '-1']
    fail = [k for k, x in enumerate(real) if x is not None]
    keys = sorted({tuple(k) for k in (tuple(map(tuple, x)) for x in real if x) for k in k} |
                  {tuple(k) for x in nulls for y in x if y for k in map(tuple, y)})
    kix = {k: i for i, k in enumerate(keys)}

    def mat(L):
        M = np.zeros((len(L), len(keys)), dtype=np.int8)
        for t, x in enumerate(L):
            if x:
                for k in x:
                    M[t, kix[tuple(k)]] = 1
        return M
    R = mat(real)                      # tablets x keys
    Ns = np.stack([mat(x) for x in nulls])   # reps x tablets x keys
    rc = R.sum(0); nc = Ns.sum(1)      # keys ; reps x keys
    p = (1 + (nc >= rc).sum(0)) / (1 + len(nulls))
    # family-wise: for each null rep, its min p vs the other reps
    ranks = []
    for r in range(len(nulls)):
        others = np.delete(nc, r, axis=0)
        pr = (1 + (others >= nc[r]).sum(0)) / (1 + len(others))
        pr[nc[r] == 0] = 1
        ranks.append(pr.min())
    ranks = np.array(ranks)
    p[rc == 0] = 1
    order = np.argsort(p)
    out = ['== %s: %d failing tablets, %d hypotheses seen (real or null); %d explain >=1 real tablet'
           % (name, len(fail), len(keys), int((rc > 0).sum()))]
    for i in order[:15]:
        pfw = (1 + (ranks <= p[i]).sum()) / (1 + len(ranks))
        out.append('  %-45s real %2d null mean %.2f  p %.4f  family-wise p %.3f'
                   % (' '.join(map(str, keys[i])), rc[i], nc[:, i].mean(), p[i], pfw))
    # split-half: pick best key on half A (vs null on half A), count on half B real vs null
    rng = random.Random(1)
    rows = [t for t in range(R.shape[0]) if real[t] is not None]
    gains = []; picks = collections.Counter()
    for s in range(nsplit):
        rng.shuffle(rows)
        A = rows[:len(rows) // 2]; B = rows[len(rows) // 2:]
        ra = R[A].sum(0); na = Ns[:, A].sum(1)
        pa = (1 + (na >= ra).sum(0)) / (1 + len(nulls)); pa[ra == 0] = 1
        best = int(np.argmin(pa + 1e-6 * -ra))
        picks[keys[best]] += 1
        rb = R[B, best].sum(); nb = Ns[:, B, best].sum(1).mean()
        gains.append(rb - nb)
    g = np.array(gains)
    out.append('  split-half (%d splits): held-out gain of the half-A winner %.2f (positive in %.0f%%); most-picked: %s'
               % (nsplit, g.mean(), 100 * (g > 0).mean(),
                  '; '.join('%s x%d' % (' '.join(map(str, k)), v) for k, v in picks.most_common(4))))
    return '\n'.join(out), {keys[i]: (int(rc[i]), float(p[i])) for i in order[:40]}


if __name__ == '__main__':
    txt = []
    res = {}
    for name in d:
        t, r = analyse(name)
        txt.append(t); res[name] = {' '.join(map(str, k)): v for k, v in r.items()}
    print('\n'.join(txt))
    json.dump(res, open(os.path.join(CK, 'c2_top.json'), 'w'))
