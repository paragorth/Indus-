"""v52 cycle 3b: WHERE DOES THE ATTRIBUTE LIVE?  For each corpus and target variable v: the best SINGLE
slot over a pool of schemas (the cycle-1 top 30, the cycle-2 climbed 8, 400 fresh random schemas) is
chosen by its gain on half A; its held-out gain on half B (bits/token over the prior) is compared with
the held-out gain of the WHOLE WORD (type counts with a prior-shrinkage, unseen -> prior).
CONC = best slot / whole word. A field code puts each attribute in one slot (CONC ~ 1 or above, since the
slot generalises to unseen keys); a text whose words carry attributes as vocabulary spreads it (CONC << 1)."""
import sys, os, pickle, random
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v52_lib as L
from v52_cycle1 import build

TV = dict(V=['SEC', 'LANG', 'HAND', 'POS', 'NBR', 'NXT'], VI=['SEC', 'LANG', 'HAND', 'POS', 'NBR', 'NXT'],
          VMARK=['SEC', 'LANG', 'HAND', 'POS', 'NBR', 'NXT'], VSHUF=['SEC', 'LANG', 'HAND', 'POS', 'NBR', 'NXT'],
          PL6=['SEC', 'POS', 'NBR'], PL3=['SEC', 'POS', 'NBR'],
          GORILA=['SITE', 'SUPPORT', 'PERIOD'], UNICODE=['BLOCK', 'CASE', 'DECOMP'])


def cond_ll(x_tr, y_tr, x_te, y_te, ny, lp, beta=4.0):
    nx = int(max(x_tr.max(), x_te.max())) + 1
    T = np.zeros((nx, ny)); np.add.at(T, (x_tr, y_tr), 1)
    P = (T + beta * np.exp2(lp)[None, :]) / (T.sum(1, keepdims=True) + beta)
    return np.log2(P[x_te, y_te])


def run(name):
    C = build(name); A = C['A']; B = ~A
    pool = []
    for tag in ('c1', 'c2'):
        S = pickle.load(open(os.path.join(L.CK, f'{tag}_summary.pkl'), 'rb'))
        if name in S:
            pool += [t['sch'] for t in S[name]['top']]
    for i in range(400):
        pool.append(L.random_schema(random.Random(77 + 7919 * i), C['top_units']))
    res = {}
    ys = {v: C['vars'][v] for v in TV[name]}
    best = {v: (-1e9, None, None) for v in ys}
    for sch in pool:
        codes, _ = L.slot_codes(C, sch)
        for v, y in ys.items():
            ny = int(y.max()) + 1
            pr = np.bincount(y[A], minlength=ny) + 0.5; lp = np.log2(pr / pr.sum())
            for s, c in enumerate(codes):
                gA = cond_ll(c[A], y[A], c[A], y[A], ny, lp).mean() - lp[y[A]].mean()
                if gA > best[v][0]:
                    gB = cond_ll(c[A], y[A], c[B], y[B], ny, lp).mean() - lp[y[B]].mean()
                    best[v] = (gA, gB, (sch, s))
    tt = C['tok_type']
    for v, y in ys.items():
        ny = int(y.max()) + 1
        pr = np.bincount(y[A], minlength=ny) + 0.5; lp = np.log2(pr / pr.sum())
        gw = cond_ll(tt[A], y[A], tt[B], y[B], ny, lp).mean() - lp[y[B]].mean()
        seen = np.isin(tt[B], np.unique(tt[A])).mean()
        res[v] = dict(slotA=float(best[v][0]), slotB=float(best[v][1]), word=float(gw), conc=float(best[v][1] / gw) if gw > 1e-3 else float('nan'),
                      seen=float(seen), sch=best[v][2])
    return name, res


if __name__ == '__main__':
    out = {}
    with Pool(2) as P:
        for name, res in P.imap_unordered(run, list(TV)):
            out[name] = res
            for v, r in res.items():
                print(f"{name:8s}{v:8s} slotB {r['slotB']:.3f} word {r['word']:.3f} conc {r['conc']:.2f} seen {r['seen']:.2f}", flush=True)
    pickle.dump(out, open(os.path.join(L.CK, 'c3b_summary.pkl'), 'wb'))
    print('all done')
