"""v12 cycle 2: hidden two-state switch over word streams (HMM over streams).
Per corpus, per fold (folio parity): fit by EM on the training fold
   k1  : one stream (context = previous word)
   H2  : 2-state HMM, context = previous word (same parameter count as W2)
   W2  : 2-stream WEAVE, context = previous word of the SAME stream
 (2 random restarts each, best training likelihood kept), score held-out bits/token.
Statistic: W2 - H2 (bits/token). A woven text must give W2 >> H2; a single text W2 <= H2.
Also stored: switch matrix A of W2 (P(stay)), stream shares, and for W2 the per-stream vocabulary entropy."""
import sys, os, json, time, math
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v12_lib as V, v12_hmm as H

ITERS = 10


def fit_best(tr, k, kind, seeds=(1, 2)):
    best = None
    for sd in seeds:
        m = H.StreamModel(k, kind, seed=sd); m.fit(tr, iters=ITERS)
        if best is None or m.train_bits > best.train_bits: best = m
    return best


def job(arg):
    name, fold = arg
    ck = os.path.join(V.CK, f'c2_{name}_{fold}.json')
    if os.path.exists(ck): return json.load(open(ck))
    t0 = time.time()
    C = V.corpus(name); tr, te = V.split(C, fold)
    out = {'name': name, 'fold': fold}
    m1 = fit_best(tr, 1, 'H', seeds=(1,)); out['k1'] = m1.score(te)
    mh = fit_best(tr, 2, 'H'); out['H2'] = mh.score(te); out['H2_A'] = mh.A; out['H2_pi'] = mh.pi
    mw = fit_best(tr, 2, 'W'); out['W2'] = mw.score(te); out['W2_A'] = mw.A; out['W2_pi'] = mw.pi
    out['sec'] = time.time() - t0
    json.dump(out, open(ck, 'w'), indent=1)
    print(name, fold, round(out['k1'], 3), round(out['H2'], 3), round(out['W2'], 3), 'W2-H2', round(out['W2'] - out['H2'], 3),
          'Astay', [round(mw.A[s][s], 2) for s in range(2)], round(out['sec']), flush=True)
    return out


if __name__ == '__main__':
    names = sys.argv[1:] or (V.VOY + V.POS + V.NEG + ['Gloss-pair'])
    jobs = [(n, f) for n in names for f in (0, 1)]
    with Pool(2) as p:
        R = p.map(job, jobs, chunksize=1)
    json.dump(R, open(os.path.join(V.CK, 'cycle2.json'), 'w'), indent=1)
    print(f"{'corpus':14s} {'k1':>8s} {'H2':>8s} {'W2':>8s} {'H2-k1':>7s} {'W2-k1':>7s} {'W2-H2':>7s}  W2 P(stay)")
    for n in names:
        rs = [r for r in R if r['name'] == n]
        a = {k: sum(r[k] for r in rs) / 2 for k in ('k1', 'H2', 'W2')}
        print(f"{n:14s} {a['k1']:8.3f} {a['H2']:8.3f} {a['W2']:8.3f} {a['H2']-a['k1']:7.3f} {a['W2']-a['k1']:7.3f} {a['W2']-a['H2']:7.3f}  "
              + ' '.join(str([round(r['W2_A'][s][s], 2) for s in range(2)]) for r in rs))
