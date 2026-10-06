"""v80 cycle 2: the row as a left-to-right column HMM (cells may continue over several words, blank cells vanish).
Random configurations (K, emission smoothing, seed) fitted on one leaf half, scored on the other (both directions).
Gain = held-out bits/token over the edge law (first/second/last/interior). Nulls: interior-shuffled corpora.
usage: python3 v80_c2.py corpus [corpus ...]   (NAME~sh = interior-shuffle null; VOY_ZL@H = section subset)"""
import sys, json, os, time, random
import numpy as np
import v80_lib as L

C = L.corpora()
KS = (1, 2, 3, 4, 5, 6, 8, 10, 13)
for nm in sys.argv[1:]:
    out = os.path.join(L.CK, 'c2', nm.replace('~', '_').replace('@', '-') + '.json')
    if os.path.exists(out): continue
    base = nm.split('~')[0].split('@')[0]
    P = C[base][1]
    if '@' in nm:
        sec = nm.split('@')[1].split('~')[0]; P = [p for p in P if p['sec'] in sec]
    if nm.endswith('~sh'): P = L.shuffle_interior(P, 8801)
    T = L.flatten(P)
    t0 = time.time(); res = []
    rng = random.Random(802)
    for rep in ('pre2', 'full'):
        M, lens, half, page = L.line_batches(T, rep)
        t = T['r_' + rep]; V = int(t.max() + 1)
        for K in KS:
            for r in range(3 if K > 1 else 1):
                lam = rng.uniform(0.4, 0.9); seed = rng.randrange(10 ** 6)
                g = []; ins = []
                for dh in (0, 1):
                    d = T['half'] == dh
                    pg = np.bincount(t[d], minlength=V) + 0.5; pg = pg / pg.sum()
                    h = L.RowHMM(K, V, pg, seed=seed, lam=lam).fit(M[half == dh], lens[half == dh], 25)
                    llh = h.loglik(M[half != dh], lens[half != dh]); n_h = (M[half != dh] >= 0).sum()
                    lld = h.loglik(M[half == dh], lens[half == dh]); n_d = (M[half == dh] >= 0).sum()
                    eh = L.edge_law_ll(T, rep, d, ~d).sum(); ed = L.edge_law_ll(T, rep, d, d).sum()
                    g.append((llh.sum() - eh) / n_h); ins.append((lld.sum() - ed) / n_d)
                res.append(dict(rep=rep, K=K, lam=lam, seed=seed, gain=float(np.mean(g)), gains=g, ins=float(np.mean(ins))))
    json.dump(res, open(out, 'w'))
    print(nm, '%.0fs' % (time.time() - t0), flush=True)
