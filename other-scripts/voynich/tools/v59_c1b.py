import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
"""v59 cycle 1b: cross-fitted random transducer search with a wider rule space, plus a test of
the v30 form-level A->B rewrite under the context-retrieval (meaning-preservation) score.

Screening: train pages of each side are split into two folds; a rule set counts only by the
smaller of its two fold gains (min-of-folds), so a gain must appear in two independent page
samples before it is believed. Greedy stacking uses the same criterion. Held-out test after.
"""
import sys, json, time, random, zlib
import numpy as np
from multiprocessing import Pool
from v59_lib import *
from v59_c1 import pairs, TRUTH
import v30_lib

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
KEEP = ['P_Vplant', 'C_Isid_scribal', 'C_Cz_old', 'N_AA', 'U_Cz_Lat', 'V_AB', 'V_AB_IT', 'V_BA', 'N_Bshuf']
P = {}


def v30_rules(fn, k):
    r = json.load(open(os.path.join(ROOT, 'data', 'v30_ckpt', fn)))['path'][-1]['rules'][:k]
    rb = v30_lib.rules_by_first([tuple(x) for x in r])
    return lambda w: v30_lib.apply_word(w, rb)


def run(name):
    s1, s2 = P[name]
    t0 = time.time()
    s1tr, s1te = split_pages(s1, 11)
    s2tr, s2te = split_pages(s2, 12)
    f1a, f1b = split_pages(s1tr, 13); f2a, f2b = split_pages(s2tr, 14)
    Sa, Sb, Ste = Scorer(f2a), Scorer(f2b), Scorer(s2te)
    Pa, Pb, Pte = Sa.prepare(f1a), Sb.prepare(f1b), Ste.prepare(s1te)
    ba, bb, bte = Sa.score_T(Pa, []), Sb.score_T(Pb, []), Ste.score_T(Pte, [])
    fold = lambda T: min(Sa.score_T(Pa, T) - ba, Sb.score_T(Pb, T) - bb)
    test = lambda T: Ste.score_T(Pte, T) - bte
    RS = RuleSampler(s1tr, s2tr, seed=zlib.crc32(name.encode()), n_top=60, kmax=5)
    rows = []
    for i in range(N):
        T = RS.T(3)
        rows.append((fold(T), test(T), T))
    dfo = np.array([r[0] for r in rows]); dte = np.array([r[1] for r in rows])
    order = np.argsort(-dfo)
    rng = np.random.default_rng(0)
    null = np.array([dte[rng.choice(len(dte), 20, replace=False)].mean() for _ in range(2000)])
    z = (dte[order[:20]].mean() - null.mean()) / (null.std() + 1e-12)
    cand = []
    for j in order[:60]:
        for r in rows[j][2]:
            if r not in cand:
                cand.append(r)
    T, best = [], 0.0
    for _ in range(8):
        g = [(fold(T + [r]), r) for r in cand if r not in T]
        if not g:
            break
        v, r = max(g, key=lambda x: x[0])
        if v <= best + 1e-4:
            break
        T.append(r); best = v
    res = {'name': name, 'base': [ba, bb, bte], 'n_fold_pos': int((dfo > 0).sum()),
           'top20_dte': float(dte[order[:20]].mean()), 'z_top20': float(z),
           'max_dte_among_foldpos': float(dte[dfo > 0].max()) if (dfo > 0).any() else None,
           'greedy_T': T, 'greedy_fold': best, 'greedy_test': test(T) if T else 0.0,
           'top_sets': [(float(dfo[j]), float(dte[j]), rows[j][2]) for j in order[:15]]}
    tr = TRUTH.get(name)
    if tr:
        res['truth_fold'] = fold(tr); res['truth_test'] = test(tr)
    if name in ('V_AB', 'V_AB_IT', 'V_BA'):
        fn = {'V_AB': 'g3_V_AB_s0.json', 'V_AB_IT': 'g3_V_AB_s1.json', 'V_BA': 'g3_V_BA_s0.json'}[name]
        res['v30'] = {k: (fold(v30_rules(fn, k)), test(v30_rules(fn, k))) for k in (1, 2, 3, 5, 10, 20, 30)}
        # null for v30: random rule sets of the same sizes from the cycle's sampler
        res['rand_test_by_k'] = {k: float(np.mean([test(RS.T(1) * 0 + [RS.rule() for _ in range(k)]) for _ in range(40)]))
                                 for k in (1, 3, 10, 30)}
    res['secs'] = time.time() - t0
    json.dump(res, open(os.path.join(CK, f'c1b_{name}.json'), 'w'), ensure_ascii=False)
    print(name, f"fold+ {res['n_fold_pos']} top20 dte {res['top20_dte']:+.4f} z {z:+.1f} greedy {best:+.4f}/{res['greedy_test']:+.4f} "
          f"truth {res.get('truth_fold')} {res.get('truth_test')} v30 {res.get('v30')} {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    P = {k: v for k, v in pairs().items() if k in KEEP}
    with Pool(2) as pool:
        list(pool.imap_unordered(run, KEEP))
