#!/usr/bin/env python3
"""LA-58 cycle 3b: do the Hagia Triada deposits share more vocabulary with each other than with other sites?
Pair overlap (shared types / sqrt(types), top-30 excluded) from the la58 panel; null: 1,000 deposit-label
shuffles (sizes kept). Same comparison for LB: KN find areas vs other sites is not available at this
granularity, so the control is the shuffle itself plus a planted CAPSAT world on the deposit frame."""
import sys, os, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la58_common import *
from la58_run import corpus, nk_of, load_bank

docs, K, ab, _ = corpus('LADEP')
ht = [i for i, a in enumerate(ab) if a.startswith('HT')]


def pairs(s):
    p = K * 13; out = {}
    for k in range(K):
        for l in range(k + 1, K):
            out[(k, l)] = s[p]; p += 1
    return out


def score(s):
    P = pairs(s)
    a = [v for (k, l), v in P.items() if k in ht and l in ht]
    b = [v for (k, l), v in P.items() if (k in ht) != (l in ht)]
    c = [v for (k, l), v in P.items() if k not in ht and l not in ht]
    return np.mean(a), np.mean(b), np.mean(c)


real = score(stats(docs, K))
rng = random.Random(seed('la58-c3b'))
null = []
for i in range(1000):
    s = [x['site'] for x in docs]; rng.shuffle(s)
    null.append(score(stats([dict(x, site=q) for x, q in zip(docs, s)], K)))
null = np.array(null)
d_real = real[0] - real[1]; d_null = null[:, 0] - null[:, 1]
res = dict(real=list(map(float, real)), null_mean=null.mean(0).tolist(), p_within_vs_cross=float((d_null >= d_real).mean()),
           p_within_low=float((null[:, 0] <= real[0]).mean()))
# planted check: capital at HT-R13 with HT rooms as satellite offices, other sites independent
S, T, meta = load_bank('LADEP')
cand = np.where(T[:, 0] == 1)[0]
pl = []
for j in range(30):
    th = T[np.random.RandomState(j).choice(cand)].copy()
    th[2:2 + K] = [1 if k == 0 else (2 if k in ht else 0) for k in range(K)]
    th[2 + K:2 + 2 * K] = [0 if (k in ht and k != 0) else -1 for k in range(K)]
    s, _ = simulate(K, nk_of(docs, K), 1, seed('la58-c3b-pl-%d' % j), force_theta=th)
    a, b, c = score(s[0]); pl.append(a - b)
res['planted_capsat_diff'] = [float(np.median(pl)), float(np.mean(np.array(pl) > np.quantile(d_null, 0.95)))]
print(json.dumps(res)); jdump(res, 'c3b_results.json')
