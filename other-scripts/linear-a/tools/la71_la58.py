#!/usr/bin/env python3
"""la71 cycle 2: the la58 descriptive B (Hagia Triada deposits share word types with each other more than
with other sites; real 0.099 vs 0.040 under deposit shuffles) re-run on one corpus version.
Same statistic as la58_c3b.py (pair overlap from the la58 panel, 1,000 deposit-label shuffles, seed
'la58-c3b'); the planted CAPSAT part is skipped (its simulation bank is not kept).  Run via la71_run.py."""
import sys, os, json, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la58_common as L
L.SO = os.path.join(HERE, '..', 'data', 'la71_ckpt', 'la58_sim.so')
from la58_common import stats, seed
from la58_run import corpus

docs, K, ab, _ = corpus('LADEP')
ht = [i for i, a in enumerate(ab) if a.startswith('HT')]


def score(s):
    p = K * 13; P = {}
    for k in range(K):
        for l in range(k + 1, K):
            P[(k, l)] = s[p]; p += 1
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
res = dict(ver=os.environ.get('LA71_VERSION'), ndocs=len(docs), real=list(map(float, real)), null_mean=null.mean(0).tolist(),
           p_within_vs_cross=float((d_null >= d_real).mean()), p_within_low=float((null[:, 0] <= real[0]).mean()))
print(json.dumps(res))
json.dump(res, open(os.path.join(HERE, '..', 'data', 'la71_ckpt', 'c2_la58_%s.json' % res['ver']), 'w'))
