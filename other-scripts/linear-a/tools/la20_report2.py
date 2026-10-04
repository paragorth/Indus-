#!/usr/bin/env python3
"""LA-20 cycle 2 report: mixtures of K hidden orders. Statistic = best held-out gain over K=1..4
(nats over coin-flip order); the same max is taken in every shuffled replicate."""
import json, os, glob
import numpy as np
from la20_common import CK
for p in sorted(glob.glob(os.path.join(CK, 'c2_mix_*.json'))):
    d = json.load(open(p))
    if not d['null']: continue
    real = {int(k): v for k, v in d['real'].items()}
    nmax = np.array([max(x.values()) for x in d['null']])
    rmax = max(real.values()); kbest = max(real, key=real.get)
    n1 = np.array([x['1'] for x in d['null']]) if '1' in d['null'][0] else np.array([x[1] for x in d['null']])
    pm = [max(v.values()) for v in d['pmix'].values()]
    pk = [max(v, key=v.get) for v in d['pmix'].values()]
    p1 = [v['1'] for v in d['pmix'].values()]
    print(os.path.basename(p), 'nulls', len(nmax), '| real gain by K', {k: round(v, 2) for k, v in real.items()},
          '| best K', kbest, 'max %.2f vs null max %.2f +- %.2f, P %.3f' % (rmax, nmax.mean(), nmax.std(), (np.sum(nmax >= rmax) + 1) / (len(nmax) + 1)),
          '| K=1 real %.2f vs null %.2f +- %.2f, P %.3f' % (real[1], n1.mean(), n1.std(), (np.sum(n1 >= real[1]) + 1) / (len(n1) + 1)),
          '| planted 2-order: best K', pk, 'gain', [round(x, 1) for x in pm], 'K=1', [round(x, 1) for x in p1])
