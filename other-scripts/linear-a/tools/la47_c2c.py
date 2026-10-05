#!/usr/bin/env python3
"""la47 cycle 2c: power of the direct size test (2a): plant a size shift of d (log units) on the
commodity tokens' residual size (or on a random 15 % of tokens), re-run the within-page permutation test.
Output data/la47_ckpt/c2c.json"""
import json, os
import numpy as np
import la47_common as C, la47_c2 as M


def ptest(pid, s, m, rng, reps=1000):
    obs = s[m].mean() - s[~m].mean(); groups = [np.where(pid == p)[0] for p in np.unique(pid)]; null = []
    for _ in range(reps):
        sp = s.copy()
        for k in groups: sp[k] = sp[rng.permutation(k)]
        null.append(sp[m].mean() - sp[~m].mean())
    return float((np.abs(np.array(null)) >= abs(obs)).mean())


if __name__ == '__main__':
    rng = np.random.default_rng(3)
    P = M.geo_pages()
    data = [(pi, i['sres'], C.la45_class(i['id'])) for pi, p in enumerate(P) for i, s, o in C.featurize(p) if i['k'] == 'w' and i.get('sres') is not None]
    pid = np.array([d[0] for d in data]); s0 = np.array([d[1] for d in data]); cl = np.array([d[2] for d in data])
    out = {'sd_residual': round(float(s0.std()), 3), 'n': len(s0)}
    for d in (0.05, 0.1, 0.15, 0.2, 0.3):
        m = cl == 'commodity'; s = s0 + d * m
        out['commodity_shift_%.2f' % d] = ptest(pid, s, m, rng)
        m2 = rng.random(len(s0)) < 0.15; s = s0 + d * m2
        out['random15pct_shift_%.2f' % d] = ptest(pid, s, m2, rng)
    json.dump(out, open(os.path.join(C.CK, 'c2c.json'), 'w'), indent=1); print(out)
