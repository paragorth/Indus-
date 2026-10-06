#!/usr/bin/env python3
"""la66 cycle 3b: global test with the direct (pe58) estimator.  For every feature on >= 4 documents, the
within-document x stratum factor and a document-bootstrap 95 % interval; count features whose interval
excludes 1, in the real corpus and in 40 corpora with quantities shuffled within document x stratum.
Linear A, Linear B at LA size (3 draws), planted LA (3 seeds)."""
import os, json
import numpy as np
from la66_lib import *
from la66_c3 import all_factors


def nsig(rows, B=200, seed=0):
    F = all_factors(rows, B=B, seed=seed)
    return sum(1 for v in F.values() if v[1] > 0 or v[2] < 0), len(F), F


def test(rows, n_null=40, tag=''):
    k, n, F = nsig(rows)
    ks = [nsig(null_q_doc(rows, 7000 + s), B=100, seed=s)[0] for s in range(n_null)]
    p = (sum(x >= k for x in ks) + 1) / (n_null + 1)
    sig = sorted([(f, round(float(np.exp(v[0])), 2), v[3]) for f, v in F.items() if v[1] > 0 or v[2] < 0], key=lambda x: x[1])
    return dict(tag=tag, k=k, n=n, null_mean=float(np.mean(ks)), null_max=int(max(ks)), p=p, sig=sig)


if __name__ == '__main__':
    L = la_rows()
    out = [test(L, tag='LA')]
    for s in range(3):
        out.append(test(thin(lb_rows(), len(L), s), tag=f'LB{s}'))
        P, eff = plant(L, 900 + s)
        r = test(P, tag=f'PLANT{s}'); r['planted'] = eff; out.append(r)
    for r in out:
        print(r['tag'], 'sig', r['k'], '/', r['n'], 'null', round(r['null_mean'], 1), 'max', r['null_max'], 'p', round(r['p'], 3))
        if r['tag'] == 'LA':
            print('  ', r['sig'])
    json.dump(out, open(os.path.join(CK, 'c3b.json'), 'w'))
