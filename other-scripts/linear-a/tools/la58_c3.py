#!/usr/bin/env python3
"""LA-58 cycle 3.
A. Model adequacy: is each real corpus inside the cloud of simulated paperwork? Nearest-neighbour distance of the
   real panel vs the NN distances of 500 held-out simulated worlds (percentile; high = real is an outlier).
B. Deposits as units (HT Room 13, Villa Magazine, Casa rooms, roundel deposit, KH, KN, PH, ZA, sanctuaries...;
   two sides of one object merged): ABC-RF on a 60k bank. Are the Hagia Triada deposits one office with
   satellites (same institution) or separate houses? Null: 10 deposit-label shuffles.
"""
import sys, os, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la58_common import *
from la58_run import corpus, nk_of, load_bank
from la58_fit import train, read
from la58_c2 import zscale


def adequacy(bank, cname, ntest=500, rs=0):
    S, T, meta = load_bank(bank)
    S = np.nan_to_num(S, nan=-2)
    med, mad = zscale(S); Z = (S - med) / mad
    rng = np.random.RandomState(rs); te = rng.choice(len(S), ntest, replace=False)
    mask = np.ones(len(S), bool); mask[te] = False; R = Z[mask]
    nn = np.array([np.sqrt(((R - Z[i]) ** 2).sum(1)).min() for i in te])
    docs, K, ab, _ = corpus(cname)
    zr = (np.nan_to_num(stats(docs, K), nan=-2) - med) / mad
    d = np.sqrt(((R - zr) ** 2).sum(1)).min()
    return dict(real_nn=float(d), sim_nn_median=float(np.median(nn)), pct=float((nn < d).mean()))


def main():
    res = {}
    for b, c in [('LA', 'LA'), ('LA', 'LASHUF0'), ('LB', 'LB'), ('UR', 'UR'), ('LADEP', 'LADEP')]:
        res['adequacy_' + c] = adequacy(b, c); print(c, res['adequacy_' + c], flush=True)
    rf, cal = train('LADEP')
    res['DEP_calib'] = cal
    docs, K, ab, _ = corpus('LADEP')
    res['DEP_real'] = read(rf, stats(docs, K), K)
    res['DEP_shuf'] = [read(rf, stats(corpus('LADEP%d' % i)[0], K), K) for i in range(10)]
    res['abbr'] = ab
    jdump(res, 'c3_results.json')


if __name__ == '__main__':
    main()
