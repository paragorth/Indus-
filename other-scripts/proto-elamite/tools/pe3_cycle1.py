"""pe3 cycle 1: fingerprints of phonetic spelling, PE middles vs calibrated name lists.

All corpora are cut to the same size (N distinct strings) and the same length
profile (PE middles, bins 2/3/4/5+), R replicates.  Each statistic is reported
raw and as z against a within-corpus null:
  fin_topk, ini_topk, order_cons, aa_rate : within-string shuffle (keeps each string's signs)
  gap_share, pos_free                      : global token shuffle (keeps lengths and unigrams)
"""
import json, random, sys, os
from pe3_common import *

N = int(os.environ.get('PE3_N', 360))
R = int(os.environ.get('PE3_R', 20))
NNULL = 30
OUT = os.path.join(PEDATA, 'pe3_cycle1.json')


def stats(c, rng):
    d = {}
    d.update(inventory(c, rng=rng))
    d.update(final_closure(c))
    d.update(order_consistency(c))
    d.update(phonotactic_gaps(c))
    d.update(repeat_rate(c))
    d.update(positional_freedom(c))
    return d


NULLMAP = {'fin_topk': 'w', 'ini_topk': 'w', 'order_cons': 'w', 'aa_rate': 'w',
           'gap_excess': 'g', 'adj_mi': 'g', 'pos_free': 'g'}


def run(name, corpus, profile, rng):
    rows = []
    for r in range(R):
        smp, short = matched_sample(corpus, profile, N, rng)
        obs = stats(smp, rng)
        nw = [stats(shuffle_within(smp, rng), rng) for _ in range(NNULL)]
        ng = [stats(shuffle_global(smp, rng), rng) for _ in range(NNULL)]
        z = {}
        for k, which in NULLMAP.items():
            nl = [x[k] for x in (nw if which == 'w' else ng)]
            nl = [x for x in nl if x == x]
            if obs[k] == obs[k] and len(nl) > 2:
                z['z_' + k], _ = zscore(obs[k], nl)
        obs.update(z)
        obs['short'] = short
        rows.append(obs)
    keys = rows[0].keys()
    mean = {k: sum(r[k] for r in rows if k in r and r[k] == r[k]) /
            max(1, sum(1 for r in rows if k in r and r[k] == r[k])) for k in keys}
    return mean


def main():
    rng = random.Random(1)
    pe = pe_middles('A')
    prof = length_profile(pe)
    corpora = {'PE_mid_A': (pe, 'PE'), 'PE_full_B': (pe_middles('B'), 'PE'),
               'PE_mid_C': (pe_middles('C'), 'PE')}
    for k, (p, ty) in CALIB.items():
        corpora[k] = (load_calib(k), ty)
    res = {}
    for k, (c, ty) in corpora.items():
        m = run(k, c, prof, rng)
        m['type'] = ty
        m['n_distinct'] = len(c)
        res[k] = m
        print(k, ty, len(c), {a: round(b, 3) for a, b in m.items() if isinstance(b, float)}, flush=True)
    json.dump({'N': N, 'R': R, 'profile': prof, 'res': res}, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
