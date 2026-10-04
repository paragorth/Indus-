"""pe14 cycle 3b: FIXED-RECORD WEAVES. For period p = 2-6: number of tablets in which
one identical entry (same sign string) recurs at every p-th position >= 3 times in a
row (with a different entry right after it) -- the signature of a fixed p-line record
(pe12: 'name M288, 2' / 'M376, 1/4'). Real count vs (a) 1,000 within-tablet shuffles
and (b) 200 DIP surrogates (joint adjacency null of cycle 2/3, so the pe13 dip
cannot make it). Positive controls: Ur III Drehem lines, planted P3SIZE / P4FIRST.
Also lists the tablets and the recurring entries.
usage: python3 pe14_cycle3b.py <config>
"""
import json, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe14_common import CK  # noqa
from pe14_cycle1 import corpus, run_spaced  # noqa
from pe14_cycle3 import prep, tune, dip_perm  # noqa

PS = [2, 3, 4, 5, 6]


def counts(T):
    return np.array([sum(1 for t in T if run_spaced(t['u'], p) >= 3) for p in PS])


def main(cfg):
    T = [t for t in corpus(cfg) if len(t['u']) >= 3]
    rng = random.Random(sum(map(ord, cfg)))
    obs = counts(T)
    sh = np.array([counts([{'u': rng.sample(t['u'], len(t['u']))} for t in T]) for _ in range(1000)])
    _, tabs = prep(T)
    rho, _ = tune(tabs, rng)
    dp = []
    for _ in range(200):
        dp.append(counts([{'u': [t['u'][i] for i in dip_perm(tb, rho, rng)]} for t, tb in zip(T, tabs)]))
    dp = np.array(dp)
    which = {p: [(t['id'], len(t['u']), run_spaced(t['u'], p)) for t in T if run_spaced(t['u'], p) >= 3] for p in PS}
    out = {'cfg': cfg, 'ntab': len(T), 'rho': rho, 'obs': obs.tolist(),
           'shuf_m': sh.mean(0).tolist(), 'shuf_p': [float((1 + np.sum(sh[:, j] >= obs[j])) / 1001) for j in range(len(PS))],
           'dip_m': dp.mean(0).tolist(), 'dip_p': [float((1 + np.sum(dp[:, j] >= obs[j])) / 201) for j in range(len(PS))],
           'which': which}
    json.dump(out, open(os.path.join(CK, 'c3b_%s.json' % cfg), 'w'), indent=1)
    for j, p in enumerate(PS):
        print(cfg, 'p', p, 'obs', obs[j], 'shuf %.1f (p %.3f)' % (out['shuf_m'][j], out['shuf_p'][j]),
              'dip %.1f (p %.3f)' % (out['dip_m'][j], out['dip_p'][j]))


if __name__ == '__main__':
    main(sys.argv[1])
