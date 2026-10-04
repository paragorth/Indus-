"""v40 helper: Mantel-Haenszel odds ratio of the B-member (t, sh, benched) within (frame x page) strata, for
paragraph-first line vs other lines, and right vs left half of the line (non-first lines only). 95% CI by page
bootstrap (200). Both transcriptions."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v40_lib import *
from v40_cycle1 import vpairs


def mh(t, cond, keep):
    by = defaultdict(lambda: [[0, 0], [0, 0]])
    for x in t:
        if keep(x):
            by[(x['frame'], x['page'])][int(cond(x))][x['y']] += 1
    num = den = 0.0
    for (a0, a1), (b0, b1) in by.values():
        n = a0 + a1 + b0 + b1
        if a0 + a1 and b0 + b1:
            num += b1 * a0 / n; den += b0 * a1 / n
    return num / den if den else float('nan')


rng = np.random.default_rng(7)
for nm in ('ZL3b', 'IT2a'):
    P = load_voynich(nm); T = extract(P, vpairs(), V_TALL)
    for pn in ('CS', 'KT', 'BENCH', 'PF'):
        t = [x for x in T if x['pair'] == pn]
        pages = sorted(set(x['page'] for x in t)); byp = defaultdict(list)
        for x in t:
            byp[x['page']].append(x)
        tests = {'first-line': (lambda x: x['ps'] == 1, lambda x: True),
                 'right-half': (lambda x: x['xrel'] > 0.5, lambda x: x['ps'] == 0),
                 'first-word': (lambda x: x['wi'] == 0, lambda x: x['ps'] == 0)}
        out = []
        for k, (c, kp) in tests.items():
            o = mh(t, c, kp)
            bs = []
            for _ in range(200):
                s = [x for p in rng.choice(pages, len(pages)) for x in byp[p]]
                bs.append(mh(s, c, kp))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            out.append('%s OR %.2f [%.2f-%.2f]' % (k, o, lo, hi))
        print(nm, pn, '; '.join(out), flush=True)
