#!/usr/bin/env python3
"""la25 cycle 4: how big must a burnt archive be for the clock to read it?

Planted power as archive size grows (x1, x4, x16, x64 the LA document counts), with base rates
free and known (oracle). Strong clock only (gamma > 1.2, terroir sigma < 0.4). This sets a bound
for any future find: what size of LM IB archive would give a usable destruction month.
"""
import json, sys
import numpy as np
from la25_common import *
from la25_c2 import bank2, power

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
out = open(os.path.join(CK, 'c4.log'), 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()

if __name__ == '__main__':
    X = profile(la_entries(), ['HT', 'KH', 'ZA', 'TY'], CORE); N0 = X.sum(1)
    beta0 = np.random.default_rng(31).normal(0, 1.5, len(CORE)); beta0[-1] = 0
    res = {}
    for mult in (1, 4, 16, 64):
        N = (N0 * mult).tolist()
        for tag, extra in (('free', {}), ('oracle', dict(beta0=beta0))):
            kw = dict(N=N, cats=CORE, groups=[1] * 4, fixed={}, s_off=0.0, **extra)
            B = bank2(400 + mult + (tag == 'oracle'), n=NB, **kw)
            pw = power(B, 500 + mult, **kw, n=100)
            # perfect calendar knowledge too: tau/floor/window are still drawn; report as is
            P(f'SIZE x{mult} ({int(sum(N))} doc-commodity units) base={tag}: mass on true {pw[0]:.3f} (chance .083) MAP+-1 {pw[1]:.3f} (chance .25) cov80 {pw[2]:.2f}')
            res[f'{mult}_{tag}'] = pw
    json.dump(res, open(os.path.join(CK, 'c4.json'), 'w'))
