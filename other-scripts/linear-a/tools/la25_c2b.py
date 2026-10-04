#!/usr/bin/env python3
"""la25 cycles 1-2 completion (the 10^6 runs hit the job time limit): LB planted power lines of
cycle 1, and the transfer runs of cycle 2 not yet logged (s_off 0.3 shuffle null, random-calendar
model choice, s_off 1.0). Banks of 3x10^5."""
import json, sys
import numpy as np
from la25_common import *
from la25_c2 import bank2, post, power, SPRING

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
out = open(os.path.join(CK, 'c2b.log'), 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()

la = la_entries(); lb = lb_entries()
LB4 = ['KN', 'PY', 'TH', 'MY']
XLBe = profile(lb, LB4, EXT); XLB4 = profile(lb, LB4, CORE)
for tag, X, cats in (('LB-ext', XLBe, EXT), ('LB-core', XLB4, CORE)):
    kw = dict(N=X.sum(1).tolist(), cats=cats, groups=[0] * 4, fixed={}, s_off=0.0)
    B = bank2(700 + len(cats), n=NB, **kw)
    pw = power(B, 710 + len(cats), **kw, n=100)
    P(f'[c1] PLANTED {tag}-size, free base rates, strong clock: mass {pw[0]:.3f} (chance .083) MAP+-1 {pw[1]:.3f} cov80 {pw[2]:.2f}')

XLB = profile(lb, ['KN', 'PY'], CORE); XLA = profile(la, ['HT', 'KH', 'ZA', 'TY'], CORE)
X6 = np.concatenate([XLB, XLA]); N6 = X6.sum(1).tolist(); A6 = ['KN', 'PY', 'HT', 'KH', 'ZA', 'TY']
res = {}
for s_off in (0.3, 1.0):
    banks = {}
    for mode in ('agro', 'random'):
        kw = dict(N=N6, cats=CORE, groups=[0, 0, 1, 1, 1, 1], fixed={0: SPRING, 1: SPRING}, s_off=s_off, mode=mode)
        B = bank2(800 + int(s_off * 10) + (mode == 'random'), n=NB, **kw)
        banks[mode] = B
        Pm, ps, idx = post(B, X6)
        P(f'[c2] TRANSFER s_off={s_off} calendar={mode}: P(LA same month)={ps:.2f}')
        for a in range(2, 6):
            p = Pm[a]
            P(f'   {A6[a]}: MAP {MONTHS[int(p.argmax())]} p={p.max():.3f} 80% [{fmt_months(cred_set(p))}] KL {kl_unif(p):.3f}  ' + ' '.join(f'{x:.2f}' for x in p))
        res[f'{s_off}_{mode}'] = Pm.tolist()
        if mode == 'agro':
            rng = np.random.default_rng(5)
            klr = sum(kl_unif(Pm[a]) for a in range(2, 6)); ks = []
            for _ in range(200):
                Xs = X6.copy()
                for a in range(2, 6): Xs[a] = rng.permutation(Xs[a])
                Ps, _, _ = post(B, Xs); ks.append(sum(kl_unif(Ps[a]) for a in range(2, 6)))
            P(f'   SHUFFLE null (LA commodity labels permuted per site): summed KL real {klr:.3f} vs median {np.median(ks):.3f}, P {(np.array(ks) >= klr).mean():.3f}')
    S = np.concatenate([banks['agro']['S'], banks['random']['S']])
    idx, _ = abc(S, summ(X6[None])[0], 1000, banks['agro']['scale'])
    P(f'   MODEL CHOICE agronomic vs random calendars (LB+LA joint): P(agro) {(idx < NB).mean():.3f} (prior 0.5)')
json.dump(res, open(os.path.join(CK, 'c2b.json'), 'w'))
