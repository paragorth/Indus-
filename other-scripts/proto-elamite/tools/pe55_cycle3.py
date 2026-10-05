"""pe55 cycle 3: does a sign's activity replicate between disjoint halves of the archive?
usage: pe55_cycle3.py pe|ur3thin   -> data/pe55_ckpt/c3_<src>.json
For 20 random splits, fit (no area labels) separately on half A and half B. Per key: coarse class
(person / stock / none) and period (d / m / y) in each half. Agreement A vs B over keys, against the
same procedure on data whose capacity values are shuffled across lines (10 shuffles x 20 splits).
Also the unit u of each half (scale replication), and per-sign class tallies (the readout)."""
import sys, json, math, time
import numpy as np
from collections import Counter, defaultdict
from pe55_lib import *

src = sys.argv[1]
rng = np.random.default_rng(553)
t0 = time.time()


def fit(D, n_hyp=1500):
    R = random_search(D, rng, n_hyp, False, n_rand_lab=10)
    b = int(np.argmax(R['s_best']))
    return float(R['logu'][b]), R['lab_best'][b]


def cls(a):
    return 'none' if a < 0 else ACTS[a].split('-')[0]


def agree(D, n_split=20):
    ag, tot, ag_full, us, tally = 0, 0, 0, [], defaultdict(Counter)
    for s in range(n_split):
        A = split_tabs(D, rng)
        ua, la = fit(subset(D, A, True))
        ub, lb = fit(subset(D, A, False))
        us.append((ua, ub))
        for i, k in enumerate(D['keys']):
            na = int((subset(D, A, True)['k'] == i).sum())
            nb = int((subset(D, A, False)['k'] == i).sum())
            for lab_, n_ in ((la[i], na), (lb[i], nb)):
                if n_ >= 2:
                    tally[k][ACTS[lab_] if lab_ >= 0 else 'none'] += 1
            if na >= 2 and nb >= 2 and la[i] >= 0 and lb[i] >= 0:
                tot += 1
                ag += cls(la[i]) == cls(lb[i])
                ag_full += la[i] == lb[i]
    return dict(agree=ag, tot=tot, agree_full=ag_full, us=us,
                tally={k: dict(v) for k, v in tally.items()})


if src == 'pe':
    D = build_arrays(pe_pairs(), 'kc', 3)
    truth = None
else:
    P, _ = opaque(ur3_pairs())
    D0 = build_arrays(P, 'kc', 3)
    tabs = np.unique(D0['tab'])
    sel = set(rng.choice(tabs, 220, replace=False))
    Dt = subset(D0, sel, True)
    cnt = Counter(Dt['k'].tolist())
    keys = [D0['keys'][i] for i in sorted(cnt) if cnt[i] >= 3]
    D = build_arrays(Dt['P'], 'kc', 3, keys)
    truth = {}
    for p in D['P']:
        truth.setdefault(p['kc'], Counter())[p['truth']] += 1
    truth = {k: v.most_common(1)[0][0] for k, v in truth.items()}
real = agree(D)
print('real', real['agree'], real['tot'], real['agree_full'], flush=True)
nulls = []
for j in range(10):
    r = agree(shuffle_q(D, rng))
    nulls.append((r['agree'], r['tot'], r['agree_full'], r['us']))
    print('null', j, r['agree'], r['tot'], r['agree_full'], flush=True)
res = dict(src=src, n_pairs=len(D['lr']), keys=D['keys'], real=real, nulls=nulls, truth=truth,
           secs=time.time() - t0)
json.dump(res, open(os.path.join(CK, f'c3_{src}.json'), 'w'), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('done', res['secs'])
