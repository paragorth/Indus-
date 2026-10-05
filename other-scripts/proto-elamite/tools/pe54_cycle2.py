"""pe54 cycle 2: do composition links PREDICT held-out sightings?
Tablet-split halves: the linking hypothesis is searched on the train half only; each held-out sighting's
classes are predicted one at a time from its best-linked train partner (chosen on its OTHER classes),
in bits per cell against the background (value of a random train entry of the same class).
Baselines: same hypothesis but a random partner (random linking, same demographic rules); a random
hypothesis (random roles and rates); nearest neighbour by plain composition with identity dynamics
(dt 0 only, i.e. 'flock unchanged') to see whether demography adds anything; nulls: shuffled classes.
usage: pe54_cycle2.py pe|ur3|plant  [n_splits]"""
import sys
import numpy as np
from pe54_lib import *
import pe54_lib as L

part = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rng = np.random.default_rng(5402)


def one_split(X, tabs, strings, n_hyp, climb):
    C = Corpus(X, tabs, strings)
    ut = np.unique(tabs)
    tr_t = set(rng.choice(ut, size=len(ut) // 2, replace=False))
    train = np.array([i for i in range(C.n) if tabs[i] in tr_t])
    test = np.array([i for i in range(C.n) if tabs[i] not in tr_t])
    if len(train) < 6 or len(test) < 3:
        return None
    Ctr = Corpus(X[train], tabs[train], [strings[i] for i in train])
    top, _ = search(Ctr, n_hyp, rng, keep=20, climb=climb)
    s, r, th = top[0]
    out = {}
    out['fit'] = heldout_bits(C, r, th, train, test)[0]
    out['rand_partner'] = heldout_bits(C, r, th, train, test, rng=rng, random_partner=True)[0]
    rr, tt = sample_roles(C.K, rng), sample_rates(rng)
    out['rand_hyp'] = heldout_bits(C, rr, tt, train, test)[0]
    old = L.DTS
    L.DTS = (0,)
    ident = np.where(X.sum(0) > 0, 1, 0)        # every class its own F0 role, no dynamics
    out['static_nn'] = heldout_bits(C, ident, dict(th, k=th['k']), train, test)[0]
    L.DTS = old
    # string check: does the chosen partner share the string more than chance?
    return out, r


res = dict(part=part, splits=[])
if part == 'pe':
    R, X, tabs, st = pe_corpus()
    nh, cl = 3000, 3
elif part == 'ur3':
    d, X, tabs, st, yr = ur3_corpus()
    nh, cl = 600, 2
else:
    X, tabs, st, yrs, truth, th0 = plant_archive(rng, n_flocks=150, years=8, survive=0.04)
    st = list(st)
    nh, cl = 2000, 3
for sp in range(NS):
    o = one_split(X, tabs, st, nh, cl)
    if o is None:
        continue
    o0 = one_split(shuffle_within_class(X, rng), tabs, st, nh, cl)
    row = dict(real=o[0], roles=o[1], shuffled=o0[0] if o0 else None)
    res['splits'].append(row)
    print(sp, {k: round(v, 3) for k, v in o[0].items()}, o[1], 'shuf', {k: round(v, 3) for k, v in o0[0].items()} if o0 else None, flush=True)
    dump(res, os.path.join(CK, 'c2_%s.json' % part))
print('done', part)
