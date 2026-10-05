"""pe44: shortfall signature.  For each group of quantities, rate of values just BELOW a round
anchor (SHORT) and just ABOVE (OVER) minus the same rates under the smooth same-notation null
(pe23 smooth_null, log-uniform x/1.5..x1.5, 300 reps).  A quota system that is usually missed
should give SHORT excess > OVER excess.  Ur III known kinds are the reference."""
import json, os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe44_common import *  # noqa
from pe44_cycle1 import KIND


def rates(vals, den):
    f = np.array([value_feats(int(v), den) for v in vals])
    big = np.array(vals) >= den[1]
    if big.sum() == 0:
        return np.nan, np.nan, np.nan
    return f[big, 3].mean(), f[big, 4].mean(), f[big, 0].mean()


def group_stat(vals, den, rng, reps=200):
    s, o, r1 = rates(vals, den)
    ns, no, nr = [], [], []
    for _ in range(reps):
        w = P23.smooth_null(vals, rng)
        a, b, c = rates(w, den)
        ns.append(a); no.append(b); nr.append(c)
    ns, no, nr = map(np.array, (ns, no, nr))
    asym = (s - o) - (ns - no)
    z = asym / ((ns - no).std() + 1e-9)
    return {'n': int(sum(np.array(vals) >= den[1])), 'short': s, 'over': o, 'null_short': ns.mean(),
            'null_over': no.mean(), 'asym': asym.mean() if hasattr(asym, 'mean') else asym, 'z': float(np.mean(z)),
            'round_ex': r1 - nr.mean()}


if __name__ == '__main__':
    rng = np.random.default_rng(3)
    u = ur3_tabs()
    groups = {}
    for t, d in u.items():
        for r in d['recs']:
            k = KIND.get(r['lab'])
            if r['lab'] in ('LIVESTOCK', 'PEOPLE', 'GRAIN', 'RATION', 'DEBIT', 'ESTIMATE', 'DEFICIT'):
                groups.setdefault(('UR', r['lab'], r['sys']), []).append(r['val'])
    pe = pe_tabs()
    for t, recs in pe.items():
        for r in recs:
            groups.setdefault(('PE', 'ALL', r['sys']), []).append(r['val'])
    res = []
    for (c, lab, sy), vals in sorted(groups.items()):
        if len(vals) > 20000:
            vals = list(rng.choice(vals, 20000, replace=False))
        st = group_stat(vals, DEN[sy], rng)
        st.update({'corpus': c, 'lab': lab, 'sys': sy})
        res.append(st)
        print(c, lab, sy, {k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()}, flush=True)
    json.dump(res, open(os.path.join(CK, 'short.json'), 'w'), indent=1)
