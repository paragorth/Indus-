#!/usr/bin/env python3
"""la64 cycle 3b: per-document version of the 'men then commodities' test (one person total per
document; null draws one random person total per document and random quantities)."""
import sys, json, math
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from la64_lib import *
LA = load_la()
r = json.load(open(os.path.join(CK, 'c3.json')))
clu = r['cluster']
allp = [p['n'] for d in LA for p in d['P'] if p['tot']] + [sum(p['n'] for p in d['P'] if not p['tot']) for d in LA if d['P']]
allp = [p for p in allp if p > 0]
rng = np.random.default_rng(642)
out = {}
for fv in (0.25, 0.5):
    for c in ('CYP', 'NI', 'VIN', 'OLE'):
        docs = sorted({x['doc'] for x in clu if x['c'] == c})
        if len(docs) < 2:
            continue
        tot = {dd: sum(x['n'] + fv * len(x['fr']) for x in clu if x['c'] == c and x['doc'] == dd) for dd in docs}
        per = {dd: [x['persons'] for x in clu if x['doc'] == dd][0] for dd in docs}
        lr = np.log([tot[dd] / per[dd] for dd in docs])
        av = [e['n'] + fv * len(e['fr']) for d in LA for e in d['E'] if e['c'] == c and not e['tot'] and e['n'] + len(e['fr']) > 0]
        # null: each document's commodity total = sum of k random entries (k as observed), persons random
        ks = {dd: sum(1 for x in clu if x['c'] == c and x['doc'] == dd) for dd in docs}
        nl = []
        for _ in range(20000):
            v = [math.log(sum(rng.choice(av) for _ in range(ks[dd])) / rng.choice(allp)) for dd in docs]
            nl.append(np.std(v))
        nl = np.array(nl)
        m, s = DAY[la_class(c)]
        out['%s|%g' % (c, fv)] = dict(docs=docs, per_head_units=[round(float(math.exp(x)), 4) for x in lr],
                                      logsd=float(lr.std()), p_null=float((nl <= lr.std()).mean()),
                                      implied_u_day=round(float(m / math.exp(lr.mean())), 2),
                                      corr_log_q_persons=float(np.corrcoef(np.log([tot[dd] for dd in docs]), np.log([per[dd] for dd in docs]))[0, 1]) if len(docs) > 2 else None)
json.dump(out, open(os.path.join(CK, 'c3b.json'), 'w'), indent=1)
for k, v in out.items():
    print(k, v)
