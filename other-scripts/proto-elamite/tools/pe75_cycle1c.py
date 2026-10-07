"""pe75 cycle 1c: test the frozen 11 mm height ripple (data/pe75_frozen_quantum.json)."""
import csv, json, os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe75_common as C
from pe75_cycle1b import qg, QS, fit_mix, draw

csv.field_size_limit(10 ** 9)
mus = {}
for row in csv.DictReader(open(C.CAT, encoding='utf-8')):
    if row['period'].startswith('Proto-Elamite'):
        mus['P%06d' % int(row['id_text'])] = (row['museum_no'], row['collection'])
R = [r for r in C.pe_table() if r['complete_cat'] and r['h'] and r['w'] and r['t']]
win = (QS >= 10.7) & (QS <= 11.3)


def excess(x, rng, K=2, NN=200, strata=None):
    x = np.asarray(x, float)
    z = qg(x)
    if strata is None:
        par = fit_mix(np.log(x), K, rng)
        N = np.array([qg(draw(par, len(x), rng)) for _ in range(NN)])
    else:
        pars = {s: fit_mix(np.log(x[strata == s]), 1, rng) for s in set(strata)}
        N = []
        for _ in range(NN):
            y = np.empty(len(x))
            for s, p in pars.items():
                m = strata == s
                y[m] = draw(p, m.sum(), rng)
            N.append(qg(y))
        N = np.array(N)
    ex = (z - N.mean(0)) / (N.std(0) + 1e-9)
    return round(float(ex[win].max()), 2), round(float(QS[ex.argmax()]), 1), round(float(ex.max()), 2)


rng = np.random.default_rng(7501)
out = {}
grp = {}
for r in R:
    m = mus.get(r['id'], ('', ''))[0]
    g = 'Louvre_Sb' if m.startswith('Sb') else ('other' if m else 'unknown')
    grp.setdefault(g, []).append(r)
for g, rs in grp.items():
    if len(rs) >= 60:
        out['P1:' + g] = (len(rs), excess([r['h'] for r in rs], rng))
h = np.array([r['h'] for r in R]); o = np.array([r['obv'] for r in R])
bins = np.digitize(o, [4, 6, 8, 11])
out['P2:h_lineBinNull'] = (len(R), excess(h, rng, strata=bins))
out['P3:w'] = (len(R), excess([r['w'] for r in R], rng))
out['P0:h_all'] = (len(R), excess(h, rng))
for k, v in out.items():
    print(k, v)
print({g: len(v) for g, v in grp.items()})
import collections
print(collections.Counter(mus.get(r['id'], ('', ''))[1][:40] for r in R).most_common(6))
json.dump(out, open(os.path.join(C.CK, 'cycle1c.json'), 'w'), indent=1)
