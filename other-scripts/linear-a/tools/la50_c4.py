#!/usr/bin/env python3
"""la50 cycle 4: the SA-RA2 basket alone (la46) and the shared LA/LB staple order.
(a) Baskets on tablets that contain SA-RA2: Aegean fit vs 10^5 random calendars, permutations.
(b) Does the LA basket order agree with the LB best order more than with random orders?
    (Kendall-type concordance of LA pairs with the LB best permutation of c1 vs all permutations.)
"""
import itertools, json
from la50_common import *
rng = np.random.default_rng(504)
OUT = os.path.join(CK, 'c4.log'); open(OUT, 'w').close()
c = json.load(open(os.path.join(DATA, 'corpus.json')))
sara = {base_id(r['id']) for r in c if any(t['t'] == 'word' and t['s'] == ['SA', 'RA₂'] for t in r['tokens'])}
la = load_la()
for nm, docs in [('SA-RA2 tablets', [d for d in la if d['id'] in sara]), ('non-SA-RA2 LA', [d for d in la if d['id'] not in sara])]:
    us = units(docs, 'basket')
    it = sorted({x for _, o in us for x in o if x in AEGEAN})
    a, b, w, _ = pairs(us, it)
    fa, sa = conc(cal_vec(AEGEAN, it)[None], a, b, w, return_start=True)
    R = rng.uniform(0, 12, (100000, len(it)))
    fr = np.concatenate([conc(R[i:i+20000], a, b, w) for i in range(0, 100000, 20000)])
    fp = conc(np.array(list(itertools.permutations(cal_vec(AEGEAN, it)))), a, b, w)
    log(OUT, '%s: %d tablets, %d baskets, %d pairs, items %s | Aegean fit %.3f (start %s) | P_random %s | P_perm %s (%d perms) | north %.3f'
        % (nm, len(docs), len(us), len(w), ','.join(it), fa[0], MONTHS[int(sa[0])], fmt((fr >= fa[0]).mean()), fmt((fp >= fa[0]).mean()), len(fp),
           conc(cal_vec(NORTH, it)[None], a, b, w)[0]))
# (b) LA vs LB order
lbord = ['GRA', 'OLIV', 'FIC', 'VIN', 'OLE']   # LB-KN+PY best permutation restricted to crops (c1)
us = units(la, 'basket')
it = ['GRA', 'OLIV', 'FIC', 'VIN', 'OLE']
a, b, w, _ = pairs(us, it)
def lin(order):
    rk = np.array([order.index(x) for x in it], float)
    return float((((rk[a] < rk[b]) * 1.0) @ w) / w.sum())
v = lin(lbord); allp = np.array([lin(list(p)) for p in itertools.permutations(it)])
aeg = sorted(it, key=lambda x: (AEGEAN[x] - 4.0) % 12)
log(OUT, 'LA baskets vs LB staple order %s: concordance %.3f, rank %d/120, P %s; vs Aegean calendar order from May %s: %.3f, rank %d/120'
    % ('>'.join(lbord), v, int((allp > v).sum()) + 1, fmt((allp >= v).mean()), '>'.join(aeg), lin(aeg), int((allp > lin(aeg)).sum()) + 1))
# with OLE/OLIV merged as one olive class
us2 = [(i, list(dict.fromkeys(['OLV' if x in ('OLE', 'OLIV') else x for x in o]))) for i, o in us]
it2 = ['GRA', 'OLV', 'FIC', 'VIN']; a, b, w, _ = pairs(us2, it2)
def lin2(order):
    rk = np.array([order.index(x) for x in it2], float)
    return float((((rk[a] < rk[b]) * 1.0) @ w) / w.sum())
allp = np.array([lin2(list(p)) for p in itertools.permutations(it2)])
for nm, o in [('LB staple GRA>OLV>FIC>VIN', ['GRA', 'OLV', 'FIC', 'VIN']), ('Aegean GRA>FIC>VIN>OLV', ['GRA', 'FIC', 'VIN', 'OLV'])]:
    log(OUT, 'olive merged: LA vs %s: %.3f rank %d/24' % (nm, lin2(o), int((allp > lin2(o)).sum()) + 1))
