#!/usr/bin/env python3
"""la50 cycle 1: does written commodity order follow the Aegean farming year?

For each corpus and unit (basket, tablet list): concordance of written order with the Aegean
calendar (maximised over the administrative year start). Nulls: (a) 10^5 random calendars
(each item a uniform month); (b) every permutation of the Aegean month values over the items;
(c) the northern-European calendar; (d) 2,000 within-unit shuffles of written order.
Planted control: LA baskets re-ordered by the Aegean calendar with probability p.
"""
import itertools, json
from la50_common import *

rng = np.random.default_rng(50)
OUT = os.path.join(CK, 'c1.log'); open(OUT, 'w').close()
NR = 100000


def analyse(name, docs, unit, items_pool=None, n_shuf=2000, quiet=False, drop_pair=None):
    us = units(docs, unit)
    if drop_pair:
        us = [(i, [x for x in o if x != drop_pair]) for i, o in us]
    present = sorted({x for _, o in us for x in o if x in AEGEAN})
    a, b, w, g = pairs(us, present)
    if len(w) == 0: return None
    aeg = cal_vec(AEGEAN, present)
    fa, sa = conc(aeg[None], a, b, w, return_start=True)
    fa, sa = fa[0], sa[0]
    fn = conc(cal_vec(NORTH, present)[None], a, b, w)[0]
    R = rng.uniform(0, 12, (NR, len(present)))
    fr = np.concatenate([conc(R[i:i + 20000], a, b, w) for i in range(0, NR, 20000)])
    perms = np.array(list(itertools.permutations(aeg)))
    fp = conc(perms, a, b, w)
    # shuffles of written order within each unit
    fs = []
    for k in range(n_shuf):
        us2 = [(i, list(rng.permutation(o))) for i, o in us]
        a2, b2, w2, _ = pairs(us2, present)
        fs.append(conc(aeg[None], a2, b2, w2)[0])
    fs = np.array(fs)
    # best permutation (the order the data want)
    bi = np.argmax(fp)
    best_order = [present[j] for j in np.argsort((perms[bi] - conc(perms[bi:bi + 1], a, b, w, return_start=True)[1][0]) % 12)]
    res = dict(name=name, unit=unit, n_units=len(us), n_pairs=len(w), items=present, fit=fa,
               start=MONTHS[int(sa) % 12] + ('' if sa == int(sa) else '+'),
               P_random=float((fr >= fa).mean()), rand_med=float(np.median(fr)),
               P_perm=float((fp >= fa).mean()), n_perm=len(fp), perm_rank=int((fp > fa).sum()) + 1,
               north=fn, north_pct=float((fr >= fn).mean()),
               P_shuf=float((fs >= fa).mean()), shuf_med=float(np.median(fs)),
               best_perm_fit=float(fp[bi]), best_order=best_order)
    if not quiet:
        log(OUT, '%s %s: units %d pairs %d items %s | Aegean fit %.3f (start %s) | random median %.3f P %s | '
                 'perm rank %d/%d P %s | north %.3f (P_rand %s) | shuffle median %.3f P %s | best perm %.3f order %s'
            % (name, unit, len(us), len(w), ','.join(present), fa, res['start'], res['rand_med'],
               fmt(res['P_random']), res['perm_rank'], len(fp), fmt(res['P_perm']), fn, fmt(res['north_pct']),
               res['shuf_med'], fmt(res['P_shuf']), res['best_perm_fit'], '>'.join(best_order)))
    return res


def planted(docs, p, reps=20):
    """re-order known items in each LA basket by the Aegean calendar from start May (prob p)."""
    out = []
    for r in range(reps):
        D2 = []
        for d in docs:
            nb = []
            for bkt in d['baskets']:
                if rng.random() < p:
                    kn = [x for x in bkt if x in AEGEAN]
                    kn_sorted = sorted(kn, key=lambda x: (AEGEAN[x] - 4.0) % 12)
                    it = iter(kn_sorted)
                    nb.append([next(it) if x in AEGEAN else x for x in bkt])
                else:
                    nb.append(list(rng.permutation(bkt)))
            D2.append(dict(d, baskets=nb))
        res = analyse('PLANT p=%.1f' % p, D2, 'basket', n_shuf=200, quiet=True)
        out.append((res['fit'], res['P_random'], res['P_perm'], res['P_shuf']))
    o = np.array(out)
    log(OUT, 'PLANTED p=%.1f (%d reps): fit %.3f, P_random median %s (<0.05 in %d/%d), P_perm median %s, P_shuf median %s'
        % (p, reps, o[:, 0].mean(), fmt(np.median(o[:, 1])), (o[:, 1] < 0.05).sum(), reps, fmt(np.median(o[:, 2])),
           fmt(np.median(o[:, 3]))))
    return o.tolist()


if __name__ == '__main__':
    la = load_la(); lb = load_lb(); kn = load_lb(('KN',)); py = load_lb(('PY',))
    ht = [d for d in la if d['id'].startswith('HT')]
    R = []
    log(OUT, 'Calendar sources: ' + CAL_SRC)
    log(OUT, 'Aegean months: ' + ', '.join('%s %s' % (k, MONTHS[int(v)]) for k, v in AEGEAN.items()))
    for nm, d in [('LA', la), ('LA-HT', ht), ('LB-KN+PY', lb), ('LB-KN', kn), ('LB-PY', py)]:
        for u in ('basket', 'tablet'):
            r = analyse(nm, d, u)
            if r: R.append(r)
    # LB without livestock->wool (flock + wool target records are administrative, not seasonal)
    for nm, d in [('LB-KN+PY noLANA', lb)]:
        r = analyse(nm, d, 'basket', drop_pair='LANA'); R.append(r)
        r = analyse('LB-KN+PY noLIV', d, 'basket', drop_pair='LIV'); R.append(r)
    P = {p: planted(la, p) for p in (0.3, 0.5, 0.8)}
    json.dump({'results': R, 'planted': P}, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)
