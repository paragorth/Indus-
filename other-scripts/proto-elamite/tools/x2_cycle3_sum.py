#!/usr/bin/env python3
"""X-2 cycle 3 summary: LA <-> LB identity control, LA <-> Ur III via conventional labels,
split-half concordance of LA <-> PE, triangle consistency, and the control's precision at the
family-wise stability threshold (from cycle 2)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, random
from collections import Counter, defaultdict
import numpy as np
import x2_common as X

G = json.load(open(X.GOLD_FILE))
GOLD = G['gold']
rows3 = X.jsonl(os.path.join(X.DX, 'c3_worlds.jsonl'))
rows2 = X.jsonl(os.path.join(X.DX, 'c2_worlds.jsonl'))
by3 = defaultdict(list)
for r in rows3:
    by3[r['name']].append(r)
by2 = defaultdict(list)
for r in rows2:
    by2[r['name']].append(r)
print({k: len(v) for k, v in by3.items()})
M = ('prof', 'joint', 'flood', 'freq')
out = {}


def modal(d):
    c = Counter(d); t, n = c.most_common(1)[0]
    return t, n / sum(c.values())


def ok(d, bt):
    return sum(d.values()) >= bt / 2


def pooled_modal(rs, m, key='top'):
    agg = defaultdict(Counter)
    for r in rs:
        for s, d in r['res'][m][key].items():
            agg[s].update(d)
    return {s: modal(d) for s, d in agg.items()}


# ---------------- (a) LA <-> LB identity control
LA_LB_GOLD = ['GRA', 'OLE', 'VIN', 'OLIV', 'VIR', 'NI', 'CYP', 'AROM', 'CAP', 'TELA', 'HIDE']
print('\n(a) LA <-> LB, identity of conventional labels (top-1 of the modal partner)')
for m in M:
    def acc(r):
        h = n = 0
        for s, d in r['res'][m]['top'].items():
            if s in LA_LB_GOLD and ok(d, 100):
                t, st = modal(d)
                tg = set(r['res'][m]['btop'].keys())
                if s in tg:
                    n += 1; h += (t == s)
        return h, n
    real = [acc(r) for r in by3['la_lb']]; sh = [acc(r) for r in by3['la_lb_s']]
    rh = np.mean([h for h, n in real]); rn = np.mean([n for h, n in real])
    sv = np.array([h for h, n in sh])
    pm = pooled_modal(by3['la_lb'], m)
    o = {'hits_real': float(rh), 'n_gold': float(rn), 'hits_shuf_mean': float(sv.mean()),
         'P': float((1 + (sv >= rh).sum()) / (1 + len(sv))),
         'modal': {s: (pm[s][0], round(pm[s][1], 2)) for s in LA_LB_GOLD if s in pm}}
    out['la_lb|' + m] = o
    print(m, {k: v for k, v in o.items() if k != 'modal'}, o['modal'])

# ---------------- (a2) LA <-> Ur III with LA labels read through the LB gold
print('\n(a2) LA <-> Ur III, gold = LB gold of the same label')
for m in M:
    pm = pooled_modal(by3['la_ur'], m)
    per = []
    for r in by3['la_ur']:
        h = n = 0
        for s, d in r['res'][m]['top'].items():
            if s in GOLD and ok(d, 100):
                tg = set(r['res'][m]['btop'].keys())
                if any(x in tg for x in GOLD[s]):
                    n += 1; h += modal(d)[0] in GOLD[s]
        per.append((h, n))
    # chance: random target among the world's commodity targets
    ch = []
    for r in by3['la_ur']:
        tg = list(r['res'][m]['btop'].keys())
        for s, d in r['res'][m]['top'].items():
            if s in GOLD and ok(d, 100) and any(x in tg for x in GOLD[s]):
                ch.append(sum(x in GOLD[s] for x in tg) / len(tg))
    o = {'hits': float(np.mean([h for h, n in per])), 'n': float(np.mean([n for h, n in per])),
         'chance_hits': float(np.sum(ch) / len(per)) if per else None,
         'modal': {s: (pm[s][0], round(pm[s][1], 2)) for s in GOLD if s in pm}}
    out['la_ur|' + m] = o
    print(m, {k: v for k, v in o.items() if k != 'modal'}, o['modal'])

# ---------------- (b) split halves
print('\n(b) split-half concordance of LA -> PE modal partners')
for m in M:
    def conc(r):
        a, b = r['halves'][0][m]['top'], r['halves'][1][m]['top']
        com = [s for s in a if s in b and ok(a[s], 50) and ok(b[s], 50)]
        if not com:
            return None
        return np.mean([modal(a[s])[0] == modal(b[s])[0] for s in com]), len(com), \
            {s: modal(a[s])[0] == modal(b[s])[0] for s in com}
    real = [conc(r) for r in by3['split']]; real = [x for x in real if x]
    sh = [conc(r) for r in by3['split_s']]; sh = [x for x in sh if x]
    rv = np.mean([x[0] for x in real]); sv = np.array([x[0] for x in sh])
    per = defaultdict(list)
    for x in real:
        for s, v in x[2].items():
            per[s].append(v)
    o = {'conc_real': float(rv), 'n_items': float(np.mean([x[1] for x in real])), 'conc_shuf': float(sv.mean()),
         'P': float((1 + (sv >= rv).sum()) / (1 + len(sv))),
         'per_item': {s: (round(float(np.mean(v)), 2), len(v)) for s, v in per.items() if len(v) >= 10}}
    # which partner replicates in both halves for the main items
    agree = defaultdict(Counter)
    for r in by3['split']:
        a, b = r['halves'][0][m]['top'], r['halves'][1][m]['top']
        for s in a:
            if s in b and ok(a[s], 50) and ok(b[s], 50) and modal(a[s])[0] == modal(b[s])[0]:
                agree[s][modal(a[s])[0]] += 1
    o['agree'] = {s: c.most_common(2) for s, c in agree.items()}
    out['split|' + m] = o
    print(m, {k: v for k, v in o.items() if k not in ('per_item', 'agree')})
    print('   per item', o['per_item'])
    print('   agreeing partners', o['agree'])

# ---------------- (c) triangle
print('\n(c) triangle: LA X -> PE Y (c2), X -> LB (a), Y -> Ur III (pe_ur); consistent if (LB, UR3) is gold')
for m in ('prof', 'joint', 'flood'):
    lape = by2['lape'][0]['res'][m]['top']
    la_lb = pooled_modal(by3['la_lb'], m)
    pe_ur = pooled_modal(by3['pe_ur'], m)
    pairs = [(s, modal(d)[0]) for s, d in lape.items() if ok(d, 1000)]
    def consistent(x, y):
        if x not in la_lb or y not in pe_ur:
            return None
        lb, ur = la_lb[x][0], pe_ur[y][0]
        return ur in GOLD.get(lb, [])
    cs = [(x, y, la_lb.get(x, ('?',))[0], pe_ur.get(y, ('?',))[0], consistent(x, y)) for x, y in pairs]
    real = sum(1 for c in cs if c[4]); n = sum(1 for c in cs if c[4] is not None)
    pes = [y for x, y in pairs]
    rng = random.Random(1); null = []
    pe_all = list(pe_ur.keys())
    for _ in range(5000):
        null.append(sum(1 for x, y in pairs if consistent(x, rng.choice(pe_all))))
    null = np.array(null)
    o = {'consistent': real, 'n': n, 'null_mean': float(null.mean()), 'P': float((1 + (null >= real).sum()) / (1 + len(null))),
         'chains': [c for c in cs if c[4] is not None]}
    out['triangle|' + m] = o
    print(m, {k: v for k, v in o.items() if k != 'chains'})
    for c in o['chains']:
        print('   ', c)
    print('   PE -> Ur III modal:', {k: (v[0], round(v[1], 2)) for k, v in pe_ur.items()})

# ---------------- (d) control precision at the family-wise threshold (cycle 2 worlds)
print('\n(d) control: precision of modal pairs above the family-wise stability threshold')
for m in ('prof', 'joint', 'flood'):
    for ctl in ('ctl_lape', 'ctl_pe'):
        fam = []
        for r in by2[ctl + '_s']:
            v = [modal(d)[1] for d in r['res'][m]['top'].values() if ok(d, r['res'][m]['bt'])]
            fam.append(max(v) if v else 0)
        thr = float(np.percentile(fam, 95))
        hit = []; chance = []
        for r in by2[ctl]:
            tg = list(r['res'][m]['btop'].keys())
            for s, d in r['res'][m]['top'].items():
                if s in GOLD and ok(d, r['res'][m]['bt']) and any(x in tg for x in GOLD[s]):
                    t, st = modal(d)
                    chance.append(sum(x in GOLD[s] for x in tg) / len(tg))
                    if st >= thr:
                        hit.append((t in GOLD[s], s, t))
        o = {'fam_thr': thr, 'n_above': len(hit), 'prec_above': float(np.mean([h[0] for h in hit])) if hit else None,
             'chance_per_pair': float(np.mean(chance)), 'above_examples': Counter('%s>%s%s' % (h[1], h[2], '+' if h[0] else '-') for h in hit).most_common(8)}
        out['ctlfam|%s|%s' % (m, ctl)] = o
        print(m, ctl, o)
json.dump(out, open(os.path.join(X.DX, 'c3_summary.json'), 'w'), indent=1, default=str)
