#!/usr/bin/env python3
"""la64 cycle 3: (a) is the Linear B subunit recovery physics or internal consistency (nonsense
rulers)?  (b) the Hagia Triada 'men, then small commodity amounts' documents read as per-head issues:
implied unit sizes per period, against a numbers-shuffled null and against one outside physical object
(ZA Zb 3 pithos, VIN 32; measured unit ~12.5 l = ~400 l content: Notti 2025; Younger max ~1,000 l).
(c) frozen predictions for inscribed vessels."""
import sys, json, math
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from la64_lib import *

OUT = os.path.join(CK, 'c3.json')
res = {}
G = np.linspace(*LOGU, 241)


def save():
    json.dump(res, open(OUT, 'w'), indent=1)


# (a) LB dT under nonsense rulers
LB = load_lb()
A = build(LB, lb_class)
i = A['keys'].index('GRA')
m = A['k'][A['pe']] == i
B = dict(A, pe=A['pe'][m], pn=A['pn'][m], pk=A['pk'][m])
fj = A['frkeys'].index('dT')
TH = np.exp(np.linspace(math.log(0.01), math.log(0.95), 40))


def best_t(ruler):
    out = []
    for t in TH:
        th = np.array([LB_TRUE_FR.get(f, 0.1) for f in A['frkeys']]); th[fj] = t
        L = np.zeros((len(G), len(A['keys']))); L[:, i] = G
        out.append(float(score(B, L, np.tile(th, (len(G), 1)), ruler, use=('E1',)).max()))
    out = np.array(out)
    return float(TH[np.argmax(out)]), float(out.max() - np.median(out))


phys = best_t(None)
rul = [best_t(nonsense_ruler(np.random.default_rng(2000 + r))) for r in range(30)]
res['lb_dT'] = dict(phys=phys, rulers=rul, frac_rulers_within_1p5=float(np.mean([0.1 / 1.5 <= r[0] <= 0.15 for r in rul])))
save()
print('a', res['lb_dT']['phys'], res['lb_dT']['frac_rulers_within_1p5'], flush=True)

# (b) HT 'men then commodities' documents: person total (KU-RO of the VIR block) and the commodities after it
LA = load_la()
clu = []
for d in LA:
    tp = [p for p in d['P'] if p['tot']]
    if not tp:
        continue
    for e in d['E']:
        if not e['tot']:
            clu.append(dict(doc=d['id'], c=e['c'], n=e['n'], fr=e['fr'], persons=tp[0]['n']))
res['cluster'] = clu
FRV = 0.5   # nuisance: every fraction sign set to 0.5 and to 0.1-0.9 below


def implied(c, q, persons, per):
    m, s = DAY[la_class(c)]
    return m * {'d': 1, 'm': 30, 'y': 360}[per] * persons / q


rows = defaultdict(list)
for x in clu:
    for fv in (0.1, 0.5, 0.9):
        q = x['n'] + fv * len(x['fr'])
        if q > 0:
            rows[(x['c'], fv)].append(implied(x['c'], q, x['persons'], 'd'))
res['implied_day'] = {'%s|%g' % k: [round(v, 2) for v in vs] for k, vs in rows.items()}
# VIN against ZA Zb 3 (12.5 l per unit if VIN 32 = the pithos content of ~400 l; <= 31 l if <= 1,000 l)
vin = [x for x in clu if x['c'] == 'VIN']
obs = [implied('VIN', x['n'] + 0.5 * len(x['fr']), x['persons'], 'd') for x in vin]
gm = float(np.exp(np.mean(np.log(obs))))
# null: VIN quantities and person totals drawn from all LA VIN entries and all LA person totals
allv = [e['n'] + 0.5 * len(e['fr']) for d in LA for e in d['E'] if e['c'] == 'VIN' and not e['tot'] and e['n'] + len(e['fr']) > 0]
allp = [p['n'] for d in LA for p in d['P'] if p['tot']] + [sum(p['n'] for p in d['P'] if not p['tot']) for d in LA if d['P']]
allp = [p for p in allp if p > 0]
rng = np.random.default_rng(64)
null = []
for _ in range(20000):
    o = [implied('VIN', rng.choice(allv), rng.choice(allp), 'd') for _ in range(len(obs))]
    null.append(float(np.exp(np.mean(np.log(o)))))
null = np.array(null)
res['vin_vs_zazb3'] = dict(n=len(obs), implied=[round(v, 2) for v in obs], geomean=gm,
                           in_6_25=bool(6.25 <= gm <= 25), p_null_in_6_25=float(((null >= 6.25) & (null <= 25)).mean()),
                           null_q=[float(np.quantile(null, q)) for q in (0.05, 0.5, 0.95)],
                           month_unit=gm * 30, year_unit=gm * 360,
                           pithos_pred_day=[32 * min(obs), 32 * max(obs)], pithos_pred_month=[32 * 30 * min(obs), 32 * 30 * max(obs)])
# leave-one-document-out
res['vin_lodo'] = {x['doc']: round(float(np.exp(np.mean(np.log([o for o, y in zip(obs, vin) if y['doc'] != x['doc']])))), 2) for x in vin}
# same idea for the other commodities after the men (day reading), with the same shuffle null
oth = {}
for c in ('CYP', 'NI', 'OLE', 'GRA'):
    xs = [x for x in clu if x['c'] == c and x['n'] + len(x['fr']) > 0]
    if len(xs) < 2:
        continue
    ob = [implied(c, x['n'] + 0.5 * len(x['fr']), x['persons'], 'd') for x in xs]
    av = [e['n'] + 0.5 * len(e['fr']) for d in LA for e in d['E'] if e['c'] == c and not e['tot'] and e['n'] + len(e['fr']) > 0]
    nl = np.array([np.std(np.log([implied(c, rng.choice(av), rng.choice(allp), 'd') for _ in xs])) for _ in range(5000)])
    oth[c] = dict(n=len(ob), geomean=float(np.exp(np.mean(np.log(ob)))), logsd=float(np.std(np.log(ob))),
                  p_null_logsd_le=float((nl <= np.std(np.log(ob))).mean()))
res['other_after_men'] = oth
save()
print(json.dumps({k: res[k] for k in ('vin_vs_zazb3', 'vin_lodo', 'other_after_men')}, indent=0)[:3000])
