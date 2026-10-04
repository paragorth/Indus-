"""pe23 cycle 2.
(a) Ur III magnitude-matched separation: capacity types only (measured / allocated / estimated),
    values 30..6000 sila, each type resampled to the same log2-size histogram.
(b) PE absolute fingerprints per system (CNT, CAP, AMBCAP) vs the smooth null, both value sets,
    magnitude-matched to Ur III where possible; nearest Ur III type.
(c) Power with the composite indicator: planted recall vs group size.
(d) REAL PE scan: keys final / first / anysign / hdr / office / site / surface / size / nsign / sys,
    features ROUND LOW FIVE LEAD1 REP COMP, strata 'mag' and 'tab', BH over the whole family.
    Calibration: same family on 3 within-(system,size) value-shuffled corpora.
"""
import sys, json, math
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pe23_common import *
from pe23_scan import scan
import pe23_cycle1_lib as L

rng = np.random.default_rng(232)
R = {}
ur = ur3_entries()
TYPE = {'GRAIN': 'measured', 'RATION': 'allocated', 'ESTIMATE': 'estimated', 'TARGET': 'estimated',
        'LIVESTOCK': 'counted', 'PEOPLE': 'counted'}

# (a) magnitude matched capacity types
cap = [dict(e, lab=TYPE[e['cls']]) for e in ur if e['cls'] in ('GRAIN', 'RATION', 'ESTIMATE', 'TARGET')
       and 30 <= e['val'] <= 6000]
bins = defaultdict(lambda: defaultdict(list))
for e in cap:
    bins[int(math.log2(e['val']))][e['lab']].append(e)
matched = []
for b, d in bins.items():
    if len(d) < 3:
        continue
    m = min(len(v) for v in d.values())
    for lab_, v in d.items():
        idx = rng.choice(len(v), m, replace=False)
        matched += [v[i] for i in idx]
print('matched', Counter(e['lab'] for e in matched))
R['a'] = {'n': dict(Counter(e['lab'] for e in matched))}
accs, ctrl = [], []
for rep in range(3):
    acc, conf, cents, norm = L.separation(matched, ['measured', 'allocated', 'estimated'], rng, f'MATCHED rep{rep}', gs=40)
    accs.append(acc)
    if rep == 0:
        R['a']['conf'] = conf
    sh = [dict(e) for e in matched]
    labs = [e['lab'] for e in sh]
    rng.shuffle(labs)
    for e, l in zip(sh, labs):
        e['lab'] = l
    ctrl.append(L.separation(sh, ['measured', 'allocated', 'estimated'], rng, f'SHUF rep{rep}', gs=40)[0])
R['a']['acc'], R['a']['ctrl'] = accs, ctrl

# (b) PE absolute fingerprints
R['b'] = {}
urf = {}
for c in ('LIVESTOCK', 'PEOPLE', 'GRAIN', 'RATION', 'ESTIMATE', 'TARGET'):
    xs = [e for e in ur if e['cls'] == c]
    if len(xs) > 3000:
        xs = [xs[i] for i in rng.choice(len(xs), 3000, replace=False)]
    f, eff, z = effects([x['val'] for x in xs], [x['tab'] for x in xs], ur_den(c), rng, reps=100)
    urf[c] = eff
for vset in ('B', 'ALT'):
    pe = pe_entries(vset)
    for s in ('CNT', 'CAP', 'AMBCAP'):
        xs = [e for e in pe if e['sys'] == s]
        f, eff, z = effects([x['val'] for x in xs], [x['tab'] for x in xs], pe_den(s, vset), rng, reps=200)
        dist = {c: float(np.sqrt(sum((eff[k] - urf[c][k]) ** 2 for k in ('ROUND', 'LOW', 'FIVE', 'REP')
                                     if not np.isnan(eff[k])))) for c in urf}
        near = min(dist, key=dist.get)
        R['b'][f'{vset}:{s}'] = {'n': len(xs), 'real': f, 'eff': eff, 'z': z, 'dist': dist, 'nearest': near}
        print(vset, s, len(xs), ' '.join(f'{k}={f[k]:.3f}({eff[k]:+.3f},z{z[k]:+.1f})' for k in FEATS), 'nearest', near,
              {k: round(v, 2) for k, v in dist.items()})
print('UR3 effects', {c: {k: round(v, 3) for k, v in e.items()} for c, e in urf.items()})
R['b']['ur3_eff'] = urf

# (c) power with composite
pe = pe_entries('B')
cnt = [e for e in pe if e['sys'] in ('CNT', 'CAP')]
fc = Counter(e['final'] for e in cnt if e['val'] >= 10)
cands = sorted(s for s, n in fc.items() if n >= 15 and s)
rec = defaultdict(lambda: [0, 0])
fps = []
for p in (0.5, 0.25):
    for rep in range(8):
        signs = set(rng.choice(cands, 6, replace=False).tolist())
        ents = L.plant(cnt, signs, p, rng)
        res = scan(ents, keys=('final',), feats=('ROUND', 'COMP', 'LOW'), strata='mag')
        hits = {r['group'] for r in res if r['q'] < 0.05 and r['z'] > 0}
        fps.append(len(hits - signs))
        for s in signs:
            nb = fc[s]
            bucket = '15-29' if nb < 30 else '30-59' if nb < 60 else '60+'
            rec[(p, bucket)][0] += s in hits
            rec[(p, bucket)][1] += 1
R['c'] = {f'{k[0]}|{k[1]}': v for k, v in rec.items()}
R['c']['fp_per_run'] = fps
print('plant recall by size (entries >= 10):', dict(R['c']))

# (d) REAL scan
KEYS = ('final', 'first', 'anysign', 'hdr', 'office', 'site', 'surface', 'size', 'nsign', 'sys')
FE = ('ROUND', 'LOW', 'FIVE', 'LEAD1', 'REP', 'COMP')
real = [e for e in pe if e['sys'] in ('CNT', 'CAP', 'AMBCAP')]
R['d'] = {}
for st in ('mag', 'tab'):
    res = scan(real, keys=KEYS, feats=FE, strata=st, by_system=(st == 'mag'))
    sig = sorted([r for r in res if r['q'] < 0.05], key=lambda r: -abs(r['z']))
    R['d'][st] = {'ntests': len(res), 'sig': sig}
    print(f'REAL strata={st}: {len(sig)} of {len(res)} tests q<0.05')
    for r in sig[:40]:
        print('  ', r['key'], r['group'], r['sys'], r['feat'], 'n', r['n'], 'tabs', r['tabs'],
              f"obs {r['obs']:.3f} exp {r['exp']:.3f} z {r['z']:+.1f} q {r['q']:.3g}")
    # calibration
    cal = []
    for rep in range(3):
        ents = L.shuffle_values(real, rng)
        r2 = scan(ents, keys=KEYS, feats=FE, strata=st, by_system=(st == 'mag'))
        cal.append(sum(r['q'] < 0.05 for r in r2))
    R['d'][st]['calib_flags'] = cal
    print('  calibration (value-shuffled) flags:', cal)
json.dump(R, open(os.path.join(CK, 'c2.json'), 'w'), indent=1, default=float)
