"""pe23 cycle 1: positive controls.
(a) Ur III lines of KNOWN number type: fingerprint effects vs the smooth same-notation null.
(b) Separation: tablet-disjoint pseudo-groups (60 entries), nearest-centroid on effect vectors,
    train half A, test half B; control = class labels shuffled across entries before grouping.
(c) Planted estimated class in PE: 6 random final signs, p of their entries (v >= 10) rounded
    to the leading denomination; group scan (cluster-robust stratified z, BH) must find them;
    calibration = PE values permuted within (system, magnitude) strata.
"""
import sys, json, math, random
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pe23_common import *
from pe23_scan import scan

rng = np.random.default_rng(23)
R = {}
ur = ur3_entries()
CL = ['LIVESTOCK', 'PEOPLE', 'GRAIN', 'RATION', 'ESTIMATE', 'TARGET']
by = defaultdict(list)
for e in ur:
    by[e['cls']].append(e)

# (a) whole-class fingerprints
R['a'] = {}
print('(a) Ur III class fingerprints: real, null mean, z')
for c in CL:
    xs = by[c]
    if len(xs) > 4000:
        xs = [xs[i] for i in rng.choice(len(xs), 4000, replace=False)]
    v = [x['val'] for x in xs]
    t = [x['tab'] for x in xs]
    f, eff, z = effects(v, t, ur_den(c), rng, reps=200)
    R['a'][c] = {'n': len(v), 'real': f, 'eff': eff, 'z': z}
    print(c, len(v), ' '.join(f'{k}={f[k]:.3f}({eff[k]:+.3f},z{z[k]:+.1f})' for k in FEATS))

# (b) separation with pseudo-groups
GS = 60


def groups_for(entries, tabset, ngroups, rng):
    pool = [e for e in entries if e['tab'] in tabset]
    tabs = sorted({e['tab'] for e in pool})
    bt = defaultdict(list)
    for e in pool:
        bt[e['tab']].append(e)
    out = []
    for _ in range(ngroups):
        order = rng.permutation(len(tabs))
        g = []
        for i in order:
            g += bt[tabs[i]][:10]  # at most 10 per tablet
            if len(g) >= GS:
                break
        if len(g) >= GS // 2:
            out.append(g[:GS])
    return out


def vec(g, den, rng):
    f, eff, z = effects([e['val'] for e in g], [e['tab'] for e in g], den, rng, reps=40)
    return [0.0 if np.isnan(eff[k]) else eff[k] for k in FEATS]


def separation(labelled, classes, rng, tag):
    alltabs = sorted({e['tab'] for e in labelled})
    rng.shuffle(alltabs)
    A, B = set(alltabs[::2]), set(alltabs[1::2])
    X = {h: [] for h in 'AB'}
    Y = {h: [] for h in 'AB'}
    for c in classes:
        ents = [e for e in labelled if e['lab'] == c]
        for h, ts in (('A', A), ('B', B)):
            for g in groups_for(ents, ts, 30, rng):
                X[h].append(vec(g, ur_den(g[0]['cls']), rng))
                Y[h].append(c)
    XA, XB = np.array(X['A']), np.array(X['B'])
    mu, sd = XA.mean(0), XA.std(0) + 1e-9
    XA, XB = (XA - mu) / sd, (XB - mu) / sd
    cents = {c: XA[np.array(Y['A']) == c].mean(0) for c in classes if (np.array(Y['A']) == c).any()}
    pred = [min(cents, key=lambda c: ((x - cents[c]) ** 2).sum()) for x in XB]
    acc = float(np.mean([p == y for p, y in zip(pred, Y['B'])]))
    conf = Counter(zip(Y['B'], pred))
    print(tag, 'acc', round(acc, 3), 'chance', round(1 / len(classes), 3), 'nB', len(pred))
    return acc, {f'{a}->{b}': n for (a, b), n in conf.items()}, {c: list(map(float, cents[c])) for c in cents}, (mu.tolist(), sd.tolist())


TYPE = {'LIVESTOCK': 'counted', 'PEOPLE': 'counted', 'GRAIN': 'measured', 'RATION': 'allocated',
        'ESTIMATE': 'estimated', 'TARGET': 'estimated'}
lab = []
for c in CL:
    xs = by[c]
    if len(xs) > 8000:
        xs = [xs[i] for i in rng.choice(len(xs), 8000, replace=False)]
    for e in xs:
        lab.append(dict(e, lab=TYPE[c]))
types = ['counted', 'measured', 'allocated', 'estimated']
R['b'] = {}
accs, ctrl = [], []
for rep in range(3):
    acc, conf, cents, norm = separation(lab, types, rng, f'REAL rep{rep}')
    accs.append(acc)
    if rep == 0:
        R['b']['confusion'] = conf
        R['b']['centroids'] = cents
        R['b']['norm'] = norm
    sh = [dict(e) for e in lab]
    labs = [e['lab'] for e in sh]
    rng.shuffle(labs)
    for e, l in zip(sh, labs):
        e['lab'] = l
    a2, _, _, _ = separation(sh, types, rng, f'SHUF rep{rep}')
    ctrl.append(a2)
R['b']['acc'] = accs
R['b']['ctrl'] = ctrl
print('separation acc', accs, 'shuffled', ctrl)

# (c) planted estimated class in PE
pe = pe_entries('B')
cnt = [e for e in pe if e['sys'] in ('CNT', 'CAP')]
fc = Counter(e['final'] for e in cnt if e['val'] >= 10)
cands = [s for s, n in fc.items() if n >= 15 and s]
R['c'] = []


def plant(ents, signs, p, rng):
    out = []
    for e in ents:
        e = dict(e)
        if e['final'] in signs and e['val'] >= 10 and rng.random() < p:
            den = pe_den(e['sys'])
            hi = max(d for d in den if d <= e['val'])
            e['val'] = int(max(1, round(e['val'] / hi)) * hi)
        out.append(e)
    return out


for p in (0.5, 0.25):
    for rep in range(5):
        signs = set(rng.choice(sorted(cands), 6, replace=False).tolist())
        ents = plant(cnt, signs, p, rng)
        res = scan(ents, keys=('final',), feats=('ROUND',), strata='mag')
        hits = {r['group'] for r in res if r['q'] < 0.05 and r['z'] > 0}
        tp = len(hits & signs)
        fp = len(hits - signs)
        R['c'].append({'p': p, 'planted': sorted(signs), 'tp': tp, 'fp': fp, 'fp_signs': sorted(hits - signs)})
        print(f'plant p={p} rep{rep}: recovered {tp}/6, other flagged {fp}', sorted(hits - signs)[:8])

# calibration: values permuted within (system, magnitude bin)
for rep in range(5):
    ents = [dict(e) for e in cnt]
    st = defaultdict(list)
    for i, e in enumerate(ents):
        st[(e['sys'], int(math.log2(e['val'])))].append(i)
    for idx in st.values():
        vals = [ents[i]['val'] for i in idx]
        rng.shuffle(vals)
        for i, v in zip(idx, vals):
            ents[i]['val'] = v
    res = scan(ents, keys=('final',), feats=('ROUND',), strata='mag')
    hits = [r['group'] for r in res if r['q'] < 0.05]
    R['c'].append({'calib_shuffle': rep, 'flagged': len(hits), 'ntests': len(res)})
    print('calibration shuffle', rep, 'flagged', len(hits), 'of', len(res))
# real, same scan (for reference)
res = scan(cnt, keys=('final',), feats=('ROUND',), strata='mag')
R['c_real'] = [r for r in res if r['q'] < 0.05]
print('REAL final-sign ROUND flags:', [(r['group'], round(r['z'], 1)) for r in R['c_real']])
json.dump(R, open(os.path.join(CK, 'c1.json'), 'w'), indent=1, default=float)
