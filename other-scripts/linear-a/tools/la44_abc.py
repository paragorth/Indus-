#!/usr/bin/env python3
"""LA-44 ABC read-out: python3 la44_abc.py BANKGLOB N_TARGET TAG [panel=all|vf] [njobs]
Random-forest projection of the panel (OOB class votes for morphology and syllable class, OOB regression predictions
for continuous history parameters), then rejection ABC in that projected space (nearest K sims) -> posterior.
Targets at n=583: LA, 5 LB random type subsamples, 5 globally shuffled LA, 3 position-class shuffled LA.
Targets at n=107: Cypriot (Idalion), 5 LA and 5 LB subsamples at 107, 3 shuffled Cypriot."""
import sys, os, json, glob, random, math, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la44_common as A
import la5_common as C5
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

bank, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else 'all'
nj = int(sys.argv[5]) if len(sys.argv) > 5 else 2
excl = set(int(x) for x in sys.argv[6].split(',')) if len(sys.argv) > 6 and sys.argv[6] else set()   # morph classes left out
# modes: all = raw panel; vf = value-free raw; delta = panel minus the panel of a global sign shuffle of the same
# list (only statistics that a shuffle can change); both = raw + delta; vfdelta = value-free delta.
INV = {'len2', 'len3', 'len4', 'len5p', 'H', 'zslope', 'nsign', 'mlen', 'sdlen'} | {k for k in A.panel_names() if k[:2] in ('v_', 's_')}
DEL = [k for k in A.panel_names() if k not in INV]
names = {'all': A.panel_names(), 'vf': A.VALUE_FREE, 'delta': ['d_' + k for k in DEL],
         'both': A.panel_names() + ['d_' + k for k in DEL],
         'vfdelta': ['d_' + k for k in A.VALUE_FREE if k not in INV],
         'robust': A.panel_names() + ['d_' + k for k in DEL]}[mode]
def feat(f, fs):
    g = dict(f)
    for k in DEL: g['d_' + k] = f[k] - fs[k]
    return [g[k] for k in names]
rows = []
for fn in sorted(glob.glob(bank)):
    for l in open(fn):
        r = json.loads(l)
        if r['f'] is not None and r['P']['morph'] not in excl: rows.append(r)
X = np.array([feat(r['f'], r['fs']) for r in rows], float)
X = np.nan_to_num(X)
P = [r['P'] for r in rows]
ym = np.array([p['morph'] for p in P]); ys = np.array([p['syl'] for p in P])
cont = ['nc0', 'nv0', 'rootlen', 'pborrow', 'nsc', 'pcoda', 'pcl', 'plate']
Yc = np.array([[p[k] for k in cont] for p in P], float)
print('sims', len(rows), 'morph counts', np.bincount(ym, minlength=6).tolist(), flush=True)

la, lb, cy = A.targets()
T = collections.OrderedDict()
if n > 300:
    T['LA'] = la
    for i in range(5): T['LB_sub%d' % i] = random.Random(100 + i).sample(lb, n)
    NS = int(os.environ.get('LA44_NSHUF', '0'))
    for i in range(max(5, NS)): T['LA_shufG%d' % i] = C5.shuffle_global(la, random.Random(200 + i))
    for i in range(max(3, NS // 2)): T['LA_shufP%d' % i] = C5.shuffle_pos(la, random.Random(300 + i))
    for i in range(NS // 4): T['LB_shufG%d' % i] = C5.shuffle_global(random.Random(100 + i % 5).sample(lb, n), random.Random(700 + i))
else:
    T['CYP'] = cy
    for i in range(5): T['LA_sub%d' % i] = random.Random(400 + i).sample(la, n)
    for i in range(5): T['LB_sub%d' % i] = random.Random(500 + i).sample(lb, n)
    for i in range(3): T['CYP_shufG%d' % i] = C5.shuffle_global(cy, random.Random(600 + i))
TX = np.nan_to_num(np.array([feat(A.panel(t), A.panel(C5.shuffle_global(t, random.Random(999 + j))))
                             for j, t in enumerate(T.values())], float))

if mode == 'robust':
    # drop every statistic on which the KNOWN control (mean of the LB subsamples) lies outside the central 95 % of the
    # simulations; the rule looks only at LB, never at LA or the shuffles.
    lbrows = [j for j, tn in enumerate(T) if tn.startswith('LB_sub')]
    lbm = TX[lbrows].mean(0)
    lo = np.percentile(X, 2.5, 0); hi = np.percentile(X, 97.5, 0)
    keep = [i for i in range(X.shape[1]) if lo[i] <= lbm[i] <= hi[i] and hi[i] > lo[i]]
    dropped = [names[i] for i in range(X.shape[1]) if i not in keep]
    print('robust: dropped', len(dropped), dropped, flush=True)
    X = X[:, keep]; TX = TX[:, keep]; names = [names[i] for i in keep]

# where does each target sit relative to the simulated cloud? (Mahalanobis-free: share of sims farther, robust z)
med = np.median(X, 0); mad = np.median(np.abs(X - med), 0) * 1.4826 + 1e-9
dz = np.sqrt((((X - med) / mad) ** 2).mean(1))
out = {'names': names, 'nsims': len(rows), 'targets': {}, 'excl': sorted(excl)}

def proj_class(y, ncls):
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, oob_score=True, n_jobs=nj, random_state=1,
                                max_features='sqrt')
    rf.fit(X, y)
    oob = np.nan_to_num(rf.oob_decision_function_)
    return rf, oob, rf.predict_proba(TX)

K = max(200, len(rows) // 100)
rng = np.random.default_rng(0)
res = {}
for lab, y, ncls, cn in (('morph', ym, 6, A.MORPH), ('syl', ys, 3, A.SYL)):
    rf, oob, tp = proj_class(y, ncls)
    acc = (rf.classes_[oob.argmax(1)] == y).mean()
    conf = np.zeros((ncls, ncls), int)
    for a, b in zip(y, rf.classes_[oob.argmax(1)]): conf[a, b] += 1
    # calibration: 400 held-out sims as pseudo-targets, rejection posterior in OOB-vote space
    cal = []
    for i in rng.choice(len(y), 400, replace=False):
        d = ((oob - oob[i]) ** 2).sum(1); d[i] = np.inf
        acc_i = np.argsort(d)[:K]; post = np.bincount(y[acc_i], minlength=ncls) / K
        cal.append((post.max(), int(post.argmax() == y[i])))  # labels are original class ids
    cal = np.array(cal)
    bins = [(lo, hi) for lo, hi in ((0, .3), (.3, .45), (.45, .6), (.6, .8), (.8, 1.01))]
    calt = [(lo, hi, int(((cal[:, 0] >= lo) & (cal[:, 0] < hi)).sum()),
             float(cal[(cal[:, 0] >= lo) & (cal[:, 0] < hi), 1].mean()) if ((cal[:, 0] >= lo) & (cal[:, 0] < hi)).any() else None)
            for lo, hi in bins]
    out[lab] = {'oob_acc': float(acc), 'confusion': conf.tolist(), 'calibration': calt,
                'importance': sorted(zip(names, rf.feature_importances_.round(4).tolist()), key=lambda z: -z[1])[:12]}
    print(lab, 'OOB acc', round(acc, 3), 'calibration', calt, flush=True)
    for j, tn in enumerate(T):
        d = ((oob - tp[j]) ** 2).sum(1); acc_i = np.argsort(d)[:K]
        post = np.bincount(y[acc_i], minlength=ncls) / K
        H = -sum(p * math.log(p) for p in post if p > 0) / math.log(ncls - (len(excl) if lab == 'morph' else 0))
        out['targets'].setdefault(tn, {})[lab] = {'votes': dict(zip([cn[c] for c in rf.classes_], tp[j].round(3).tolist())),
                                                  'post': dict(zip(cn, post.round(3).tolist())), 'Hnorm': round(H, 3),
                                                  'dist_k': float(np.sqrt(d[acc_i[-1]]))}
for j, tn in enumerate(T):
    tz = np.sqrt((((TX[j] - med) / mad) ** 2).mean())
    out['targets'][tn]['cloud_pct'] = float((dz >= tz).mean())   # share of sims at least as far from the sim median
rfr = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, max_samples=0.2, max_features=0.33, oob_score=True, n_jobs=nj, random_state=2)
rfr.fit(X, Yc)
oobr = rfr.oob_prediction_; tpr = rfr.predict(TX)
r2 = [float(np.corrcoef(oobr[:, k], Yc[:, k])[0, 1] ** 2) for k in range(len(cont))]
out['cont_r2'] = dict(zip(cont, np.round(r2, 3).tolist()))
sd = oobr.std(0) + 1e-9
for j, tn in enumerate(T):
    d = (((oobr - tpr[j]) / sd) ** 2).sum(1); acc_i = np.argsort(d)[:K]
    acc = Yc[acc_i]
    out['targets'][tn]['cont'] = {k: [round(float(np.percentile(acc[:, i], q)), 3) for q in (10, 50, 90)]
                                  for i, k in enumerate(cont)}
    out['targets'][tn]['cont_prior_median'] = {k: round(float(np.median(Yc[:, i])), 3) for i, k in enumerate(cont)}
    # history menu: how often each sound change appears in accepted vs all sims
    hc = collections.Counter(h for i in acc_i for h in set(P[i]['hist'])); ha = collections.Counter(h for p in P for h in set(p['hist']))
    out['targets'][tn]['hist_lift'] = {h: round((hc[h] / K) / (ha[h] / len(P)), 2) for h in sorted(ha)}
print('cont R2', out['cont_r2'])
for tn, v in out['targets'].items():
    print(tn, 'morph', v['morph']['post'], 'H', v['morph']['Hnorm'], '| syl', v['syl']['post'], '| cloud', round(v['cloud_pct'], 3))
json.dump(out, open(os.path.join(A.CK, 'abc_%s.json' % tag), 'w'), indent=1)
