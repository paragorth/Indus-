"""v75: LOSO confusion of the survivors (which kind does a held-out real system land on?). Checks whether the kind the
Voynich lands on is an attractor that held-out systems of other kinds also fall into.
Usage: python3 v75_confusion.py TAG REP NMODELS"""
import os, sys, pickle, json
import numpy as np
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_cls as C, v75_lib as X

tag, rep, nm = sys.argv[1], sys.argv[2], int(sys.argv[3])
R = pickle.load(open(os.path.join(X.CK, 'run_%s.pkl' % tag), 'rb'))
os.environ['V75_FEATS'] = R.get('featfile', 'feats')
f, rows, F = C.load(rep)
RI = [i for i, r in enumerate(rows) if r['kind'] != '?']
base = np.array([rows[i]['base'] for i in RI]); kind = [rows[i]['kind'] for i in RI]; role = [rows[i]['role'] for i in RI]
conf = defaultdict(Counter); sysland = defaultdict(Counter)
rng = np.random.default_rng(0)
surv = list(R['surv']); sel = [surv[i] for i in rng.choice(len(surv), min(nm, len(surv)), replace=False)]
for mi in sel:
    m = R['models'][mi]
    for b in sorted(set(base)):
        tr = [j for j in range(len(RI)) if base[j] != b and (R['with_gen'] or kind[j] != 'GEN')]
        te = [j for j in range(len(RI)) if base[j] == b and role[j] == 'real']
        if not te: continue
        fit = C.Fitted(m, F[RI][tr], [kind[j] for j in tr], [base[j] for j in tr])
        pred, _ = fit.predict(F[RI][te])
        for p in pred: conf[kind[te[0]]][p] += 1; sysland[b][p] += 1
out = {k: {kk: round(v / sum(c.values()), 3) for kk, v in c.most_common()} for k, c in conf.items()}
into = Counter()
for k, c in conf.items():
    for kk, v in c.items():
        if kk != k: into[kk] += v / sum(c.values())
out['_attractor_mass'] = {k: round(v, 2) for k, v in into.most_common()}
out['_systems'] = {b: c.most_common(2) for b, c in sysland.items()}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(X.CK, 'confusion_%s_%s.json' % (tag, rep)), 'w'), indent=1)
