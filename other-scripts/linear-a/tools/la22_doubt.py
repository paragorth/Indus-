#!/usr/bin/env python3
"""la22: do frozen restorations flag the readings SigLA later changed?
Score = frozen out-of-fold P(lineara's own reading | context) at each sign position (int mask).
AUC of (1 - score) for positions SigLA reads differently vs positions SigLA reads identically.
Null: 10,000 random draws of the same number of identical positions (permutation of labels).
Shuffled-tablet model: P(read) taken from its frozen top-10 (0 if absent)."""
import json, os, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la22_data
from la22_sigla_test import nw, norm_code, norm_name
OUT = os.path.join(la22_data.DATA, 'la22')
txt = open(os.path.join(OUT, 'la22_predictions.json')).read()
assert hashlib.sha256(txt.encode()).hexdigest() == open(os.path.join(OUT, 'la22_predictions.sha256')).read().split()[0]
F = json.loads(txt)
S = json.load(open(os.path.join(la22_data.CK, 'sigla.json')))
D = la22_data.load(); sn = {norm_name(k): k for k in S}
pos = {(p['doc'], p['tok'], p['j']): p for p in F['pos'] if p['mode'] == 'int'}
posS = {(p['doc'], p['tok'], p['j']): p for p in F['pos_shuf'] if p['mode'] == 'int'}
vocab = {c for d in D for t in d['toks'] if t['t'] == 'w' for c in t['c']}
from collections import Counter
fq = Counter(c for d in D for t in d['toks'] if t['t'] == 'w' for c in t['c'])
lab = []; sc = []; scS = []; scF = []
for d in D:
    k = sn.get(norm_name(d['id']))
    if not k: continue
    a = [(c, ti, j) for ti, t in enumerate(d['toks']) if t['t'] == 'w' for j, c in enumerate(t['c'])]
    b = [norm_code(o['code']) for o in S[k]['occ'] if o['role'] == 'syllabogram']
    if not a or not b: continue
    for i, j in nw([x[0] for x in a], b):
        if i is None or j is None: continue
        key = (d['id'], a[i][1], a[i][2])
        if key not in pos: continue
        if a[i][0] != b[j] and b[j] not in vocab: continue      # SigLA reads a non-syllabic / unknown sign: alignment noise
        lab.append(a[i][0] != b[j]); sc.append(pos[key]['p_read'])
        ps = dict((c, p) for c, p in posS.get(key, {'top10': []})['top10'])
        scS.append(ps.get(a[i][0], 0.0)); scF.append(fq[a[i][0]])
lab = np.array(lab); sc = np.array(sc); scS = np.array(scS); scF = np.array(scF, float)
# residual: model score divided by the frequency-only probability (does context add anything beyond sign frequency?)
scR = sc / (scF / scF.sum() * 0 + scF / sum(fq.values()))
def auc(s, l):
    from scipy.stats import rankdata
    r = rankdata(-s); n1 = l.sum(); n0 = (~l).sum()
    return (r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rng = np.random.default_rng(5)
res = {'n_changed': int(lab.sum()), 'n_same': int((~lab).sum())}
for nm, s in (('model', sc), ('shuffled_tablet_model', scS), ('frequency_only', scF), ('model_over_frequency', scR)):
    a0 = auc(s, lab); null = [auc(s, rng.permutation(lab)) for _ in range(10000)]
    res[nm] = {'auc': round(float(a0), 3), 'null_mean': round(float(np.mean(null)), 3), 'p': round(float((np.array(null) >= a0).mean()), 4),
               'median_p_read_changed': round(float(np.median(s[lab])), 4), 'median_p_read_same': round(float(np.median(s[~lab])), 4)}
json.dump(res, open(os.path.join(OUT, 'la22_doubt.json'), 'w'), indent=1)
print(json.dumps(res))
