#!/usr/bin/env python3
"""LA-39 cycle 4b: power of the within-document profile test at Linear A size.
Linear B known-meaning modifiers are cut down to the size of LA's best-attested type (OLE+KI:
17 tokens in 13 mixed documents): random mixed documents are drawn until the type has >= 17 tokens;
within-document permutation (5,000); detection = any token feature at P < 0.05/83 (LA's Holm floor).
"""
import sys, os, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la39_common as L

rng = np.random.default_rng(3942)
TOK = ['logq', 'noq', 'relpos', 'afterword', 'relq']
A = L.to_arrays(L.lb_rows())
F = [A['feats'].index(f) for f in TOK]
out = {}
for b, m in [('OVIS', 'm'), ('OVIS', 'f'), ('CAP', 'f'), ('CAP', 'm'), ('BOS', 'm'), ('SUS', 'f'), ('TELA', 'TE'), ('TELA', 'PU')]:
    kb = np.where(A['base'] == b)[0]
    strata = defaultdict(list)
    for i in kb: strata[A['doc'][i]].append(i)
    strata = [np.array(v) for v in strata.values()
              if any(A['mod'][i] == m for i in v) and any(A['mod'][i] != m for i in v)]
    det = 0; reps = 20
    for r in range(reps):
        order = rng.permutation(len(strata)); pick = []; n = 0
        for k in order:
            pick.append(strata[k]); n += sum(A['mod'][i] == m for i in strata[k])
            if n >= 17: break
        idx = np.concatenate(pick); lab = np.array([A['mod'][i] == m for i in idx])
        X = A['X'][np.ix_(idx, F)]
        o = X[lab].mean(0) - X[~lab].mean(0)
        offs = np.cumsum([0] + [len(s) for s in pick]); labs = [lab[offs[k]:offs[k + 1]] for k in range(len(pick))]
        ge = np.zeros(len(F)); le = np.zeros(len(F)); NP = 5000
        for _ in range(NP):
            l = np.concatenate([rng.permutation(x) for x in labs])
            v = X[l].mean(0) - X[~l].mean(0); ge += v >= o - 1e-12; le += v <= o + 1e-12
        p = 2 * np.minimum(ge, le) / NP
        det += bool((p < 0.05 / 83).any())
    out['%s+%s' % (b, m)] = (det, reps, len(strata))
    print(b, m, 'detected', det, '/', reps, 'mixed docs available', len(strata), flush=True)
json.dump(out, open(os.path.join(L.CK, 'c4b.json'), 'w'))
