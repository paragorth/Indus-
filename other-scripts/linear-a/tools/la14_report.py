#!/usr/bin/env python3
"""LA-14 cycle-1 report: AUC tables, survival of feature groups against nulls."""
import json, os, sys
from collections import defaultdict
import numpy as np
import la14_common as C

tag = sys.argv[1] if len(sys.argv) > 1 else 'c1'
R = [json.loads(l) for l in open(os.path.join(C.OUT, tag + '.jsonl'))]
A = defaultdict(list)
for r in R:
    g = r['g'][0] if len(r['g']) == 1 else 'ALL' + r['clf']
    if len(r['g']) == 1 and r['clf'] != 'LR': continue
    A[(r['c'], r['f'], r['role'], g)].append(r['auc'])

GS = C.ALLG + ['ALLLR', 'ALLRF']


def ref(c, f, g):
    if f in C.GEN_FORGERS: return A.get((c, f, 'null', g), [])
    return A.get((c, 'COPY0', 'real', g), [])


def surv(c, f, g):
    x = np.array(A.get((c, f, 'real', g), [])); n = np.array(ref(c, f, g))
    if len(x) < 3 or len(n) < 3: return None
    d = x.mean() - n.mean(); se = np.sqrt(x.var(ddof=1) / len(x) + n.var(ddof=1) / len(n)) + 1e-9
    return x.mean(), n.mean(), d, d / se


out = []
for c in ['LA', 'PLA', 'LB', 'FW_MK2', 'FW_FLAT', 'FW_NEUR']:
    out.append('\n=== corpus %s : mean CV AUC over seeds (real role); [null/COPY0 ref]; * = survives (t>3 and d>0.03)' % c)
    out.append('%-11s ' % 'forger' + ' '.join('%7s' % g[:7] for g in GS))
    for f in C.GEN_FORGERS + C.CopyEdit.TYPES:
        row = []
        for g in GS:
            s = surv(c, f, g)
            if s is None: row.append('%7s' % '-'); continue
            row.append('%6.3f%s' % (s[0], '*' if (s[3] > 3 and s[2] > 0.03) else ' '))
        out.append('%-11s ' % f + ' '.join(row))
        if f in C.GEN_FORGERS:
            out.append('%-11s ' % ('  null') + ' '.join('%6.3f ' % np.mean(A[(c, f, 'null', g)]) if A.get((c, f, 'null', g)) else '%7s' % '-' for g in GS))
print('\n'.join(out))
json.dump({'|'.join(k): v for k, v in A.items()}, open(os.path.join(C.OUT, tag + '_auc.json'), 'w'))
