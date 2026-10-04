#!/usr/bin/env python3
"""LA-14 cycle-2 report: random pairings (feature subsets x learners x seeds).
For each (corpus, forger) fit AUC ~ group dummies + learner dummies (OLS); the group coefficient is its marginal
contribution to catching the forger. Survival: coefficient on the real corpus minus coefficient on the reference
(FW_FLAT world for FLAT; COPY0 on the same corpus for copy forgers), bootstrap over tasks (seeds) for the SE."""
import json, os
from collections import defaultdict
import numpy as np
import la14_common as C

R = [json.loads(l) for l in open(os.path.join(C.OUT, 'c2.jsonl'))]
G = C.ALLG; L = C.CLFS
by = defaultdict(list)
for r in R: by[(r['c'], r['f'])].append(r)


def fit(rows):
    X = np.array([[1.0] + [g in r['g'] for g in G] + [r['clf'] == l for l in L[1:]] for r in rows], float)
    y = np.array([r['auc'] for r in rows])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[1:1 + len(G)], b


def boot(rows, B=300, seed=0):
    rng = np.random.default_rng(seed)
    seeds = sorted({r['s'] for r in rows}); bs = defaultdict(list)
    for r in rows: bs[r['s']].append(r)
    out = []
    for _ in range(B):
        pick = rng.choice(seeds, len(seeds))
        out.append(fit([r for s in pick for r in bs[s]])[0])
    return np.array(out)


out = ['C2. Random pairings: %d discriminators in %d tasks' % (len(R), len({(r['c'], r['f'], r['s']) for r in R}))]
out.append('\nMean AUC by learner and best single pairing, per corpus/forger')
for k in sorted(by):
    rows = by[k]
    m = {l: np.mean([r['auc'] for r in rows if r['clf'] == l]) for l in L if any(r['clf'] == l for r in rows)}
    out.append('  %-8s %-11s n=%3d mean %.3f max %.3f | %s' % (k[0], k[1], len(rows), np.mean([r['auc'] for r in rows]),
               max(r['auc'] for r in rows), ' '.join('%s %.3f' % x for x in m.items())))
out.append('\nGroup marginal contribution to AUC (OLS coefficient x1000), with survival vs reference (* = d > 3 SE and d > 0.02)')
out.append('%-20s ' % 'corpus/forger' + ' '.join('%7s' % g[:7] for g in G))
refs = {}
for (c, f), rows in sorted(by.items()):
    if c.startswith('FW_') and f != 'FLAT': continue
    b = fit(rows)[0]
    if f in C.CopyEdit.TYPES and f != 'COPY0': ref = (c, 'COPY0')
    elif f == 'FLAT' and not c.startswith('FW_'): ref = ('FW_FLAT', 'FLAT')
    else: ref = None
    cell = []
    if ref and ref in by:
        bb = boot(rows); br = boot(by[ref], seed=1); d = b - fit(by[ref])[0]; se = np.sqrt(bb.var(0) + br.var(0))
        for i in range(len(G)): cell.append('%+6.0f%s' % (1000 * d[i], '*' if d[i] > 3 * se[i] and d[i] > 0.02 else ' '))
        out.append('%-20s ' % ('%s/%s - ref' % (c, f)) + ' '.join(cell))
    else:
        out.append('%-20s ' % ('%s/%s raw' % (c, f)) + ' '.join('%+6.0f ' % (1000 * x) for x in b))
txt = '\n'.join(out); print(txt)
open(os.path.join(C.OUT, 'c2_report.txt'), 'w').write(txt + '\n')
