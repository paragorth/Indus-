#!/usr/bin/env python3
"""LA-46 cycle-3 report: sides merged, 300 architectures incl. topic forgers. Family pass rates under three rules
(strict max-group, ensemble mean, weakest group) vs the forger worlds; casts = connected components of CO residues."""
import sys, json, collections
from scipy.stats import fisher_exact
from la46_common import *

TH = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
NAMES = ['LA', 'W1', 'W2', 'SH1', 'PL1', 'PL2', 'PB1', 'PB2', 'LB1', 'LB2']
NULL = ['W1', 'W2']
IDX = {'max': 4, 'mean': 6, 'min': 7}
C = {}
for n in NAMES:
    fn = os.path.join(CK, 'c3_%s.json' % n)
    if os.path.exists(fn):
        C[n] = json.load(open(fn))
print('docs:', {n: C[n]['n_docs'] for n in C})
for rule, ix in IDX.items():
    print('\nrule %s, p < %g: pass/tested per family' % (rule, TH))
    print('%-4s' % '', ' '.join('%10s' % n for n in C))
    for f in FAMS:
        row = []
        for n in C:
            t = sum(1 for r in C[n]['res'] if r[0][0] == f)
            p = sum(1 for r in C[n]['res'] if r[0][0] == f and r[ix] < TH)
            row.append('%4d/%-5d' % (p, t))
        print('%-4s' % f, ' '.join(row))
    # Fisher vs pooled worlds, per family, for LA / LB / PL
    nl = [n for n in NULL if n in C]
    for f in ('CO', 'ES', 'WL', 'TOT', 'SW'):
        wt = sum(1 for n in nl for r in C[n]['res'] if r[0][0] == f)
        wp = sum(1 for n in nl for r in C[n]['res'] if r[0][0] == f and r[ix] < TH)
        s = []
        for n in ('LA', 'PL1', 'PL2', 'LB1', 'LB2', 'SH1'):
            if n not in C:
                continue
            t = sum(1 for r in C[n]['res'] if r[0][0] == f)
            p = sum(1 for r in C[n]['res'] if r[0][0] == f and r[ix] < TH)
            if t and wt:
                P = fisher_exact([[p, t - p], [wp, wt - wp]], alternative='greater')[1]
                s.append('%s %.2f P=%.2g' % (n, p / t, P))
        print('   %s worlds %d/%d=%.2f | %s' % (f, wp, wt, wp / max(1, wt), '; '.join(s)))
# planted recall
for n in ('PL1', 'PL2', 'PB1', 'PB2'):
    if n in C:
        T = set(tuple(t) for t in C[n]['truth'])
        P = {tuple(r[0]): r for r in C[n]['res']}
        for rule, ix in IDX.items():
            print('%s planted found (%s): %d/%d' % (n, rule, sum(1 for t in T if t in P and P[t][ix] < TH), len(T)))
# casts
for n in ('LA', 'LB1', 'LB2', 'W1', 'W2', 'PL1'):
    if n not in C:
        continue
    G = collections.defaultdict(set)
    for r in C[n]['res']:
        if r[0][0] == 'CO' and r[6] < TH:
            a, b = r[0][1], r[0][2]
            G[a].add(b); G[b].add(a)
    seen = set(); comps = []
    for v in G:
        if v in seen:
            continue
        st = [v]; comp = set()
        while st:
            x = st.pop()
            if x in comp:
                continue
            comp.add(x); st.extend(G[x])
        seen |= comp; comps.append(sorted(comp))
    comps.sort(key=len, reverse=True)
    print('\n%s casts (CO residues, ensemble-mean rule): %d components, sizes %s' % (
        n, len(comps), [len(c) for c in comps]))
    docs = [dict(d, toks=[tuple(x) for x in d['toks']]) for d in json.load(open(os.path.join(CK, 'c3_%s_docs.json' % n)))]
    for c in comps[:12]:
        ds = [d['id'] for d in docs if sum(any(t[0] == 'W' and t[1] == w for t in d['toks']) for w in c) >= 2]
        print('   ', c, '| docs:', ds[:10])
if 'LA' in C:
    print('\nLA residues (mean rule p < %g), non-CO:' % TH)
    for r in sorted(C['LA']['res'], key=lambda r: r[6]):
        if r[6] < TH and r[0][0] != 'CO':
            print('   ', r[0], 'R', r[1], 'Emean %.2f Emax %.2f pmax %.1e' % (r[5], r[2], r[4]))
