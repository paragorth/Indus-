#!/usr/bin/env python3
"""PE-79 cycle 1 read-out: leave-one-civilisation-out transfer per PE-grammar architecture.

For held-out civ h: rank the 100,000 grafts by mean z on the other six civilisations (real labels; NULL j:
the six with labels permuted, rep j), keep the top 500, report their mean percentile on h (real labels).
Trusted architecture: real transfer above all 3 nulls on EVERY held-out civ (khipu included) AND the planted
graft recovered on PLANT (percentile >= 0.999).
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe79_common as C
from pe79_c1 import ARCHS, NH, NPERM

TOP = 500


def pct_of(zcol, sel):
    order = np.argsort(np.argsort(zcol[:NH]))
    return (order[sel] / (NH - 1)).mean()


def loco(Z, names, rep=0):
    ix = {n: i for i, n in enumerate(names)}
    out = {}
    for h in C.KNOWN:
        tr = [ix['%s:%d' % (k, rep)] for k in C.KNOWN if k != h]
        ztr = Z[tr, :NH].mean(0)
        sel = np.argsort(-ztr)[:TOP]
        out[h] = pct_of(Z[ix['%s:0' % h]], sel)
    return out


def main(half='A'):
    rows = []
    for ai in range(len(ARCHS)):
        fn = os.path.join(C.CK, 'c1_%s_%02d.npz' % (half, ai))
        if not os.path.exists(fn):
            continue
        d = np.load(fn)
        Z = d['Z'].astype(np.float32); names = list(d['names'])
        real = loco(Z, names, 0)
        nulls = [loco(Z, names, j) for j in range(1, NPERM + 1)]
        pz = Z[names.index('PLANT:0')]
        ppl = (pz[:NH] < pz[-1]).mean()
        win = {h: real[h] > max(n[h] for n in nulls) for h in C.KNOWN}
        rows.append(dict(ai=ai, arch=ARCHS[ai], real=real, nullmax={h: max(n[h] for n in nulls) for h in C.KNOWN},
                         plant=ppl, nwin=sum(win.values()), trusted=all(win.values()) and ppl >= 0.999))
    json.dump(rows, open(os.path.join(C.CK, 'report1_%s.json' % half), 'w'), indent=0, default=float)
    for r in sorted(rows, key=lambda r: -r['nwin']):
        a = r['arch']
        print('%02d n%d p%d l%d b%d lam%.2f | plant %.4f | win %d/7 %s | ' % (
            r['ai'], a['ncls'], a['pos'], a['lpos'], a['prev'], a['lam'], r['plant'], r['nwin'],
            'TRUST' if r['trusted'] else '     ') +
            ' '.join('%s %.2f/%.2f' % (h, r['real'][h], r['nullmax'][h]) for h in C.KNOWN))
    tr = [r for r in rows if r['trusted']]
    print('trusted', len(tr), 'of', len(rows))
    for h in C.KNOWN:
        print(h, 'median real %.3f  median nullmax %.3f  wins %d/%d' % (
            np.median([r['real'][h] for r in rows]), np.median([r['nullmax'][h] for r in rows]),
            sum(r['real'][h] > r['nullmax'][h] for r in rows), len(rows)))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'A')


def family(half='A'):
    """hold out a whole civilisation family (CUN = PC UR3 OB EB OA; AEG = LB; AND = KH): train on the others."""
    fam = {'CUN': ['PC', 'UR3', 'OB', 'EB', 'OA'], 'AEG': ['LB'], 'AND': ['KH']}
    res = []
    for ai in range(len(ARCHS)):
        d = np.load(os.path.join(C.CK, 'c1_%s_%02d.npz' % (half, ai)))
        Z = d['Z'].astype(np.float32); names = list(d['names']); ix = {n: i for i, n in enumerate(names)}
        row = {}
        for f, hs in fam.items():
            vals = []
            for rep in range(NPERM + 1):
                tr = [ix['%s:%d' % (k, rep)] for k in C.KNOWN if k not in hs]
                sel = np.argsort(-Z[tr, :NH].mean(0))[:TOP]
                vals.append(np.mean([pct_of(Z[ix['%s:0' % h]], sel) for h in hs]))
            row[f] = (vals[0], max(vals[1:]))
        res.append(row)
    for f in fam:
        r = np.array([x[f][0] for x in res]); n = np.array([x[f][1] for x in res])
        print('family %s held out: real median %.3f (min %.3f max %.3f), null-max median %.3f, wins %d/48' % (
            f, np.median(r), r.min(), r.max(), np.median(n), (r > n).sum()))
    allw = sum(all(x[f][0] > x[f][1] for f in fam) for x in res)
    print('architectures winning all three families:', allw)
    return res
