#!/usr/bin/env python3
"""pe56 cycle 2 report: held-out transfer of half-A residues to half-B tablets (real vs forged)."""
import sys, json, os, collections
from pe56_common import CK


def main(names):
    for name in names:
        fn = os.path.join(CK, 'c2_%s.json' % name)
        if not os.path.exists(fn):
            print(name, 'missing'); continue
        d = json.load(open(fn))
        T = set(map(tuple, d['truth'] or []))
        rW = eW = rC = eC = 0
        nW = 0
        rel = collections.defaultdict(lambda: [0, 0.0, 0])
        fam = collections.defaultdict(lambda: [0, 0.0])
        for sp in d['splits']:
            for k, r, e in sp['W']:
                rW += r; eW += e; nW += 1
                rel[tuple(k)][0] += r; rel[tuple(k)][1] += e; rel[tuple(k)][2] += 1
                fam[k[0]][0] += r; fam[k[0]][1] += e
            for k, r, e in sp['C']:
                rC += r; eC += e
        print('== %s: %d splits, %d residues; held-out real/forged %d/%.1f = %.2f; control set %d/%.1f = %.2f' % (
            name, len(d['splits']), nW, rW, eW, rW / max(eW, 1e-9), rC, eC, rC / max(eC, 1e-9)))
        print('   families:', {f: '%d/%.1f' % tuple(v) for f, v in fam.items()})
        for k, (r, e, n) in sorted(rel.items(), key=lambda x: -x[1][0]):
            print('     %-4s %-16s %-14s splits %d  B real %3d forged %6.1f ratio %.2f%s' % (
                k[0], k[1], k[2], n, r, e, r / max(e, 1e-9), ' PLANTED' if k in T else ''))


if __name__ == '__main__':
    main(sys.argv[1:])
