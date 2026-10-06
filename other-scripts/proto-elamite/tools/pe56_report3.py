#!/usr/bin/env python3
"""pe56 cycle 3 report: constraint type of each residue from the oracle forgers.
Type rule (p < 1e-3 = survives that oracle):
  ORDER     survives PERM (own entries permuted)                     -> order of entries matters
  BINDING   dies under PERM, survives VPERM                         -> string-numeral binding / arithmetic
  HEADER    dies under PERM and VPERM, dies under HDR               -> explained by the header's own pool
  TOPIC     dies under HDR? no; dies under TOPIC                    -> vocabulary co-occurrence
  TABLET    survives HDR and TOPIC but dies under PERM and VPERM    -> a whole-tablet set constraint
"""
import sys, json, os, collections
from pe56_common import CK

TH = 1e-3


def ctype(r):
    s = {m: r['p_' + m] < TH for m in ['PERM', 'VPERM', 'HDR', 'TOPIC']}
    if s['PERM'] and s['VPERM']:
        return 'ORDER+BIND'
    if s['PERM']:
        return 'ORDER'
    if s['VPERM']:
        return 'BINDING'
    if not s['HDR']:
        return 'HEADER'
    if not s['TOPIC']:
        return 'TOPIC'
    return 'TABLET'


def main(names):
    for name in names:
        fn = os.path.join(CK, 'c3_%s.json' % name)
        if not os.path.exists(fn):
            print(name, 'missing'); continue
        d = json.load(open(fn))
        T = set(map(tuple, d['truth'] or []))
        rows = d['rows']
        cnt = collections.Counter()
        print('== %s: %d keys (%d planted)' % (name, len(rows), len(T)))
        for r in sorted(rows, key=lambda r: (r['p_forger'] or 1)):
            t = ctype(r)
            pl = tuple(r['key']) in T
            cnt[(t, pl)] += 1
            print('   %-4s %-18s %-12s R %3d | forg %6.2f p %.0e | PERM %6.2f %.0e | VPERM %6.2f %.0e | HDR %6.2f %.0e | TOPIC %6.2f %.0e | %s%s' % (
                r['key'][0], r['key'][1], r['key'][2], r['R'], r['E_forger'] or -1, r['p_forger'] or 1,
                r['E_PERM'], r['p_PERM'], r['E_VPERM'], r['p_VPERM'], r['E_HDR'], r['p_HDR'],
                r['E_TOPIC'], r['p_TOPIC'], t, ' PLANTED' if pl else ''))
        print('   types:', dict(cnt))


if __name__ == '__main__':
    main(sys.argv[1:])
