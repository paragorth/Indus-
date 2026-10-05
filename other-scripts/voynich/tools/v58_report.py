"""v58: summarise cycle json files (controls, Voynich scan, order search)."""
import os, sys, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v58_lib as L

def c1():
    r = json.load(open(os.path.join(L.CK, 'cycle1.json')))
    g = defaultdict(list)
    for k, v in r.items():
        a, b, n, s, rep = k.split('|')
        g[(a, int(n), s)].append(v)
    print('CONTROL pair | window | setting | POS zx/zb (mean) | NEG zx/zb (max)')
    for (a, n, s), vs in sorted(g.items()):
        print('%-14s %4d %s  POS zx %6.1f zb %6.1f | NEG zx max %5.2f zb max %5.2f' % (
            a, n, s, np.mean([v['pos']['zx'] for v in vs]), np.mean([v['pos']['zb'] for v in vs]),
            max(v['neg']['zx'] for v in vs), max(v['neg']['zb'] for v in vs)))
    negs = [v['neg']['zb'] for v in r.values()]
    print('NEG zb distribution: n=%d mean %.2f sd %.2f max %.2f; >=3: %d' % (len(negs), np.mean(negs), np.std(negs), max(negs), sum(z >= 3 for z in negs)))

def c2(fn='cycle2.json'):
    r = json.load(open(os.path.join(L.CK, fn)))
    T = json.load(open(os.path.join(L.CK, 'texts.json')))
    by = defaultdict(list)
    for k, v in r.items():
        s, c = k.split('|'); by[s].append((c, v))
    allz = []
    for s, vs in by.items():
        vs.sort(key=lambda t: -min(t[1]['zb'], t[1]['zx']))
        print('\n== %s (n=%d) candidates %d' % (s, vs[0][1]['n'], len(vs)))
        for c, v in vs[:6]:
            print('  %-22s %-8s zx %5.2f zb %5.2f zy %5.2f  obs %.1f c %.2f m %d' % (c, T.get(c, {}).get('genre', '?'), v['zx'], v['zb'], v['zy'], v['obs'], v['c'], v['m']))
        allz += [min(v['zb'], v['zx']) for c, v in vs]
    allz = np.array(allz)
    print('\nall section x candidate tests: %d; min(zx,zb) mean %.2f sd %.2f max %.2f; >=2: %d, >=3: %d' % (len(allz), allz.mean(), allz.std(), allz.max(), (allz >= 2).sum(), (allz >= 3).sum()))

if __name__ == '__main__':
    for a in sys.argv[1:]:
        if a == 'c1': c1()
        elif a.startswith('c2'): c2(a.split(':')[1] if ':' in a else 'cycle2.json')
