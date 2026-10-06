"""v76 cycle 3 report."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v76_lib as V
GEN = ['WSHUF', 'MK2', 'SELFCIT', 'JUNC', 'SC10', 'PARCOPY', 'PARCOPYID']


def load(n):
    f = os.path.join(V.CK, 'c3_%s.json' % n)
    return json.load(open(f)) if os.path.exists(f) else None


def sub(r, k):
    x = r.get(k)
    return 'n/a' if not x else 'pairs %d FIX %.3f SLOTR %.2f PSD %+.3f' % (x['npairs'], x['FIX'], x['SLOTR'], x['PSD'] or 0)


def idl(r):
    x = r['ID']; L = r['LINK']
    return 'ID pages %d conc %s (n %d) adj %s rnd %s (n %d) | LINK groups %d tests %d linked %d rate %.3f' % (
        x['npages'], '%.3f' % x['conc'] if x['conc'] is not None else 'n/a', x['nconc'],
        '%.3f' % x['adj'] if x['adj'] is not None else 'n/a', '%.3f' % x['rnd'] if x['rnd'] is not None else 'n/a', x['nadj'],
        L['groups'], L['tests'], L['linked'], L['rate'])


def main():
    for real in ('ZL3b', 'IT2a', 'CUL', 'BRU', 'API'):
        R = load(real)
        if R is None: continue
        print('==', real, 'REAL S:', sub(R, 'S'), '| H0:', sub(R, 'H0'), '|', idl(R))
        for gn in GEN:
            for f in sorted(glob.glob(os.path.join(V.CK, 'c3_%s__%s_*.json' % (real, gn)))):
                G = json.load(open(f))
                print('   ', os.path.basename(f)[3:-5], 'S:', sub(G, 'S'), '| H0:', sub(G, 'H0'), '|', idl(G))


if __name__ == '__main__':
    main()
