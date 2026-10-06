"""v76 cycle 2 report: slot roles inside near-copy pairs, real vs generator panel."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v76_lib as V
GEN = ['WSHUF', 'MK2', 'SELFCIT', 'JUNC', 'SC10', 'PARCOPY', 'PARCOPYID']
KEYS = ['npairs', 'FIX', 'SLOTR', 'POSH', 'PSD', 'PSF']


def load(n):
    f = os.path.join(V.CK, 'c2_%s.json' % n)
    return json.load(open(f)) if os.path.exists(f) else None


def fmt(r):
    return 'pairs %d FIX %.3f SLOTR %.2f POSH %+.2f PSD %+.3f±%.3f PSF %.3f' % (
        r['npairs'], r['FIX'], r['SLOTR'], r['POSH'], r['PSD'] or 0, r['PSD_se'] or 0, r['PSF'] or 0)


def main():
    out = {}
    for real in ('ZL3b', 'IT2a', 'CUL', 'BRU', 'API'):
        R = load(real)
        if R is None: continue
        for which in ('ALL', 'H1'):
            s = '%s %s REAL %s' % (real, which, fmt(R[which]))
            gv = {}
            for gn in GEN:
                vals = [load(os.path.basename(f)[3:-5]) for f in sorted(glob.glob(os.path.join(V.CK, 'c2_%s__%s_*.json' % (real, gn))))]
                vals = [v[which] for v in vals if v]
                if not vals: continue
                gv[gn] = {k: float(np.mean([v[k] or 0 for v in vals])) for k in KEYS}
                s += ' || %s pairs %d FIX %.3f SLOTR %.2f POSH %+.2f PSD %+.3f' % (gn, gv[gn]['npairs'], gv[gn]['FIX'], gv[gn]['SLOTR'], gv[gn]['POSH'], gv[gn]['PSD'])
            out[real + '_' + which] = dict(real=R[which], gen=gv)
            print(s)
    json.dump(out, open(os.path.join(V.CK, 'c2_report.json'), 'w'))


if __name__ == '__main__':
    main()
