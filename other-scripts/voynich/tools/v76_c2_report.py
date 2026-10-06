"""v76 cycle 2 report: slot roles inside near-copy pairs, real vs generator panel."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v76_lib as V
GEN = ['WSHUF', 'MK2', 'SELFCIT', 'JUNC', 'SC10', 'PARCOPY', 'PARCOPYID']
KEYS = ['npairs', 'FIX', 'SLOTR', 'POSH', 'PSD', 'PSF', 'OWND']
TAG = os.environ.get('V76_C2TAG', 'c2')


def load(n):
    f = os.path.join(V.CK, '%s_%s.json' % (TAG, n))
    return json.load(open(f)) if os.path.exists(f) else None


def fmt(r):
    return 'pairs %d FIX %.3f SLOTR %.2f POSH %+.2f PSD %+.3f±%.3f OWND %+.3f±%.3f' % (
        r['npairs'], r['FIX'], r['SLOTR'], r['POSH'], r['PSD'] or 0, r['PSD_se'] or 0, r.get('OWND') or 0, r.get('OWND_se') or 0)


def main():
    out = {}
    for real in ('ZL3b', 'IT2a', 'GC', 'CUL', 'BRU', 'API'):
        R = load(real)
        if R is None: continue
        for which in ('ALL', 'H1'):
            s = '%s %s REAL %s' % (real, which, fmt(R[which]))
            gv = {}
            for gn in GEN:
                vals = [load(os.path.basename(f)[len(TAG) + 1:-5]) for f in sorted(glob.glob(os.path.join(V.CK, '%s_%s__%s_*.json' % (TAG, real, gn))))]
                vals = [v[which] for v in vals if v]
                if not vals: continue
                gv[gn] = {k: float(np.mean([v.get(k) or 0 for v in vals])) for k in KEYS}
                s += ' || %s FIX %.3f SLOTR %.2f POSH %+.2f PSD %+.3f OWND %+.3f' % (gn, gv[gn]['FIX'], gv[gn]['SLOTR'], gv[gn]['POSH'], gv[gn]['PSD'], gv[gn]['OWND'])
            out[real + '_' + which] = dict(real=R[which], gen=gv)
            print(s)
    json.dump(out, open(os.path.join(V.CK, '%s_report.json' % TAG), 'w'))


if __name__ == '__main__':
    main()
