"""v76 cycle 1 report: template-cohesion summary per corpus vs its fitted generators."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v76_lib as V

GEN = ['WSHUF', 'MK2', 'SELFCIT', 'JUNC', 'SC10', 'PARCOPY', 'PARCOPYID']


def stats(name):
    d = np.load(os.path.join(V.CK, 'c1_%s.npz' % name)); sc, n = d['sc'], d['npair']
    ok0 = n[:, 1] >= 5
    top = np.argsort(-np.where(ok0, sc[:, 1], -1e9))[:50]
    rep = float(np.mean(sc[top, 2]))
    both = int(((sc[:, 1] > 3) & (sc[:, 2] > 3)).sum())
    return dict(p999=float(np.percentile(sc[:, 0], 99.9)), mx=float(sc[:, 0].max()), rep=rep, both=both,
                med=float(np.median(sc[:, 0])))


def main(fn=None):
    fits = json.load(open(os.path.join(V.CK, 'c1_fits.json')))
    lines = []
    for real in ('ZL3b', 'IT2a', 'CUL', 'BRU', 'API'):
        if not os.path.exists(os.path.join(V.CK, 'c1_%s.npz' % real)): continue
        r = stats(real)
        g = {}
        for gn in GEN:
            vals = [stats(os.path.basename(f)[3:-4]) for f in sorted(glob.glob(os.path.join(V.CK, 'c1_%s__%s_*.npz' % (real, gn))))]
            if vals: g[gn] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        f = fits[real]
        s = '%s (PARCOPY fit c=%.2f m=%.2f; top-1 sim real %.3f, no-copy law %.3f): REAL med %.2f p99.9 %.1f max %.1f H0->H1 replication %.2f, templates >3 in both halves %d' % (
            real, f['c'], f['m'], f['target'], f.get('nocopy', float('nan')), r['med'], r['p999'], r['mx'], r['rep'], r['both'])
        s += ' || ' + '; '.join('%s med %.2f p99.9 %.1f rep %.2f both %d' % (k, v['med'], v['p999'], v['rep'], v['both']) for k, v in g.items())
        lines.append((real, r, g, s))
        print(s)
    json.dump([(a, b, c) for a, b, c, _ in lines], open(os.path.join(V.CK, 'c1_report.json'), 'w'))
    return lines


if __name__ == '__main__':
    main()
