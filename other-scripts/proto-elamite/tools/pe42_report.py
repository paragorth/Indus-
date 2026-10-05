"""Summaries for pe42 cycles."""
import json, sys, collections
import numpy as np
from pe42_common import CK, FAMS, os


def c1():
    d = json.load(open(os.path.join(CK, 'c1.json')))
    real = d['real']
    nul = collections.defaultdict(lambda: collections.defaultdict(list))
    pl = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, kind, seed, r in d['res']:
        if kind.startswith('plant:'):
            pl[name][kind[6:]].append(r)
        else:
            nul[name][kind].append(r)
    out = []
    for name in ('PE', 'PC', 'UR3'):
        rc = real[name]['counts']
        out.append('== %s: %d failing, %d explained by some family' % (name, rc['_fail'], rc['_any']))
        for f in FAMS + ['_any']:
            r = rc.get(f, 0)
            row = '%-8s real %3d' % (f, r)
            for kind in ('noise', 'totals', 'shuffle'):
                v = np.array([x.get(f, 0) for x in nul[name][kind]])
                p = (1 + (v >= r).sum()) / (1 + len(v))
                row += ' | %s %.1f p=%.3f' % (kind, v.mean(), p)
            out.append(row)
        out.append('-- planted (5 reps x <=30 tablets): recovery = planted family among explainers; enrichment vs noise null')
        for f, L in pl[name].items():
            n = sum(x['n'] for x in L); h = sum(x['hit'] for x in L)
            fk = ['OMITH', 'OMITO'] if f == 'OMIT' else [f]
            re_ = sum(sum(x['real'].get(k, 0) for k in fk) for x in L)
            no = sum(sum(x['noise'].get(k, 0) for k in fk) for x in L)
            # which other families light up most in the planted set
            oth = collections.Counter()
            for x in L:
                for k, v in x['real'].items():
                    if not k.startswith('_') and k not in fk:
                        oth[k] += v - x['noise'].get(k, 0)
            top = ', '.join('%s %+d' % kv for kv in oth.most_common(2))
            out.append('  %-8s n=%3d recovered %3d (%.2f); planted-family count %d vs noise %d; others excess: %s'
                       % (f, n, h, h / max(n, 1), re_, no, top))
    return '\n'.join(out)



def c3():
    from pe42_cycle3 import FAM3
    res = json.load(open(os.path.join(CK, 'c3.json')))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, kind, seed, r in res:
        by[name][kind].append(r)
    out = []
    for name in ('PE', 'PC', 'UR3'):
        r = by[name]['real'][0]
        out.append('== %s: %d failing, %d fixed by a single omission/double (set aside)' % (name, r.get('_fail', 0), r.get('_single', 0)))
        for f in FAM3:
            row = '  %-7s real %3d' % (f, r.get(f, 0))
            for k in ('totals', 'noise'):
                v = np.array([x.get(f, 0) for x in by[name][k]])
                row += ' | %s %.2f p=%.3f' % (k, v.mean(), (1 + (v >= r.get(f, 0)).sum()) / (1 + len(v)))
            out.append(row)
        # chunk ratio: RUN2 / (RUN2 + PAIR2), real vs nulls
        def ratio(x):
            a, b = x.get('RUN2', 0), x.get('PAIR2', 0)
            return a / (a + b) if a + b else np.nan
        rr = ratio(r)
        for k in ('totals', 'noise'):
            v = np.array([ratio(x) for x in by[name][k]]); v = v[~np.isnan(v)]
            out.append('  RUN2 share real %.2f vs %s %.2f (n=%d)' % (rr, k, v.mean() if len(v) else np.nan, len(v)))
        for k in [k for k in by[name] if k.startswith('plant')]:
            L = by[name][k]
            out.append('  %s: %s ; RUN2 share %.2f' % (k, ', '.join('%s %d' % (f, sum(x.get(f, 0) for x in L)) for f in FAM3 + ['_fail']),
                                                    np.nanmean([ratio(x) for x in L])))
    return '\n'.join(out)


if __name__ == '__main__':
    print(globals()[sys.argv[1]]())
