#!/usr/bin/env python3
"""PE-79: frozen per-sign graft roles vs what is established for PE (scoring only, after freezing).

Checks: pe59 frozen classes (MEASURED+COUNTED, PERSON, PREFIX), header signs (FINDINGS test d: M157,
|M327+M342|, M327), M288 (per-head allotment, entry-final), M153 last-line compounds (pe71 B), pe38/pe59
'totals have no sign'.  Dumb baselines: a sign's own header-line rate, line-final rate, last-line rate.
NULL for every AUC: the role score columns of the permuted-label null (pe79_c2 N) and 2,000 random sign sets.
"""
import os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe79_common as C
import pe61_common as P61


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return float('nan')
    from scipy.stats import rankdata
    r = rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def slot_stats(signs):
    P = C.pe_docs()
    st = collections.defaultdict(collections.Counter)
    for d in P:
        ls = C.lines_of(d)
        for li, l in enumerate(ls):
            ts = [x[1] for x in l if x[0] == 'T']
            hdr = not any(x[0] == 'N' for x in l)
            for k, s in enumerate(ts):
                st[s]['n'] += 1; st[s]['hdr'] += hdr; st[s]['fin'] += (k == len(ts) - 1 and not hdr)
                st[s]['last'] += (li == len(ls) - 1)
    return {b: np.array([st[s][b] / max(1, st[s]['n']) for s in signs]) for b in ('hdr', 'fin', 'last')}


def compare(D, N, signs, label=''):
    cl = P61.pe59_classes()
    sets = {'MEAS+CNT': cl['MEASURED'] | cl['COUNTED'], 'PERSON': cl['PERSON'], 'PREFIX': cl['PREFIX'],
            'HEADER': {'M157', '|M327+M342|', 'M327', '|M327+X|'}, 'M153c': {s for s in signs if s.startswith('|M153+')}}
    ss = slot_stats(signs)
    rng = np.random.default_rng(C.seed('pe79-cmp'))
    out = []
    for name, role, base in [('MEAS+CNT', 'COM', 'fin'), ('MEAS+CNT', 'UNI', 'fin'), ('PERSON', 'PER', 'fin'),
                             ('PREFIX', 'PER', 'fin'), ('HEADER', 'HDR', 'hdr'), ('M153c', 'TOT', 'last')]:
        y = np.array([s in sets[name] for s in signs])
        if y.sum() == 0:
            out.append('%s as %s: no signs in top 120' % (name, role)); continue
        a = auc(D[C.RI[role]], y); an = auc(N[C.RI[role]], y); ab = auc(ss[base], y)
        rnd = [auc(D[C.RI[role]], rng.permutation(y)) for _ in range(2000)]
        p = (np.sum(np.array(rnd) >= a) + 1) / 2001
        out.append('%s (n %d) as %s: AUC %.3f (perm-label null %.3f; random-set p %.3f; slot baseline %s %.3f)' % (
            name, y.sum(), role, a, an, p, base, ab))
    for s in ['M288', 'M157', '|M327+M342|', 'M153', '|M153+M342|', '|M153+X|', 'M388', 'M054', 'M124', 'M297', 'M376']:
        if s in signs:
            j = signs.index(s)
            ranks = {r: int((D[C.RI[r]] > D[C.RI[r], j]).sum()) + 1 for r in C.ROLES}
            out.append('%s: ' % s + ' '.join('%s #%d' % (r, ranks[r]) for r in C.ROLES))
        else:
            out.append('%s: not in top 120' % s)
    # correlation of each role column with the slot baselines (how much is layout)
    from scipy.stats import spearmanr
    for r in C.ROLES:
        out.append('%s rho vs hdr %.2f fin %.2f last %.2f' % (r, spearmanr(D[C.RI[r]], ss['hdr'])[0],
                                                             spearmanr(D[C.RI[r]], ss['fin'])[0],
                                                             spearmanr(D[C.RI[r]], ss['last'])[0]))
    print('\n'.join(label + x for x in out))
    return out


if __name__ == '__main__':
    import pe79_c2 as C2
    half = sys.argv[1] if len(sys.argv) > 1 else 'A'
    M, signs, names = C2.marg(half)
    D, N, pos, elig = C2.role_d(M, names)
    compare(D, N, signs)
