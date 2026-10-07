#!/usr/bin/env python3
"""PE-79 cycle 3: replicate with the judge trained on the OTHER PE half (B) and kill with a judge trained on
half A with signs shuffled over all slots (S).  Stability of per-sign graft roles (A vs B, A vs S), the C- leads
(M122, M260, M217), the leave-one-family-out transfer, and a layout residual test."""
import os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe79_common as C
import pe79_c2 as C2
import pe79_compare as CMP
import pe79_report1 as R1


def get(h):
    M, signs, names = C2.marg(h)
    D, N, pos, elig = C2.role_d(M, names)
    return D, N, signs


def main():
    res = {h: get(h) for h in 'ABS'}
    sA = res['A'][2]
    for h in 'BS':
        D, N, s = res[h]
        common = [x for x in sA if x in s]
        ia = [sA.index(x) for x in common]; ib = [s.index(x) for x in common]
        rhos = []
        for r in C.ROLES:
            ri = C.RI[r]
            rhos.append(spearmanr(res['A'][0][ri, ia], D[ri, ib])[0])
        nr = [spearmanr(res['A'][1][C.RI[r], ia], N[C.RI[r], ib])[0] for r in C.ROLES]
        print('A vs %s (%d shared signs) rho per role: %s | null-vs-null: %s' % (
            h, len(common), ' '.join('%s %.2f' % (r, x) for r, x in zip(C.ROLES, rhos)),
            ' '.join('%.2f' % x for x in nr)))
        for lead in ['M122', 'M260', 'M217', 'M069', 'M354']:
            if lead in s:
                j = s.index(lead)
                print('  %s in %s: ' % (lead, h) + ' '.join('%s #%d' % (r, int((D[C.RI[r]] > D[C.RI[r], j]).sum()) + 1)
                                                         for r in ('COM', 'UNI', 'TOT')))
    # layout residual: COM column regressed on slot stats + log frequency; residual AUC for pe59 MEAS+CNT
    import pe61_common as P61
    cl = P61.pe59_classes(); mc = cl['MEASURED'] | cl['COUNTED']
    import collections
    freq = collections.Counter(x[1] for d in C.pe_docs() for x in d['toks'] if x[0] == 'T')
    for h in 'ABS':
        D, N, s = res[h]
        ss = CMP.slot_stats(s)
        X = np.column_stack([np.ones(len(s)), ss['hdr'], ss['fin'], ss['last'], np.log([freq[x] for x in s])])
        y = np.array([x in mc for x in s])
        for r in ('COM', 'UNI'):
            v = D[C.RI[r]]
            beta = np.linalg.lstsq(X, v, rcond=None)[0]
            res_ = v - X @ beta
            print('%s %s residual AUC MEAS+CNT %.3f (raw %.3f)' % (h, r, CMP.auc(res_, y), CMP.auc(v, y)))
    for h in 'BS':
        print('family hold-out, judge', h)
        R1.family(h)


if __name__ == '__main__':
    main()
