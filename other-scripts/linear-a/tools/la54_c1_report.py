#!/usr/bin/env python3
"""LA-54 cycle 1 report: real vs nulls vs controls."""
import os, json, glob
import numpy as np
import la54_common as C


def L(j):
    f = os.path.join(C.CK, 'c1_%s.json' % j)
    return json.load(open(f)) if os.path.exists(f) else None


def main():
    R = {os.path.basename(f)[3:-5]: json.load(open(f)) for f in glob.glob(os.path.join(C.CK, 'c1_*.json'))}
    lsh = [r['agg'] for k, r in R.items() if k.startswith('LSH_')]
    nsh = [r['agg'] for k, r in R.items() if k.startswith('NSH_')]
    out = []
    for key in ('acc', 'bacc', 'bits_gain'):
        nl = np.array([a[key] for a in lsh]) if lsh else np.array([])
        ns = np.array([a[key] for a in nsh]) if nsh else np.array([])
        row = [key]
        for j in ('LA', 'LA_NOWORD', 'LA_NUMONLY', 'LA_SITEONLY', 'LA_big'):
            if j in R:
                v = R[j]['agg'][key]
                p = (1 + (nl >= v).sum()) / (1 + len(nl)) if len(nl) else float('nan')
                row.append('%s=%.3f(P_lsh=%.3f)' % (j, v, p))
        if len(nl): row.append('LSH mean %.3f sd %.3f max %.3f (n=%d)' % (nl.mean(), nl.std(), nl.max(), len(nl)))
        if len(ns): row.append('NSH mean %.3f sd %.3f max %.3f (n=%d)' % (ns.mean(), ns.std(), ns.max(), len(ns)))
        out.append(' | '.join(row))
    if 'LA' in R:
        out.append('LA majority %.3f; groups %s; fam %s' % (R['LA']['agg']['maj'], {k: round(v, 3) for k, v in R['LA']['groups'].items()},
                                                         {k: round(v, 3) for k, v in R['LA']['fam'].items()}))
    for pre in ('LB', 'UR'):
        rs = [r for k, r in R.items() if k.startswith(pre + '_')]
        if rs:
            out.append('%s control: ' % pre + '; '.join('acc %.3f bacc %.3f bits %.3f maj %.3f classes %d' % (r['agg']['acc'], r['agg']['bacc'], r['agg']['bits_gain'], r['agg']['maj'], len(r['classes'])) for r in rs))
            out.append('   %s groups: %s' % (pre, {g: round(float(np.mean([r['groups'][g] for r in rs])), 3) for g in C.GROUPS}))
    for pre in ('PL', 'PLN', 'PLS', 'PLW'):
        rs = [r for k, r in R.items() if k.startswith(pre + '_') and not k.startswith('PLN_') or (pre == 'PLN' and k.startswith('PLN_'))]
        rs = [r for k, r in R.items() if k.split('_')[0] == pre]
        if rs:
            rec = [x['recall'] for r in rs for x in r['plant'] if x['recall'] is not None]
            prc = [x['precision'] for r in rs for x in r['plant']]
            pt = [x['pplant_true'] - x['pplant_other'] for r in rs for x in r['plant']]
            out.append('%s: PLANT recall mean %.3f (n=%d holdouts), precision %.3f, P(PLANT|true)-P(PLANT|other) %.3f' % (pre, np.mean(rec), len(rec), np.mean(prc), np.mean(pt)))
    print('\n'.join(out))


if __name__ == '__main__':
    main()
