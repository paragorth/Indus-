"""pe21 cycle 2: which kinds of PE tablets were sized to their text?

(a) Ur III label refinement: RUNNING (several day dates, no total, no balance) vs SUMMARY
    (grand total, balanced account or month span) vs DAILY.
(b) Floor check: the slope on tablets with >= 5 and >= 8 lines (a minimum lump size flattens
    the slope for short texts).
(c) PE splits: number system, seal, region/site, pe15 office, total, header, spill.
    Group beta vs the rest; null = permute group labels within content bins (quintiles of
    log glyphs), 1,000 draws, so range differences cannot make a group look planned.
(d) Totals: obverse line density (lines per mm of height, one-column obverses) of tablets
    whose entries stop on the obverse and whose total is alone on the reverse, against
    tablets whose entries run onto the reverse (obverse certainly full) and tablets with
    no total.  Planned obverse -> same density as full obverses.
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe21_common import pe_tablets, ur3_tablets, ur3_kind, dom_sys, CK, write_rows
from pe21_cycle1 import arr, beta, boot_beta, describe

rng = np.random.default_rng(212)


def ur3_fine(t):
    k = ur3_kind(t)
    if k != 'COMPILED':
        return k
    if t['total'] or t['nigka'] or t['span']:
        return 'SUMMARY'
    if t['ndays'] >= 2:
        return 'RUNNING'
    return 'MULTIMONTH'


def group_test(P, lab, content='glyphs', n=1000):
    A, C = arr(P, content)
    bins = np.digitize(C, np.percentile(C, [20, 40, 60, 80]))
    lab = np.array(lab)
    out = {}
    for g in sorted(set(lab)):
        m = lab == g
        if m.sum() < 15 or (~m).sum() < 15:
            continue
        d = beta(A[m], C[m]) - beta(A[~m], C[~m])
        ds = []
        for _ in range(n):
            lp = lab.copy()
            for b in range(5):
                ii = np.where(bins == b)[0]
                lp[ii] = rng.permutation(lp[ii])
            mm = lp == g
            ds.append(beta(A[mm], C[mm]) - beta(A[~mm], C[~mm]))
        ds = np.array(ds)
        out[g] = {'n': int(m.sum()), 'beta': beta(A[m], C[m]), 'd': float(d),
                  'p': float((np.abs(ds - ds.mean()) >= abs(d - ds.mean())).mean()),
                  'z': float((d - ds.mean()) / ds.std())}
    return out


def main():
    res = {}
    U = [t for t in ur3_tablets() if t['intact'] and t['n_lines'] >= 2]
    for k in ['DAILY', 'RUNNING', 'SUMMARY', 'MULTIMONTH']:
        T = [t for t in U if ur3_fine(t) == k]
        if len(T) >= 15:
            res['ur3_' + k] = describe(T, 'n_lines')
            res['ur3_' + k]['med_lines'] = float(np.median([t['n_lines'] for t in T]))
    P = [t for t in pe_tablets() if t['intact'] and t['n_lines'] >= 2]
    for m in [5, 8]:
        res['pe_min%d' % m] = describe([t for t in P if t['n_lines'] >= m], 'n_lines')
        res['ur3D_min%d' % m] = describe([t for t in U if ur3_fine(t) == 'DAILY' and t['n_lines'] >= m], 'n_lines')
        res['ur3S_min%d' % m] = describe([t for t in U if ur3_fine(t) == 'SUMMARY' and t['n_lines'] >= m], 'n_lines')
    labs = {
        'system': [dom_sys(t) for t in P],
        'sealed': ['SEALED' if t['sealed'] else 'UNSEALED' for t in P],
        'region': [t['region'] for t in P],
        'office': [t['office'] for t in P],
        'total': ['TOTAL' if t['n_tot'] else 'NOTOTAL' for t in P],
        'header': ['HEADER' if t['header'] else 'NOHEADER' for t in P],
        'spill': ['SPILL' if t['spill'] else 'OBVONLY' for t in P],
    }
    for k, lab in labs.items():
        res['split_' + k] = group_test(P, lab)
    # (d) totals position
    one = [t for t in P if t['cols_obv'] == 1 and t['obv'] >= 2]
    dens = lambda T: np.array([t['obv'] / t['h'] for t in T])
    G = {'SPILL': [t for t in one if t['spill']],
         'OBV+REVTOTAL': [t for t in one if not t['spill'] and 'reverse' in t['tot_face']],
         'OBV_NOTOTAL': [t for t in one if not t['spill'] and not t['n_tot']]}
    dd = {k: dens(v) for k, v in G.items()}
    res['totals'] = {k: {'n': len(v), 'median_lines_per_cm': float(np.median(v) * 10),
                         'q25': float(np.percentile(v, 25) * 10), 'q75': float(np.percentile(v, 75) * 10)}
                     for k, v in dd.items()}

    def mw(a, b, n=2000):
        d = np.median(a) - np.median(b)
        x = np.concatenate([a, b])
        ds = []
        for _ in range(n):
            p = rng.permutation(len(x))
            ds.append(np.median(x[p[:len(a)]]) - np.median(x[p[len(a):]]))
        return float(d * 10), float((np.abs(ds) >= abs(d)).mean())
    res['totals']['revtotal_vs_spill'] = mw(dd['OBV+REVTOTAL'], dd['SPILL'])
    res['totals']['revtotal_vs_nototal'] = mw(dd['OBV+REVTOTAL'], dd['OBV_NOTOTAL'])
    # obverse fill of REVTOTAL tablets: how many lines short of the SPILL density
    full = np.median(dd['SPILL'])
    short = [full * t['h'] - t['obv'] for t in G['OBV+REVTOTAL']]
    res['totals']['revtotal_lines_short_median'] = float(np.median(short))
    json.dump(res, open(os.path.join(CK, 'c2.json'), 'w'), indent=1)

    f = lambda r: 'beta %.2f [%.2f,%.2f], r %.2f, n %d' % (r['beta'], r['ci'][0], r['ci'][1], r['r'], r['n'])
    rows = [['PE-21.2a', 'Ur III kinds refined: DAILY / RUNNING (several day dates, no total) / SUMMARY (grand total, balance or month span) / MULTIMONTH; beta on lines',
             '; '.join('%s %s, median %d lines' % (k, f(res['ur3_' + k]), res['ur3_' + k]['med_lines'])
                       for k in ['DAILY', 'RUNNING', 'SUMMARY', 'MULTIMONTH'] if 'ur3_' + k in res), ''],
            ['PE-21.2b', 'Floor check: only tablets with >= 5 and >= 8 lines (PE vs Ur III DAILY and SUMMARY)',
             '; '.join('>=%d lines: PE %s | DAILY %s | SUMMARY %s' % (m, f(res['pe_min%d' % m]), f(res['ur3D_min%d' % m]), f(res['ur3S_min%d' % m])) for m in [5, 8]), '']]
    for k in labs:
        s = res['split_' + k]
        rows.append(['PE-21.2c-' + k, 'PE split by %s: group beta (glyphs) vs rest; null permutes labels within glyph quintiles (1,000)' % k,
                     '; '.join('%s n %d beta %.2f (d %+.2f, z %+.1f, p %.3f)' % (g, v['n'], v['beta'], v['d'], v['z'], v['p']) for g, v in s.items()), ''])
    tt = res['totals']
    rows.append(['PE-21.2d', 'Totals: obverse lines per cm (one-column obverses) for entries-on-obverse + total-alone-on-reverse vs spill-over tablets (obverse full) vs no-total; median-difference permutation',
                 '; '.join('%s n %d median %.2f lines/cm [%.2f-%.2f]' % (k, v['n'], v['median_lines_per_cm'], v['q25'], v['q75']) for k, v in tt.items() if isinstance(v, dict))
                 + '; REVTOTAL-SPILL %.2f (p %.3f); REVTOTAL-NOTOTAL %.2f (p %.3f); REVTOTAL obverses are %.1f lines short of a full obverse (median)' % (
                     tt['revtotal_vs_spill'] + tt['revtotal_vs_nototal'] + (tt['revtotal_lines_short_median'],)), ''])
    write_rows(os.path.join(CK, 'c2_rows.txt'), rows)
    for r in rows:
        print(' | '.join(r))


if __name__ == '__main__':
    main()
