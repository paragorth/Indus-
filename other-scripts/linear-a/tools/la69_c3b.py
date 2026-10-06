#!/usr/bin/env python3
"""LA-69 cycle 3 readout: cloze reader calibration, Linear A placement, cross-site outsiders, LA document types."""
import os, sys, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la69_common as C

O = pickle.load(open(os.path.join(C.CK, 'c3_cloze.pkl'), 'rb'))
rng = np.random.default_rng(6903)
CAL = ['LB_shape', 'LB_draft', 'UR3', 'OA']


def arr(key):
    docs, res = O[key]
    n = np.array([r[0] for r in res], float); o = np.array([r[1] for r in res], float); i = np.array([r[2] for r in res], float)
    lt = np.log(np.array([d['nT'] for d in docs], float))
    return docs, n, o, i, lt


def rate(n, h, idx=None):
    idx = np.arange(len(n)) if idx is None else idx
    return h[idx].sum() / max(1, n[idx].sum())


def boot(n, h, B=500):
    v = [rate(n, h, rng.integers(0, len(n), len(n))) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])


def diffP(a, b, stat, B=1000):
    """two-sample doc-level permutation on stat(n,h...)"""
    obs = stat(*a) - stat(*b)
    pooled = [np.r_[x, y] for x, y in zip(a, b)]
    na = len(a[0]); c = 0
    for _ in range(B):
        p = rng.permutation(len(pooled[0]))
        A = [x[p[:na]] for x in pooled]; Bb = [x[p[na:]] for x in pooled]
        if abs(stat(*A) - stat(*Bb)) >= abs(obs) - 1e-12:
            c += 1
    return obs, (c + 1) / (B + 1)


def lenadj(keys):
    """per-document outsider rate regressed on log length over the calibration NOTE/FINAL docs."""
    X, Y, W = [], [], []
    for k in keys:
        docs, n, o, i, lt = arr(k)
        m = n > 0
        X += list(lt[m]); Y += list(o[m] / n[m]); W += list(n[m])
    b = np.polyfit(X, Y, 1, w=np.sqrt(W))
    return b


if __name__ == '__main__':
    out = []
    print('%-22s %5s %7s  %-14s %7s %7s' % ('set', 'docs', 'outsdr', '95% CI', 'insidr', 'gap'))
    for k in O:
        docs, n, o, i, lt = arr(k)
        lo, hi = boot(n, o, 300)
        print('%-22s %5d %7.3f  [%.3f,%.3f] %7.3f %7.3f' % ('/'.join(k), len(docs), rate(n, o), lo, hi, rate(n, i),
                                                         rate(n, i) - rate(n, o)))
    b = lenadj([(c, cl) for c in CAL for cl in ('NOTE', 'FINAL')])
    print('\nlength slope of outsider rate on log words:', b)
    R = lambda n, h: h.sum() / max(1, n.sum())
    G = lambda n, o, i: (i.sum() - o.sum()) / max(1, n.sum())

    def adj(n, o, lt):
        return (o - n * (b[0] * lt + b[1])).sum() / max(1, n.sum())
    print('\nCalibration (FINAL - NOTE): outsider rate, length-adjusted outsider, insider-outsider gap; doc permutation P')
    for c in CAL + ['PL']:
        pairs = [('FINAL', 'NOTE')] if c != 'PL' else []
        for a_, b_ in pairs:
            A = arr((c, a_)); Bq = arr((c, b_))
            d1, p1 = diffP((A[1], A[2]), (Bq[1], Bq[2]), R)
            d2, p2 = diffP((A[1], A[2], A[4]), (Bq[1], Bq[2], Bq[4]), adj)
            d3, p3 = diffP((A[1], A[2], A[3]), (Bq[1], Bq[2], Bq[3]), G)
            print('  %-9s outsider %+.3f (P %.3f)  len-adj %+.3f (P %.3f)  gap %+.3f (P %.3f)' % (c, d1, p1, d2, p2, d3, p3))
            out.append((c, d1, p1, d2, p2, d3, p3))
    for c in CAL + ['LA']:
        fin = (c, 'FINAL') if c != 'LA' else ('LA', 'LA')
        pl = (c, 'PLANT') if c != 'LA' else ('LA', 'LA_PLANT')
        A = arr(fin); Bq = arr(pl)
        d1, p1 = diffP((A[1], A[2]), (Bq[1], Bq[2]), R)
        d3, p3 = diffP((A[1], A[2], A[3]), (Bq[1], Bq[2], Bq[3]), G)
        print('  PLANT %-9s source - planted: outsider %+.3f (P %.3f)  gap %+.3f (P %.3f)' % (c, d1, p1, d3, p3))
    # LA placement on length-adjusted outsider and gap
    print('\nLength-adjusted outsider rate and gap by set')
    for k in O:
        if k[0].startswith('X_'):
            continue
        docs, n, o, i, lt = arr(k)
        print('  %-22s adj %+.3f  gap %.3f' % ('/'.join(k), adj(n, o, lt), G(n, o, i)))
    # cross-site
    print('\nCross-site outsiders (same-site pool vs other-site pool, equal pool size)')
    for k in sorted({k[0] for k in O if k[0].startswith('X_')}):
        sites = sorted({s.split(':')[0] for (kk, s) in O if kk == k})
        for s in sites:
            A = arr((k, s + ':same')); Bq = arr((k, s + ':other'))
            print('  %-6s %-16s same %.3f other %.3f ratio %.2f (docs %d)' % (k, s[:16], R(A[1], A[2]), R(Bq[1], Bq[2]),
                                                                          R(Bq[1], Bq[2]) / max(1e-9, R(A[1], A[2])), len(A[0])))
    # Linear A document types
    print('\nLinear A document types: length-adjusted outsider rate and gap vs the rest of LA (doc permutation)')
    docs, n, o, i, lt = arr(('LA', 'LA'))
    keys = {
        'site=HT': [d['site'] == 'Haghia Triada' for d in docs],
        'site=Khania': [d['site'] == 'Khania' for d in docs],
        'site=Zakros': [d['site'] == 'Zakros' for d in docs],
        'site=Phaistos': [d['site'] == 'Phaistos' for d in docs],
        'site=other': [d['site'] not in ('Haghia Triada', 'Khania', 'Zakros', 'Phaistos') for d in docs],
        'support!=Tablet': [d['support'] != 'Tablet' for d in docs],
        'scribe known': [bool(d.get('scribe')) for d in docs],
        'long (>=10 words)': [d['nT'] >= 10 for d in docs],
    }
    for name, m in keys.items():
        m = np.array(m)
        if m.sum() < 5 or (~m).sum() < 5:
            continue
        d2, p2 = diffP((n[m], o[m], lt[m]), (n[~m], o[~m], lt[~m]), adj)
        d3, p3 = diffP((n[m], o[m], i[m]), (n[~m], o[~m], i[~m]), G)
        print('  %-18s n %3d  adj-outsider %+.3f (P %.3f)  gap %+.3f (P %.3f)' % (name, m.sum(), d2, p2, d3, p3))
    pickle.dump(out, open(os.path.join(C.CK, 'c3b.pkl'), 'wb'))
