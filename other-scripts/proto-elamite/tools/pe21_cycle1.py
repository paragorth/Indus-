"""pe21 cycle 1: does the clay follow the text?

Statistics (intact tablets with catalogue height and width, >= 2 text lines):
  beta  = elasticity of clay area on text amount: slope of log(h*w) on log(content).
          Planned (size chosen for a known text) -> beta near 1; clay chosen blind -> 0.
  pi    = planned share from a two-component mixture: P: logA ~ N(a + b logC, s1),
          U: logA ~ N(mu, s2) independent of content.  EM, 20 random starts.
  blank = share with a '$ blank' note; edge = share with writing on an edge (overflow).
Nulls: (i) reassign texts across tablets of the same kind (keeps the size distribution);
       (ii) for kind differences, permute kind labels.
Controls: Ur III DAILY (single-day receipts, written on the spot) vs COMPILED (multi-day,
month-span, balanced or grand-total accounts, compiled from earlier records), with a
content-range-matched subset; planted corpora built from PE content with known pi.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe21_common import pe_tablets, ur3_tablets, ur3_kind, CK, LOOPS, write_rows

rng = np.random.default_rng(21)


def arr(T, content):
    A = np.log(np.array([t['h'] * t['w'] for t in T], float))
    C = np.log(np.array([max(t[content], 1) for t in T], float))
    return A, C


def beta(A, C):
    v = np.var(C)
    return float(np.cov(A, C, bias=True)[0, 1] / v) if v > 0 else 0.0


def boot_beta(A, C, n=400):
    idx = rng.integers(0, len(A), (n, len(A)))
    return np.percentile([beta(A[i], C[i]) for i in idx], [5, 95])


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def mixture(A, C, starts=20, iters=200):
    best = None
    for s in range(starts):
        r = np.random.default_rng(s)
        pi = r.uniform(0.2, 0.8)
        b = r.uniform(0.5, 1.2)
        a = np.mean(A) - b * np.mean(C)
        s1 = r.uniform(0.1, 0.3)
        mu, s2 = np.mean(A), np.std(A)
        for _ in range(iters):
            p1 = pi * npdf(A, a + b * C, s1)
            p2 = (1 - pi) * npdf(A, mu, s2)
            g = p1 / (p1 + p2 + 1e-300)
            pi = np.clip(g.mean(), 1e-3, 1 - 1e-3)
            W = g.sum() + 1e-9
            mc, ma = (g * C).sum() / W, (g * A).sum() / W
            vc = (g * (C - mc) ** 2).sum() / W
            b = (g * (C - mc) * (A - ma)).sum() / W / max(vc, 1e-6)
            b = float(np.clip(b, 0.3, 2.0))   # P must be a real size-follows-text line
            a = ma - b * mc
            s1 = max(np.sqrt((g * (A - a - b * C) ** 2).sum() / W), 0.05)
            W2 = (1 - g).sum() + 1e-9
            mu = ((1 - g) * A).sum() / W2
            s2 = max(np.sqrt(((1 - g) * (A - mu) ** 2).sum() / W2), 0.05)
        ll = np.log(pi * npdf(A, a + b * C, s1) + (1 - pi) * npdf(A, mu, s2) + 1e-300).sum()
        if best is None or ll > best[0]:
            best = (ll, dict(pi=float(pi), b=b, a=float(a), s1=float(s1), mu=float(mu), s2=float(s2)), g)
    return best


def null_beta(A, C, n=500):
    return np.array([beta(A, rng.permutation(C)) for _ in range(n)])


def describe(T, content):
    A, C = arr(T, content)
    b = beta(A, C)
    lo, hi = boot_beta(A, C)
    nb = null_beta(A, C)
    mx = mixture(A, C)
    return {'n': len(T), 'beta': b, 'ci': [float(lo), float(hi)], 'z_null': float((b - nb.mean()) / nb.std()),
            'r': float(np.corrcoef(A, C)[0, 1]), 'sdC': float(np.std(C)), 'pi': mx[1]['pi'], 'mix': mx[1],
            'blank': float(np.mean([(t['blank_obv'] + t['blank_rev']) > 0 for t in T])),
            'blank_obv': float(np.mean([t['blank_obv'] > 0 for t in T])),
            'edge': float(np.mean([t['edge'] > 0 for t in T]))}


def perm_diff(T1, T2, content, n=1000):
    A1, C1 = arr(T1, content)
    A2, C2 = arr(T2, content)
    d = beta(A2, C2) - beta(A1, C1)
    A, C = np.concatenate([A1, A2]), np.concatenate([C1, C2])
    n1 = len(A1)
    ds = []
    for _ in range(n):
        p = rng.permutation(len(A))
        ds.append(beta(A[p[n1:]], C[p[n1:]]) - beta(A[p[:n1]], C[p[:n1]]))
    ds = np.array(ds)
    return float(d), float((np.abs(ds) >= abs(d)).mean())


def planted(P, pi_true, content='glyphs', s1=0.18):
    A, C = arr(P, content)
    a = np.mean(A) - np.mean(C)
    lab = rng.random(len(A)) < pi_true
    A2 = np.where(lab, a + 1.0 * C + rng.normal(0, s1, len(A)), rng.permutation(A))
    return A2, C, lab


def auc(score, lab):
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = lab.sum(); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    U = [t for t in ur3_tablets() if t['intact'] and t['n_lines'] >= 2]
    P = [t for t in pe_tablets() if t['intact'] and t['n_lines'] >= 2]
    res = {}
    D = [t for t in U if ur3_kind(t) == 'DAILY']
    Cm = [t for t in U if ur3_kind(t) == 'COMPILED']
    lo, hi = np.percentile([t['n_lines'] for t in D], [5, 95])
    Cm_m = [t for t in Cm if lo <= t['n_lines'] <= hi]
    for content in ['n_lines', 'glyphs']:
        res['ur3_daily_' + content] = describe(D, content)
        res['ur3_comp_' + content] = describe(Cm, content)
        res['ur3_comp_matched_' + content] = describe(Cm_m, content)
        res['ur3_diff_' + content] = perm_diff(D, Cm, content)
        res['ur3_diff_matched_' + content] = perm_diff(D, Cm_m, content)
        res['pe_' + content] = describe(P, content)
    # planted recovery
    pl = []
    for pi_true in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for rep in range(4):
            A2, C, lab = planted(P, pi_true)
            mx = mixture(A2, C, starts=8)
            pl.append({'pi_true': pi_true, 'beta': beta(A2, C), 'pi_hat': mx[1]['pi'],
                       'auc': auc(mx[2], lab)})
    res['planted'] = pl
    # same mixture on Ur III kinds for calibration of pi
    json.dump(res, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)

    def f(r):
        return 'beta %.2f [%.2f,%.2f] (null z %+.1f), r %.2f, pi %.2f, blank %.2f, obv-blank %.2f, edge %.2f, n %d' % (
            r['beta'], r['ci'][0], r['ci'][1], r['z_null'], r['r'], r['pi'], r['blank'], r['blank_obv'], r['edge'], r['n'])
    pls = {}
    for p in pl:
        pls.setdefault(p['pi_true'], []).append(p)
    pltxt = '; '.join('pi %.2f -> beta %.2f, pi_hat %.2f, AUC %s' % (
        k, np.mean([x['beta'] for x in v]), np.mean([x['pi_hat'] for x in v]),
        ('%.2f' % np.nanmean([x['auc'] for x in v])) if 0 < k < 1 else '-') for k, v in pls.items())
    rows = [
        ['PE-21.1a', 'Ur III control, DAILY single-day receipts (intact, CDLI dims): clay elasticity beta = slope log(h*w) on log(lines); within-kind reassignment null; 400 bootstraps; 2-component mixture',
         f(res['ur3_daily_n_lines']), 'reference "written on the spot"'],
        ['PE-21.1b', 'Ur III control, COMPILED (multi-day, month span, nig2-ka9-ak, grand total) same stats; label-permutation test of the beta difference; also COMPILED restricted to the DAILY line range',
         f(res['ur3_comp_n_lines']) + '; diff %.2f (p %.3f); range-matched: %s; diff %.2f (p %.3f)' % (
             res['ur3_diff_n_lines'][0], res['ur3_diff_n_lines'][1], f(res['ur3_comp_matched_n_lines']),
             res['ur3_diff_matched_n_lines'][0], res['ur3_diff_matched_n_lines'][1]),
         'see verdict in cycle summary'],
        ['PE-21.1c', 'Same on glyph counts (graphemes / signs + numeral impressions)',
         'DAILY ' + f(res['ur3_daily_glyphs']) + ' | COMPILED ' + f(res['ur3_comp_glyphs']) + ' | diff %.2f (p %.3f), matched diff %.2f (p %.3f)' % (
             res['ur3_diff_glyphs'] + res['ur3_diff_matched_glyphs']), ''],
        ['PE-21.1d', 'PLANTED: PE glyph counts with clay set to fit (area ~ glyphs, sd 0.18) for a share pi of tablets, real areas shuffled for the rest; 4 replicates per pi',
         pltxt, ''],
        ['PE-21.1e', 'PE (intact tablets, all genres): same statistics on lines and on glyphs',
         'lines: ' + f(res['pe_n_lines']) + ' | glyphs: ' + f(res['pe_glyphs']), ''],
    ]
    write_rows(os.path.join(CK, 'c1_rows.txt'), rows)
    for r in rows:
        print(' | '.join(r))


if __name__ == '__main__':
    main()
