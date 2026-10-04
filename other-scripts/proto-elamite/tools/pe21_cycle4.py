"""pe21 cycle 4: which PE tablets sit on a 'sized-to-fit' line, and what do they record?

Mixture with the planned slope FIXED at 1 (clay area proportional to text, the Ur III DAILY
behaviour): P: logA ~ N(a + logC, s1) vs U: logA ~ N(mu, s2) independent of text.  Fitted on
PE (glyphs, tablets with >= 5 lines to avoid the minimum-lump floor), Ur III DAILY and SUMMARY
(lines) and a planted PE corpus (pi 0.5).  PE tablets with posterior > 0.8 are compared with the
rest on system, office, seal, total, region; null = 2,000 label permutations within glyph quintiles.
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe21_common import pe_tablets, ur3_tablets, dom_sys, CK, write_rows
from pe21_cycle1 import arr, npdf, auc
from pe21_cycle2 import ur3_fine

rng = np.random.default_rng(214)


def mix1(A, C, starts=12, iters=300):
    best = None
    for s in range(starts):
        r = np.random.default_rng(s)
        pi, s1 = r.uniform(0.2, 0.8), r.uniform(0.1, 0.4)
        a = np.median(A - C) + r.normal(0, 0.2)
        mu, s2 = A.mean(), A.std()
        for _ in range(iters):
            p1 = pi * npdf(A, a + C, s1); p2 = (1 - pi) * npdf(A, mu, s2)
            g = p1 / (p1 + p2 + 1e-300)
            pi = float(np.clip(g.mean(), 1e-3, 1 - 1e-3))
            W = g.sum() + 1e-9
            a = float((g * (A - C)).sum() / W)
            s1 = max(float(np.sqrt((g * (A - a - C) ** 2).sum() / W)), 0.05)
            W2 = (1 - g).sum() + 1e-9
            mu = float(((1 - g) * A).sum() / W2)
            s2 = max(float(np.sqrt(((1 - g) * (A - mu) ** 2).sum() / W2)), 0.05)
        ll = float(np.log(pi * npdf(A, a + C, s1) + (1 - pi) * npdf(A, mu, s2) + 1e-300).sum())
        # likelihood of the U-only model (pi = 0) for a likelihood-ratio gain
        ll0 = float(np.log(npdf(A, A.mean(), A.std())).sum())
        if best is None or ll > best[0]:
            best = (ll, {'pi': pi, 'a': a, 's1': s1, 'mu': mu, 's2': s2, 'gain': ll - ll0}, g)
    return best


def main():
    res, rows = {}, []
    P = [t for t in pe_tablets() if t['intact'] and t['n_lines'] >= 5]
    A, C = arr(P, 'glyphs')
    ll, par, g = mix1(A, C)
    res['pe'] = par
    U = [t for t in ur3_tablets() if t['intact'] and t['n_lines'] >= 5]
    for k in ['DAILY', 'SUMMARY', 'RUNNING']:
        T = [t for t in U if ur3_fine(t) == k]
        Au, Cu = arr(T, 'n_lines')
        res['ur3_' + k] = mix1(Au, Cu)[1]
        res['ur3_' + k]['n'] = len(T)
    # planted
    pl = []
    for rep in range(5):
        lab = rng.random(len(A)) < 0.5
        A2 = np.where(lab, np.median(A - C) + C + rng.normal(0, 0.2, len(A)), rng.permutation(A))
        _, pp, gg = mix1(A2, C, starts=6)
        pl.append((pp['pi'], auc(gg, lab)))
    res['planted'] = pl
    # shuffled control on real PE (texts re-dealt): pi should collapse
    sh = [mix1(A, rng.permutation(C), starts=6)[1]['pi'] for _ in range(5)]
    res['shuffled_pi'] = sh
    hi = g > 0.8
    res['n_hi'] = int(hi.sum())
    bins = np.digitize(C, np.percentile(C, [20, 40, 60, 80]))
    feats = {'system': [dom_sys(t) for t in P], 'office': [t['office'] for t in P],
             'sealed': [str(t['sealed']) for t in P], 'total': [str(bool(t['n_tot'])) for t in P],
             'region': [t['region'] for t in P]}
    comp = {}
    for k, lab in feats.items():
        lab = np.array(lab)
        for v in sorted(set(lab)):
            m = lab == v
            if m.sum() < 10:
                continue
            obs = (m & hi).sum() / max(hi.sum(), 1)
            nul = []
            for _ in range(2000):
                hp = hi.copy()
                for b in range(5):
                    ii = np.where(bins == b)[0]
                    hp[ii] = rng.permutation(hp[ii])
                nul.append((m & hp).sum() / max(hp.sum(), 1))
            nul = np.array(nul)
            comp['%s=%s' % (k, v)] = (round(float(obs), 3), round(float(nul.mean()), 3),
                                      round(float((obs - nul.mean()) / (nul.std() + 1e-9)), 1))
    res['composition'] = comp
    top = sorted(zip(g, [t['id'] for t in P], [t['n_lines'] for t in P], [dom_sys(t) for t in P],
                     [t['office'] for t in P]), reverse=True)
    res['top'] = [(round(float(a), 3), b, c, d, e) for a, b, c, d, e in top[:25]]
    json.dump(res, open(os.path.join(CK, 'c4.json'), 'w'), indent=1)
    sig = {k: v for k, v in comp.items() if abs(v[2]) >= 2.5}
    rows.append(['PE-21.4a', 'Fixed-slope (area proportional to text) mixture: Ur III DAILY / SUMMARY / RUNNING (>= 5 lines, lines); PE (>= 5 lines, glyphs); planted PE pi 0.5 (5 reps); real PE with texts re-dealt (5)',
                 'Ur III DAILY pi %.2f (gain %.0f), SUMMARY pi %.2f (gain %.0f), RUNNING pi %.2f (gain %.0f); PE pi %.2f (gain %.0f, n %d); planted pi_hat %s, AUC %s; re-dealt PE pi %s' % (
                     res['ur3_DAILY']['pi'], res['ur3_DAILY']['gain'], res['ur3_SUMMARY']['pi'], res['ur3_SUMMARY']['gain'],
                     res['ur3_RUNNING']['pi'], res['ur3_RUNNING']['gain'], par['pi'], par['gain'], len(P),
                     [round(x[0], 2) for x in pl], [round(x[1], 2) for x in pl], [round(x, 2) for x in sh]), ''])
    rows.append(['PE-21.4b', 'What the on-line PE tablets (posterior > 0.8) record: share of each system/office/seal/total/region among them vs label permutation within glyph quintiles (2,000)',
                 '%d of %d tablets on the line; |z| >= 2.5: %s; all: %s' % (res['n_hi'], len(P), sig or 'none',
                                                                         '; '.join('%s %.2f vs %.2f (z %+.1f)' % (k, *v) for k, v in comp.items())), ''])
    rows.append(['PE-21.4c', 'Top PE tablets by posterior (id, lines, system, office)', '; '.join('%s %.2f %dl %s %s' % (b, a, c, d, e) for a, b, c, d, e in res['top'][:12]), ''])
    write_rows(os.path.join(CK, 'c4_rows.txt'), rows)
    for r in rows:
        print(' | '.join(r))


if __name__ == '__main__':
    main()
