"""v13 cycle 2: drift-proof tests of text-follows-drawing.

A. Offset profile: partial r between page i's text and the image of page i+k (same section,
   folio order) for k = -8..8. Text that tracks its own drawing gives a spike at k = 0;
   drift gives a smooth hump.
B. Same-leaf contrast: for leaves whose recto and verso both have text and an image, set the
   recto-minus-verso text difference against the recto-minus-verso image difference. Same
   parchment, same session, same section/hand mostly, so drift cancels. Null = random sign
   flip of the image difference per leaf (exact-style, 5000x), family-wise by max |r|.
Controls: same contrast with the image differences of the NEXT leaf (wrong leaf, close
in order); planted token texts whose length / qo-share follow the image (must pass);
planted pure-drift texts (length follows folio position only; must fail).
"""
import json, os, sys, math, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v13_lib import *
from v13_cycle1 import build, planted, sec_lang
from collections import Counter, defaultdict

OUTD = os.path.join(DER, 'v13')
IF, TF = IMG_FEATS, TXT_FEATS


def order_key(fol):
    m = re.match(r'f(\d+)([rv])', fol)
    return int(m.group(1)) * 2 + (m.group(2) == 'v')


def mats(rows, tf=None):
    tf = tf or TF
    X = np.log1p(np.maximum(np.array([[r['img'][f] for f in IF] for r in rows]), 0) * 100)
    Y = np.array([[r['txt'][t] for t in tf] for r in rows], float)
    return X, Y


def corr_cols(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-12); B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    return A.T @ B / len(A)


def leaf_pairs(rows, shift=0):
    by = {r['folio']: r for r in rows}
    leaves = sorted(set(int(re.match(r'f(\d+)', r['folio']).group(1)) for r in rows))
    pairs = []
    for L in leaves:
        a, b = by.get('f%dr' % L), by.get('f%dv' % L)
        if a and b and a['meta']['illus'] == b['meta']['illus']:
            pairs.append((a, b))
    if shift:
        imgs = [(p[0]['img'], p[1]['img']) for p in pairs]
        pairs = [((dict(a, img=imgs[(i + shift) % len(pairs)][0])), dict(b, img=imgs[(i + shift) % len(pairs)][1]))
                 for i, (a, b) in enumerate(pairs)]
    return pairs


def leaf_test(pairs, nperm=5000, seed=0, tf=None):
    tf = tf or TF
    ok = [p for p in pairs if all(np.isfinite(p[0]['txt'][t]) and np.isfinite(p[1]['txt'][t]) for t in tf)]
    Xa, _ = mats([p[0] for p in ok], tf); Xb, _ = mats([p[1] for p in ok], tf)
    Ya = np.array([[p[0]['txt'][t] for t in tf] for p in ok]); Yb = np.array([[p[1]['txt'][t] for t in tf] for p in ok])
    dX, dY = Xa - Xb, Ya - Yb
    # no centring: under H0 (exchangeable sides) differences are symmetric around 0
    def r_nc(A, B):
        return (A.T @ B) / np.sqrt(np.outer((A ** 2).sum(0), (B ** 2).sum(0)) + 1e-12)
    R = r_nc(dX, dY)
    rng = np.random.default_rng(seed)
    mx = []; ex = np.zeros_like(R)
    for _ in range(nperm):
        s = rng.choice([-1.0, 1.0], size=len(ok))[:, None]
        Rn = r_nc(dX * s, dY)
        ex += np.abs(Rn) >= np.abs(R); mx.append(np.abs(Rn).max())
    mx = np.array(mx)
    P = (ex + 1) / (nperm + 1)
    Pfw = np.vectorize(lambda v: (np.sum(mx >= abs(v)) + 1) / (nperm + 1))(R)
    return dict(n=len(ok), R=R, P=P, Pfw=Pfw, max95=float(np.quantile(mx, .95)))


def top(res, k=6, tf=None):
    tf = tf or TF
    R = res['R']
    idx = sorted(((abs(R[i, j]), i, j) for i in range(R.shape[0]) for j in range(R.shape[1])), reverse=True)[:k]
    return ['%s~%s r=%+.2f p=%.4f pFW=%.3f' % (IF[i], tf[j], R[i, j], res['P'][i, j], res['Pfw'][i, j]) for _, i, j in idx]


if __name__ == '__main__':
    rows, gc = build()
    rows.sort(key=lambda r: order_key(r['folio']))
    log = []
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s); log.append(s)
    # ---- A. offset profile within section x language (folio order)
    groups = defaultdict(list)
    for r in rows:
        groups[sec_lang(r)].append(r)
    cells = [('draw_edges', 'g_chsh'), ('draw_bytes', 'g_chsh'), ('draw_ink', 'g_m'), ('draw_edges', 'log_words')]
    prof = {}
    for k in range(-8, 9):
        Xs, Ys, metas = [], [], []
        for g in groups.values():
            n = len(g)
            if n < 12:
                continue
            for i, r in enumerate(g):
                j = i + k
                if 0 <= j < n:
                    Xs.append(g[j]); Ys.append(r); metas.append(r['meta'])
        X, _ = mats(Xs); _, Y = mats(Ys)
        Z = design(metas)
        okc = np.all(np.isfinite(Y), axis=1)
        R = partial_r_matrix(Z[okc], X[okc], Y[okc])
        prof[k] = (R, int(okc.sum()))
    P('A. OFFSET PROFILE (partial r, text of page i vs image of page i+k, same section x lang)')
    for a, b in cells:
        i, j = IF.index(a), TF.index(b)
        P('  %s~%s: ' % (a, b) + ' '.join('%+d:%+.2f' % (k, prof[k][0][i, j]) for k in range(-8, 9)))
    tot = {k: float(np.sqrt((prof[k][0] ** 2).mean())) for k in prof}
    P('  RMS r over all 378 cells: ' + ' '.join('%+d:%.3f' % (k, tot[k]) for k in range(-8, 9)))
    # ---- B. same-leaf contrast
    pairs = leaf_pairs(rows)
    P('B. SAME-LEAF CONTRAST: %d leaves (sections %s)' % (len(pairs), dict(Counter(p[0]['meta']['illus'] for p in pairs))))
    res = leaf_test(pairs, seed=1)
    P('  real: max|r|95=%.3f cells p<0.01: %d/%d' % (res['max95'], int((res['P'] < 0.01).sum()), res['P'].size))
    for s in top(res, 8): P('    ', s)
    json.dump({'R': res['R'].tolist(), 'P': res['P'].tolist(), 'Pfw': res['Pfw'].tolist()}, open(os.path.join(OUTD, 'c2_leaf.json'), 'w'))
    hp = [p for p in pairs if p[0]['meta']['illus'] == 'H']
    resh = leaf_test(hp, seed=2)
    P('  herbal leaves n=%d: max|r|95=%.3f cells p<0.01: %d' % (resh['n'], resh['max95'], int((resh['P'] < 0.01).sum())))
    for s in top(resh, 5): P('    ', s)
    # wrong-leaf control
    for sh in (1, 2, 5):
        rw = leaf_test(leaf_pairs(rows, shift=sh), seed=10 + sh)
        P('  WRONG LEAF (image diffs of leaf +%d): cells p<0.01: %d/%d, min pFW %.3f' % (sh, int((rw['P'] < 0.01).sum()), rw['P'].size, rw['Pfw'].min()))
    # planted: image-driven texts
    for bl, bq in ((0.2, 0.3), (0.3, 0.45), (0.5, 0.7)):
        hl = hq = 0; rl = []; rq = []
        for t in range(6):
            pr = planted(rows, gc, bl, bq, seed=t)
            rp = leaf_test(leaf_pairs(pr), nperm=1000, seed=20 + t)
            i1, j1 = IF.index('draw_edges'), TF.index('log_words'); i2, j2 = IF.index('frac_green'), TF.index('w_qo')
            rl.append(rp['R'][i1, j1]); rq.append(rp['R'][i2, j2])
            hl += rp['Pfw'][i1, j1] < .05; hq += rp['Pfw'][i2, j2] < .05
        P('  PLANTED image-driven beta_len=%.2f beta_q=%.2f: len r=%.2f FW-detect %d/6; q r=%.2f FW-detect %d/6' % (bl, bq, np.mean(rl), hl, np.mean(rq), hq))
    # planted: pure drift (text length follows folio position, not the image)
    pos = np.array([order_key(r['folio']) for r in rows], float); pos = (pos - pos.mean()) / pos.std()
    fp = 0
    for t in range(6):
        r_ = random.Random(t)
        vocab = [w for w in gc for _ in range(gc[w])]
        dr = []
        for k, r in enumerate(rows):
            n = max(5, int(round(len(r['P']) * math.exp(0.6 * math.sin(3 * pos[k])))))
            ws = [r_.choice(vocab) for _ in range(n)]
            dr.append(dict(r, txt=text_features(ws, r['L'], max(1, n // 9), gc, r_)))
        rp = leaf_test(leaf_pairs(dr), nperm=1000, seed=40 + t)
        fp += rp['Pfw'].min() < .05
    P('  PLANTED pure drift: any cell FW<0.05 in %d/6 trials' % fp)
    open(os.path.join(OUTD, 'c2_log.txt'), 'w').write('\n'.join(log) + '\n')
