"""v34 cycle 3b: the within-line glyph gradient found in cycle 3 (a, r, p up and e, k, sh, q
down toward the right of the line, in non-final words): (i) does it grow with crowding
(per-line right-minus-middle rate vs tightness, within-page shuffle null)? (ii) is it a
position-in-x effect or a words-from-the-end effect (same-index words compared between
lines of different reach)? Planted control: a gradient that is purely x-driven."""
import sys, os, json
import numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v34_lib as V
GL = ['a', 'r', 'p', 'e', 'k', 'sh', 'q', 'm', 'g']

if __name__ == '__main__':
    vo = [l for l in V.voynich_lines() if l['ok'] and len(l['words']) >= 5]
    F, T, G = {g: [] for g in GL}, [], []
    X = {g: [] for g in GL}
    for l in vo:
        n = len(l['words']); span = l['x1'][-1] - l['x0'][0]
        c = {0: Counter(), 1: Counter(), 2: Counter()}
        for k in range(1, n - 1):
            x = ((l['x0'][k] + l['x1'][k]) / 2 - l['x0'][0]) / span
            c[min(2, int(3 * x))].update(V.glyphs(l['words'][k]))
        n1, n2 = sum(c[1].values()), sum(c[2].values())
        if not n1 or not n2:
            continue
        for g in GL:
            F[g].append(c[2][g] / n2 - c[1][g] / n1)
        T.append(-l['slack']); G.append(l['page'])
    o = V.multi_perm(T, F, G, 2000)
    print('(i) right-minus-middle glyph rate vs tightness:', {k: (round(v['z'], 2), round(v['p_fw'], 3)) for k, v in o.items()})
    # (ii) penultimate word: rate of glyph in the penultimate word vs its absolute x (page-relative)
    rows = []
    for l in vo:
        pen = l['words'][-2]; gs = V.glyphs(pen)
        xr = ((l['x0'][-2] + l['x1'][-2]) / 2 - l['L']) / (l['R'] - l['L'])
        rows.append((xr, gs, l['page'], len(l['words'])))
    Fx = {g: [gs.count(g) / len(gs) for _, gs, _, _ in rows] for g in GL}
    o2 = V.multi_perm([r[0] for r in rows], Fx, [r[2] for r in rows], 2000, covar=np.array([[r[3] for r in rows]], float))
    print('(ii) penultimate-word glyph rate vs its x position (fixed word index, nwords covariate):',
          {k: (round(v['z'], 2), round(v['p_fw'], 3)) for k, v in o2.items()})
    json.dump({'crowding': o, 'pen_x': o2}, open(os.path.join(V.CKPT, 'cycle3b.json'), 'w'), indent=1)
