"""v95 cycle 2: A HIDDEN MACHINE / A CONSERVED QUANTITY.  If words are moves of a machine or entries of a balance, some signed
weighting of the glyphs sums to (nearly) the same value over every line (or paragraph, or page): the unit totals are
UNDER-dispersed relative to slot-swapped twins (same slot word laws, same line lengths, co-ordination destroyed).
R(c) = c' S_real c / c' S_null c (S = covariance of unit totals of glyph x position counts).  fit: smallest generalised
eigenvector on train folios (null = twins 0-3), tested on held-out folios against twins 4-7.  guess: random signed integer
weights; also a register-machine legality score (prefix sums never below 0 and the line ends at 0).
usage: python3 v95_c2.py CORPUS [NHYP]
"""
import sys, os, json, random, time
import numpy as np
from scipy.linalg import eigh
import v95_lib as L

UNITS = ['line', 'para', 'page']
NT = 8


def totals(pages, unit, h):
    S = []
    for p in pages:
        if L.half(p['id']) != h: continue
        cur = None
        for l in p['lines']:
            v = sum(L.wvec(w) for w in l['w'])
            if unit == 'line': S.append(v); continue
            if unit == 'para' and l['ps'] and cur is not None: S.append(cur); cur = None
            cur = v if cur is None else cur + v
        if unit != 'line' and cur is not None: S.append(cur)
    S = np.array(S)
    return np.cov(S.T), len(S)


def lines_flat(pages, h):
    X = []; ends = []
    for p in pages:
        if L.half(p['id']) != h: continue
        for l in p['lines']:
            for w in l['w']: X.append(L.wvec(w))
            ends.append(len(X))
    return np.array(X), np.array(ends)


def hypotheses(n, seed=952):
    rng = random.Random(seed); H = []
    for i in range(n):
        unit = rng.choice(UNITS); m = rng.randint(2, 10); gs = sorted(rng.sample(range(L.K - 1), m))
        split = rng.random() < 0.4
        if split:
            cols = sorted(rng.sample([pos * L.K + g for g in gs for pos in range(3)], rng.randint(m, 3 * m)))
            groups = [[c] for c in cols]
        else:
            groups = [[pos * L.K + g for pos in range(3)] for g in gs]
        mode = 'fit' if rng.random() < 0.6 else 'guess'
        vals = [rng.choice([-2, -1, -1, 1, 1, 2]) for _ in groups]
        H.append(dict(i=i, unit=unit, groups=groups, mode=mode, vals=vals))
    return H


def proj(groups):
    P = np.zeros((L.F, len(groups)))
    for j, g in enumerate(groups):
        for c in g: P[c, j] = 1
    return P


def legal(Xf, ends, c):
    """share of lines whose running total never goes below 0 and ends at 0 (a cleared board)."""
    v = Xf @ c; cs = np.cumsum(v); starts = np.r_[0, ends[:-1]]
    base = np.r_[0, cs][starts]
    ok = 0; n = 0
    for s, e, b in zip(starts, ends, base):
        if e - s < 2: continue
        seg = cs[s:e] - b; n += 1
        ok += (seg.min() >= -1e-9) and abs(seg[-1]) < 1e-9
    return ok / max(n, 1)


def run(name, nh):
    out = os.path.join(L.CK, 'c2_%s.json' % name)
    if os.path.exists(out): return
    t0 = time.time(); C = L.corpus(name)
    vs = [C] + [L.slot_swap(C, 9700 + s) for s in range(NT)]
    Sg = {(u, h): [totals(v, u, h) for v in vs] for u in UNITS for h in (0, 1)}
    LF = [lines_flat(v, 1) for v in vs]
    rows = []
    for hyp in hypotheses(nh):
        P = proj(hyp['groups']); r = {'i': hyp['i']}
        S0 = [P.T @ s @ P for s, n in Sg[(hyp['unit'], 0)]]; S1 = [P.T @ s @ P for s, n in Sg[(hyp['unit'], 1)]]
        if Sg[(hyp['unit'], 0)][0][1] < 20 or Sg[(hyp['unit'], 1)][0][1] < 20: rows.append(r); continue
        if hyp['mode'] == 'fit':
            Nf = sum(S0[1:5]) / 4.0
            keep = np.diag(Nf) > 1e-6 * max(np.trace(Nf), 1e-9)
            if keep.sum() < 1: rows.append(r); continue
            rr = 1e-3 * np.trace(Nf[np.ix_(keep, keep)]) / keep.sum()
            try:
                w, U = eigh(S0[0][np.ix_(keep, keep)] + rr * np.eye(keep.sum()), Nf[np.ix_(keep, keep)] + rr * np.eye(keep.sum()))
            except Exception:
                rows.append(r); continue
            c = np.zeros(len(keep)); c[keep] = U[:, 0]
            r['Rtr'] = float((c @ S0[0] @ c) / (c @ Nf @ c))
        else:
            c = np.array(hyp['vals'], float)
            Nf = sum(S0[1:5]) / 4.0
            r['Rtr'] = float((c @ S0[0] @ c) / max(c @ Nf @ c, 1e-12))
        tv = [float(c @ s @ c) for s in S1]
        r['te'] = tv                                   # real, twins 0-7 (test half); twins 4-7 are the test null
        r['c'] = [round(float(x), 4) for x in c]
        if hyp['mode'] == 'guess' and hyp['unit'] == 'line':
            cf = P @ c
            r['legal'] = [legal(X, e, cf) for X, e in LF]
        rows.append(r)
    json.dump(dict(name=name, nh=nh, rows=rows, sec=time.time() - t0), open(out, 'w'))
    print(name, 'done', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5000)
