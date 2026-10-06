"""v81 cycle 3: (a) JUNCTION-PRESERVING NULL. Does the onset (or the 2nd glyph) of word i predict the onset of word
i+1 beyond word i's last glyph? Null: the context glyph of word i is re-drawn from another line-interior token of the
same section with the same last glyph (20 re-rolls), so the ending->next-onset junction and every marginal stay
exact and only the front-of-word link is broken. Held-out (leaf 0 -> leaf 1) gain of p(o_i+1 | end_i, ctx_i) over
p(o_i+1 | end_i).
(b) ONSET PHRASES. If initials spell a text, onset n-grams (n = 4..7, inside lines) recur across pages like phrases.
Count distinct n-grams of the leaf-1 half seen on >= 2 pages, real vs within-line shuffle and same-position swap.
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v81_lib as L
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool

DEFS = [dict(v='e1c_line', k=1, part=None, m=None), dict(v='noq_line', k=1, part=None, m=None),
        dict(v='e1c_line', k=2, part=None, m=None)]
CC = None


def init():
    global CC
    CC = L.pload('comp.pkl')


def ctx_arrays(c, d, slot):
    if slot == 'on': return L.onset(c, d)
    G = c.G[d['v']]   # glyph 2 (index 1) of the variant string
    _, x = np.unique(G[:, 1].astype(np.int64), return_inverse=True); return x


def gain(c, o, ctx, tr, te):
    K = int(max(o.max(), ctx.max())) + 2
    i1 = c.p1; a, b, e = ctx[i1], o[i1 + 1], c.end[i1]
    mtr, mte = tr[i1], te[i1]
    uc = np.bincount(o[tr], minlength=K).astype(float) + 0.5; pu = uc / uc.sum()
    he, pe = L._ce_cond(e[mtr], b[mtr], e[mte], b[mte], pu[b[mte]], K)
    hea, _ = L._ce_cond(e[mtr] * K + a[mtr], b[mtr], e[mte] * K + a[mte], b[mte], pe, K)
    return he.mean() - hea.mean()


def reroll(c, x, rng):
    y = x.copy(); key = c.sec * 1000 + c.end
    inner = c.pos >= 1
    for kk in np.unique(key[inner]):
        ix = np.nonzero((key == kk) & inner)[0]
        y[ix] = x[ix[rng.permutation(len(ix))]]
    return y


def phrases(c, o, te, n):
    toks = np.nonzero(te)[0]
    seen = {}
    ln, pg = c.ln, c.pg
    for i in toks:
        j = i + n - 1
        if j >= c.n or ln[j] != ln[i] or not te[j]: continue
        g = tuple(o[i:j + 1]); seen.setdefault(g, set()).add(pg[i])
    return sum(len(v) >= 2 for v in seen.values()), len(seen)


def work(name):
    c = CC[name]; tr, te = L.masks(c, 'hold'); rng = np.random.default_rng(813)
    out = {}
    for di, d in enumerate(DEFS):
        o = L.onset(c, d)
        for slot in ('on', 'g2'):
            ctx = ctx_arrays(c, d, slot)
            r = gain(c, o, ctx, tr, te)
            nul = [gain(c, o, reroll(c, ctx, rng), tr, te) for _ in range(20)]
            out[(di, slot)] = (r, float(np.mean(nul)), float(np.std(nul)))
        for n in (4, 5, 6, 7):
            real = phrases(c, o, te, n)
            w = [phrases(c, o[P], te, n) for P in c.perm[:2]]
            cc = [phrases(c, o[P], te, n) for P in c.cperm[:2]]
            out[(di, 'ph', n)] = (real, w, cc)
    return name, out


if __name__ == '__main__':
    names = list(L.pload('comp.pkl'))
    res = {}
    with Pool(2, initializer=init) as P:
        for name, out in P.imap_unordered(work, names):
            res[name] = out
            s = []
            for di in range(len(DEFS)):
                for slot in ('on', 'g2'):
                    r, m, sd = out[(di, slot)]; s.append('d%d%s %+.4f (z %.1f)' % (di, slot, r - m, (r - m) / (sd + 1e-9)))
                for n in (4, 6):
                    real, w, cc = out[(di, 'ph', n)]
                    s.append('d%d ph%d %d/%d w%.0f c%.0f' % (di, n, real[0], real[1], np.mean([x[0] for x in w]), np.mean([x[0] for x in cc])))
            print(name, ' | '.join(s), flush=True)
    pickle.dump(res, open(os.path.join(L.CK, 'c3.pkl'), 'wb'))
