"""v81 cycle 3b: is the front-of-word link adjacency or line/page mood? Two tighter nulls for V-81.3a:
(c) context re-drawn only from the SAME PAGE with the same last glyph; (d) lag-2 control: the context is the onset
of word i-1 (with word i's ending kept), which shares the line mood but is not adjacent to word i+1."""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v81_lib as L, v81_c3 as C3
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool
CC = None
def init():
    global CC
    CC = L.pload('comp.pkl')

def reroll_page(c, x, rng):
    y = x.copy(); key = c.pg * 1000 + c.end; inner = c.pos >= 1
    for kk in np.unique(key[inner]):
        ix = np.nonzero((key == kk) & inner)[0]
        y[ix] = x[ix[rng.permutation(len(ix))]]
    return y

def gain_lag(c, o, ctx, tr, te, lag):
    """predict o[i+1] from end[i] and ctx[i+1-lag]; pairs need i+1-lag in the same line and >= position 1."""
    K = int(max(o.max(), ctx.max())) + 2
    i1 = c.p1
    j = i1 + 1 - lag
    ok = (j >= 0) & (c.ln[np.maximum(j, 0)] == c.ln[i1]) & (c.pos[np.maximum(j, 0)] >= 1)
    i1 = i1[ok]; j = j[ok]
    a, b, e = ctx[j], o[i1 + 1], c.end[i1]
    mtr, mte = tr[i1], te[i1]
    uc = np.bincount(o[tr], minlength=K).astype(float) + 0.5; pu = uc / uc.sum()
    he, pe = L._ce_cond(e[mtr], b[mtr], e[mte], b[mte], pu[b[mte]], K)
    hea, _ = L._ce_cond(e[mtr] * K + a[mtr], b[mtr], e[mte] * K + a[mte], b[mte], pe, K)
    return he.mean() - hea.mean()

def work(name):
    c = CC[name]; tr, te = L.masks(c, 'hold'); rng = np.random.default_rng(814); out = {}
    for di, d in enumerate(C3.DEFS):
        o = L.onset(c, d)
        r1 = C3.gain(c, o, o, tr, te)
        nulp = [C3.gain(c, o, reroll_page(c, o, rng), tr, te) for _ in range(20)]
        # lag-2 restricted to the same pair set as lag 1 (i >= 2 in line) for a fair comparison
        g1 = gain_lag(c, o, o, tr, te, 1); g2 = gain_lag(c, o, o, tr, te, 2)
        g1r = np.mean([gain_lag(c, o, C3.reroll(c, o, rng), tr, te, 1) for _ in range(10)])
        g2r = np.mean([gain_lag(c, o, C3.reroll(c, o, rng), tr, te, 2) for _ in range(10)])
        out[di] = dict(page=(r1 - np.mean(nulp), (r1 - np.mean(nulp)) / (np.std(nulp) + 1e-9)), lag1=g1 - g1r, lag2=g2 - g2r)
    return name, out

if __name__ == '__main__':
    names = list(L.pload('comp.pkl')); res = {}
    with Pool(2, initializer=init) as P:
        for name, out in P.imap_unordered(work, names):
            res[name] = out
            print(name, ' | '.join('d%d samepage %+.4f (z %.1f) lag1 %+.4f lag2 %+.4f' % (di, v['page'][0], v['page'][1], v['lag1'], v['lag2']) for di, v in out.items()), flush=True)
    pickle.dump(res, open(os.path.join(L.CK, 'c3b.pkl'), 'wb'))
