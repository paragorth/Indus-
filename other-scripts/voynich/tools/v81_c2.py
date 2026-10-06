"""v81 cycle 2: IS THE REST OF THE WORD SURFACE?  The initials hypothesis predicts that once the onset is fixed the rest
of the word (tail) is chosen without reference to page, neighbours or the next word. Re-roll test: inside each
(section, onset class) pool, tails are permuted across tokens (20 re-rolls) - this is the hypothesis' own generator.
Statistics on the real vs re-rolled text (whole corpus, plug-in, top-60 tail types + other):
  S1 page information of the tail      I(page; tail)            [given onset via the pool]
  S2 tail-to-tail coupling in a line   I(tail_i; tail_i+1)
  S3 tail-to-next-onset (junction)     I(tail_i; onset_i+1)
  S4 onset-to-next-tail                I(onset_i; tail_i+1)
For each: excess bits = real - mean(reroll), z.  Also the slot inversion: same random-definition scoring as cycle 1
(lag1+skip+tri, held out) on onset, middle and coda slots."""
import os, sys, json, random, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v81_lib as L
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool

FN = 'v81_cycle2.txt'
DEFS = [dict(v='e1c_line', k=1, part=None, m=None), dict(v='e1c_line', k=2, part=None, m=None),
        dict(v='noq_line', k=1, part=None, m=None)]
CC = None


def init():
    global CC
    CC = L.pload('comp.pkl')


def strings(c, v):
    import v72_lib as V
    out = []
    for t, w in enumerate(c.W):
        out.append(L.variant(w, c.pos[t] == 0, False, v))
    return out


def tail_ids(S, k, top=60):
    from collections import Counter
    T = [s[k:] or '$' for s in S]
    cnt = Counter(T); keep = {x: i for i, (x, _) in enumerate(cnt.most_common(top))}
    return np.array([keep.get(x, top) for x in T])


def mi(x, y):
    return L._mi(x, y)


def stats(c, o, t, inner):
    i1 = c.p1
    return dict(S1=mi(c.pg[inner], t[inner]), S2=mi(t[i1], t[i1 + 1]), S3=mi(t[i1], o[i1 + 1]), S4=mi(o[i1], t[i1 + 1]))


def work(args):
    name, di = args
    c = CC[name]; d = DEFS[di]
    S = strings(c, d['v'])
    o = L.onset(c, d); t = tail_ids(S, d['k'])
    inner = c.pos >= 0
    real = stats(c, o, t, inner)
    rng = np.random.default_rng(812)
    key = c.sec * 100000 + o
    R = []
    for r in range(20):
        tt = t.copy()
        for kk in np.unique(key):
            ix = np.nonzero(key == kk)[0]
            tt[ix] = t[ix[rng.permutation(len(ix))]]
        R.append(stats(c, o, tt, inner))
    res = {}
    for s in real:
        v = np.array([x[s] for x in R])
        res[s] = (real[s] - v.mean(), (real[s] - v.mean()) / (v.std() + 1e-9))
    return name, di, res


def slot_feats(c, d, slot, stage):
    """onset definition applied to a different slot of the variant string: 'on' first k, 'mid' units 1..k,
    'coda' last k units."""
    import v72_lib as V
    A = c.G[d['v']]
    if slot == 'on':
        o = L.onset(c, d)
    else:
        S = strings(c, d['v'])
        M = np.full((c.n, 3), L.AIX['$'], np.int16)
        for i, s in enumerate(S):
            x = s[1:1 + d['k']] if slot == 'mid' else s[-d['k']:]
            for j, ch in enumerate(x): M[i, j] = L.AIX.get(ch, L.AIX['_'])
        G0 = c.G[d['v']]; c.G[d['v']] = M
        o = L.onset(c, d); c.G[d['v']] = G0
    tr, te = L.masks(c, stage)
    f = L.stream_feats(c, o, tr, te)
    return f


def work_slot(args):
    i, d = args
    out = {}
    for name in CC:
        out[name] = {s: slot_feats(CC[name], d, s, 'hold') for s in ('on', 'mid', 'coda')}
    return i, out


if __name__ == '__main__':
    names = list(L.pload('comp.pkl'))
    jobs = [(n, di) for n in names for di in range(len(DEFS))]
    res = {}
    with Pool(2, initializer=init) as P:
        for n, di, r in P.imap_unordered(work, jobs):
            res.setdefault(n, {})[di] = r
            print(n, di, ' '.join('%s %+.4f (z %.1f)' % (k, a, b) for k, (a, b) in r.items()), flush=True)
        rng = random.Random(8120)
        defs = [L.random_def(rng) for _ in range(200)]
        slot = {}
        for i, out in P.imap_unordered(work_slot, list(enumerate(defs))):
            slot[i] = out
            if i % 20 == 0: print('slot', i, flush=True)
    pickle.dump(dict(reroll=res, slot=slot, defs=defs), open(os.path.join(L.CK, 'c2.pkl'), 'wb'))
