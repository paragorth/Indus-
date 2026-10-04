"""v9 cycle 3: does the best-fitting volvelle reproduce the Voynich LINE structure?

Generate text with the Voynich held-out line lengths from each fitted model, and measure
six line statistics on real held-out lines and on each generated corpus (10 replicates):
  J_in    MI(suffix of word t ; prefix of word t+1) inside a line   (junction coupling)
  J_brk   same across the line break (last word -> first word of next line)
  OPEN    JSD(prefix distribution of line-first word || other words)
  CLOSE   JSD(suffix distribution of line-last word || other words)
  WW      MI(prefix ; suffix) inside one word
  CLUMP   per-line count of the commonest prefix symbol: variance / binomial variance
Models: V1 volvelle (independent rings, rule R1, reset), V2 volvelle (rule R2: step depends
on previous word's suffix class), MX per-ring Markov-1 x previous-suffix class, W2 word
trigram on filler tuples.  MI values are plug-in minus the mean of 20 within-corpus shuffles.
"""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
from v9_lib import *
import v9_cycle1 as C1
import v9_cycle2 as C2

OUT = C1.OUT
K, M = 3, 12
S = M + 1


def mi(x, y, shuffles=20, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    def plug(a, b):
        J = np.zeros((S + 1, S + 1)); np.add.at(J, (a, b), 1); J /= J.sum()
        px, py = J.sum(1), J.sum(0); m = J > 0
        return float((J[m] * np.log2(J[m] / np.outer(px, py)[m])).sum())
    rng = np.random.RandomState(seed)
    base = np.mean([plug(x, rng.permutation(y)) for _ in range(shuffles)])
    return plug(x, y) - base


def jsd(a, b):
    p = np.bincount(a, minlength=S + 1) + 0.5; q = np.bincount(b, minlength=S + 1) + 0.5
    p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    kl = lambda u, v: float((u * np.log2(u / v)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def stats(O):
    L = [[tuple(O[l, t]) for t in range(O.shape[1]) if O[l, t, 0] >= 0] for l in range(O.shape[0])]
    L = [x for x in L if x]
    si, pi_ = [], []
    for x in L:
        for a, b in zip(x[:-1], x[1:]):
            si.append(a[2]); pi_.append(b[0])
    sb = [L[i][-1][2] for i in range(len(L) - 1)]; pb = [L[i + 1][0][0] for i in range(len(L) - 1)]
    first = [x[0][0] for x in L]; rest = [w[0] for x in L for w in x[1:]]
    last = [x[-1][2] for x in L if len(x) > 1]; nonlast = [w[2] for x in L for w in x[:-1]]
    allw = [w for x in L for w in x]
    top = Counter(w[0] for w in allw).most_common(1)[0][0]
    p = np.mean([w[0] == top for w in allw])
    cnt = np.array([sum(w[0] == top for w in x) for x in L]); n = np.array([len(x) for x in L])
    clump = float(((cnt - n * p) ** 2).sum() / (n * p * (1 - p)).sum())
    return dict(J_in=mi(si, pi_), J_brk=mi(sb, pb), OPEN=jsd(first, rest), CLOSE=jsd(last, nonlast),
                WW=mi([w[0] for w in allw], [w[2] for w in allw]), CLUMP=clump)


# ---------------------------------------------------------------- generators
def gen_volvelle(rings, lens, rng, rule2=False):
    """rings: list of (lab, q (C,n), pi); rule2: kernel class = previous word's suffix class."""
    T = int(max(lens)); O = -np.ones((len(lens), T, K), dtype=np.int64)
    for l, ln in enumerate(lens):
        st = [rng.choice(len(r[0]), p=r[2]) for r in rings]
        for t in range(ln):
            if t:
                c = rings[0][3](O[l, t - 1, 2]) if rule2 else 0
                st = [(s + rng.choice(len(r[0]), p=r[1][c])) % len(r[0]) for s, r in zip(st, rings)]
            O[l, t] = [r[0][s] for s, r in zip(st, rings)]
    return O


def fit_counts(O, ctxf):
    cnt = [defaultdict(Counter) for _ in range(K)]
    for l in range(O.shape[0]):
        for t in range(O.shape[1]):
            if O[l, t, 0] < 0: break
            for k in range(K):
                key = ('S',) if t == 0 else (O[l, t - 1, k], ctxf(O[l, t - 1, 2]))
                cnt[k][key][O[l, t, k]] += 1
    return cnt


def gen_mx(cnt, lens, rng, ctxf):
    T = int(max(lens)); O = -np.ones((len(lens), T, K), dtype=np.int64)
    for l, ln in enumerate(lens):
        for t in range(ln):
            for k in range(K):
                key = ('S',) if t == 0 else (O[l, t - 1, k], ctxf(O[l, t - 1, 2]))
                c = cnt[k].get(key) or cnt[k][('S',)]
                v = np.array(list(c.keys())); w = np.array(list(c.values()), float)
                O[l, t, k] = v[rng.choice(len(v), p=w / w.sum())]
    return O


def gen_w2(Ttr, lens, rng):
    c3, c2, c1 = defaultdict(Counter), defaultdict(Counter), Counter()
    for L in Ttr:
        h = ['<s>', '<s>'] + L
        for i in range(2, len(h)):
            c3[(h[i - 2], h[i - 1])][h[i]] += 1; c2[h[i - 1]][h[i]] += 1; c1[h[i]] += 1
    def draw(c):
        v = list(c.keys()); w = np.array(list(c.values()), float)
        return v[rng.choice(len(v), p=w / w.sum())]
    T = int(max(lens)); O = -np.ones((len(lens), T, K), dtype=np.int64)
    for l, ln in enumerate(lens):
        h = ['<s>', '<s>']
        for t in range(ln):
            u = rng.rand()
            c = c3.get((h[-2], h[-1]))
            if c and u < 0.8: w = draw(c)
            elif c2.get(h[-1]) and u < 0.95: w = draw(c2[h[-1]])
            else: w = draw(c1)
            O[l, t] = w; h.append(w)
    return O


def best_c1(name, k):
    n = C2.best_n(name, k)
    return pickle.load(open(os.path.join(OUT, 'c1_%s_%d_n%d.pkl' % (name, k, n)), 'rb'))


def run(name, reps=10):
    fn = os.path.join(OUT, 'c3_%s.pkl' % name)
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    O, lens, voc, truth, tr, te = C1.data(name)
    real = stats(O[te])
    ctx, top = ctx_array(O, 2, 4, None)
    m = {s: i for i, s in enumerate(top)}
    ctxf = lambda s: m.get(int(s), 3)
    R1 = []
    for k in range(K):
        r = best_c1(name, k)
        R1.append((np.array(r['lab']), np.array(r['q']), np.array(r['pi']), ctxf))
    R2 = []
    for k in range(K):
        r = pickle.load(open(os.path.join(OUT, 'c2_R2_%s_%d.pkl' % (name, k)), 'rb'))
        lab = np.array(r['lab']); q = np.array(r['q'])
        R2.append((lab, q, np.array(r['pi']), ctxf))
    cnt = fit_counts(O[tr], ctxf)
    Ttr = tuples(O[tr])
    lens_te = [int((O[l, :, 0] >= 0).sum()) for l in te]
    out = {'real': real}
    rng = np.random.RandomState(0)
    for mod in ('V1', 'V2', 'MX', 'W2'):
        sims = []
        for rep in range(reps):
            if mod == 'V1': G = gen_volvelle(R1, lens_te, rng)
            elif mod == 'V2': G = gen_volvelle(R2, lens_te, rng, rule2=True)
            elif mod == 'MX': G = gen_mx(cnt, lens_te, rng, ctxf)
            else: G = gen_w2(Ttr, lens_te, rng)
            sims.append(stats(G))
        out[mod] = {k: (float(np.mean([s[k] for s in sims])), float(np.std([s[k] for s in sims]))) for k in real}
        print(name, mod, {k: round(v[0], 4) for k, v in out[mod].items()}, flush=True)
    print(name, 'real', {k: round(v, 4) for k, v in real.items()}, flush=True)
    pickle.dump(out, open(fn, 'wb'))
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(2) as P:
        res = P.map(run, list(C1.CORPORA), chunksize=1)
    pickle.dump(dict(zip(C1.CORPORA, res)), open(os.path.join(OUT, 'c3_all.pkl'), 'wb'))
    print('ALL DONE')
