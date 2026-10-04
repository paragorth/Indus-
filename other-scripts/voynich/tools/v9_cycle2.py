"""v9 cycle 2: richer turning rules and a one-disc (whole-word) volvelle.

(a) R2 'step depends on the previous word': each ring's step kernel is conditioned on the
    class of the previous word's suffix filler (4 classes: 3 commonest suffixes + rest).
    Comparator with the same conditioning: within-ring Markov-1 x previous-suffix class.
(b) One-disc volvelle: the whole word (filler tuple; 40 commonest + OTHER) sits on one
    ring of n cells (Llull-style sector table); comparator: word Markov-1/2 on the same
    41-symbol alphabet.
(c) Within-line shuffle null (Voynich): words shuffled inside each line; volvelle and
    Markov gains over M0 must collapse if they come from word order.
Corpora: planted (positive control), Voynich ZL3b, verbose Latin, verbose Italian.
Checkpoints: data/results/v9/c2_*.pkl
"""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
from v9_lib import *
import v9_cycle1 as C1
from multiprocessing import Pool

OUT = C1.OUT
K, M = 3, 12
CORPORA = ('planted', 'voynich', 'vs_latin', 'vs_italian')
NW = 40


def best_n(name, k):
    best = None
    for n in C1.SIZES:
        fn = os.path.join(OUT, 'c1_%s_%d_n%d.pkl' % (name, k, n))
        if os.path.exists(fn):
            r = pickle.load(open(fn, 'rb'))
            if best is None or r['hard_tr'] > best[1]:
                best = (n, r['hard_tr'])
    return best[0] if best else 40


def word_symbols(O, tr):
    T = tuples(O[tr])
    top = [w for w, _ in Counter(w for L in T for w in L).most_common(NW)]
    idx = {w: i for i, w in enumerate(top)}
    W = -np.ones(O.shape[:2], dtype=np.int64)
    for l in range(O.shape[0]):
        for t in range(O.shape[1]):
            if O[l, t, 0] < 0: break
            W[l, t] = idx.get(tuple(O[l, t]), NW)
    return W, top


def shuffled(O, seed=0):
    rng = np.random.RandomState(seed)
    O2 = O.copy()
    for l in range(O.shape[0]):
        n = int((O[l, :, 0] >= 0).sum())
        O2[l, :n] = O[l, rng.permutation(n)]
    return O2


def job(args):
    kind, name, k = args
    fn = os.path.join(OUT, 'c2_%s_%s_%d.pkl' % (kind, name, k))
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    O, lens, voc, truth, tr, te = C1.data(name)
    t0 = time.time()
    res = dict(kind=kind, name=name, k=k)
    if kind == 'R2':
        S = M + 1; n = best_n(name, k)
        ctx, top = ctx_array(O, 2, 4, None)
        otr, ote = O[tr, :, k], O[te, :, k]
        h, hll, sll = soft_em(otr, ctx[tr], S, n, C=4, iters=150, seed=0)
        res.update(n=n, hard_tr=hll, soft_tr=sll,
                   hard_te=forward_backward(h, ote, ctx[te], None, True)[0],
                   soft_te=forward_backward(h.soft, ote, ctx[te], None, True)[0],
                   q=h.q.tolist(), lab=h.lab.tolist())
        res['MX1'] = markov_ll(otr, ote, S, order=1, ctx_train=ctx[tr], ctx_test=ctx[te])
        res['MX0'] = markov_ll(otr, ote, S, order=0, ctx_train=ctx[tr], ctx_test=ctx[te])
    elif kind == 'disc':
        W, top = word_symbols(O, tr)
        S = NW + 1; n = 80
        wtr, wte = W[tr], W[te]
        z1, z2 = np.zeros_like(wtr), np.zeros_like(wte)
        h, hll, sll = soft_em(wtr, z1, S, n, iters=150, seed=0)
        res.update(n=n, hard_tr=hll, soft_tr=sll,
                   hard_te=forward_backward(h, wte, z2, None, True)[0],
                   soft_te=forward_backward(h.soft, wte, z2, None, True)[0],
                   q=h.q.tolist(), lab=h.lab.tolist(), words=[list(map(int, w)) for w in top])
        for o in (0, 1, 2):
            res['W%d' % o] = markov_ll(wtr, wte, S, order=o)
    elif kind == 'shuf':
        S = M + 1; n = best_n(name, k)
        O2 = shuffled(O)
        otr, ote = O2[tr, :, k], O2[te, :, k]
        z1, z2 = np.zeros_like(otr), np.zeros_like(ote)
        h, hll, sll = soft_em(otr, z1, S, n, iters=150, seed=0)
        res.update(n=n, hard_te=forward_backward(h, ote, z2, None, True)[0],
                   soft_te=forward_backward(h.soft, ote, z2, None, True)[0])
        for o in (0, 1):
            res['M%d' % o] = markov_ll(otr, ote, S, order=o)
    res['secs'] = time.time() - t0
    pickle.dump(res, open(fn, 'wb'))
    print('done', kind, name, k, round(time.time() - t0), 's', flush=True)
    return res


if __name__ == '__main__':
    jobs = [('disc', c, 0) for c in CORPORA]
    jobs += [('R2', c, k) for c in CORPORA for k in range(K)]
    jobs += [('shuf', 'voynich', k) for k in range(K)] + [('shuf', 'planted', k) for k in range(K)]
    with Pool(2) as P:
        res = P.map(job, jobs, chunksize=1)
    pickle.dump(res, open(os.path.join(OUT, 'c2_all.pkl'), 'wb'))
    print('ALL DONE')
