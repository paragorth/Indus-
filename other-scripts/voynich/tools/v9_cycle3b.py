"""v9 cycle 3b: (a) positive control for the one-disc (whole-word) volvelle of cycle 2:
a planted one-disc corpus (40 cells, 41 word symbols, kernel 0.6 on one step + 0.4 spread,
reset each line, Voynich line lengths) searched blind with the same settings (n = 80);
(b) transcription check: the cycle-1 3-ring volvelle on IT2a (n = 48)."""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
from v9_lib import *
import v9_cycle1 as C1
from multiprocessing import Pool

OUT = C1.OUT


def planted_disc(lens, S=41, n=40, seed=5):
    rng = np.random.RandomState(seed)
    p = 1 / np.arange(1, S + 1) ** 0.8
    lab = np.array(init_labels(p * 1000, n, rng))
    a = rng.randint(1, n)
    q = np.full(n, 0.4 / n); q[a] += 0.6
    pi = rng.dirichlet(np.ones(n) * 0.3)
    T = int(max(lens)); W = -np.ones((len(lens), T), dtype=np.int64)
    for l, ln in enumerate(lens):
        s = rng.choice(n, p=pi)
        for t in range(ln):
            if t: s = (s + rng.choice(n, p=q)) % n
            W[l, t] = lab[s]
    return W, (lab, q, pi)


def job(arg):
    fn = os.path.join(OUT, 'c3b_%s.pkl' % arg)
    if os.path.exists(fn):
        return pickle.load(open(fn, 'rb'))
    t0 = time.time()
    if arg == 'disc':
        O, lens, voc, truth, tr, te = C1.data('voynich')
        W, (lab, q, pi) = planted_disc(lens)
        S = 41
        wtr, wte = W[tr], W[te]
        z1, z2 = np.zeros_like(wtr), np.zeros_like(wte)
        h, hll, sll = soft_em_multi(wtr, z1, S, 80, seeds=4, burn=40, keep=1, more=120)
        orr = Ring(lab, S); orr.q = q[None]; orr.pi = pi
        res = dict(hard_te=forward_backward(h, wte, z2, None)[0], soft_te=forward_backward(h.soft, wte, z2, None)[0],
                   oracle_te=forward_backward(orr, wte, z2, None)[0], q=h.q.tolist(), true_step=int(np.argmax(q)))
        for o in (0, 1, 2):
            res['W%d' % o] = markov_ll(wtr, wte, S, order=o)
        res['ntok'] = int((wte >= 0).sum())
    else:  # IT2a ring k
        k = int(arg[-1])
        L = C1.corpus('voynich_IT')
        m = slot_model_cached('voynich_IT', L, 3)
        O, lens, voc = encode(L, m, 12)
        tr, te = split_pages(L)
        otr, ote = O[tr, :, k], O[te, :, k]
        z1, z2 = np.zeros_like(otr), np.zeros_like(ote)
        h, hll, sll = soft_em_multi(otr, z1, 13, 48, seeds=6, burn=40, keep=2, more=120)
        res = dict(hard_te=forward_backward(h, ote, z2, None)[0], soft_te=forward_backward(h.soft, ote, z2, None)[0],
                   q=h.q.tolist(), voc=voc[k], ntok=int((ote >= 0).sum()))
        for o in (0, 1):
            res['M%d' % o] = markov_ll(otr, ote, 13, order=o)
    res['secs'] = time.time() - t0
    pickle.dump(res, open(fn, 'wb'))
    print('done', arg, round(res['secs']), flush=True)
    return res


if __name__ == '__main__':
    # IT2a slot model (glyph-order learning) is slow: build it once before the pool
    L = C1.corpus('voynich_IT'); slot_model_cached('voynich_IT', L, 3)
    with Pool(2) as P:
        P.map(job, ['disc', 'IT0', 'IT1', 'IT2'], chunksize=1)
    print('ALL DONE')
