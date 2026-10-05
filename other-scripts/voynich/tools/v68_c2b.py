"""v68 cycle 2b: the stepwise test and an out-of-fit prediction.

(1) SMOOTH kernel: the step preference is a Gaussian with drift, b(d) = beta1*x + beta2*x^2 (beta2 <= 0),
    x = d/(K-1).  Melody = mostly stepwise: a smooth unimodal kernel must keep the free model's gain.
    The full Toeplitz kernel (v68_c2) can encode arbitrary extreme-class pairings (seen in German).
(2) WORD SPAN (not used in fitting): the fit only sees junctions between DIFFERENT words.  If the latent line
    is a pitch line, a word's own entry and exit classes must lie close on it (a neume spans a few steps):
    corr(s(w), e(w)) over kept word types, frequency-weighted, against a permutation null.
(3) WSHUF null: the same fit on text with words shuffled inside each line keeps line-likeness and
    removes adjacency.
usage: python3 v68_c2b.py corpus K restarts
"""
import json, math, os, random, sys
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v68_c2 as C2
import v68_lib as V


class Smooth(C2.Model):
    def cond(self, C):
        K = self.K
        a = np.log(C.sum(0) + 1); a -= a.mean()
        D = np.arange(K)[None, :] - np.arange(K)[:, None] + K - 1
        x = (np.arange(2 * K - 1) - (K - 1)) / (K - 1)
        beta = np.array([0.0, -1.0])
        rs = C.sum(1)
        for it in range(300):
            b = beta[0] * x + beta[1] * x ** 2
            L = a[None, :] + b[D]; L -= L.max(1, keepdims=True)
            P = np.exp(L); P /= P.sum(1, keepdims=True)
            G = C - rs[:, None] * P
            ga = G.sum(0); gb = np.bincount(D.ravel(), weights=G.ravel(), minlength=2 * K - 1)
            lr = 5.0 / max(1.0, C.sum())
            a += lr * ga
            beta += lr * 2 * np.array([(gb * x).sum(), (gb * x ** 2).sum()])
            beta[1] = min(beta[1], 0.0)
        b = beta[0] * x + beta[1] * x ** 2
        L = a[None, :] + b[D]; L -= L.max(1, keepdims=True)
        P = np.exp(L); P /= P.sum(1, keepdims=True)
        self.kernel = b; self.beta = beta
        return np.log(P + 1e-12)


def span(m, keep, btr, rng, nperm=500):
    wc = Counter(w for _, _, w in btr)
    xs, xe, w8 = [], [], []
    for w in keep:
        if w in m.si and w in m.ei:
            xs.append(m.cs[m.si[w]]); xe.append(m.ce[m.ei[w]]); w8.append(wc.get(w, 1))
    xs, xe, w8 = np.array(xs, float), np.array(xe, float), np.array(w8, float)

    def wcorr(a, b):
        ma, mb = (a * w8).sum() / w8.sum(), (b * w8).sum() / w8.sum()
        return ((a - ma) * (b - mb) * w8).sum() / math.sqrt(((a - ma) ** 2 * w8).sum() * ((b - mb) ** 2 * w8).sum() + 1e-12)
    obs = wcorr(xs, xe)
    null = [wcorr(xs, rng.permutation(xe)) for _ in range(nperm)]
    return obs, float(np.mean(null)), float(np.std(null)), len(xs)


def run(name, K, R, seed=0):
    rng = np.random.default_rng(seed); prng = random.Random(seed)
    tr, te, truth = C2.corpus(name, prng)
    keep = C2.vocab(tr)
    btr, bte = C2.build(tr, keep), C2.build(te, keep)
    res = {'name': name, 'K': K, 'ntr': len(btr), 'nkeep': len(keep)}
    frees = [C2.Model(K, False, rng).fit(btr) for _ in range(R)]
    fb = max(frees, key=C2.full_ll)
    res['free_gain'] = fb.score(btr, bte)
    cands = [Smooth(K, True, rng).fit(btr) for _ in range(R)]
    for dim in (0, 1):
        for fl in (False, True):
            cands.append(Smooth(K, True, rng).fit(btr, init=C2.ca_init(btr, K, dim, fl)))
    for f in sorted(frees, key=C2.full_ll)[-2:]:
        cands.append(Smooth(K, True, rng).fit(btr, init=C2.seriate(f)))
    bm = max(cands, key=C2.full_ll)
    for it in range(8):
        ie = {x: (int(rng.integers(0, K)) if rng.random() < 0.15 else int(bm.ce[i])) for x, i in bm.ei.items()}
        is_ = {x: (int(rng.integers(0, K)) if rng.random() < 0.15 else int(bm.cs[i])) for x, i in bm.si.items()}
        m2 = Smooth(K, True, rng).fit(btr, init=(ie, is_))
        if C2.full_ll(m2) > C2.full_ll(bm):
            bm = m2
    res['smooth_gain'] = bm.score(btr, bte)
    res['R_smooth'] = res['smooth_gain'] / max(1e-9, res['free_gain'])
    res['beta'] = [round(float(b), 3) for b in bm.beta]
    o, mu, sd, n = span(bm, keep, btr, rng)
    res['span_corr'] = o; res['span_z'] = (o - mu) / max(sd, 1e-9); res['span_n'] = n
    if truth:
        from scipy.stats import spearmanr
        xs, ys = [], []
        for w in keep:
            if w in truth and w in bm.si:
                xs.append(bm.cs[bm.si[w]]); ys.append(truth[w][0])
        res['entry_rho'] = float(abs(spearmanr(xs, ys)[0]))
    # keep the assignment for cycle 3
    json.dump({'ce': {str(x): int(bm.ce[i]) for x, i in bm.ei.items()},
               'cs': {str(x): int(bm.cs[i]) for x, i in bm.si.items()}, 'beta': list(map(float, bm.beta))},
              open(os.path.join(V.CK, 'c2b_assign_%s_K%d.json' % (name, K)), 'w'))
    return res


if __name__ == '__main__':
    name, K, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    r = run(name, K, R)
    print(json.dumps(r))
    with open(os.path.join(V.CK, 'c2b_results.jsonl'), 'a') as f:
        f.write(json.dumps(r) + '\n')
