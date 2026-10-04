"""v16 cycle 2: annealed free segmentation (no rule at all).

Delete the spaces and search directly over all boundary sets for the minimum MDL2 description length
(adaptive Chinese-restaurant token code + lexicon spelled with an adaptive glyph-bigram code), by
simulated annealing with single-boundary toggles (Metropolis, T from 1.5 bits to 0.02 over 30 sweeps, then
4 greedy sweeps; a first run with T0 = 6 bits and 16 sweeps did not converge, kept in cycle2_hot/). Two starts per corpus: the written spacing and a random spacing (density 0.22).
Same 10 corpora as cycle 1 (Voynich ZL/IT, glyph-shuffle and Markov-2 nulls, true-spaced Latin/Italian,
planted-spaced Latin/Italian). Line starts are fixed boundaries.
Output per run: MDL2 of written vs found, boundary F vs written (and vs the true words for planted
controls), unit counts, length, Zipf, junction excess, and the final boundary mask (npz) for cycle 3.
"""
import os, sys, json, time, math, random
import numpy as np
from math import lgamma
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v16_lib as L
from v16_cycle1 import build, NAMES

OUT = os.path.join(L.RESDIR, 'cycle2')
os.makedirs(OUT, exist_ok=True)
SWEEPS = int(os.environ.get('V16_SWEEPS', 30))
T0 = float(os.environ.get('V16_T0', 1.5))


class Seg:
    def __init__(self, c, st, alpha):
        self.c = c; self.sym = c.sym.tolist(); self.n = c.n
        self.b = st.astype(bool).tolist()
        self.ls = c.ls.tolist()
        self.A = c.A; self.k = c.A + 1
        self.la = math.log(alpha); self.alpha = alpha
        self.cnt = {}; self.N = 0; self.S = 0.0
        self.t = {}; self.row = {}; self.lex = 0.0
        self.lgk2 = lgamma(self.k / 2); self.lg05 = lgamma(0.5)
        s = np.flatnonzero(st); e = np.r_[s[1:], c.n]
        for a, z in zip(s.tolist(), e.tolist()):
            self.add(tuple(self.sym[a:z]))

    # ---- lexicon spelling (adaptive bigram KT), contributions are additive per cell/row ----
    def _lexw(self, w, sgn):
        A = self.A; d = 0.0
        prev = A
        for x in list(w) + [A]:
            key = (prev, x)
            t = self.t.get(key, 0); r = self.row.get(prev, 0)
            t2 = t + sgn; r2 = r + sgn
            d += -(lgamma(t2 + 0.5) - lgamma(t + 0.5)) + (lgamma(r2 + self.k / 2) - lgamma(r + self.k / 2))
            self.t[key] = t2; self.row[prev] = r2
            prev = x
        self.lex += d

    def add(self, w):
        n = self.cnt.get(w, 0)
        if n == 0:
            self._lexw(w, +1)
        else:
            self.S += math.log(n)          # lgamma(n+1) - lgamma(n)
        self.cnt[w] = n + 1; self.N += 1

    def remove(self, w):
        n = self.cnt[w]
        if n == 1:
            del self.cnt[w]; self._lexw(w, -1)
        else:
            self.cnt[w] = n - 1; self.S -= math.log(n - 1)
        self.N -= 1

    def cost(self):
        K = len(self.cnt); a = self.alpha
        crp = -(K * self.la + lgamma(a) - lgamma(a + self.N) + self.S)
        return (crp + self.lex) / math.log(2)

    def bounds(self, i):
        l = i - 1
        while not self.b[l]: l -= 1
        r = i + 1
        while r < self.n and not self.b[r]: r += 1
        return l, r

    def sweep(self, T, rng):
        sym = self.sym
        cur = self.cost()
        order = list(range(self.n)); rng.shuffle(order)
        acc = 0
        for i in order:
            if self.ls[i]: continue
            l, r = self.bounds(i)
            w1 = tuple(sym[l:i]); w2 = tuple(sym[i:r]); w = w1 + w2
            if self.b[i]:
                self.remove(w1); self.remove(w2); self.add(w)
                new = self.cost(); d = new - cur
                if d <= 0 or (T > 0 and rng.random() < math.exp(-d / T)):
                    self.b[i] = False; cur = new; acc += 1
                else:
                    self.remove(w); self.add(w1); self.add(w2)
            else:
                self.remove(w); self.add(w1); self.add(w2)
                new = self.cost(); d = new - cur
                if d <= 0 or (T > 0 and rng.random() < math.exp(-d / T)):
                    self.b[i] = True; cur = new; acc += 1
                else:
                    self.remove(w1); self.remove(w2); self.add(w)
        return cur, acc


def best_alpha(c, st):
    s, e = L.tokens(c, st)
    hv = L.token_hash(c, s, e)
    _, cnt = np.unique(hv, return_counts=True)
    N = float(cnt.sum()); K = len(cnt); sg = float(np.sum(L._gl(cnt)))
    return min(L._ALPHAS, key=lambda a: -(K * np.log(a) + L._gl(a) - L._gl(a + N) + sg))


def run(job):
    name, start = job
    tag = f'{name}__{start}'
    path = os.path.join(OUT, tag + '.json')
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    c = build(name)
    nrng = np.random.default_rng(3)
    st0 = c.written.copy() if start == 'written' else (nrng.random(c.n) < 0.22) | c.ls
    alpha = float(best_alpha(c, c.written))
    seg = Seg(c, st0, alpha)
    rng = random.Random(11)
    temps = [T0 * (0.02 / T0) ** (j / max(1, SWEEPS - 1)) for j in range(SWEEPS)] + [0, 0, 0, 0]
    log = []
    for j, T in enumerate(temps):
        cur, acc = seg.sweep(T, rng)
        log.append((T, cur / c.n, acc))
        print(tag, j, round(T, 3), round(cur / c.n, 4), acc, round(time.time() - t0), flush=True)
    st = np.array(seg.b, bool)
    srng = np.random.default_rng(5)
    found = L.score(c, st, srng); written = L.score(c, c.written, srng)
    out = dict(name=name, start=start, alpha=alpha, n=c.n, log=log, found=found, written=written,
               secs=time.time() - t0)
    if c.truth is not None:
        out['truth'] = L.score(c, c.truth, srng)
    np.savez_compressed(os.path.join(OUT, tag + '.npz'), st=st)
    json.dump(out, open(path, 'w'), indent=1, default=float)
    return out


if __name__ == '__main__':
    jobs = [(n, 'written') for n in NAMES] + [(n, 'random') for n in ('V-ZL3b', 'Latin', 'NULL-markov2')]
    # longest first is not needed; interleave so the Voynich and its controls finish early
    pri = ['V-ZL3b', 'Latin', 'Latin-planted-every5', 'NULL-markov2', 'Italian', 'Italian-planted-VC',
           'V-IT2a', 'Latin-planted-VC', 'Italian-planted-every5', 'NULL-glyphshuf']
    jobs.sort(key=lambda j: (j[1] == 'random', pri.index(j[0])))
    with Pool(2) as p:
        p.map(run, jobs, chunksize=1)
