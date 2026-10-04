"""v12 latent stream models (cycle 2).
Each line = sequence of words; hidden stream label s_t in {0..k-1}, first-order Markov switch (pi, A).
Emission P(word_t | s_t, context) from a per-stream CtxModel (v12_lib), where the context is
  'W' (weave):  the last EARLIER word in the line carrying the SAME stream label (START if none)
  'H' (HMM):    the immediately preceding word (START at line start), whatever its label
Both have exactly the same parameters (k emission tables + switch); W differs only in WHICH word is the
context after a switch. A woven text should be fitted far better by W; a single text by H.
Exact forward-backward over states (s_t, last index of every stream). EM with fractional counts."""
import math, random
from collections import defaultdict
import v12_lib as V


def _states_init(k, kind):
    return


class StreamModel:
    def __init__(self, k, kind, seed=0):
        self.k, self.kind, self.rng = k, kind, random.Random(seed)
        self.pi = [1.0 / k] * k
        self.A = [[1.0 / k] * k for _ in range(k)]
        self.M = None

    # ---- lattice for one line -------------------------------------------------
    def _lattice(self, F, em):
        """returns list over t of dict state -> list of (prev_state, s, ctx_index); state = (s, lasts)"""
        k, n = self.k, len(F)
        layers = []
        L0 = {}
        for s in range(k):
            lasts = tuple(0 if j == s else -1 for j in range(k)) if self.kind == 'W' else (s,)
            st = (s, lasts)
            L0[st] = [(None, s, -1)]
        layers.append(L0)
        for t in range(1, n):
            Lt = defaultdict(list)
            for ps in layers[-1]:
                s0, lasts = ps
                for s in range(k):
                    if self.kind == 'W':
                        ctx = lasts[s]
                        nl = tuple(t if j == s else lasts[j] for j in range(k))
                        Lt[(s, nl)].append((ps, s, ctx))
                    else:
                        Lt[(s, (s,))].append((ps, s, t - 1))
            layers.append(dict(Lt))
        return layers

    def _em_line(self, F, collect):
        k, n = self.k, len(F)
        cache = {}
        def e(t, s, c):
            key = (t, s, c)
            v = cache.get(key)
            if v is None:
                v = 2.0 ** self.M[s].lp(F[t], F[c] if c >= 0 else V.START); cache[key] = v
            return v
        layers = self._lattice(F, e)
        alpha = []; logZ = 0.0
        a0 = {}
        for st, ins in layers[0].items():
            _, s, c = ins[0]; a0[st] = self.pi[s] * e(0, s, c)
        z = sum(a0.values()); logZ += math.log2(z); a0 = {q: v / z for q, v in a0.items()}
        alpha.append(a0)
        for t in range(1, n):
            at = {}
            for st, ins in layers[t].items():
                tot = 0.0
                for ps, s, c in ins:
                    tot += alpha[-1][ps] * self.A[ps[0]][s] * e(t, s, c)
                at[st] = tot
            z = sum(at.values()); logZ += math.log2(z); at = {q: v / z for q, v in at.items()}
            alpha.append(at)
        if not collect:
            return logZ
        beta = [None] * n
        beta[n - 1] = {st: 1.0 for st in layers[n - 1]}
        for t in range(n - 1, 0, -1):
            bt = defaultdict(float); tot = 0.0
            for st, ins in layers[t].items():
                for ps, s, c in ins:
                    v = self.A[ps[0]][s] * e(t, s, c) * beta[t][st]
                    bt[ps] += v
            z = sum(bt.values())
            beta[t - 1] = {q: v / z for q, v in bt.items()}
        # posteriors
        for t in range(n):
            if t == 0:
                g = {st: alpha[0][st] * beta[0][st] for st in layers[0]}
                z = sum(g.values())
                for st, ins in layers[0].items():
                    _, s, c = ins[0]; w = g[st] / z
                    collect['pi'][s] += w; collect['em'][s].append((F[0], V.START, w))
                    if 'post' in collect: collect['post'].append((0, s, w, None))
            else:
                xs = []
                for st, ins in layers[t].items():
                    for ps, s, c in ins:
                        xs.append((ps[0], s, c, alpha[t - 1][ps] * self.A[ps[0]][s] * e(t, s, c) * beta[t][st]))
                z = sum(x[3] for x in xs)
                agg = defaultdict(float)
                for s0, s, c, v in xs:
                    w = v / z
                    collect['A'][s0][s] += w
                    agg[(s, c)] += w
                if 'post' in collect:
                    sw = defaultdict(float)
                    for s0, s, c, v in xs: sw[(s0, s)] += v / z
                    for (s0, s), w in sw.items(): collect['post'].append((t, s, w, s0))
                for (s, c), w in agg.items():
                    if w > 1e-4:
                        collect['em'][s].append((F[t], F[c] if c >= 0 else V.START, w))
        return logZ

    def _mstep(self, col):
        k = self.k
        tp = sum(col['pi']) or 1
        self.pi = [(x + 0.1) / (tp + 0.1 * k) for x in col['pi']]
        self.A = []
        for s in range(k):
            r = col['A'][s]; t = sum(r)
            self.A.append([(x + 0.1) / (t + 0.1 * k) for x in r])
        self.M = []
        for s in range(k):
            m = V.CtxModel()
            for x, c, w in col['em'][s]:
                m.add(x, c, w)
            self.M.append(m.finalize())

    def fit(self, lines, iters=12, verbose=False):
        Fs = [[V.feats(w) for w in l['words']] for l in lines if l['words']]
        k = self.k
        # random soft initialisation: each token's stream weights random
        col = {'pi': [0.0] * k, 'A': [[0.0] * k for _ in range(k)], 'em': [[] for _ in range(k)]}
        for F in Fs:
            prev = None
            lab = [self.rng.randrange(k) for _ in F]
            for t, x in enumerate(F):
                ws = [self.rng.random() + (2.0 if j == lab[t] else 0) for j in range(k)]; z = sum(ws)
                for j in range(k):
                    if self.kind == 'W':
                        c = max([u for u in range(t) if lab[u] == j], default=-1)
                    else:
                        c = t - 1
                    col['em'][j].append((x, F[c] if c >= 0 else V.START, ws[j] / z))
                if t == 0: col['pi'][lab[t]] += 1
                else: col['A'][lab[t - 1]][lab[t]] += 1
        self._mstep(col)
        hist = []
        for it in range(iters):
            col = {'pi': [0.0] * k, 'A': [[0.0] * k for _ in range(k)], 'em': [[] for _ in range(k)]}
            ll = sum(self._em_line(F, col) for F in Fs)
            self._mstep(col)
            hist.append(ll / sum(len(F) for F in Fs))
            if verbose: print(self.kind, k, it, round(hist[-1], 4), flush=True)
            if it > 2 and abs(hist[-1] - hist[-2]) < 2e-4: break
        self.train_bits = hist[-1]
        return hist

    def score(self, lines):
        Fs = [[V.feats(w) for w in l['words']] for l in lines if l['words']]
        return sum(self._em_line(F, None) for F in Fs) / sum(len(F) for F in Fs)

    def decode(self, words):
        """per position t: P(s_t = s) and P(switch at t) (t >= 1)."""
        F = [V.feats(w) for w in words]
        col = {'pi': [0.0] * self.k, 'A': [[0.0] * self.k for _ in range(self.k)], 'em': [[] for _ in range(self.k)], 'post': []}
        self._em_line(F, col)
        P = [[0.0] * self.k for _ in F]; sw = [0.0] * len(F)
        for t, s, w, s0 in col['post']:
            P[t][s] += w
            if s0 is not None and s0 != s: sw[t] += w
        return P, sw
