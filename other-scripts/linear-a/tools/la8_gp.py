#!/usr/bin/env python3
"""LA-8 genetic programming over typed-slot probabilistic automata.

Machine: word classes (each lexicon type -> one class; each novel-word key -> one class, which makes
that class 'open'), 1-3 hidden states per class (slots), state-to-state transitions fit by Baum-Welch.
Fitness (lower = better): held-out bits (2-fold CV by document) + description length (bits).
Usage: la8_gp.py CORPUS SEED GENS [--tag TAG]
Checkpoints the population every 50 generations to data/la8/gp_<CORPUS>_<TAG>_s<SEED>.json and resumes.
"""
import ctypes, json, math, os, random, sys, time
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'la8')
lib = ctypes.CDLL(os.path.join(HERE, 'la8_core.so'))
lib.score.restype = ctypes.c_double
I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C')
F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C')
lib.score.argtypes = [ctypes.c_int, ctypes.c_int, I32, I32, I32, I32, F64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      I32, I32, I32, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                      ctypes.c_int, F64, F64, F64]
BETA, ALPHA0, GAMMA, DELTA, ITERS = 0.1, 0.5, 0.5, 0.1, 6
MAXK, MAXST = 30, 3
DLMODE = os.environ.get('LA8_DL', 'v2')


class Data:
    def __init__(self, docs, fold_seed=0):
        self.docs = docs
        cnt = Counter(t[0] for d in docs for t in d['toks'])
        self.lex = sorted(w for w, n in cnt.items() if n >= 2)
        self.lexid = {w: i for i, w in enumerate(self.lex)}
        novel = [t for d in docs for t in d['toks'] if t[0] not in self.lexid]
        kc = Counter(t[2] for t in novel)
        def keyof(t):
            k = t[2]
            if kc.get(k, 0) >= 3: return k
            return t[1] + ':other' if t[1] == 'W2' else t[1]
        self.keys = sorted({keyof(t) for t in novel} | {'W2:other'})
        self.keyid = {k: i for i, k in enumerate(self.keys)}
        ntypes_kind = Counter(t[1] for t in {tuple(t) for d in docs for t in d['toks']})
        tt, tk, lp, off = [], [], [], [0]
        for d in docs:
            for t in d['toks']:
                tt.append(self.lexid.get(t[0], -1)); tk.append(self.keyid.get(keyof(t), 0))
                lp.append(-math.log(2.0 * max(1, ntypes_kind[t[1]])))
            off.append(len(tt))
        rng = random.Random(fold_seed)
        self.fold = np.array([rng.randint(0, 1) for _ in docs], dtype=np.int32)
        self.ttype = np.array(tt, dtype=np.int32); self.tkey = np.array(tk, dtype=np.int32)
        self.logp0 = np.array(lp, dtype=np.float64); self.off = np.array(off, dtype=np.int32)
        self.T, self.V, self.NK = len(tt), len(self.lex), len(self.keys)
        self.lexcount = np.array([cnt[w] for w in self.lex])
        # context neighbours for a guided mutation: left/right token-kind and frequent-neighbour profile
        ctx = defaultdict(Counter)
        top = {w for w, _ in cnt.most_common(40)}
        for d in docs:
            ts = [t[0] for t in d['toks']]; ks = [t[1] for t in d['toks']]
            for i, w in enumerate(ts):
                if w not in self.lexid: continue
                L = ts[i-1] if i > 0 else '<s>'; Rr = ts[i+1] if i + 1 < len(ts) else '</s>'
                ctx[w]['L:' + (L if L in top else ks[i-1] if i > 0 else '<s>')] += 1
                ctx[w]['R:' + (Rr if Rr in top else ks[i+1] if i + 1 < len(ts) else '</s>')] += 1
        feats = sorted({f for c in ctx.values() for f in c}); fid = {f: i for i, f in enumerate(feats)}
        X = np.zeros((self.V, len(feats)))
        for w, c in ctx.items():
            for f, n in c.items(): X[self.lexid[w], fid[f]] = n
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        S = X @ X.T; np.fill_diagonal(S, -1)
        self.nn = np.argsort(-S, axis=1)[:, :6]
        kot = {t[0]: t[1] for d in docs for t in d['toks']}
        self.kind_of_type = [kot[w] for w in self.lex]
        self.kind_idx = defaultdict(list); self.keykind_idx = defaultdict(list)
        for i, w in enumerate(self.lex): self.kind_idx[kot[w]].append(i)
        for i, k in enumerate(self.keys): self.keykind_idx[k.split(':')[0]].append(i)
        self.kind_idx = {k: np.array(v) for k, v in self.kind_idx.items()}; self.keykind_idx = {k: np.array(v) for k, v in self.keykind_idx.items()}

    def evaluate(self, g, mode=0):
        out = np.zeros(4); K = g['K']
        S = int(sum(g['nst']))
        tr = np.zeros((S + 1) * (S + 1)); em = np.zeros((S + 1) * (S + 1))
        bits = lib.score(self.T, len(self.docs), self.off, self.fold, self.ttype, self.tkey, self.logp0, self.V, self.NK, K,
                         g['tc'], g['kc'], g['nst'], BETA, ALPHA0, GAMMA, DELTA, ITERS, mode, out, tr, em)
        if mode == 1: return out, tr.reshape(S + 1, S + 1), em.reshape(S + 1, S + 1)
        return bits, out

    def dl(self, g, edges):
        K = g['K']; S = int(sum(g['nst'])); lk = math.log2(max(K, 2))
        if DLMODE == 'strict':
            assign = (self.V + self.NK) * lk
        else:   # v2: class labels entropy-coded given the token kind (W1/W2/L/NUM/FRAC/S) + parameter cost
            assign = 0.0
            for grp, labs in ((self.kind_idx, g['tc']), (self.keykind_idx, g['kc'])):
                for idx in grp.values():
                    c = Counter(labs[idx].tolist()); n = len(idx)
                    assign += -sum(v * math.log2(v / n) for v in c.values()) + 0.5 * math.log2(n + 1) * (len(c) - 1) + lk
        return assign + S * lk + K * 2 + edges * (2 * math.log2(S + 2) + 0.5 * math.log2(self.T))


def canon(g):
    used = sorted(set(g['tc'].tolist()) | set(g['kc'].tolist()))
    m = {c: i for i, c in enumerate(used)}
    tc = np.array([m[c] for c in g['tc']], dtype=np.int32); kc = np.array([m[c] for c in g['kc']], dtype=np.int32)
    nst = np.array([g['nst'][c] for c in used], dtype=np.int32)
    return {'K': len(used), 'tc': tc, 'kc': kc, 'nst': nst}


def random_genome(D, rng):
    K = rng.randint(2, 20)
    tc = np.array([rng.randrange(K) for _ in range(D.V)], dtype=np.int32)
    kc = np.array([rng.randrange(K) for _ in range(D.NK)], dtype=np.int32)
    nst = np.array([1 if rng.random() < 0.7 else rng.randint(2, MAXST) for _ in range(K)], dtype=np.int32)
    return canon({'K': K, 'tc': tc, 'kc': kc, 'nst': nst})


def mutate(D, g, rng):
    g = {'K': g['K'], 'tc': g['tc'].copy(), 'kc': g['kc'].copy(), 'nst': list(g['nst'])}
    for _ in range(1 + min(4, int(rng.expovariate(0.9)))):
        K = g['K']; r = rng.random()
        if r < 0.30:   # move one type (frequency-weighted half the time)
            w = rng.randrange(D.V) if rng.random() < 0.5 else int(rng.choices(range(D.V), D.lexcount)[0])
            if rng.random() < 0.06 and K < MAXK: g['tc'][w] = K; g['K'] += 1; g['nst'].append(1)
            else: g['tc'][w] = rng.randrange(K)
        elif r < 0.50:  # guided: move a type into the class of a context neighbour (or pull neighbour in)
            w = rng.randrange(D.V); u = int(D.nn[w][rng.randrange(D.nn.shape[1])])
            if rng.random() < 0.5: g['tc'][w] = g['tc'][u]
            else: g['tc'][u] = g['tc'][w]
        elif r < 0.60:  # move a novel key
            k = rng.randrange(D.NK)
            if rng.random() < 0.1 and K < MAXK: g['kc'][k] = K; g['K'] += 1; g['nst'].append(1)
            else: g['kc'][k] = rng.randrange(K)
        elif r < 0.70 and K > 1:  # merge two classes
            a, b = rng.sample(range(K), 2)
            g['tc'][g['tc'] == b] = a; g['kc'][g['kc'] == b] = a
        elif r < 0.80 and K < MAXK:  # split a class
            a = rng.randrange(K); idx = np.where(g['tc'] == a)[0]
            if len(idx) >= 2:
                sel = idx[np.array([rng.random() < 0.5 for _ in idx])]
                g['tc'][sel] = K; g['K'] += 1; g['nst'].append(1)
                ks = np.where(g['kc'] == a)[0]
                for k in ks:
                    if rng.random() < 0.5: g['kc'][k] = K - 0 if False else g['K'] - 1
        elif r < 0.92:  # change number of states (slots) of a class
            a = rng.randrange(K); g['nst'][a] = max(1, min(MAXST, g['nst'][a] + rng.choice((-1, 1))))
        else:  # move a whole block: all types of one kind in class a -> class b
            a, b = rng.randrange(K), rng.randrange(K)
            idx = np.where(g['tc'] == a)[0]
            if len(idx) and D.kind_of_type:
                kd = D.kind_of_type[int(rng.choice(idx))]
                for i in idx:
                    if D.kind_of_type[i] == kd: g['tc'][i] = b
    g['nst'] = np.array(g['nst'], dtype=np.int32)
    return canon(g)


def fitness(D, g):
    bits, out = D.evaluate(g)
    return bits + D.dl(g, out[2]), bits, out


def to_json(g): return {'K': int(g['K']), 'tc': g['tc'].tolist(), 'kc': g['kc'].tolist(), 'nst': g['nst'].tolist()}
def from_json(j): return {'K': j['K'], 'tc': np.array(j['tc'], dtype=np.int32), 'kc': np.array(j['kc'], dtype=np.int32), 'nst': np.array(j['nst'], dtype=np.int32)}


def run(corpus, seed, gens, tag='main', mu=16, lam=32, stall=250, log=True):
    docs = json.load(open(os.path.join(OUT, 'corpus_%s.json' % corpus)))
    D = Data(docs)
    ck = os.path.join(OUT, 'gp_%s_%s_s%d.json' % (corpus, tag, seed))
    rng = random.Random(seed * 7919 + hash(corpus) % 1000)
    pop, gen0, best_hist, restarts, last_imp = None, 0, [], 0, 0
    if os.path.exists(ck):
        j = json.load(open(ck))
        if j.get('done'): return j
        pop = [(p['f'], from_json(p['g'])) for p in j['pop']]; gen0 = j['gen']; best_hist = j['hist']; restarts = j['restarts']; last_imp = j.get('last_imp', gen0)
        rng.seed(seed * 1000 + gen0)
    else:
        pop = []
        for _ in range(mu):
            g = random_genome(D, rng); pop.append((fitness(D, g)[0], g))
    t0 = time.time(); evals = 0
    for gen in range(gen0, gens):
        pop.sort(key=lambda x: x[0])
        kids = []
        for _ in range(lam):
            a = min(rng.sample(pop, 3), key=lambda x: x[0])[1]
            c = mutate(D, a, rng); kids.append((fitness(D, c)[0], c)); evals += 1
        allp = sorted(pop + kids, key=lambda x: x[0])
        # dedupe identical fitness
        newp, seen = [], set()
        for f, g in allp:
            k = round(f, 4)
            if k in seen: continue
            seen.add(k); newp.append((f, g))
            if len(newp) == mu: break
        if newp[0][0] < pop[0][0] - 1e-6: last_imp = gen
        pop = newp
        if gen - last_imp > stall:   # random restart of all but the 2 best
            restarts += 1; last_imp = gen
            pop = pop[:2] + [(fitness(D, g)[0], g) for g in (random_genome(D, rng) for _ in range(mu - 2))]
        best_hist.append(round(pop[0][0], 2))
        if (gen + 1) % 50 == 0 or gen + 1 == gens:
            json.dump({'corpus': corpus, 'seed': seed, 'gen': gen + 1, 'hist': best_hist, 'restarts': restarts, 'last_imp': last_imp,
                       'pop': [{'f': f, 'g': to_json(g)} for f, g in pop], 'done': gen + 1 == gens}, open(ck, 'w'))
            if log: print('%s s%d gen %d best %.1f K=%d S=%d restarts %d (%.1fs, %d evals)' % (corpus, seed, gen + 1, pop[0][0], pop[0][1]['K'], sum(pop[0][1]['nst']), restarts, time.time() - t0, evals), flush=True)
    return json.load(open(ck))


if __name__ == '__main__':
    corpus, seed, gens = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tag = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == '--tag' else 'main'
    run(corpus, seed, gens, tag)
