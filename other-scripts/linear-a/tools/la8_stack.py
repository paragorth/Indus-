#!/usr/bin/env python3
"""LA-8 stage 2: does a stack (recursion) pay? Over the fixed word classes of the best stage-1 machine,
evolve the slot structure (1-3 states per class) with or without a one-counter stack (each state may
push or pop on entry; END needs an empty stack; depth <= 4). Equal compute for both arms.
Fitness: held-out transition bits (2-fold CV) + DL. Recursion gain = best FSA fitness - best PDA fitness.
Usage: la8_stack.py CORPUS ARM(fsa|pda) SEED GENS [truth]"""
import ctypes, json, math, os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la8_gp as G
from la8_truth import truth_genome

lib = G.lib
I32, F64 = G.I32, G.F64
lib.score_pda.restype = ctypes.c_double
lib.score_pda.argtypes = [ctypes.c_int, ctypes.c_int, I32, I32, I32, ctypes.c_int, I32, I32, ctypes.c_double, ctypes.c_int,
                          ctypes.c_int, F64, F64, F64]


class SData:
    def __init__(self, corpus, use_truth=False, tag='main'):
        self.D = D = G.Data(json.load(open(os.path.join(G.OUT, 'corpus_%s.json' % corpus))))
        if use_truth:
            g, _ = truth_genome(D, json.load(open(os.path.join(G.OUT, 'planted_truth.json')))[corpus])
        else:
            g = G.from_json(json.load(open(os.path.join(G.OUT, 'best_%s_%s.json' % (corpus, tag))))['genome'])
        self.g = g; self.K = g['K']
        self.tc = np.array([int(g['tc'][t]) if t >= 0 else int(g['kc'][k]) for t, k in zip(D.ttype, D.tkey)], dtype=np.int32)

    def ev(self, nst, act, mode=0):
        out = np.zeros(4); S = int(sum(nst)); tr = np.zeros((S + 1) * (S + 1)); vd = np.zeros(1)
        b = lib.score_pda(self.D.T, len(self.D.docs), self.D.off, self.D.fold, self.tc, self.K, nst, act, G.DELTA, G.ITERS, mode, out, tr, vd)
        return b, out, tr.reshape(S + 1, S + 1)

    def fit(self, nst, act):
        b, out, _ = self.ev(nst, act)
        S = int(sum(nst)); lk = math.log2(max(self.K, 2))
        dl = S * lk + out[1] * (2 * math.log2(S + 2) + 0.5 * math.log2(self.D.T)) + sum(1 + math.log2(S + 1) for a in act if a)
        return b + dl, b, out


def mutate(nst, act, arm, rng):
    nst = list(nst); act = list(act)
    # act is per state; rebuild when nst changes
    first = np.cumsum([0] + nst)[:-1].tolist()
    per = [act[first[c]:first[c] + nst[c]] for c in range(len(nst))]
    for _ in range(1 + min(3, int(rng.expovariate(1.0)))):
        c = rng.randrange(len(nst)); r = rng.random()
        if arm == 'fsa' or r < 0.4:
            d = rng.choice((-1, 1))
            if 1 <= nst[c] + d <= G.MAXST:
                nst[c] += d
                if d > 0: per[c].append(0)
                else: per[c].pop(rng.randrange(len(per[c])))
        else:
            j = rng.randrange(nst[c]); per[c][j] = rng.choice((0, 1, 2)) if rng.random() < 0.7 else 0
    act = [a for p in per for a in p]
    return np.array(nst, dtype=np.int32), np.array(act, dtype=np.int32)


def run(corpus, arm, seed, gens, use_truth=False, mu=8, lam=16):
    tag = ('truth_' if use_truth else '') + arm
    ck = os.path.join(G.OUT, 'stack_%s_%s_s%d.json' % (corpus, tag, seed))
    if os.path.exists(ck) and json.load(open(ck)).get('done'): return
    SD = SData(corpus, use_truth)
    rng = random.Random(seed * 31 + len(corpus))
    base = np.array(SD.g['nst'], dtype=np.int32)
    gen0 = 0
    if os.path.exists(ck):
        j = json.load(open(ck)); gen0 = j['gen']
        pop = [(p['f'], np.array(p['nst'], dtype=np.int32), np.array(p['act'], dtype=np.int32)) for p in j['pop']]
    else:
        pop = []
        for i in range(mu):
            nst = base.copy() if i == 0 else np.array([rng.randint(1, G.MAXST) if rng.random() < 0.3 else 1 for _ in range(SD.K)], dtype=np.int32)
            act = np.zeros(int(nst.sum()), dtype=np.int32)
            if arm == 'pda' and i > 0: act = np.array([rng.choice((0, 0, 0, 1, 2)) for _ in act], dtype=np.int32)
            pop.append((SD.fit(nst, act)[0], nst, act))
    t0 = time.time()
    for gen in range(gen0, gens):
        kids = []
        for _ in range(lam):
            a = min(rng.sample(pop, 2), key=lambda x: x[0])
            n2, a2 = mutate(a[1], a[2], arm, rng)
            kids.append((SD.fit(n2, a2)[0], n2, a2))
        allp = sorted(pop + kids, key=lambda x: x[0]); pop, seen = [], set()
        for x in allp:
            k = round(x[0], 3)
            if k in seen: continue
            seen.add(k); pop.append(x)
            if len(pop) == mu: break
        if (gen + 1) % 25 == 0 or gen + 1 == gens:
            f, nst, act = pop[0]
            fb, b, out = SD.fit(nst, act)
            res = {'corpus': corpus, 'arm': arm, 'truth': use_truth, 'gen': gen + 1, 'done': gen + 1 == gens,
                   'pop': [{'f': x[0], 'nst': x[1].tolist(), 'act': x[2].tolist()} for x in pop],
                   'best': {'f': f, 'heldout_trans_bits': b, 'edges': out[1], 'depth_mass': out[2], 'S': int(nst.sum()),
                            'push': int((act == 1).sum()), 'pop': int((act == 2).sum())}}
            if arm == 'pda' and (act > 0).any():   # ablation: same slots, stack removed, refit
                fz, bz, _ = SD.fit(nst, np.zeros_like(act)); res['best']['ablate_stack_f'] = fz; res['best']['ablate_stack_bits'] = bz
            json.dump(res, open(ck, 'w'))
            print('%s %s s%d gen %d f %.1f S=%d push %d pop %d depth_mass %.1f (%.0fs)' % (corpus, tag, seed, gen + 1, f, nst.sum(), (act == 1).sum(), (act == 2).sum(), out[2], time.time() - t0), flush=True)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), len(sys.argv) > 5 and sys.argv[5] == 'truth')
