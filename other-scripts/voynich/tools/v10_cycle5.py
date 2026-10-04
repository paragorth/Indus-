"""v10 cycle 5: is the 'return to the glyph two lines up' (lag-2/3 excess over a Markov-1 surrogate, cycle 4) real
second-order memory, or just paragraph composition (some paragraphs are t- or q-heavy) combined with the lag-1 rule?
Null = Markov-weighted permutation: inside each paragraph, lines 2..n are re-ordered by MCMC (pairwise swaps,
Metropolis on the product of global lag-1 transition probabilities T(a->b), 40 sweeps), so each paragraph keeps its
exact glyph multiset AND the chain keeps (approximately) the lag-1 preference; second-order memory is not kept.
Statistics: lag-2 / lag-3 same-glyph rate, lag-2 MI, ABA count, per-glyph 2-step return.
Controls: planted order-2 chain (cycle 4 plant: copy the glyph two up with p = 0.25) must be flagged; a planted
'composition + Markov-1' chain (each paragraph's body re-sampled from the global Markov-1 table, then reweighted to
that paragraph's own composition by the same MCMC from a random start) must not be flagged."""
import sys, os, json, random, math
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v10_lib import *
from v10_cycle4 import load as load4, fit, body_seqs

REPS = int(os.environ.get('REPS', 200))

def logT(paras):
    T = defaultdict(Counter)
    for s in body_seqs(paras):
        for a, b in zip(s, s[1:]): T[a][b] += 1
    syms = sorted({x for s in body_seqs(paras) for x in s})
    return {(a, b): math.log((T[a][b] + 0.5) / (sum(T[a].values()) + 0.5 * len(syms))) for a in syms for b in syms}

def mcmc_perm(s, L, rng, sweeps=40):
    s = list(s); n = len(s)
    if n < 3: return s
    def loc(i):  # log-weight of edges touching position i
        t = 0.0
        if i > 0: t += L[(s[i - 1], s[i])]
        if i < n - 1: t += L[(s[i], s[i + 1])]
        return t
    rng.shuffle(s)
    for _ in range(sweeps * n):
        i, j = rng.sample(range(n), 2)
        if s[i] == s[j]: continue
        idx = {i, j, max(0, i - 1), max(0, j - 1)}
        before = sum(L[(s[k], s[k + 1])] for k in idx if k < n - 1)
        s[i], s[j] = s[j], s[i]
        after = sum(L[(s[k], s[k + 1])] for k in idx if k < n - 1)
        if after < before and rng.random() >= math.exp(after - before): s[i], s[j] = s[j], s[i]
    return s

def stats(seqs):
    R = {}
    for lag in (1, 2, 3):
        P = [(s[i], s[i + lag]) for s in seqs for i in range(len(s) - lag)]
        R['lag%d same' % lag] = sum(a == b for a, b in P) / len(P)
    P2 = [(s[i], s[i + 2]) for s in seqs for i in range(len(s) - 2)]; R['lag2 mi'] = mi(P2)
    R['ABA rate'] = sum(s[i] == s[i + 2] != s[i + 1] for s in seqs for i in range(len(s) - 2)) / len(P2)
    c = Counter(); r = Counter()
    for s in seqs:
        for i in range(len(s) - 2): c[s[i]] += 1; r[s[i]] += s[i] == s[i + 2]
    for g in 'dyoqstSC': R['ret2 ' + g] = r[g] / max(1, c[g])
    return R

_D = {}
def data(name, plant):
    if (name, plant) in _D: return _D[(name, plant)]
    if plant == 'comp_markov':
        paras = chain_paras(name); L = logT(paras); rng = random.Random(11)
        for p in paras: p['chain'] = [p['chain'][0]] + mcmc_perm(p['chain'][1:], L, rng)
    else:
        paras = load4(name, plant)
    _D[(name, plant)] = (body_seqs(paras), logT(paras))
    return _D[(name, plant)]

def job(a):
    name, plant, seed = a; seqs, L = data(name, plant); rng = random.Random(seed)
    return stats([mcmc_perm(s, L, rng) for s in seqs])

if __name__ == '__main__':
    with Pool(2) as pool:
        for name, plant in (('ZL3b', None), ('IT2a', None), ('ZL3b', 'order2'), ('ZL3b', 'comp_markov')):
            ck = os.path.join(CK, 'c5_%s_%s.json' % (name, plant))
            if os.path.exists(ck): print(name, plant, open(ck).read()); continue
            seqs, L = data(name, plant); o = stats(seqs)
            res = pool.map(job, [(name, plant, s) for s in range(REPS if plant is None else 100)])
            S = {k: zstat(o[k], [r[k] for r in res]) for k in o}
            json.dump(S, open(ck, 'w'), indent=1); print(name, plant, json.dumps(S), flush=True)
