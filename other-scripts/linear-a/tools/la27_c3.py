#!/usr/bin/env python3
"""LA-27 cycle 3: (a) Linear B control for the ladder search; (b) do the Linear A fraction
values (la1 candidates) fall on physical sub-unit weights?

(a) LB weight notation (DAMOS): in a written amount 'U1 c1 U2 c2', a subordinate unit's count
    c2 must stay below the ratio U1:U2 (mixed radix). The search sees only the counts at each
    subordinate pair (L>M, LANA>M, M>N, N>P, M>P); a ladder assigns ratios (L:M, LANA:M, M:N,
    N:P, P-under-M = M:N x N:P). Score = held-out (2-fold by document, 20 splits) log-lik of
    the counts under uniform{1..r-1} with error floor 1e-3; 10^4 random ladders (each ratio
    log-uniform on 2..60). Truth (Bennett 1950): L:M 30, M:N 4; LANA:M 3 is Michailidou's
    reading; N:P unknown (Bennett's guess 12). Recovered = modal ratio of the best ladders.
    Planted: LA-sized counts (same n per pair) drawn uniform below a random ratio vector.
    Shuffle: counts permuted across unit pairs.
(b) Fraction value sets: conventional (J 1/2, E 1/4, F 1/8, K 1/16, D 1/5, B 1/3, A=H 1/6,
    JE 3/4), attack-1 binary (L 7/16, E 3/8, J 5/16, A=F=H=JE 1/4, B=D=K=Y 1/8, L2=L6 1/16,
    L4 1/32), pure binary {1/2,1/4,1/8,1/16}, thirds {1/3,2/3,1/6}. Score = share of a set's
    distinct values v whose mass B x v has a core physical weight within 4 %; base B = 61 g
    unit, 65.5 g, 488 g mina. Null: 10^4 random sets of the same size (distinct p/q, q <= 16,
    0 < v < 1). Planted: one mass per value of a random set added (3 % noise) to the masses.
    Jitter: masses x U(0.85, 1.15).
"""
import json, math, os, re, sys
from collections import Counter, defaultdict
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la27_ckpt')
rng = np.random.default_rng(327)
RANK = {'L': 5, '*118': 5, 'LANA': 4, 'M': 3, 'N': 2, 'P': 1, 'Q': 0}
PAIRS = ['L>M', 'LANA>M', 'M>N', 'N>P', 'M>P']

def lb_counts():
    rec = []
    for l in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(l); s = d.get('content') or ''
        for line in s.split('\n'):
            t = re.findall(r'(\*118|LANA|\b[LMNPQ]\b)\s+(\d+)', line)
            for (u0, c0), (u, c) in zip(t, t[1:]):
                u0 = 'L' if u0 == '*118' else u0
                k = u0 + '>' + u
                if RANK[u0] > RANK[u] and k in PAIRS: rec.append((d['id'], k, int(c)))
    return rec

def ratios_of(L):  # L = (LM, LANAM, MN, NP)
    return {'L>M': L[0], 'LANA>M': L[1], 'M>N': L[2], 'N>P': L[3], 'M>P': L[2] * L[3]}

def loglik(recs, L, eps=1e-3):
    r = ratios_of(L); s = 0.0
    for _, k, c in recs:
        s += math.log((1 - eps) / (r[k] - 1) + eps / 100) if c < r[k] else math.log(eps / 100)
    return s

def search(recs, nl=10000, splits=20):
    lad = [tuple(int(round(math.exp(rng.uniform(math.log(2), math.log(60))))) for _ in range(4))
           for _ in range(nl)]
    docs = sorted({r[0] for r in recs}); tot = np.zeros(nl)
    for sp in range(splits):
        perm = rng.permutation(len(docs)); A = {docs[i] for i in perm[::2]}
        te = [r for r in recs if r[0] not in A]
        tot += np.array([loglik(te, L) for L in lad])
    top = np.argsort(-tot)[:50]
    est = []
    for j, name in enumerate(['L:M', 'LANA:M', 'M:N', 'N:P']):
        # for each ratio: value maximising the held-out score when the others are free
        best = Counter(lad[i][j] for i in top).most_common(3)
        est.append((name, best))
    return dict(best=lad[int(np.argmax(tot))], modal_top50=est)

def part_a():
    recs = lb_counts()
    out = dict(n={k: sum(1 for r in recs if r[1] == k) for k in PAIRS},
               max={k: max([r[2] for r in recs if r[1] == k] or [0]) for k in PAIRS})
    out['real'] = search(recs)
    # planted: same docs and pairs, counts uniform below a random ratio vector
    pl = []
    for _ in range(3):
        T = tuple(int(x) for x in rng.integers(3, 40, 4))
        r = ratios_of(T)
        fake = [(d, k, int(rng.integers(1, r[k]))) for d, k, _ in recs]
        res = search(fake, nl=4000, splits=8)
        pl.append(dict(truth=T, best=res['best'], modal=res['modal_top50']))
    out['planted'] = pl
    # shuffle: counts permuted across pairs
    cs = [r[2] for r in recs]; rng.shuffle(cs)
    out['shuffled'] = search([(d, k, c) for (d, k, _), c in zip(recs, cs)], nl=4000, splits=8)
    return out

SETS = {'conventional': ['1/2', '1/4', '1/8', '1/16', '1/5', '1/3', '1/6', '3/4'],
        'attack1_binary': ['7/16', '3/8', '5/16', '1/4', '1/8', '1/16', '1/32'],
        'pure_binary': ['1/2', '1/4', '1/8', '1/16'],
        'thirds': ['1/3', '2/3', '1/6']}

def score(vals, masses, B, tol=.04):
    m = np.asarray(masses)
    return float(np.mean([np.min(np.abs(m / (B * v) - 1)) < tol for v in vals]))

def rand_set(k):
    pool = sorted({Fr(p, q) for q in range(2, 17) for p in range(1, q)})
    idx = rng.choice(len(pool), k, replace=False)
    return [float(pool[i]) for i in idx]

def part_b():
    W = json.load(open(os.path.join(D, 'la27_weights.json')))
    m = [w['g'] for w in W if w['core']]
    out = {}
    for B in (61.0, 65.5, 488.0):
        for name, S in SETS.items():
            v = [float(Fr(x)) for x in S]
            obs = score(v, m, B)
            null = np.array([score(rand_set(len(v)), m, B) for _ in range(10000)])
            jit = np.mean([score(v, list(np.array(m) * rng.uniform(.85, 1.15, len(m))), B) for _ in range(200)])
            hits = [x for x, vv in zip(S, v) if np.min(np.abs(np.array(m) / (B * vv) - 1)) < .04]
            out['%s@%g' % (name, B)] = dict(score=obs, null_mean=round(float(null.mean()), 3),
                                            p=float((null >= obs).mean()), jitter_mean=round(float(jit), 3), hits=hits)
        # planted power: add one mass per value of a random 6-value set
        pw = []
        for _ in range(200):
            V = rand_set(6)
            mm = m + [B * x * (1 + rng.normal(0, .03)) for x in V]
            obs = score(V, mm, B)
            null = np.array([score(rand_set(6), mm, B) for _ in range(300)])
            pw.append((null >= obs).mean() < .05)
        out['planted_power@%g' % B] = float(np.mean(pw))
    return out

if __name__ == '__main__':
    res = dict(a=part_a()); print(json.dumps(res['a'], default=str), flush=True)
    res['b'] = part_b(); print(json.dumps(res['b'], indent=0))
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), indent=1, default=str)
