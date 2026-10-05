"""Cycle 2b: split-half re-test with selection on half A restricted to
sign-conditioned hypotheses explaining >= 2 half-A tablets, ranked by excess
over the noise-null mean; counted on half B (real vs null mean)."""
import json, random, collections, sys
import numpy as np
from pe42_common import CK, os

d = json.load(open(os.path.join(CK, 'c2_keys.json')))


def signcond(k):
    return any(str(s).startswith(('S:', 'L:')) for s in k) or (k[0] == 'WHOLE' and k[-1] != '*')


def run(name, nsplit=300, seed=3):
    D = d[name]; real = D['-1']; nulls = [D[k] for k in D if k != '-1']
    rows = [t for t, x in enumerate(real) if x is not None]
    R = [set(map(tuple, map(lambda z: tuple(map(str, z)), x))) if x else set() for x in real]
    N = [[set(tuple(map(str, z)) for z in x) if x else set() for x in n] for n in nulls]
    rng = random.Random(seed); gains = []; picks = collections.Counter(); none = 0
    for s in range(nsplit):
        rng.shuffle(rows); A = rows[:len(rows) // 2]; B = rows[len(rows) // 2:]
        ca = collections.Counter(k for t in A for k in R[t] if signcond(k))
        cand = [k for k, v in ca.items() if v >= 2]
        if not cand:
            none += 1; gains.append(0.0); continue
        na = {k: np.mean([sum(k in n[t] for t in A) for n in N]) for k in cand}
        best = max(cand, key=lambda k: (ca[k] - na[k], ca[k]))
        picks[best] += 1
        rb = sum(best in R[t] for t in B)
        nb = np.mean([sum(best in n[t] for t in B) for n in N])
        gains.append(rb - nb)
    g = np.array(gains)
    return ('%s: held-out gain %.2f (positive %.0f%%, no candidate %d/%d); picks %s'
            % (name, g.mean(), 100 * (g > 0).mean(), none, nsplit,
               '; '.join('%s x%d' % (' '.join(k), v) for k, v in picks.most_common(3))))


if __name__ == '__main__':
    for name in d:
        print(run(name))
