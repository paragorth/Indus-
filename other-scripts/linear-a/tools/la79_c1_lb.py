"""LB shared vocabulary with a position-preserving null (first, middle, last signs shuffled separately)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la79_common as C, la79_claims as K
rng = np.random.default_rng(8)
D = C.load(); LEX = K.lb_lexicon()
def z(docs, n=300):
    T = sorted(K.types_of(docs, 3))
    obs = sum(t in LEX for t in T)
    f = [t[0] for t in T]; l = [t[-1] for t in T]; m = [s for t in T for s in t[1:-1]]
    null = []
    for _ in range(n):
        ff = rng.permutation(f); ll = rng.permutation(l); mm = list(rng.permutation(m)); k = 0; c = 0
        for i, t in enumerate(T):
            nm = len(t) - 2; c += (tuple([ff[i]] + mm[k:k + nm] + [ll[i]]) in LEX); k += nm
        null.append(c)
    null = np.array(null)
    return obs, round(null.mean(), 2), round((obs - null.mean()) / max(null.std(), .5), 2), len(T)
print('full', z(D))
y = np.array([d['year'] for d in D])
for cut in (1950, 1976):
    print('after', cut, z([d for d, yy in zip(D, y) if yy > cut]))
for g in C.GROUPS:
    print('site', g, z([d for d in D if d['g'] == g]))
print('non-HT', z([d for d in D if d['g'] != 'HT']))
