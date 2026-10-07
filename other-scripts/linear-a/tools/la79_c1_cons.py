"""Robustness of consonant-group avoidance: drop pure-vowel group; within-word shuffle null; per site group; read-only."""
import sys, os, numpy as np, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la79_common as C, la79_claims as K
rng = np.random.default_rng(7)

def words(docs, readonly=False):
    W = []
    for d in docs:
        for w in d['words']:
            if readonly and not w['read']:
                continue
            s = [x for x in w['s'] if K.consonant(x) is not None]
            if len(s) >= 2:
                W.append((d['g'], s))
    return W

def stat(W, g, novowel):
    c = 0
    for _, s in W:
        for a, b in zip(s, s[1:]):
            if a != b and g[a] == g[b] and not (novowel and g[a] == 'V'):
                c += 1
    return c

def test(W, g, novowel, nperm=200):
    obs = stat(W, g, novowel)
    null = []
    for _ in range(nperm):
        WW = [(x, list(rng.permutation(s))) for x, s in W]
        null.append(stat(WW, g, novowel))
    null = np.array(null)
    return obs, null.mean(), (obs - null.mean()) / max(null.std(), .5)

D = C.load()
signs = sorted({x for d in D for w in d['words'] for x in w['s'] if K.consonant(x) is not None})
g = {s: K.consonant(s) for s in signs}
vals = [g[s] for s in signs]
for lab, W in (('all', words(D)), ('read', words(D, True))):
    print(lab, 'within-word null, incl V', test(W, g, False), 'excl V', test(W, g, True))
W = words(D)
for grp in C.GROUPS:
    Wg = [x for x in W if x[0] == grp]
    print(grp, len(Wg), test(Wg, g, True))
# decoys under within-word null (excl V where real V): 100 permuted maps
dz = []
for i in range(100):
    gd = dict(zip(signs, rng.permutation(vals)))
    dz.append(test(W, gd, False, nperm=40)[2])
print('decoy z (within-word null) mean %.2f, 5%% %.2f, min %.2f' % (np.mean(dz), np.percentile(dz, 5), np.min(dz)))
