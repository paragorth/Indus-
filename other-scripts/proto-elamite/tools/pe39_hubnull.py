"""Hub-preserving null for seed agreement: in each seed, the source -> target
lexicon is permuted among source signs of similar frequency (+-8 ranks), which
keeps every seed's target popularity (hubs) but breaks the link to the sign.
Count of stable sources (modal target in >= k seeds) vs this null."""
import sys, os, json, glob, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe39_common import CK
MINN = 15
runs = collections.defaultdict(list)
for tag in ('c2', 'c3'):
    for fn in glob.glob(os.path.join(CK, '%s_*_prof_*.json' % tag)):
        r = json.load(open(fn)); runs[r['set']].append(r)


def n_stable(lexs, k):
    tg = collections.defaultdict(list)
    for L in lexs:
        for s, t in L.items():
            tg[s].append(t)
    return sum(1 for s, ts in tg.items() if collections.Counter(ts).most_common(1)[0][1] >= k)


out = {}
rng = random.Random(0)
for st, R in sorted(runs.items()):
    lexs = [{s: t[0] for s, t in r['lexAB'].items() if t[4] >= MINN} for r in R]
    freq = {s: t[4] for r in R for s, t in r['lexAB'].items() if t[4] >= MINN}
    order = sorted(freq, key=lambda s: -freq[s])
    res = {}
    for k in range(2, len(R) + 1):
        obs = n_stable(lexs, k)
        nul = []
        for it in range(300):
            pl = []
            for L in lexs:
                srcs = [s for s in order if s in L]
                perm = list(srcs)
                for i in range(len(perm)):           # local shuffle within +-8 ranks
                    j = min(len(perm) - 1, max(0, i + rng.randint(-8, 8)))
                    perm[i], perm[j] = perm[j], perm[i]
                pl.append({s: L[p] for s, p in zip(srcs, perm)})
            nul.append(n_stable(pl, k))
        p = (1 + sum(x >= obs for x in nul)) / 301
        res[k] = (obs, round(float(np.mean(nul)), 2), round(p, 4))
        print(st, 'k>=%d/%d' % (k, len(R)), 'obs', obs, 'hub-null mean %.2f' % np.mean(nul), 'p %.4f' % p)
    out[st] = res
json.dump(out, open(os.path.join(CK, 'hubnull_c3.json'), 'w'))
