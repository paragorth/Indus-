"""pe71 cycle 2c: control for c2b(c) -- V/n and top share at n = 50 of full sign strings from NON-closing lines
(unnumbered non-closing lines; numbered entry lines), 50 draws each; and the partner values of closing M153 compounds.
usage: pe71_c2c.py"""
import collections
import numpy as np
from pe71_lib import load, comps
from pe71_c2b import closing

rng = np.random.default_rng(7123)
R = load()
bare, ent = [], []
for r in R:
    k, _ = closing(r)
    for i, l in enumerate(r['lines']):
        if not l['signs'] or i == k:
            continue
        (ent if l['nums'] else bare).append(' '.join(l['signs']))
for nm, v in (('bare_nonclosing', bare), ('entry', ent)):
    pr = []
    for _ in range(50):
        c = collections.Counter(v[j] for j in rng.choice(len(v), 50, replace=False))
        pr.append((len(c) / 50, c.most_common(1)[0][1] / 50))
    print(nm, len(v), 'V/n %.3f top %.3f' % tuple(np.mean(pr, 0)))
vals = collections.Counter()
lots = collections.defaultdict(set)
for r in R:
    k, s = closing(r)
    if s:
        for t in s:
            if t.startswith('|') and 'M153' in comps(t):
                p = '+'.join(x for x in comps(t) if x != 'M153'); vals[p] += 1; lots[p].add(r['vol'])
print('closing M153 partner values', dict(vals), {k: sorted(v) for k, v in lots.items()}, 'V/n %.3f' % (len(vals) / sum(vals.values())))

# within-tablet position control: if each M153+{X,M342} token sat on a random signed line (or a random UNNUMBERED signed
# line) of its own tablet, how many would be closing lines?
obs = e_any = e_bare = 0.0
nt = 0
for r in R:
    k, _ = closing(r)
    sl = [i for i, l in enumerate(r['lines']) if l['signs']]
    bl = [i for i in sl if not r['lines'][i]['nums']]
    for i, l in enumerate(r['lines']):
        for t in set(l['signs']):
            if t in ('|M153+X|', '|M153+M342|'):
                nt += 1; obs += i == k
                e_any += (k is not None) / len(sl)
                e_bare += (k is not None) / len(bl) if bl else 0
print('within-tablet control: tokens %d closing %d; expected if on a random signed line %.1f, random unnumbered line %.1f' % (nt, obs, e_any, e_bare))
ps = []
for r in R:
    k, _ = closing(r)
    bl = [i for i, l in enumerate(r['lines']) if l['signs'] and not l['nums']]
    for l in r['lines']:
        for t in set(l['signs']):
            if t in ('|M153+X|', '|M153+M342|'):
                ps.append((k is not None) / len(bl))
ps = np.array(ps)
sim = (rng.random((100000, len(ps))) < ps).sum(1)
print('Poisson-binomial p (random unnumbered line) = %.5f' % (sim >= 24).mean())
