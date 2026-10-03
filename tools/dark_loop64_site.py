"""S-DARK-64 cycle 2b: grain-matched check for the Ur III city control. Indus elements (dark_loop64 parser) x SITE, one per distinct text per site,
MI/H(site) vs site labels permuted within type x length (200x). Usage: python3 tools/dark_loop64_site.py <seq_raw|seq_strong|seq_all>"""
import sys, collections, random
lv = sys.argv[1]
sys.argv = ['x', '0', lv, '200']
src = open('/home/user/Indus-/tools/dark_loop64.py').read()
g = {'__file__': '/home/user/Indus-/tools/dark_loop64.py', '__name__': 'x'}
exec(compile(src, 'dl64', 'exec'), g)
OBJ = g['OBJ']; mi_loyal = g['mi_loyal']; rnd = random.Random(7)
for oc in ('seal', 'tablet'):
    seen = set(); U = []
    for o in OBJ:
        if o['oc'] != oc or not o['el']: continue
        k = (o['site'], o['seq'])
        if k in seen: continue
        seen.add(k); U.append(o)
    sites = collections.Counter(o['site'] for o in U)
    obs = mi_loyal([(o['el'], o['site']) for o in U], None)
    S = collections.defaultdict(list)
    for i, o in enumerate(U): S[(o['typ'], min(o['L'], 7))].append(i)
    nm = []; nmin = []
    for _ in range(200):
        lab = [None] * len(U)
        for k, idx in S.items():
            l = [U[i]['site'] for i in idx]; rnd.shuffle(l)
            for i, x in zip(idx, l): lab[i] = x
        a = mi_loyal([(o['el'], l) for o, l in zip(U, lab)], None); nm.append(a[0]); nmin.append(a[3])
    p = (1 + sum(x >= obs[0] for x in nm)) / (1 + len(nm))
    print(f'{lv} {oc}s by SITE: {len(U)} distinct texts, sites {dict(sites.most_common(5))}; MI {obs[0]:.1f}% vs {sum(nm)/len(nm):.1f}% (excess {obs[0]-sum(nm)/len(nm):+.1f}, P={p:.3f}); loyal elements {obs[1]:.3f} of {obs[2]}; loyal to non-dominant site {obs[3]} vs {sum(nmin)/len(nmin):.1f}')
