"""pe33 control: Ur III sealed tablets where seal content (legend title words) and tablet content are both
known. Same statistic and seal-block null as the PE test, at full size and subsampled to PE size
(63 tablets / 39 seals)."""
import json, sys, collections, os
import numpy as np
from pe33_common import assoc, CK

SCR = sys.argv[1]   # scratchpad dir with ur3_legends.json and x2/corpus_UR3.json
NSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 60
L = json.load(open(os.path.join(SCR, 'ur3_legends.json')))
C = {c['id']: c for c in json.load(open(os.path.join(SCR, 'x2', 'corpus_UR3.json')))}
KEY = {'kuruszda': 'kuruszda', 'szusz3': 'szusz3', 'sipa': 'sipa', 'muhaldim': 'muhaldim', 'szabra': 'szabra',
       'gu4': 'gu4', 'udu': 'udu', 'ugula': 'ugula', 'banda3': 'nu-banda3', 'ku6': 'ku6', 'gigir': 'gigir', 'i3': 'i3'}
rows = []
for x in L:
    p = 'P' + x['pid']
    if p not in C:
        continue
    toks = [s for l in x['lines'] for s in l]
    mot = {v for k, v in KEY.items() if k in toks}
    t = C[p]
    con = set()
    for e in t['entries']:
        if e['com']:
            con.add('S:' + e['com'])
        for d in e['des'][:3]:
            for w in d.split('-'):
                if w.isalpha():
                    con.add('S:' + w)
    if not con:
        continue
    n = len(t['entries'])
    rows.append(dict(id=p, seal=' / '.join(' '.join(l) for l in x['lines']), motif=mot, content=con,
                     vol=t['site'], size=0 if n <= 3 else 1 if n <= 8 else 2 if n <= 20 else 3))
print(len(rows), 'Ur III sealed tablets with legend + content', flush=True)
MOT = sorted(set(KEY.values()))


def mats(R, min_n=4):
    fc = collections.Counter(f for r in R for f in r['content'])
    n = len(R)
    F = sorted(f for f, c in fc.items() if min_n <= c <= n - min_n)
    mot = [m for m in MOT if min_n <= sum(m in r['motif'] for r in R) <= n - min_n]
    X = np.array([[f in r['content'] for f in F] for r in R], dtype=np.int8)
    Y = np.array([[m in r['motif'] for m in mot] for r in R], dtype=np.int8)
    return X, F, Y, mot


def seal_perm(Y, R, rng):
    seals = sorted({r['seal'] for r in R}); first = {}
    for i, r in enumerate(R):
        first.setdefault(r['seal'], i)
    sv = collections.defaultdict(list)
    for s in seals:
        sv[R[first[s]]['vol']].append(s)
    new = {}
    for v, S in sv.items():
        P = rng.permutation(len(S))
        for a, b in zip(S, P):
            new[a] = Y[first[S[b]]]
    return np.array([new[r['seal']] for r in R])


def test(R, nperm, rng):
    X, F, Y, mot = mats(R)
    if Y.shape[1] == 0 or X.shape[1] == 0:
        return None
    _, z = assoc(X, Y); az = np.abs(z)
    mx = np.array([np.abs(assoc(X, seal_perm(Y, R, rng))[1]).max() for _ in range(nperm)])
    thr = np.quantile(mx, .95)
    links = sorted([(round(float(az[i, j]), 2), mot[i], F[j]) for i, j in zip(*np.where(az > thr))], reverse=True)
    return dict(n=len(R), seals=len({r['seal'] for r in R}), max=float(az.max()), p_max=float((mx >= az.max()).mean()),
                n_links=len(links), links=links[:12], motifs=mot)


rng = np.random.default_rng(7)
# full size: one Drehem-free random 3000-tablet sample (keeps it lean)
idx = rng.choice(len(rows), min(3000, len(rows)), replace=False)
full = test([rows[i] for i in idx], 200, rng)
print('FULL', {k: v for k, v in full.items() if k != 'links'}, full['links'][:8], flush=True)
# PE-sized subsamples: 39 seals with 1-10 tablets, 63 tablets, drawn so that >=50% carry a motif word
byseal = collections.defaultdict(list)
for r in rows:
    byseal[r['seal']].append(r)
seals = [s for s, v in byseal.items() if any(r['motif'] for r in v)]
other = [s for s, v in byseal.items() if not any(r['motif'] for r in v)]
sub = []
for k in range(NSUB):
    rr = np.random.default_rng(100 + k)
    S = list(rr.choice(seals, 30, replace=False)) + list(rr.choice(other, 9, replace=False))
    R = []
    for s in S:
        v = byseal[s]
        R += [v[i] for i in rr.choice(len(v), min(len(v), int(rr.integers(1, 4))), replace=False)]
    R = R[:63]
    o = test(R, 200, rr)
    if o:
        sub.append(o)
        print('SUB', k, o['n'], o['seals'], round(o['max'], 2), o['p_max'], o['n_links'], o['links'][:3], flush=True)
pw = np.mean([o['p_max'] < 0.05 for o in sub])
print('power at PE size (p_max<0.05):', pw, 'of', len(sub))
json.dump(dict(full=full, sub=sub, power=float(pw)), open(f'{CK}/ur3.json', 'w'), indent=1)
