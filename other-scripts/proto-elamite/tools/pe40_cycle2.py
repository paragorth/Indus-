"""pe40 cycle 2: WHO SERVED LAST, WHO SERVES NEXT.  If tablets are rota steps that list the same households,
the shared households on two tablets should come in the same cyclic order but starting at a different point
(a nonzero rotation).  For every tablet pair sharing >= 3 player units (one per entry), classify the order of
shared units on tablet B relative to A: identical, reversed, nonzero cyclic rotation, other.
Nulls: entry order shuffled within each tablet (500x).  Planted: on 15% of tablets the entries carrying
player units are rewritten as rotations of a fixed 6-unit cycle (shared across those tablets).
Units: (a) rare signs (2..5% tablets), first occurrence per entry; (b) whole entry names (minus class sign)."""
import os as _o
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'): _o.environ[_v] = '1'
import json, collections, itertools
import numpy as np
from pe40_common import *


def tablet_units(kind):
    T = load(); out = []
    for t in T:
        if 'Susa' not in (t['provenience'] or ''):
            continue
        ents = []
        for l in t['lines']:
            if not l['numerals'] or l.get('header_comment'):
                continue
            s = [base(x) for x in l['signs'] if is_sign(x)]
            if not s or len(s) != len(l['signs']):
                continue
            if s[-1] in CLASS and len(s) > 1:
                s = s[:-1]
            ents.append(s)
        if ents:
            out.append((t['id'], ents))
    if kind == 'sign':
        df = collections.Counter(x for _, E in out for x in set(y for e in E for y in e))
        hi = int(0.05 * len(out))
        f = lambda e: [x for x in e if 2 <= df[x] <= hi and x not in CLASS]
    else:
        df = collections.Counter(' '.join(e) for _, E in out for e in set(tuple(x) for x in E))
        f = lambda e: [' '.join(e)] if df[' '.join(e)] >= 2 and len(e) >= 1 else []
    res = []
    for tid, E in out:
        u = []
        for e in E:
            c = f(e)
            if c:
                u.append(c[0])
        res.append((tid, u))
    return res


def classify(a, b):
    """a, b: sequences of the same k unique shared units."""
    k = len(a)
    if a == b:
        return 'same'
    if a == b[::-1]:
        return 'rev'
    for r in range(1, k):
        if b == a[r:] + a[:r]:
            return 'rot'
    return 'other'


def pairstats(U, kmin=3):
    first = []
    for tid, u in U:
        d = {}
        for i, x in enumerate(u):
            d.setdefault(x, i)
        first.append(d)
    inv = collections.defaultdict(list)
    for i, d in enumerate(first):
        for x in d:
            inv[x].append(i)
    cnt = collections.Counter(); exp_rot = 0.0; npair = 0
    seen = set()
    cand = collections.Counter()
    for x, L in inv.items():
        for i, j in itertools.combinations(L, 2):
            cand[(i, j)] += 1
    for (i, j), c in cand.items():
        if c < kmin:
            continue
        sh = [x for x in first[i] if x in first[j]]
        a = sorted(sh, key=lambda x: first[i][x]); b = sorted(sh, key=lambda x: first[j][x])
        cnt[classify(a, b)] += 1; npair += 1
        k = len(sh)
        exp_rot += (k - 1) / math.factorial(k) if k < 15 else 0
    return dict(npair=npair, same=cnt['same'], rev=cnt['rev'], rot=cnt['rot'], other=cnt['other'], exp_rot_random=exp_rot)


def shuffle_within(U, rng):
    return [(t, list(rng.permutation(np.array(u, dtype=object)))) for t, u in U]


def plant(U, rng, frac=0.15, k=6):
    U = [(t, list(u)) for t, u in U]
    cyc = ['PLANT%d' % i for i in range(k)]
    idx = rng.choice(len(U), int(frac * len(U)), replace=False)
    for i in idx:
        r = rng.integers(0, k); m = rng.integers(3, k + 1)
        seq = (cyc[r:] + cyc[:r])[:m]
        u = U[i][1]
        pos = sorted(rng.choice(len(u) + m, m, replace=False))
        new = list(u); 
        for p, s in zip(pos, seq):
            new.insert(p, s)
        U[i] = (U[i][0], new)
    return U


if __name__ == '__main__':
    out = {}
    for kind in ('sign', 'name'):
        U = tablet_units(kind)
        rng = np.random.default_rng(7)
        real = pairstats(U)
        nulls = [pairstats(shuffle_within(U, rng)) for _ in range(300)]
        rotn = np.array([n['rot'] for n in nulls]); samen = np.array([n['same'] for n in nulls])
        plants = {}
        for fr in (0.03, 0.015):
            pl = plant(U, rng, frac=fr, k=6); plr = pairstats(pl)
            pln = np.array([pairstats(shuffle_within(pl, rng))['rot'] for _ in range(100)])
            plants[str(fr)] = dict(rot=plr['rot'], npair=plr['npair'], null=float(pln.mean()),
                                   p=float((1 + (pln >= plr['rot']).sum()) / (1 + len(pln))))
        out[kind] = dict(plants=plants, real=real, null_rot_mean=float(rotn.mean()), null_rot_sd=float(rotn.std()),
                         p_rot=float((1 + (rotn >= real['rot']).sum()) / (1 + len(rotn))),
                         null_same_mean=float(samen.mean()), p_same=float((1 + (samen >= real['same']).sum()) / (1 + len(samen))),
                         )
        print(kind, json.dumps(out[kind]))
    json.dump(out, open(os.path.join(CK, 'cycle2.json'), 'w'), indent=1)
