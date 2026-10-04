"""pe25 cycle 4: random-rule scan for lines written by another hand.

~2,500 random line predicates ("the second hand writes lines that ..."): atoms on
face, position (header, last obverse line, first/last reverse line, second half),
contained base sign, numeral unit present, no numeral, 1 sign, 3+ signs, and random
two-atom conjunctions.  For each rule P and tablet: variant agreement between P
lines and the tablet's other lines (context-differing pairs), minus the same with
the other lines of 3 same-stratum partner tablets -> H_P.  Reference: 300 random
line subsets (each line in with prob q) give H under 'same hand' and its spread
by cell count -> z.  Discovery on half A of tablets (split by tablet id hash),
the 20 lowest-z rules re-tested on half B.
Controls: (i) the same pipeline on a single-hand planted corpus (no rule should
replicate); (ii) a planted corpus where lines containing an N14 numeral, and
separately the last obverse line, are written by one of 5 checkers: the planted
rule must rank top in A and replicate in B.
"""
import sys, random, json, hashlib
import numpy as np
from pe25_common import *
import pe25_cycle1 as c1

OUT = os.path.join(HERE, '..', 'loops', 'pe25_cycle4.txt')
d = c1.d
vb = c1.vb
rng = random.Random(44)

# per tablet line records
T = []
for t in d:
    obv, rev, edge = faces(t)
    L = []
    nobv, nrev = len(obv), len(rev)
    for k, l in enumerate(obv + rev):
        face = 'obv' if k < nobv else 'rev'
        pos = k if face == 'obv' else k - nobv
        n = nobv if face == 'obv' else nrev
        L.append(dict(face=face, pos=pos, n=n, hdr=bool(face == 'obv' and pos == 0 and (l.get('header_comment') or (l['signs'] and not l['numerals']))),
                      bases=set(base(s) for s in l['signs']), units=set(u.split('@')[0] for c, u in l['numerals']),
                      nsign=len(l['signs']), tok=tokens([l], vb)))
    T.append(L)

half = {i: int(hashlib.md5(t['id'].encode()).hexdigest(), 16) % 2 for i, t in enumerate(d)}
partners = {}
for i in range(len(d)):
    js = []
    for _ in range(3):
        j = null_partner(i, c1.groups, c1.keyof, rng)
        if j is None:
            j = null_partner(i, c1.groups_s, c1.keyof_s, rng)
        if j is not None:
            js.append(j)
    partners[i] = js

# atoms
common_bases = [b for b, c in __import__('collections').Counter(b for L in T for l in L for b in l['bases']).most_common(80)]
units = ['N01', 'N14', 'N39B', 'N30C', 'N24', 'N34', 'N30D', 'N45', 'N23', 'N08A']
ATOMS = {
    'rev': lambda l: l['face'] == 'rev', 'obv': lambda l: l['face'] == 'obv',
    'hdr': lambda l: l['hdr'], 'lastobv': lambda l: l['face'] == 'obv' and l['pos'] == l['n'] - 1,
    'firstrev': lambda l: l['face'] == 'rev' and l['pos'] == 0, 'lastrev': lambda l: l['face'] == 'rev' and l['pos'] == l['n'] - 1,
    'obv2ndhalf': lambda l: l['face'] == 'obv' and l['pos'] >= l['n'] / 2, 'nonum': lambda l: not l['units'],
    'onesign': lambda l: l['nsign'] == 1, 'long': lambda l: l['nsign'] >= 3,
}
for b in common_bases:
    ATOMS['has_' + b] = (lambda b: lambda l: b in l['bases'])(b)
for u in units:
    ATOMS['num_' + u] = (lambda u: lambda l: u in l['units'])(u)


def make_rules(n, r):
    names = list(ATOMS)
    rules = [(a,) for a in names]
    seen = set(rules)
    while len(rules) < n:
        a, b = r.sample(names, 2)
        k = tuple(sorted((a, b)))
        if k not in seen:
            seen.add(k)
            rules.append(k)
    return rules


def split_tab(L, pred):
    P, Q = [], []
    for l in L:
        (P if pred(l) else Q).extend(l['tok'])
    return P, Q


def score(TT, pred, idx):
    real, nul = [], []
    for i in idx:
        P, Q = split_tab(TT[i], pred)
        if not P or not Q:
            continue
        c = cells(P, Q, True)
        if not c:
            continue
        real.extend(c)
        for j in partners[i]:
            _, Qj = split_tab(TT[j], pred)
            nul.extend(cells(P, Qj, True))
    if len(real) < 15 or not nul:
        return None
    a, a0 = np.mean(real), np.mean(nul)
    return (a - a0) / (1 - a0) if a0 < 1 else None, len(real)


def reference(TT, idx, r, n=150):
    out = []
    for k in range(n):
        q = r.choice([0.1, 0.2, 0.3, 0.5])
        salt = r.random()
        pred = (lambda q, salt: lambda l: (hash((id(l), salt)) % 1000) / 1000 < q)(q, salt)
        s = score(TT, pred, idx)
        if s and s[0] is not None:
            out.append(s)
    H = np.array([s[0] for s in out]); N = np.array([s[1] for s in out])
    mu = H.mean()
    k = np.sqrt(np.mean((H - mu) ** 2 * N))   # sd ~ k / sqrt(n)
    return mu, k


def scan(TT, label, rules, r):
    A = [i for i in range(len(TT)) if half[i] == 0]
    B = [i for i in range(len(TT)) if half[i] == 1]
    muA, kA = reference(TT, A, r)
    muB, kB = reference(TT, B, r)
    res = []
    for rule in rules:
        pred = (lambda rule: lambda l: all(ATOMS[a](l) for a in rule))(rule)
        s = score(TT, pred, A)
        if not s or s[0] is None:
            continue
        res.append((rule, s[0], s[1], (s[0] - muA) / (kA / np.sqrt(s[1]))))
    res.sort(key=lambda x: x[3])
    top = res[:20]
    rep = []
    for rule, h, n, z in top:
        pred = (lambda rule: lambda l: all(ATOMS[a](l) for a in rule))(rule)
        s = score(TT, pred, B)
        zb = (s[0] - muB) / (kB / np.sqrt(s[1])) if s and s[0] is not None else None
        rep.append(dict(rule='&'.join(rule), HA=h, nA=n, zA=z, HB=s[0] if s else None, nB=s[1] if s else None, zB=zb))
    nrep = sum(1 for x in rep if x['zB'] is not None and x['zB'] < -2)
    print(label, 'rules scored', len(res), 'ref A', round(muA, 3), 'replicated', nrep, flush=True)
    for x in rep[:8]:
        print('  ', x, flush=True)
    return dict(label=label, nscored=len(res), muA=muA, muB=muB, top=rep, nrep=nrep)


def planted(kind, seed, alpha=1.0, preds=()):
    r = random.Random(seed)
    checkers = [c1.profile(alpha, r) for _ in range(5)]
    out = []
    for L in T:
        h = c1.profile(alpha, r)
        ch = r.choice(checkers)
        NL = []
        for l in L:
            hh = ch if any(p(l) for p in preds) else h
            n = dict(l)
            n['tok'] = [(b, c1.draw(hh, b, r), k) for b, v, k in l['tok']]
            NL.append(n)
        out.append(NL)
    return out


if __name__ == '__main__':
    rules = make_rules(2500, random.Random(4))
    allres = {}
    allres['REAL'] = scan(T, 'REAL', rules, random.Random(5))
    dump('cycle4.json', allres)
    allres['S1'] = scan(planted('S1', 61), 'S1', rules, random.Random(6))
    dump('cycle4.json', allres)
    allres['PLANT_N14'] = scan(planted('P', 62, preds=(ATOMS['num_N14'],)), 'PLANT_N14', rules, random.Random(7))
    allres['PLANT_LASTOBV'] = scan(planted('P', 63, preds=(ATOMS['lastobv'],)), 'PLANT_LASTOBV', rules, random.Random(8))
    dump('cycle4.json', allres)
    def top3(x):
        return '; '.join('%s zA %.1f zB %s' % (t['rule'], t['zA'], '%.1f' % t['zB'] if t['zB'] is not None else 'na') for t in x['top'][:3])
    for k, rid in (('S1', 'PE-25.4a'), ('PLANT_N14', 'PE-25.4b'), ('PLANT_LASTOBV', 'PE-25.4c'), ('REAL', 'PE-25.4d')):
        x = allres[k]
        row(OUT, rid, 'RANDOM-RULE SCAN (%s): %d random line predicates (atoms: face, header, last obverse, first/last reverse, 2nd half, 80 contained signs, 10 numeral units, no numeral, 1 or 3+ signs; 2-atom conjunctions) scored by hand index of predicate lines vs rest of tablet minus same-stratum partners, z vs 150 random line subsets; discovery half A, top 20 re-tested on half B' % (k, x['nscored']),
            'replicated (zB < -2) %d of 20; top: %s' % (x['nrep'], top3(x)), {'S1': 'control: must replicate ~0', 'PLANT_N14': 'control: must find num_N14', 'PLANT_LASTOBV': 'control: must find lastobv', 'REAL': 'see 4e'}[k])
