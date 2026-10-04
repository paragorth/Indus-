#!/usr/bin/env python3
"""LA-24 cycle 3: SHARE-VECTOR REUSE. If a clerk apportioned by fixed shares, the same recipients
should get proportional amounts on different tablets (same shares, different totals): the
ratio a2(w)/a1(w) is the same for every shared recipient word w, and differs from 1 (equal amounts
are 'fixed allotments', la6.4d, counted separately).

Unit: a document side. Entry: a recipient word (2+ signs, not KU-RO/KI-RO, used once on that
side) with one amount of one commodity (bare numbers = BARE). For every pair of sides sharing
>= 2 recipient words in the same commodity, k = size of the largest group of shared words whose
ratios agree within 2 % (r != 1), k1 = the same for r == 1.
Statistics: number of pairs with k >= 2 and with k >= 3; summed k over pairs.
Nulls (2,000 each): P1 amounts permuted among the entries of each side (keeps every side's
values; breaks word-amount links); P2 each distinct amount replaced by a random corpus amount
within +-25 % (scale and repeats kept).
Positive control: Linear B (DAMOS) sides, words of a line as recipients, line amount of one
commodity (PY Es contains known proportional contributions). Planted: 15 LA side pairs sharing
>= 2 words get the second side's shared amounts set to round(k x first).
"""
import json, os, random, re, sys, time
import numpy as np
from collections import defaultdict, Counter
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la24_common import CK
from la6_common import la_entries, LB_COM, LIQ_COM, DRYU, LIQU

TOL = 0.02
NR = int(os.environ.get('NR', 2000))


def la_sides():
    E = la_entries(use_frac=True)
    by = defaultdict(list)
    for e in E:
        if e['role'] != 'entry' or not e['label'] or e['label'].count('-') < 1: continue
        if e['label'].startswith('*') and e['label'].count('-') < 1: continue
        coms = {k: v for k, v in e['com'].items() if v > 0}
        if len(coms) == 1:
            k, v = next(iter(coms.items()))
        elif not coms and len(e['bare']) == 1 and e['bare'][0] > 0:
            k, v = 'BARE', e['bare'][0]
        else:
            continue
        by[e['doc']].append((e['label'], k, float(v)))
    return clean(by)


def clean(by):
    out = {}
    for d, ents in by.items():
        c = Counter((w, k) for w, k, v in ents)
        ents = [(w, k, v) for w, k, v in ents if c[(w, k)] == 1]
        if len(ents) >= 2: out[d] = ents
    return out


NUM = re.compile(r'^\[?(\d+)\]?$')
STOP = {'do-so-mo', 'o-na-to', 'ke-ke-me-na', 'ko-to-na', 'pa-ro', 'to-so', 'to-sa', 'o-da-a2', 'qe', 'e-ke'}


def lb_sides():
    by = defaultdict(list)
    for line in open(os.path.join(os.path.dirname(CK), 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        for ln in (d.get('content') or '').split('\n'):
            toks = ln.split()
            words = [t for t in (re.sub(r'[\[\]⟦⟧?!]', '', x) for x in toks)
                     if re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)+', t) and t not in STOP]
            import unicodedata
            com = defaultdict(Fr); cur = None
            for i, t in enumerate(toks):
                s = ''.join(ch for ch in unicodedata.normalize('NFD', t) if unicodedata.category(ch) != 'Mn').strip('[]⟦⟧')
                b = s.split('+')[0]
                if b in LB_COM: cur = b; continue
                if cur is None: continue
                mm = NUM.match(s)
                if mm and i > 0:
                    prev = ''.join(ch for ch in unicodedata.normalize('NFD', toks[i - 1]) if unicodedata.category(ch) != 'Mn').strip('[]')
                    tab = LIQU if cur in LIQ_COM else DRYU
                    if prev in tab: com[cur] += tab[prev] * int(mm.group(1))
                    elif prev.split('+')[0].strip('[]⟦⟧') in LB_COM: com[cur] += int(mm.group(1))
            if len(com) == 1 and words:
                k, v = next(iter(com.items()))
                if v > 0:
                    for w in set(words): by[h].append((w, k, float(v)))
    return clean(by)


def pairs_of(sides):
    idx = defaultdict(set)
    for d, ents in sides.items():
        for w, k, v in ents: idx[(w, k)].add(d)
    cnt = Counter()
    for key, ds in idx.items():
        ds = sorted(ds)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)): cnt[(ds[i], ds[j])] += 1
    return [p for p, c in cnt.items() if c >= 2]


def stat(sides, pairs):
    k2 = k3 = ksum = 0; eq2 = 0; hits = []
    for a, b in pairs:
        A = {(w, k): v for w, k, v in sides[a]}; B = {(w, k): v for w, k, v in sides[b]}
        sh = [key for key in A if key in B]
        r = np.log(np.array([B[key] / A[key] for key in sh]))
        best = 0; best1 = 0
        for x in r:
            grp = np.abs(r - x) < TOL
            n = int(grp.sum())
            if abs(x) < TOL: best1 = max(best1, n)
            else: best = max(best, n)
        if best >= 2: k2 += 1; hits.append((a, b, best, len(sh)))
        if best >= 3: k3 += 1
        if best1 >= 2: eq2 += 1
        ksum += best if best >= 2 else 0
    return {'k2': k2, 'k3': k3, 'ksum': ksum, 'eq2': eq2}, hits


def perm_sides(sides, rng):
    out = {}
    for d, ents in sides.items():
        v = [e[2] for e in ents]; rng.shuffle(v)
        out[d] = [(w, k, x) for (w, k, _), x in zip(ents, v)]
    return out


def band_sides(sides, rng, pool):
    vals = np.array(sorted(pool))
    out = {}
    for d, ents in sides.items():
        mp = {}
        for w, k, v in ents:
            if v in mp: continue
            lo, hi = np.searchsorted(vals, v * np.exp(-0.25)), np.searchsorted(vals, v * np.exp(0.25), 'right')
            c = [x for x in vals[lo:hi] if x != v]
            mp[v] = rng.choice(c) if c else v
        out[d] = [(w, k, mp[v]) for w, k, v in ents]
    return out


def run(name, sides, planted=None):
    pairs = pairs_of(sides)
    obs, hits = stat(sides, pairs)
    rng = random.Random(7)
    pool = sorted({v for e in sides.values() for _, _, v in e})
    res = {'name': name, 'sides': len(sides), 'pairs': len(pairs), 'obs': obs, 'hits': hits[:40]}
    for tag, fn in (('P1', lambda: perm_sides(sides, rng)), ('P2', lambda: band_sides(sides, rng, pool))):
        acc = defaultdict(list)
        for _ in range(NR):
            s, _ = stat(fn(), pairs)
            for k, v in s.items(): acc[k].append(v)
        res[tag] = {k: {'mean': float(np.mean(v)), 'P': float((np.sum(np.array(v) >= obs[k]) + 1) / (NR + 1))}
                    for k, v in acc.items()}
    print(json.dumps({k: v for k, v in res.items() if k != 'hits'}), flush=True)
    return res


def main():
    out = {}
    la = la_sides()
    out['LA'] = run('LA', la)
    lb = lb_sides()
    out['LB'] = run('LB', lb)
    es = {d: e for d, e in lb.items() if d.startswith('PY Es')}
    out['LB_PY_Es'] = run('LB_PY_Es', es)
    # planted: 15 LA pairs, second side's shared amounts = round(k * first)
    rng = random.Random(11)
    pairs = pairs_of(la)
    pl = {d: list(e) for d, e in la.items()}
    for a, b in rng.sample(pairs, min(15, len(pairs))):
        k = rng.choice([2, 3, 0.5, 4, 5])
        A = {(w, c): v for w, c, v in pl[a]}
        pl[b] = [(w, c, (max(1.0, round(A[(w, c)] * k)) if (w, c) in A else v)) for w, c, v in pl[b]]
    out['LA_planted'] = run('LA_planted', pl)
    json.dump(out, open(os.path.join(CK, 'c3.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
