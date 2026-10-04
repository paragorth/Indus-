#!/usr/bin/env python3
"""LA-18 cycle 4: avoidance one level down - the ENTRY.
Cycles 1-2 showed alternants of one lemma share tablets (LB). Do they at least avoid the same ENTRY?
Entry = maximal run of tokens ending in numerals/logograms (a new entry starts at a word that follows a
numeral or logogram). For every tablet holding both members of a form pair (any family), count the token
pairs (a, b) that sit in the same entry; null: word tokens permuted across the entries of that tablet
(entry word counts kept), 5,000 permutations, aggregated by family. LB truth labels as in cycle 1.
Also ALL co-present word pairs (baseline) and planted entry-exclusive alternants in LA.
"""
import sys, json, random, collections
import numpy as np
from la18_common import *
from la18_c1 import lb_truth


def entries(d):
    out = [[]]; prevF = False
    for t in d['toks']:
        if t[0] in ('NL', 'D', 'B'): continue
        if t[0] == 'W':
            if prevF and out[-1]: out.append([])
            if len(t[1]) >= 2: out[-1].append(t[1])
            prevF = False
        elif t[0] == 'F': prevF = True
    return [e for e in out if e]


def run(docs, truthf, nperm, seed, label):
    rnd = np.random.default_rng(seed)
    T = collections.Counter(w for d in docs for w in set(d['words']))
    P = form_pairs(T)
    tasks = collections.defaultdict(list)  # family -> list of (doc entries, a, b)
    for d in docs:
        E = entries(d)
        if len(E) < 2: continue
        ws = set(w for e in E for w in e)
        for (a, b), f in P.items():
            pass
        # efficient: check pairs inside the doc
        wl = sorted(ws)
        for i in range(len(wl)):
            for j in range(i + 1, len(wl)):
                a, b = wl[i], wl[j]
                f = P.get((a, b)) or P.get((b, a))
                fam = f if f else 'ANY'
                if f and truthf:
                    k = (a, b) if (a, b) in P else (b, a)
                    fam = 'TRUE' if truthf(k[0], k[1], f) else 'FORM_OTHER'
                elif f:
                    fam = 'PLANT' if (a[-1] == '#E' or b[-1] == '#E') else 'FORM'
                tasks[fam].append((E, a, b))
    res = {}
    for fam, L in tasks.items():
        if fam == 'ANY' and len(L) > 1500:
            idx = rnd.choice(len(L), 1500, replace=False); L = [L[i] for i in idx]
        obs = 0; null = np.zeros(nperm)
        for E, a, b in L:
            lab = np.concatenate([[k] * len(e) for k, e in enumerate(E)])
            toks = [w for e in E for w in e]
            ia = np.array([t == a for t in toks]); ib = np.array([t == b for t in toks])
            same = lambda lb: sum(len(set(lb[ia]) & set(lb[ib])) for _ in [0])
            obs += same(lab)
            perms = np.array([rnd.permutation(lab) for _ in range(nperm)])
            pa = perms[:, ia]; pb = perms[:, ib]
            null += np.array([len(set(x) & set(y)) for x, y in zip(pa, pb)])
        mu = null.mean()
        res[fam] = dict(n=len(L), O=int(obs), E=round(float(mu), 2), ratio=round(obs / mu, 3) if mu else None,
                        p_avoid=round(float(((null <= obs).sum() + 1) / (nperm + 1)), 4),
                        p_attract=round(float(((null >= obs).sum() + 1) / (nperm + 1)), 4))
    return dict(label=label, fam=res)


def plant(docs, nw, seed):
    """entry-exclusive alternants: in tablets holding w in >= 2 entries, rewrite w in a random half of its
    entries as w+'#E' (the two forms then share the tablet but never an entry)."""
    r = random.Random(seed)
    cnt = collections.Counter()
    for d in docs:
        E = entries(d); c = collections.Counter(w for e in E for w in set(e))
        for w, k in c.items():
            if k >= 2: cnt[w] += 1
    cand = [w for w in cnt]
    ch = set(r.sample(cand, min(nw, len(cand))))
    out = []
    for d in docs:
        d = dict(d); toks = []; ent = 0; prevF = False; flip = {}
        for t in d['toks']:
            if t[0] == 'W' and prevF: ent += 1
            if t[0] == 'W': prevF = False
            elif t[0] == 'F': prevF = True
            if t[0] == 'W' and t[1] in ch:
                key = (t[1], ent)
                if key not in flip: flip[key] = r.random() < 0.5
                if flip[key]: t = ('W', t[1] + ('#E',))
            toks.append(t)
        d['toks'] = toks; d['words'] = [t[1] for t in toks if t[0] == 'W' and len(t[1]) >= 2]
        out.append(d)
    return out


def main():
    out = {}
    lb = lb_corpus()
    r = run(lb, lb_truth, 1000, 1, 'LB'); out['LB'] = r; print(r, flush=True)
    la = la_corpus()
    r = run(la, None, 5000, 2, 'LA'); out['LA'] = r; print(r, flush=True)
    pl = []
    for rep in range(4):
        pd = plant(la, 10, 70 + rep)
        r = run(pd, None, 2000, 80 + rep, 'LAplant%d' % rep)
        pl.append(r); print(r, flush=True)
    out['LA_planted'] = pl
    dump(out, 'c4.json')


if __name__ == '__main__':
    main()
