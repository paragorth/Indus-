#!/usr/bin/env python3
"""la47 cycle 3: identities meet layout.
(a) spatial profiles of word types (distribution over physical-layout cells, no identities) -> do they
    sort la45 classes (commodity / header / total / other)?  Leave-one-type-out nearest-centroid accuracy
    and bits, against order profiles and the within-page identity permutation null (200 reps).
(b) transfer: grammars searched at Haghia Triada, scored at the other sites (S vs O), and the reverse;
    nulls: identities permuted within page.
(c) totals: does a number's spatial position mark it as the total (number after KU-RO / LB to-so)?
    S vs O grammars, LA and LB.
Output data/la47_ckpt/c3.json"""
import json, os, random, collections
import numpy as np
from multiprocessing import Pool
import la47_common as C, la47_engine as E

SC = ['vline', 'linit', 'lfin', 'numline']
OC = ['rank', 'nextnum', 'prevnum']


def profiles(rows, feats, key, minc=4):
    cells = sorted({tuple(r[key][f] for f in feats) for r in rows}, key=str); ci = {c: k for k, c in enumerate(cells)}
    cnt = collections.Counter(r['type'] for r in rows)
    T = sorted(t for t, n in cnt.items() if n >= minc); ti = {t: k for k, t in enumerate(T)}
    M = np.zeros((len(T), len(cells)))
    for r in rows:
        if r['type'] in ti: M[ti[r['type']], ci[tuple(r[key][f] for f in feats)]] += 1
    return T, M


def loto(T, M, la=True):
    """leave-one-type-out nearest centroid on smoothed log-profiles; returns accuracy over non-'other' types,
    balanced accuracy over classes, and held-out log2 likelihood gain of class membership"""
    cls = np.array([C.la45_class(t) if la else C.lb_class(t) for t in T])
    X = np.log((M + 0.5) / (M + 0.5).sum(1, keepdims=True))
    K = sorted(set(cls)); hit = collections.Counter(); tot = collections.Counter()
    for k in range(len(T)):
        m = np.ones(len(T), bool); m[k] = False
        cent = {c: X[m & (cls == c)].mean(0) for c in K if (m & (cls == c)).any()}
        pred = min(cent, key=lambda c: np.sum((X[k] - cent[c]) ** 2))
        tot[cls[k]] += 1; hit[cls[k]] += pred == cls[k]
    bal = float(np.mean([hit[c] / tot[c] for c in K]))
    return bal, {c: '%d/%d' % (hit[c], tot[c]) for c in K}


def part_a(rows, rng, reps=200, la=True):
    w = [r for r in rows if r['kind'] == 'w']
    out = {}
    for name, feats, key in (('S', SC, 's'), ('O', OC, 'o')):
        T, M = profiles(w, feats, key); b, d = loto(T, M, la)
        null = []
        for _ in range(reps):
            ww = [dict(r) for r in w]
            by = collections.defaultdict(list)
            for k, r in enumerate(ww): by[r['p']].append(k)
            for ks in by.values():
                tg = [w[k]['type'] for k in ks]; rng.shuffle(tg)
                for k, t in zip(ks, tg): ww[k]['type'] = t
            T2, M2 = profiles(ww, feats, key); null.append(loto(T2, M2, la)[0])
        null = np.array(null)
        out[name] = {'balacc': round(b, 3), 'per_class': d, 'null_mean': round(float(null.mean()), 3), 'null_sd': round(float(null.std()), 3),
                     'P': round(float((null >= b).mean()), 4), 'types': len(T)}
    return out


def job(a):
    name, seed = a
    rng = random.Random(seed)
    LA = C.la_pages()
    if name == 'A_LA':
        return name, part_a(C.table(LA), rng)
    if name == 'A_LB':
        B = C.lb_pages(sites={'KN', 'PY'}); rng.shuffle(B)
        return name, part_a(C.table(B[:600], la=False), rng, reps=100, la=False)
    if name.startswith('B_'):
        P = LA if 'NULL' not in name else C.null_permute(LA, rng)
        rows = [r for r in C.table(P) if r['kind'] == 'w']
        out = {}
        for tgt in ('type', 'cls'):
            out[tgt + '_HT->other'] = E.run_fixed(rows, tgt, lambda r: r['site'] == 'Haghia Triada', G=G, seed=seed)
            out[tgt + '_other->HT'] = E.run_fixed(rows, tgt, lambda r: r['site'] != 'Haghia Triada', G=G, seed=seed)
        return name, out
    if name.startswith('C_'):
        if 'LB' in name:
            B = C.lb_pages(sites={'KN', 'PY'}); rng.shuffle(B); P = B[:400]; la = False; tw = {'to-so', 'to-sa'}
        else:
            P = LA; la = True; tw = C.TOTAL
        if 'NULL' in name: P = C.null_permute(P, rng)
        rows = C.table(P, la)
        allit = [i for p in P for i, s, o in C.featurize(p)]
        prev = None; pp = None
        for r, i in zip(rows, allit):
            if r['p'] != pp: prev = None; pp = r['p']
            if r['kind'] == 'n': r['tot'] = 'T' if (prev is not None and prev in tw) else 'N'
            else: prev = i['id']
        nr = [r for r in rows if r['kind'] == 'n']
        return name, {'tot': E.run(nr, 'tot', G=G, seed=seed), 'n_tot': sum(r['tot'] == 'T' for r in nr)}


G = int(os.environ.get('G', 1500))
if __name__ == '__main__':
    jobs = [('A_LA', 1), ('A_LB', 2), ('B_LA', 3), ('B_NULL1', 4), ('B_NULL2', 5), ('C_LA', 6), ('C_LB', 7), ('C_NULL1', 8), ('C_NULL2', 9)]
    out = {}
    with Pool(2) as pool:
        for name, o in pool.imap_unordered(job, jobs):
            out[name] = o; print(name, json.dumps(o)[:1500], flush=True)
    json.dump(out, open(os.path.join(C.CK, 'c3.json'), 'w'), indent=1)
