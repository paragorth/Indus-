#!/usr/bin/env python3
"""LA-13 cycle 3: can the substitution graph build a sign GRID by itself?
Edges = one-sign-different word-type pairs (X <-> Y), split into CROSS-site (clean pairs, cycle 1) and WITHIN-site.
An annealer places the top-N signs injectively in an R x C grid to maximise the weight of edges joining signs that
share a row or a column; held-out test: grid fitted on a random half of the word pairs, scored on the other half.
Nulls: degree-preserving edge rewiring (same weights) and site-label permutation (for cross edges).
Positive control: Linear B (DAMOS) word types, where the true grid is known (truth used ONLY to score the control).
Sound values are never used for Linear A; a post-hoc label check is reported separately and is descriptive only."""
import sys, os, json, random, math, collections, itertools
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la13_common import *

def onediff_pairs(types, minlen=2):
    idx = collections.defaultdict(list)
    for w in types:
        if len(w) < minlen: continue
        for i in range(len(w)): idx[(len(w), i, w[:i] + w[i + 1:])].append(w)
    out = []
    for (L, i, m), ws in idx.items():
        for a, b in itertools.combinations(sorted(ws), 2):
            out.append((a[i], b[i], a, b))
    return out

def edges_from(pairs):
    E = collections.Counter()
    for x, y, a, b in pairs:
        if x == y or x.startswith('*') and False: continue
        E[tuple(sorted((x, y)))] += 1
    return E

def fit_grid(E, R, C, seed, steps=150000, nmax=None):
    rnd = random.Random(seed)
    deg = collections.Counter()
    for (x, y), w in E.items(): deg[x] += w; deg[y] += w
    nmax = nmax or R * C
    signs = [s for s, _ in deg.most_common(nmax)]
    S = set(signs)
    adj = collections.defaultdict(dict)
    for (x, y), w in E.items():
        if x in S and y in S: adj[x][y] = w; adj[y][x] = w
    cells = [(r, c) for r in range(R) for c in range(C)]
    rnd.shuffle(cells)
    pos = {s: cells[i] for i, s in enumerate(signs)}
    occ = {cells[i]: (signs[i] if i < len(signs) else None) for i in range(len(cells))}
    def ok(p, q): return p[0] == q[0] or p[1] == q[1]
    def local(s, p, excl=None):
        return sum(w for t, w in adj[s].items() if t != excl and ok(p, pos[t]))
    score = sum(w for (x, y), w in E.items() if x in S and y in S and ok(pos[x], pos[y]))
    tot = sum(w for (x, y), w in E.items() if x in S and y in S)
    T0, T1 = 3.0, 0.02
    for t in range(steps):
        T = T0 * (T1 / T0) ** (t / steps)
        s = rnd.choice(signs); q = rnd.choice(cells); p = pos[s]
        if q == p: continue
        u = occ[q]
        before = local(s, p, u) + (local(u, q, s) if u else 0)
        pos[s] = q
        if u: pos[u] = p
        after = local(s, q, u) + (local(u, p, s) if u else 0)
        d = after - before
        if d >= 0 or rnd.random() < math.exp(d / T):
            occ[q] = s; occ[p] = u; score += d
        else:
            pos[s] = p
            if u: pos[u] = q
    return dict(score=score, tot=tot, frac=score / tot if tot else 0, pos=pos)

def score_grid(pos, E):
    sc = tot = 0
    for (x, y), w in E.items():
        if x in pos and y in pos:
            tot += w
            if pos[x][0] == pos[y][0] or pos[x][1] == pos[y][1]: sc += w
    return sc, tot

def rewire(E, rnd):
    stubs = []
    for (x, y), w in E.items():
        for _ in range(w): stubs += [x, y]
    rnd.shuffle(stubs)
    out = collections.Counter()
    for a, b in zip(stubs[::2], stubs[1::2]):
        if a != b: out[tuple(sorted((a, b)))] += 1
    return out

def heldout(pairs, R, C, seed):
    rnd = random.Random(seed)
    a, b = [], []
    for p in pairs: (a if rnd.random() < 0.5 else b).append(p)
    Ea, Eb = edges_from(a), edges_from(b)
    g = fit_grid(Ea, R, C, seed)
    sc, tot = score_grid(g['pos'], Eb)
    # chance: same grid with signs permuted
    keys = list(g['pos']); vals = list(g['pos'].values())
    ch = []
    for k in range(200):
        rnd.shuffle(vals); ch.append(score_grid(dict(zip(keys, vals)), Eb)[0] / max(tot, 1))
    return dict(train=g['frac'], test=sc / max(tot, 1), chance=sum(ch) / len(ch),
                p=sum(1 for c in ch if c >= sc / max(tot, 1)) / len(ch), pos=g['pos'])

CONS = lambda s: s.rstrip('0123456789')[:-1] if s.rstrip('0123456789')[-1:] in 'AEIOU' else s
def truth_ari(pos):
    """LB control only: adjusted Rand of grid rows vs true consonant and cols vs true vowel."""
    def ari(lab1, lab2):
        n = len(lab1); from math import comb
        cont = collections.Counter(zip(lab1, lab2))
        a = collections.Counter(lab1); b = collections.Counter(lab2)
        sij = sum(comb(v, 2) for v in cont.values()); sa = sum(comb(v, 2) for v in a.values()); sb = sum(comb(v, 2) for v in b.values())
        e = sa * sb / comb(n, 2); m = (sa + sb) / 2
        return (sij - e) / (m - e) if m != e else 0
    ss = [s for s in pos if s[-1:] in 'AEIOU' and not s.startswith('*')]
    rows = [pos[s][0] for s in ss]; cols = [pos[s][1] for s in ss]
    cons = [CONS(s) for s in ss]; vow = [s[-1] for s in ss]
    return dict(row_cons=round(ari(rows, cons), 3), col_vow=round(ari(cols, vow), 3),
                row_vow=round(ari(rows, vow), 3), col_cons=round(ari(cols, cons), 3))

def job(args):
    name, E, R, C, seed = args
    g = fit_grid(E, R, C, seed)
    return name, seed, g['frac'], {k: list(v) for k, v in g['pos'].items()}

def main():
    out = open(os.path.join(OUT, 'c3_report.txt'), 'w')
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()
    U = la_units(); T = types_by_group(U)
    cross = [(x, y, w1, w2) for (a, b, x, y, p, w1, w2) in pairs(T, indel=False)]
    within = [p for g in T for p in onediff_pairs(T[g])]
    LBU = lb_units(); LBT = sorted({w for u in LBU for w in u[3]})
    lbp = onediff_pairs(LBT, 3)
    # LA-sized LB: subsample LB types to the LA type count
    rnd = random.Random(5)
    lbsmall = onediff_pairs(rnd.sample(LBT, 1000), 2)
    sets = dict(LA_cross=cross, LA_within=within, LA_all=cross + within, LB_full=lbp, LB_small=lbsmall)
    for k, v in sets.items():
        E = edges_from(v); say(k, 'pairs', len(v), 'edges', len(E), 'weight', sum(E.values()))
    R, C = 12, 5
    res = {}
    ck = os.path.join(OUT, 'c3_fits.json')
    if os.path.exists(ck): res = json.load(open(ck))
    else:
        jobs = []
        for k, v in sets.items():
            E = edges_from(v)
            jobs += [(k, E, R, C, s) for s in range(2)]
            r2 = random.Random(11)
            for s in range(20): jobs.append((k + '_rewire', rewire(E, r2), R, C, 100 + s))
        with Pool(2) as pool: rr = pool.map(job, jobs)
        for name, seed, frac, pos in rr: res.setdefault(name, []).append((seed, frac, pos))
        json.dump(res, open(ck, 'w'))
    for k in sets:
        real = max(res[k], key=lambda t: t[1]); nul = [t[1] for t in res[k + '_rewire']]
        pos = {s: tuple(p) for s, p in real[2].items()}
        say(f'{k}: grid-explained edge weight {real[1]:.3f} (seeds {[round(t[1],3) for t in res[k]]}) | rewired null mean {sum(nul)/len(nul):.3f} '
            f'max {max(nul):.3f} P(>=) {sum(1 for x in nul if x >= real[1])/len(nul):.3f} | z {(real[1]-sum(nul)/len(nul))/(max(1e-9,(sum((x-sum(nul)/len(nul))**2 for x in nul)/len(nul))**.5)):.2f}')
        if k.startswith('LB'): say('    LB truth ARI (control only):', truth_ari(pos))
        else: say('    post-hoc conventional-label ARI (descriptive only, not used in fitting):', truth_ari(pos))
        rows = collections.defaultdict(list)
        for s, p in pos.items(): rows[p[0]].append(s)
        say('    rows:', [rows[r] for r in sorted(rows)])
    # held-out prediction
    ho = os.path.join(OUT, 'c3_heldout.json')
    if os.path.exists(ho): H = json.load(open(ho))
    else:
        H = {}
        hs = dict(sets)
        r3 = random.Random(17)
        for k, v in sets.items():
            ys = [p[1] for p in v]; r3.shuffle(ys)
            hs[k + '_shuf'] = [(p[0], y, p[2], p[3]) for p, y in zip(v, ys)]
        with Pool(2) as pool:
            for k, v in hs.items():
                rr = pool.starmap(heldout, [(v, R, C, s) for s in range(4)])
                H[k] = [dict(train=r['train'], test=r['test'], chance=r['chance'], p=r['p'],
                             ari=truth_ari(r['pos'])) for r in rr]
        json.dump(H, open(ho, 'w'), indent=1)
    for k, rr in H.items():
        say(f'held-out {k}:', [(round(r['train'], 3), round(r['test'], 3), round(r['chance'], 3), r['p']) for r in rr],
            'ARI(row~C, col~V):', [(r['ari']['row_cons'], r['ari']['col_vow']) for r in rr])
    out.close()

if __name__ == '__main__':
    main()
