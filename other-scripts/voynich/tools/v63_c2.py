"""v63 cycle 2: SLIP OPERATORS BY MASSIVE RANDOM GUESSING.

Every adjacent pair (A, B) within a paragraph at glyph edit distance 1-2 is given the edit script that turns
the second word into the first, A = op(B) (substitution x->y, insertion, deletion, each tagged initial / medial /
final; TRUNCk = A is B cut short by k glyphs, a false start).  An ordered copying slip makes c(op) > c(op^-1);
an order-free text makes them equal.  A hypothesis is a SET of oriented operators ("these are the scribe's
slips"); its score is sum(c(op) - c(op^-1)) / sqrt(sum(c(op) + c(op^-1))) on training folios.
Search: 20,000 random sets of 1-24 operators + greedy top-K by training z (both orientations), on random halves
of the folios; the best training hypotheses are scored on the held-out folios.  Repeated over 10 random splits.
Nulls: same pipeline on within-folio position-preserving column shuffles; lag-2 pairs.
Power: planted slips inside the Voynich (6 fixed random substitution slips; Plaoul real mix) at 0.5/1/2%.
"""
import json, os, sys, math, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v63_lib as L, v63_plant as PL


def pos_tag(i, n):
    return 'I' if i == 0 else ('F' if i >= n - 1 else 'M')


def script(B, A):
    """edit ops turning B into A (Levenshtein backtrace), tagged by position in B.  None if ed > 2."""
    if len(A) < len(B) and B[:len(A)] == A:
        return ('TRUNC%d' % min(len(B) - len(A), 4),)
    if len(B) < len(A) and A[:len(B)] == B:
        return ('EXT%d' % min(len(A) - len(B), 4),)
    la, lb = len(B), len(A)
    if abs(la - lb) > 2: return None
    D = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1): D[i][0] = i
    for j in range(lb + 1): D[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (B[i - 1] != A[j - 1]))
    if D[la][lb] == 0 or D[la][lb] > 2: return None
    ops, i, j = [], la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (B[i - 1] != A[j - 1]):
            if B[i - 1] != A[j - 1]:
                ops.append('S%s>%s@%s' % (B[i - 1], A[j - 1], pos_tag(i - 1, la)))
            i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append('D%s@%s' % (B[i - 1], pos_tag(i - 1, la))); i -= 1
        else:
            ops.append('I%s@%s' % (A[j - 1], pos_tag(i, la + 1) if i < la else 'F')); j -= 1
    return tuple(sorted(ops))


def inv_op(o):
    if o.startswith('TRUNC'): return 'EXT' + o[5:]
    if o.startswith('EXT'): return 'TRUNC' + o[3:]
    if o[0] == 'S':
        xy, p = o[1:].split('@'); x, y = xy.split('>'); return 'S%s>%s@%s' % (y, x, p)
    g, p = o[1:].split('@')
    return ('I' if o[0] == 'D' else 'D') + g + '@' + p


def counts(streams, folios, lag=1):
    """Counter of ops per folio subset: returns dict folio -> Counter(op)."""
    out = defaultdict(Counter)
    for s in streams:
        if s['folio'] not in folios: continue
        ws = s['words']
        for i in range(len(ws) - lag):
            A, B = ws[i], ws[i + lag]
            if A == B: continue
            sc = script(B, A)
            if sc is None: continue
            for o in sc:
                out[s['folio']][o] += 1
            if len(sc) == 2:
                out[s['folio']]['+'.join(sc)] += 1
    return out


def agg(cf, fols):
    c = Counter()
    for f in fols: c.update(cf.get(f, {}))
    return c


def inv_full(o):
    return '+'.join(sorted(inv_op(x) for x in o.split('+')))


def zset(c, S):
    num = sum(c[o] - c[inv_full(o)] for o in S)
    den = sum(c[o] + c[inv_full(o)] for o in S)
    return num / math.sqrt(den) if den else 0.0


def search(cf, folios, rng, nrand=20000, minc=6):
    fols = sorted(folios); rng.shuffle(fols)
    tr, te = set(fols[:len(fols) // 2]), set(fols[len(fols) // 2:])
    ctr, cte = agg(cf, tr), agg(cf, te)
    ops = sorted(o for o in set(ctr) | set(inv_full(o) for o in ctr) if ctr[o] + ctr[inv_full(o)] >= minc)
    zs = {o: zset(ctr, [o]) for o in ops}
    pos = [o for o in ops if zs[o] > 0]
    best = []
    # greedy top-K oriented operators
    rank = sorted(pos, key=lambda o: -zs[o])
    for K in (1, 2, 4, 8, 16, 32):
        S = rank[:K]
        if S: best.append((zset(ctr, S), S, 'top%d' % K))
    # random sets (orientation chosen by training sign)
    for _ in range(nrand):
        k = rng.randint(1, 24)
        S = rng.sample(pos, min(k, len(pos))) if pos else []
        if S: best.append((zset(ctr, S), S, 'rand'))
    best.sort(key=lambda x: -x[0])
    top = best[:50]
    te_z = [zset(cte, S) for _, S, _ in top]
    return dict(train_best=top[0][0], test_of_best=te_z[0], test_top50=float(np.mean(te_z)), best_set=top[0][1][:12],
                kind=top[0][2], test_top8=zset(cte, rank[:8]), test_top32=zset(cte, rank[:32]))


def run(name, streams, nsplit=10, seed=0, lag=1, nrand=20000):
    rng = random.Random(seed)
    folios = set(s['folio'] for s in streams)
    cf = counts(streams, folios, lag)
    R = [search(cf, folios, rng, nrand) for _ in range(nsplit)]
    o = dict(name=name, test_of_best=float(np.mean([r['test_of_best'] for r in R])),
             test_top50=float(np.mean([r['test_top50'] for r in R])), test_top8=float(np.mean([r['test_top8'] for r in R])),
             test_top32=float(np.mean([r['test_top32'] for r in R])), train_best=float(np.mean([r['train_best'] for r in R])),
             best_sets=[r['best_set'] for r in R[:3]])
    print(f"{name:34s} train {o['train_best']:6.2f}  held-out: best {o['test_of_best']:+.2f}  top50 {o['test_top50']:+.2f}  "
          f"top8 {o['test_top8']:+.2f}  top32 {o['test_top32']:+.2f}   e.g. {R[0]['best_set'][:5]}", flush=True)
    return o


def column_shuffle_folio(streams, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for a, s in enumerate(streams):
        for i, (k, j, n) in enumerate(s['line']):
            groups[(s['folio'], 'L' if j == n - 1 else min(j, 8))].append((a, i))
    out = [dict(s, words=list(s['words'])) for s in streams]
    for locs in groups.values():
        ws = [streams[a]['words'][i] for a, i in locs]; rng.shuffle(ws)
        for (a, i), w in zip(locs, ws): out[a]['words'][i] = w
    return out


def plant_ops(streams, rate, seed, nops=6):
    """inserted slips A = op(B) with 6 fixed random single-glyph substitutions (a 'visual confusion' table)."""
    rng = random.Random(seed)
    gl = Counter(g for s in streams for w in s['words'] for g in w)
    common = [g for g, _ in gl.most_common(14)]
    table = {}
    while len(table) < nops:
        x, y = rng.sample(common, 2)
        if x not in table: table[x] = y
    out = []
    for s in streams:
        ws, ln = [], []
        for i, w in enumerate(s['words']):
            if i > 0 and rng.random() < rate:
                idx = [k for k, g in enumerate(w) if g in table]
                if idx:
                    k = rng.choice(idx); a = list(w); a[k] = table[w[k]]
                    ws.append(tuple(a)); ln.append(s['line'][i])
            ws.append(w); ln.append(s['line'][i])
        out.append(dict(s, words=ws, line=ln))
    return out, table


def main():
    res = {}
    zl = L.voynich_paras('ZL3b'); it = L.voynich_paras('IT2a')
    lat = PL.latin_clean()
    for i, s in enumerate(lat): s['folio'] = 'p%03d' % (i // 4)   # group paragraphs as 'folios'
    res['V-ZL3b'] = run('V-ZL3b', zl)
    res['V-IT2a'] = run('V-IT2a', it)
    res['V-ZL3b lag2'] = run('V-ZL3b lag 2', zl, lag=2)
    for k in range(6):
        res[f'colshuf{k}'] = run(f'V-ZL folio column-shuffle #{k}', column_shuffle_folio(zl, k), seed=k)
    for k in range(3):
        res[f'copygen{k}'] = run(f'V-ZL copy-and-modify #{k}', PL.copygen(zl, seed=k), seed=k)
    for r in (0.005, 0.01, 0.02):
        S, tab = plant_ops(zl, r, seed=5)
        res[f'plantops{r}'] = run(f'V-ZL + 6-op slips {r*100:.1f}% {sorted(tab.items())}'[:60], S)
        S2 = PL.plant_slips(zl, r * 3, seed=5)
        for s2, s in zip(S2, zl): s2['line'] = None
        res[f'plantmix{r}'] = run(f'V-ZL + Plaoul-mix slips {r*300:.1f}%', S2)
    res['LAT clean'] = run('LAT clean (opaque)', lat)
    res['LAT + mix 3%'] = run('LAT + Plaoul-mix slips 3%', PL.plant_slips(lat, 0.03, seed=5))
    json.dump(res, open(os.path.join(L.CK, 'c2_results.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
