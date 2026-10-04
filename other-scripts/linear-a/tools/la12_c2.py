#!/usr/bin/env python3
"""LA-12 cycle 2: 'every number may be a total' + simulated-annealing ledger assembly.

Jigsaw version 2. A summary tablet records whole lists from other tablets as single lines.
TARGETS: every quantity on a Hagia Triada tablet (entries and totals, any tag).
DONORS:  multi-item sums of other tablets (a section, the whole tablet, per tag or all tags);
         blocks made of one item are excluded (two equal single numbers are not a sum).
A HIT is target == donor (k=1) or target == donor + donor from two distinct tablets (k=2),
tags compatible, no donor from the target's own tablet.
ASSEMBLY: simulated annealing chooses a set of hits in which every tablet is a donor at most
once and every target is used at most once (a tablet's list goes to one ledger). Hit weight =
1 / (number of alternative hits for that target at that k), so a unique closure counts 1 and an
ambiguous one little. 8 random restarts x 60k moves; score = best total weight.
Nulls: N1 entry numbers shuffled across tablets within tag (totals fixed), N2 iid draws; the
same counts and the same annealing. Planted: 10 random HT tablets have their whole sum written
as a new line on another random HT tablet (the 'summary line' the hypothesis predicts), and the
Linear B pool (Pylos + Knossos, 205 docs) gets the same plant.
Usage: la12_c2.py [mode] [nnull]
"""
import sys, math, random, time, pickle
from multiprocessing import Pool
from la12_common import *


def donors_targets(recs, mode, cval):
    D = []; T = []
    for rid, r in enumerate(recs):
        ent = [i for i in r['items'] if not i['tot']]
        seen = set()

        def add(its, tag, desc):
            if len(its) < 2:
                return
            s = Q()
            for i in its:
                s = s + i['q']
            if s.iszero():
                return
            k = s.key(mode, cval)
            if (k, tag) in seen:
                return
            seen.add((k, tag)); D.append((k, tag, rid, desc))
        groups = [('all', ent)] + [('sec%d' % s, [i for i in ent if i['sec'] == s]) for s in sorted({i['sec'] for i in ent})]
        for nm, its in groups:
            tags = {i['tag'] for i in its}
            add(its, tags.pop() if len(tags) == 1 else 'MIX', nm)
            for tg in {i['tag'] for i in its}:
                add([i for i in its if i['tag'] == tg], tg, nm + ':' + tg)
        for j, i in enumerate(r['items']):
            if i['q'].iszero():
                continue
            T.append({'rid': rid, 'rec': r['id'], 'idx': j, 'key': i['q'].key(mode, cval), 'tag': i['tag'],
                      'tot': i['tot'], 'int': i['q'].n, 'label': i.get('label')})
    return D, T


def hits(recs, mode, cval, kmax=2):
    D, T = donors_targets(recs, mode, cval)
    byk = defaultdict(list)
    for d in D:
        byk[d[0]].append(d)
    pairs = defaultdict(list)
    if kmax >= 2:
        n = len(D)
        for a in range(n):
            ka, ta, ra, _ = D[a]
            for b in range(a + 1, n):
                kb, tb, rb, _ = D[b]
                if ra != rb:
                    pairs[ka + kb].append((a, b))
    H = []
    for ti, t in enumerate(T):
        ok = lambda tag: t['tag'] == '*' or tag in (t['tag'], '*')
        h1 = [d for d in byk.get(t['key'], []) if d[2] != t['rid'] and ok(d[1])]
        for d in h1:
            H.append((ti, (d[2],), 1.0 / len(h1), 1, (d,)))
        if kmax >= 2:
            h2 = [(D[a], D[b]) for a, b in pairs.get(t['key'], []) if D[a][2] != t['rid'] and D[b][2] != t['rid']
                  and ok(D[a][1]) and ok(D[b][1])]
            for da, db in h2:
                H.append((ti, (da[2], db[2]), 0.5 / len(h2), 2, (da, db)))
    return H, D, T


def anneal(H, T, rng, restarts=8, moves=60000):
    """Max-weight set of hits with each target once and each donor tablet once.
    Constraint also: a tablet that is a donor cannot have its own targets used as summary lines
    for donors that include it (handled by distinct records already)."""
    if not H:
        return 0.0, []
    best = (0.0, [])
    for _ in range(restarts):
        chosen = set(); used_t = {}; used_r = {}; score = 0.0
        temp0 = 0.5
        for m in range(moves):
            temp = temp0 * (1 - m / moves) + 1e-3
            h = rng.randrange(len(H))
            if h in chosen:
                if rng.random() < math.exp(-H[h][2] / temp):
                    chosen.discard(h); score -= H[h][2]
                    del used_t[H[h][0]]
                    for r in H[h][1]:
                        del used_r[r]
                continue
            ti, rs, w, k, _ = H[h]
            conf = set()
            if ti in used_t:
                conf.add(used_t[ti])
            for r in rs:
                if r in used_r:
                    conf.add(used_r[r])
            dw = w - sum(H[c][2] for c in conf)
            if dw >= 0 or rng.random() < math.exp(dw / temp):
                for c in conf:
                    chosen.discard(c); score -= H[c][2]
                    del used_t[H[c][0]]
                    for r in H[c][1]:
                        del used_r[r]
                chosen.add(h); score += w; used_t[ti] = h
                for r in rs:
                    used_r[r] = h
        if score > best[0]:
            best = (score, sorted(chosen))
    return best


def summarize(H, T, sol):
    n1 = sum(1 for h in H if h[3] == 1); n2 = sum(1 for h in H if h[3] == 2)
    t1 = len({h[0] for h in H if h[3] == 1}); t2 = len({h[0] for h in H if h[3] == 2})
    u1 = sum(1 for ti in {h[0] for h in H if h[3] == 1} if sum(1 for h in H if h[0] == ti and h[3] == 1) == 1)
    return {'hits1': n1, 'tgt1': t1, 'uniq1': u1, 'hits2': n2, 'tgt2': t2, 'score': sol[0],
            'nsel': len(sol[1]), 'sel1': sum(1 for h in sol[1] if H[h][3] == 1)}


def _job(a):
    kind, seed, recs, mode, cval = a
    rng = random.Random(seed)
    if kind == 'N1':
        R2 = shuffle_numbers(recs, rng, totals_too=False)
    elif kind == 'N2':
        from la12_c1 import iid_numbers_entries
        R2 = iid_numbers_entries(recs, rng)
    else:
        R2 = recs
    H, D, T = hits(R2, mode, cval)
    sol = anneal(H, T, rng)
    return kind, summarize(H, T, sol)


def plant_summary(recs, mode, cval, rng, n=10):
    """Write the whole sum of n random multi-item tablets as a new line on another random tablet."""
    out = [dict(r, items=[dict(i) for i in r['items']]) for r in recs]
    cand = [i for i, r in enumerate(out) if sum(1 for x in r['items'] if not x['tot']) >= 2]
    src = rng.sample(cand, n); truth = []
    for s in src:
        ent = [i for i in out[s]['items'] if not i['tot']]
        tags = {i['tag'] for i in ent}
        tag = tags.pop() if len(tags) == 1 else '*'
        q = Q()
        for i in ent:
            q = q + i['q']
        dst = rng.choice([i for i in range(len(out)) if i != s])
        out[dst]['items'].append({'sec': max([i['sec'] for i in out[dst]['items']] + [0]), 'tag': tag, 'q': q,
                                  'tot': False, 'kind': 'entry', 'label': 'PLANT'})
        truth.append((out[s]['id'], out[dst]['id']))
    return out, truth


def compare(real, nulls, keys):
    s = []
    for k in keys:
        xs = [a[k] for a in nulls]; m = sum(xs) / len(xs); sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** .5
        p = (1 + sum(x >= real[k] for x in xs)) / (1 + len(xs))
        s.append('%s %.2f (null %.2f+-%.2f, P=%.3f)' % (k, real[k], m, sd, p))
    return '; '.join(s)


KEYS = ['hits1', 'tgt1', 'uniq1', 'hits2', 'tgt2', 'score', 'nsel', 'sel1']


def evaluate(P, recs, mode, cval, nnull, seed0, label, out):
    jobs = [('R', seed0, recs, mode, cval)] + [('N1', seed0 + 1 + i, recs, mode, cval) for i in range(nnull)] + \
           [('N2', seed0 + 9001 + i, recs, mode, cval) for i in range(nnull)]
    res = P.map(_job, jobs, chunksize=2)
    real = res[0][1]
    for nm in ('N1', 'N2'):
        out.write('[%s vs %s, %d runs] %s\n' % (label, nm, nnull, compare(real, [a for k, a in res if k == nm], KEYS)))
    out.flush()
    return real


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'V'
    nnull = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out = open(os.path.join(OUT, 'c2_%s.txt' % mode), 'w')
    t0 = time.time()
    R = la_records('HT')
    H, D, T = hits(R, mode, LA_CVAL)
    sol = anneal(H, T, random.Random(5))
    out.write('HT records %d, donors (multi-item sums) %d, targets (all quantities) %d; hits k1 %d, k2 %d\n' %
              (len(R), len(D), len(T), sum(h[3] == 1 for h in H), sum(h[3] == 2 for h in H)))
    out.write('Annealed assembly (real): score %.2f, %d joins\n' % (sol[0], len(sol[1])))
    for h in sorted(sol[1], key=lambda h: -H[h][2]):
        ti, rs, w, k, ds = H[h]; t = T[ti]
        out.write('  w=%.2f k=%d  %s line %d (%s %s %d%s) <= %s\n' % (
            w, k, t['rec'], t['idx'], t['label'], t['tag'], t['int'], ' TOTAL' if t['tot'] else '',
            ' + '.join('%s %s' % (R[d[2]]['id'], d[3]) for d in ds)))
    pickle.dump({'H': H, 'T': T, 'D': D, 'sol': sol, 'ids': [r['id'] for r in R]}, open(os.path.join(OUT, 'c2_%s.pkl' % mode), 'wb'))
    out.flush()
    with Pool(2) as P:
        evaluate(P, R, mode, LA_CVAL, nnull, 100, 'HT real', out)
        for seed in (31, 32):
            rng = random.Random(seed)
            Rp, truth = plant_summary(R, mode, LA_CVAL, rng)
            out.write('PL-summary seed %d planted %s\n' % (seed, truth))
            evaluate(P, Rp, mode, LA_CVAL, max(15, nnull // 2), seed * 100, 'PL-summary seed %d' % seed, out)
        LB = [r for r in lb_records('') if r['id'][:2] in ('PY', 'KN')]
        for seed in (41, 42):
            rng = random.Random(seed)
            pool = rng.sample(LB, 205)
            evaluate(P, pool, mode, LB_CVAL, max(15, nnull // 2), seed * 100, 'LB pool seed %d (no plant)' % seed, out)
            Rp, truth = plant_summary(pool, mode, LB_CVAL, rng)
            evaluate(P, Rp, mode, LB_CVAL, max(15, nnull // 2), seed * 100 + 50, 'LB pool seed %d + 10 planted summaries' % seed, out)
    out.write('time %.0fs\n' % (time.time() - t0))


if __name__ == '__main__':
    main()
