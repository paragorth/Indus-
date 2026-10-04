#!/usr/bin/env python3
"""LA-12 cycle 1: exhaustive cross-tablet closure search (k <= 3 blocks, plus completions).

For every written total T on a Hagia Triada tablet, count the sets of 1-3 blocks from OTHER
tablets that sum exactly to T (k1,k2,k3), and the completions own-section + 1-2 blocks (c1,c2).
Per-target and aggregate statistics:
  H_k1, H_c1, H_k2  number of targets with >= 1 closure of that kind
  L_k2, L_k3, L_c2  sum of log(1 + count)
Nulls (identical exhaustive search, so search size is corrected by construction):
  N1  entry numbers shuffled across tablets within commodity tag; written totals fixed.
  N2  entry numbers drawn iid from the site's entry distribution per tag; totals fixed.
  NB  neighbour values (reported only as a diagnostic: it is biased, see the log).
Positive controls (same nulls, same statistics):
  PL  HT tablets whose total closes internally are cut into 2-3 fragments and mixed back.
  PB  Linear B (Pylos + Knossos, 205 documents): 10 multi-entry documents get a planted
      exact total, then are cut the same way.
Usage: la12_c1.py [mode V|C] [nnull]
"""
import sys, math, random, time, pickle
from multiprocessing import Pool
from la12_common import *

STATS = ['k1', 'k2', 'k3', 'c1', 'c2']
AGG = ['H_k1', 'H_c1', 'H_k2', 'L_k2', 'L_k3', 'L_c2']


def counts(recs, mode, cval, nb=False, own=None):
    B, T = blocks_targets(recs, mode, cval)
    T = [t for t in T if t['key'] > 0]
    if own is not None:          # nulls keep the REAL own-section sum of each target
        for t in T:
            t['own'] = own.get((t['rec'], t['idx']), t['own'])
    idx = make_indexes(B)
    out = {}
    for t in T:
        r = {'rec': t['rec'], 'kind': t['kind'], 'tag': t['tag'], 'int': t['int'], 'real': target_counts(idx, t)}
        if nb:
            r['nb'] = [target_counts(idx, t, v) for v in neighbours(t, mode, cval)]
        out[(t['rec'], t['idx'])] = r
    return out, B, T, idx


def agg(vecs):
    vecs = list(vecs)
    return {'H_k1': sum(v[0] > 0 for v in vecs), 'H_c1': sum(v[3] > 0 for v in vecs),
            'H_k2': sum(v[1] > 0 for v in vecs),
            'L_k2': sum(math.log1p(v[1]) for v in vecs), 'L_k3': sum(math.log1p(v[2]) for v in vecs),
            'L_c2': sum(math.log1p(v[4]) for v in vecs)}


def _job(a):
    kind, seed, recs, mode, cval = a
    rng = random.Random(seed)
    _, T0 = blocks_targets(recs, mode, cval)
    own = {(t['rec'], t['idx']): t['own'] for t in T0}
    R2 = shuffle_numbers(recs, rng, totals_too=False) if kind == 'N1' else iid_numbers_entries(recs, rng)
    c, *_ = counts(R2, mode, cval, own=own)
    return kind, {k: v['real'] for k, v in c.items()}


def iid_numbers_entries(recs, rng):
    out = [dict(r, items=[dict(i) for i in r['items']]) for r in recs]
    pool = defaultdict(list)
    for r in recs:
        for i in r['items']:
            if not i['tot']:
                pool[i['tag']].append(i['q'])
    for r in out:
        for i in r['items']:
            if not i['tot']:
                i['q'] = rng.choice(pool[i['tag']])
    return out


def compare(real, nulls, keys=None):
    """real: {key: vec}; nulls: list of {key: vec}. Returns aggregate lines and per-target P."""
    keys = [k for k in real if (keys is None or k in keys)]
    ra = agg(real[k] for k in keys)
    na = [agg(n[k] for k in keys if k in n) for n in nulls]
    lines = []
    for s in AGG:
        xs = [a[s] for a in na]; m = sum(xs) / len(xs); sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** .5
        p = (1 + sum(x >= ra[s] for x in xs)) / (1 + len(xs))
        lines.append('%s %.1f (null %.1f+-%.1f, P=%.3f)' % (s, ra[s], m, sd, p))
    per = {}
    for k in keys:
        per[k] = [(1 + sum(n.get(k, [0] * 5)[i] >= real[k][i] for n in nulls)) / (1 + len(nulls)) for i in range(5)]
    return '; '.join(lines), per


def synth_totals(recs, rng, n):
    """Give n random records with >= 3 entries of one tag a planted exact total (sum of those entries)."""
    cand = []
    for ri, r in enumerate(recs):
        if any(i['tot'] for i in r['items']):
            continue
        tc = Counter(i['tag'] for i in r['items'])
        tg = [t for t, c in tc.items() if c >= 3]
        if tg:
            cand.append((ri, tg[0]))
    pick = rng.sample(cand, min(n, len(cand)))
    out = list(recs)
    for ri, tg in pick:
        r = recs[ri]
        its = [dict(i, sec=0) for i in r['items']]
        s = Q()
        for i in its:
            if i['tag'] == tg:
                s = s + i['q']
        its.append({'sec': 0, 'tag': tg, 'q': s, 'tot': True, 'kind': 'total', 'label': 'PLANT'})
        out[ri] = dict(r, items=its)
    return out


def run_null(P, recs, mode, cval, nnull, seed0):
    jobs = [('N1', seed0 + i, recs, mode, cval) for i in range(nnull)] + \
           [('N2', seed0 + 50000 + i, recs, mode, cval) for i in range(nnull)]
    res = P.map(_job, jobs, chunksize=4)
    return [c for k, c in res if k == 'N1'], [c for k, c in res if k == 'N2']


def planted(P, recs, mode, cval, seed, label, out, nnull):
    rng = random.Random(seed)
    cut_recs, truth = cut(recs, mode, cval, rng)
    c, *_ = counts(cut_recs, mode, cval)
    real = {k: v['real'] for k, v in c.items()}
    n1, n2 = run_null(P, cut_recs, mode, cval, nnull, seed * 1000)
    pk = {k for k in real if k[0] in {t['rec'] for t in truth}}
    det = []
    for nm, nl in (('N1', n1), ('N2', n2)):
        s_pl, per = compare(real, nl, pk)
        s_all, _ = compare(real, nl)
        out.write('[%s vs %s] planted targets (n=%d): %s\n' % (label, nm, len(pk), s_pl))
        out.write('[%s vs %s] all targets (n=%d): %s\n' % (label, nm, len(real), s_all))
        for t in truth:
            k = next((k for k in pk if k[0] == t['rec']), None)
            if k is None:
                continue
            st = (0 if t['k'] == 1 else 1) if t['keep'] == 0 else (3 if t['k'] == 1 else 4)
            det.append((nm, t['rec'], STATS[st], real[k][st], round(per[k][st], 3)))
    out.write('[%s] per planted target (null, target, true stat, count, P): %s\n' % (label, det))
    out.flush()
    return det


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'V'
    nnull = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    out = open(os.path.join(OUT, 'c1_%s.txt' % mode), 'w')
    t0 = time.time()
    R = la_records('HT')
    c, B, T, idx = counts(R, mode, LA_CVAL, nb=True)
    real = {k: v['real'] for k, v in c.items()}
    out.write('HT records %d, blocks %d, targets %d (mode %s)\n' % (len(R), len(B), len(real), mode))
    with Pool(2) as P:
        n1, n2 = run_null(P, R, mode, LA_CVAL, nnull, 1000)
        pickle.dump({'real': real, 'n1': n1, 'n2': n2}, open(os.path.join(OUT, 'c1_%s.pkl' % mode), 'wb'))
        s1, per1 = compare(real, n1)
        s2, per2 = compare(real, n2)
        out.write('REAL vs N1 (%d runs): %s\n' % (len(n1), s1))
        out.write('REAL vs N2 (%d runs): %s\n' % (len(n2), s2))
        # neighbour diagnostic
        nbr = agg(v['real'] for v in c.values() if v['nb'])
        nbm = agg([sum(n[i] for n in v['nb']) / len(v['nb']) for i in range(5)] for v in c.values() if v['nb'])
        out.write('DIAG neighbour-value means (biased: decreasing density + round totals): real %s vs nb-mean %s\n' %
                  ({k: round(x, 1) for k, x in nbr.items()}, {k: round(x, 1) for k, x in nbm.items()}))
        out.write('Per target: real [k1,k2,k3,c1,c2]; P(N1 >= real) per stat\n')
        for k, v in c.items():
            out.write('  %-12s %-7s %-5s %5d real %-28s P_N1 %s P_N2 %s\n' % (v['rec'], v['kind'], v['tag'], v['int'], v['real'],
                                                                         [round(x, 2) for x in per1[k]], [round(x, 2) for x in per2[k]]))
        # sparse joins listed
        out.write('Sparse joins (k1 and c1 hits):\n')
        for t in T:
            I = idx.get(t['tag'], idx['*']) if t['tag'] != '*' else idx['*']
            for k, tag, rid, desc in I.blocks:
                if rid == t['rid']:
                    continue
                if k == t['key']:
                    out.write('  k1 %s %s %d <= %s %s [%s] P_N1=%.2f\n' % (t['rec'], t['kind'], t['int'], R[rid]['id'], desc, tag, per1[(t['rec'], t['idx'])][0]))
                if 0 < t['own'] < t['key'] and k == t['key'] - t['own']:
                    out.write('  c1 %s %s %d <= own + %s %s [%s] P_N1=%.2f\n' % (t['rec'], t['kind'], t['int'], R[rid]['id'], desc, tag, per1[(t['rec'], t['idx'])][3]))
        out.flush()
        npl = max(20, nnull // 3)
        for seed in (11, 12, 13):
            planted(P, R, mode, LA_CVAL, seed, 'PL seed %d' % seed, out, npl)
        LB = [r for r in lb_records('') if r['id'][:2] in ('PY', 'KN')]
        for seed in (21, 22, 23):
            rng = random.Random(seed)
            pool = rng.sample(LB, 205)
            pool = synth_totals(pool, rng, 10)
            planted(P, pool, mode, LB_CVAL, seed, 'PB seed %d' % seed, out, npl)
    out.write('time %.0fs\n' % (time.time() - t0))
    out.close()


if __name__ == '__main__':
    main()
