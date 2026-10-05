#!/usr/bin/env python3
"""LA-56 cycle 3: are the 'errors' our misreading of the document's structure?

Instead of changing what the scribe did, change where we cut the document. Every KU-RO /
PO-TO-KU-RO total is tested against EVERY contiguous run of quantities on the same tablet
(both sides, in either side order, earlier totals allowed inside the run), with one optional
skipped quantity inside the run. Fraction values: D random draws (closure weight = share of
draws). A section is 're-cut' when the default run does not close and some other run does.
Statistics: number of re-cut totals; how far the best run sits from the default (start
shift, crosses an earlier total, crosses the side boundary, ends after the total); and the
label words at the new run edges.
Nulls: N1 totals permuted across sections; N2 entries dealt across tablets (la56_c1.make);
N3 noise on balanced totals. Positive control: Ur III count lists merged in random pairs
into two-section 'tablets' with the second section's cut moved by 1-3 lines (planted
re-cut), and LA with planted shifted cuts (total rewritten as the sum of a shifted window).
Usage: python3 la56_c3.py CORPUS JOB nseeds
"""
import sys, os, json, random, copy
import numpy as np
import la56_common as C
import la56_c1 as C1

D = 400


def tablet_streams(name):
    """Per tablet: ordered quantity stream(s) and the totals to test, as (stream, total_index, default (i,j))."""
    import json as J
    if name != 'LA':
        return None
    Cp = J.load(open(os.path.join(C.D_DIR, 'corpus.json')))
    by = {c['id']: c for c in Cp}
    tabs = {}
    for c in Cp: tabs.setdefault(C.tab_of(c['id']), []).append(c['id'])
    out = []
    for t, sides in tabs.items():
        sides = sorted(sides)
        per = {s: C._side_items(by[s]['tokens']) for s in sides}
        if not any(it['tot'] for s in sides for it in per[s]): continue
        orders = [sides] if len(sides) == 1 else [sides, sides[::-1]]
        out.append({'tab': t, 'orders': [[(s, it) for s in o for it in per[s]] for o in orders]})
    return out


def run_la(job, seed):
    rng = random.Random(seed * 7919 + sum(map(ord, job)))
    T = tablet_streams('LA')
    # flatten every quantity for nulls
    if job in ('N1', 'N2', 'N3', 'PCUT'):
        T = mutate(T, job, rng)
    lets = sorted({L for t in T for o in t['orders'] for _, it in o for L in it['l']})
    li = {L: i for i, L in enumerate(lets)}
    V = C.GRID[np.random.default_rng(seed).integers(0, len(C.GRID), size=(D, len(lets)))]
    res = []
    for t in T:
        for oi, o in enumerate(t['orders']):
            vals = np.zeros((D, len(o)), dtype=np.int64)
            for j, (_, it) in enumerate(o):
                lv = np.zeros(len(lets), dtype=np.int64)
                for L, c in it['l'].items(): lv[li[L]] += c
                vals[:, j] = it['v'] * C.U + V @ lv
            cs = np.concatenate([np.zeros((D, 1), dtype=np.int64), np.cumsum(vals, 1)], 1)
            for k, (side, it) in enumerate(o):
                if not it['tot']: continue
                if oi == 1 and side == o[0][0]:
                    pass
                tv = vals[:, k]
                # default run: back to previous total on the same side (KU-RO) or side start (PO-TO-KU-RO)
                st = k
                while st > 0 and o[st - 1][0] == side and not (o[st - 1][1]['tot'] and it['tot'] == 'KU-RO'):
                    st -= 1
                dflt = runsum(cs, vals, o, st, k, it['tot'] == 'PO-TO-KU-RO')
                pd = float(np.mean(dflt == tv))
                best = None
                n = len(o)
                for i in range(n):
                    for j in range(i + 1, n + 1):
                        if i <= k < j: continue        # a run may not contain its own total
                        base = cs[:, j] - cs[:, i]
                        for skip in [None] + list(range(i, j)):
                            sv = base if skip is None else base - vals[:, skip]
                            pr = float(np.mean(sv == tv))
                            if pr <= 0: continue
                            cost = (abs(i - st) + abs(j - k)) + (0 if skip is None else 1)
                            if (i, j) == (st, k) and skip is None: cost = 0
                            cand = (cost, -pr, i, j, skip)
                            if best is None or cand < best: best = cand
                if oi == 1 and pd >= 0.5: continue
                rec = {'tab': t['tab'], 'order': oi, 'side': side, 'k': k, 'kind': it['tot'], 'total': it['v'],
                       'p_default': pd, 'default': [st, k]}
                if best:
                    c, npr, i, j, sk = best
                    rec.update({'best': [i, j, sk], 'cost': c, 'p': -npr,
                                'crosses_total': any(o[x][1]['tot'] for x in range(i, j) if x != sk),
                                'crosses_side': len({o[x][0] for x in range(i, j)}) > 1,
                                'after_total': j > k + 1,
                                'edge_words': [o[i][1]['word'], o[j - 1][1]['word']],
                                'skipped_word': o[sk][1]['word'] if sk is not None else None})
                res.append(rec)
    # one record per total: keep the better of the two side orders
    keep = {}
    for r in res:
        key = (r['tab'], r['side'], r['k'] if r['order'] == 0 else None, r['total'], r['kind'])
        key = (r['tab'], r['side'], r['total'], r['kind'])
        if key not in keep or (r.get('cost', 99), -r.get('p', 0)) < (keep[key].get('cost', 99), -keep[key].get('p', 0)):
            keep[key] = r
    res = list(keep.values())
    bad = [r for r in res if r['p_default'] < 0.5]
    recut = [r for r in bad if 'best' in r]
    st = {'n_tot': len(res), 'n_bad': len(bad), 'n_recut': len(recut),
          'recut_cost1': sum(r['cost'] <= 1 for r in recut), 'recut_cost2': sum(r['cost'] <= 2 for r in recut),
          'cross_total': sum(r['crosses_total'] for r in recut), 'cross_side': sum(r['crosses_side'] for r in recut),
          'after': sum(r['after_total'] for r in recut), 'skip': sum(r['best'][2] is not None for r in recut)}
    out = {'job': job, 'seed': seed, 'stats': st, 'res': res}
    json.dump(out, open(os.path.join(C.CK, 'c3_LA_%s_%d.json' % (job, seed)), 'w'))
    print('LA', job, seed, st, flush=True)


def runsum(cs, vals, o, st, k, grand):
    s = cs[:, k] - cs[:, st]
    if not grand:
        return s
    # PO-TO-KU-RO default: entries only (earlier totals on the side excluded)
    for x in range(st, k):
        if o[x][1]['tot']: s = s - vals[:, x]
    return s


def mutate(T, job, rng):
    T = copy.deepcopy(T)
    allq = [it for t in T for _, it in t['orders'][0] if not it['tot']]
    allt = [it for t in T for _, it in t['orders'][0] if it['tot']]
    if job == 'N1':
        vals = [(it['v'], it['l']) for it in allt]; rng.shuffle(vals)
        mp = {id(it): v for it, v in zip(allt, vals)}
    elif job == 'N2':
        vals = [(it['v'], it['l']) for it in allq]; rng.shuffle(vals)
        mp = {id(it): v for it, v in zip(allq, vals)}
    elif job == 'N3':
        mp = {}
        for t in T:
            o = t['orders'][0]; st = 0
            for k, (side, it) in enumerate(o):
                if it['tot']:
                    st = k
                    while st > 0 and o[st - 1][0] == side and not o[st - 1][1]['tot']: st -= 1
                    v = sum(o[x][1]['v'] for x in range(st, k)); l = {}
                    for x in range(st, k):
                        for L, c in o[x][1]['l'].items(): l[L] = l.get(L, 0) + c
                    if rng.random() < 0.75: v = max(0, v + rng.choice([-1, 1]) * rng.choice([1, 2, 4, 9, 10, 20]))
                    mp[id(it)] = (v, l)
    elif job == 'PCUT':
        mp = {}
        for t in T:
            o = t['orders'][0]
            for k, (side, it) in enumerate(o):
                if it['tot'] and k >= 2:
                    st = k
                    while st > 0 and o[st - 1][0] == side and not o[st - 1][1]['tot']: st -= 1
                    sh = rng.choice([-2, -1, 1, 2])
                    a = max(0, min(k - 1, st + sh))
                    v = sum(o[x][1]['v'] for x in range(a, k) if not o[x][1]['tot']); l = {}
                    for x in range(a, k):
                        if o[x][1]['tot']: continue
                        for L, c in o[x][1]['l'].items(): l[L] = l.get(L, 0) + c
                    mp[id(it)] = (v, l)
    # apply to every order copy (orders hold separate deep copies; map by position)
    for t in T:
        base = t['orders'][0]
        for o in t['orders'][1:]:
            pos = {(s, it['tok']): it for s, it in o}
            for s, it in base:
                if id(it) in mp:
                    pos[(s, it['tok'])]['v'], pos[(s, it['tok'])]['l'] = mp[id(it)]
        for s, it in base:
            if id(it) in mp: it['v'], it['l'] = mp[id(it)]
    return T


if __name__ == '__main__':
    name, job = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for sd in range(n):
        if os.path.exists(os.path.join(C.CK, 'c3_%s_%s_%d.json' % (name, job, sd))): continue
        run_la(job, sd)
