#!/usr/bin/env python3
"""LA-16 cycle 2: annealing over sign-equivalence partitions + document bootstrap.
Edge weight of sign pair (X,Y) = clipped hand-contrast z (cycle-1 statistic, n>=2 word pairs). A partition of the
signs is scored by the summed weight of its within-class edges (correlation clustering), found by simulated
annealing. Nulls: the same anneal on z-matrices from hand-permuted corpora (each null row standardised against the
others). Stability: documents bootstrap-resampled B times, z recomputed (200 perms each) and re-annealed; sign pairs
co-clustered in >= 70 % of resamples are reported. LB control is scored with LB values (doublet / sameV / sameC).
"""
import sys, os, json, collections, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la16_common import *

say = say_to(os.path.join(OUT, 'c2_report.txt'))
CLIP = 4.0


def zvec(E, real, N):
    mu, sd = N.mean(0), N.std(0) + 1e-9
    z = np.clip((real - mu) / sd, -CLIP, CLIP)
    z[E.npairs_k < 2] = 0
    return z


def anneal(keys, w, rnd, iters=60000, T0=2.0):
    signs = sorted({s for k in keys for s in k if s != '_'}); si = {s: i for i, s in enumerate(signs)}
    nb = collections.defaultdict(dict)
    for (x, y), v in zip(keys, w):
        if v == 0 or x == '_' or y == '_': continue
        nb[si[x]][si[y]] = nb[si[x]].get(si[y], 0) + v; nb[si[y]][si[x]] = nb[si[y]].get(si[x], 0) + v
    act = [i for i in range(len(signs)) if nb[i]]
    lab = {i: i for i in range(len(signs))}
    score = 0.0; best = (0.0, dict(lab))
    for t in range(iters):
        T = T0 * (1 - t / iters) + 1e-3
        i = rnd.choice(act)
        # candidate classes: a neighbour's class or a new singleton
        j = rnd.choice(list(nb[i]))
        new = lab[j] if rnd.random() < 0.9 else max(lab.values()) + 1
        if new == lab[i]: continue
        d = sum(v for k, v in nb[i].items() if lab[k] == new) - sum(v for k, v in nb[i].items() if lab[k] == lab[i])
        if d >= 0 or rnd.random() < math.exp(d / T):
            lab[i] = new; score += d
            if score > best[0]: best = (score, dict(lab))
    cl = collections.defaultdict(list)
    for i, c in best[1].items(): cl[c].append(signs[i])
    return best[0], [sorted(v) for v in cl.values() if len(v) > 1]


def coclust(cls):
    S = set()
    for c in cls:
        for a in c:
            for b in c:
                if a < b: S.add((a, b))
    return S


def lb_eval(cls, tag):
    pr = coclust(cls); c = collections.Counter(lb_class(a, b) for a, b in pr)
    allsg = sorted({s for k in cls for s in k})
    say(f'   {tag}: {len(cls)} classes, {len(pr)} co-clustered pairs: ' + ', '.join(f'{k} {c[k]}' for k in ('doublet', 'sameC', 'sameV', 'other')))
    return c


def run(tag, docs, strat, nnull, B, nperm_b, rnd, lbmode=False, pf=None):
    E = Engine(docs, strat=strat, pair_filter=pf); real = E.stat(E.lab0); N = E.null(2000, rnd.randrange(10 ** 6))
    z = zvec(E, real, N)
    sc, cls = anneal(E.keys, z, rnd)
    say(f'\n== {tag}: anneal score {sc:.1f}; classes (>1 sign) {len(cls)}')
    for c in sorted(cls, key=len, reverse=True)[:12]: say('   ', ' '.join(c))
    # null anneals: each null row as pseudo-real
    mu, sd = N.mean(0), N.std(0) + 1e-9
    ns = []; ncl = []
    for b in range(nnull):
        zb = np.clip((N[b] - mu) / sd, -CLIP, CLIP); zb[E.npairs_k < 2] = 0
        s, c = anneal(E.keys, zb, rnd); ns.append(s); ncl.append(c)
    ns = np.array(ns)
    say(f'   null anneal scores mean {ns.mean():.1f} sd {ns.std():.1f} max {ns.max():.1f}; P(>=) {(ns >= sc).mean():.3f}')
    out = dict(score=sc, null=ns.tolist(), cls=cls)
    if lbmode:
        out['lb_real'] = lb_eval(cls, 'real')
        agg = collections.Counter()
        for c in ncl: agg.update(lb_class(a, b) for a, b in coclust(c))
        say('   null anneals pooled co-clustered pairs: ' + ', '.join(f'{k} {agg[k]}' for k in ('doublet', 'sameC', 'sameV', 'other')))
        out['lb_null'] = dict(agg)
    # bootstrap stability
    att = [d for d in docs if d.get('hand')]
    cc = collections.Counter()
    for b in range(B):
        bs = [att[rnd.randrange(len(att))] for _ in att]
        # duplicate docs need distinct ids only for incidence; Engine uses positions
        Eb = Engine(bs, strat=strat, pair_filter=pf); rb = Eb.stat(Eb.lab0); Nb = Eb.null(nperm_b, rnd.randrange(10 ** 6))
        _, cb = anneal(Eb.keys, zvec(Eb, rb, Nb), rnd, iters=30000)
        cc.update(coclust(cb))
    stab = [(p, n / B) for p, n in cc.most_common() if n / B >= 0.5]
    say(f'   bootstrap ({B} resamples): pairs co-clustered >= 50%: ' + ', '.join(f'{a}~{b} {f:.2f}' for (a, b), f in stab[:25]))
    if lbmode:
        c = collections.Counter(lb_class(a, b) for (a, b), f in stab if f >= 0.7)
        say('   stable (>=70%) pairs by LB class: ' + ', '.join(f'{k} {c[k]}' for k in ('doublet', 'sameC', 'sameV', 'other')))
    out['stable'] = [(list(p), f) for p, f in stab]
    return out


def main():
    rnd = random.Random(1616)
    res = {}
    lb = lb_corpus(('KN', 'PY'))
    for d in lb: d['ss'] = d['site'] + ':' + (d.get('support') or '')
    cLB, cLA = ctx_filter(contexts('LB')), ctx_filter(contexts('LA'))
    res['LB'] = run('LB KN+PY, all pairs', lb, 'ss', 30, int(os.environ.get('BLB', 20)), 150, rnd, lbmode=True)
    res['LBctx'] = run('LB KN+PY, context-sharing pairs only', lb, 'ss', 30, int(os.environ.get('BLB', 20)), 150, rnd, lbmode=True, pf=cLB)
    la = la_corpus()
    res['LA'] = run('LA hands, all pairs', la, 'site', 200, int(os.environ.get('BLA', 300)), 200, rnd)
    res['LActx'] = run('LA hands, context-sharing pairs only', la, 'site', 200, int(os.environ.get('BLA', 300)), 200, rnd, pf=cLA)
    for d in la:
        d['grp'] = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Phaistos': 'PH', 'Knossos': 'KN'}.get(d['site'], 'OTH')
        d['hand2'] = d['hand']
    # site mode: label = site group for every doc
    la2 = [dict(d, hand=d['grp']) for d in la]
    res['LA_site'] = run('LA site groups, context-sharing pairs', la2, 'support', 100, 100, 150, rnd, pf=cLA)
    json.dump(res, open(os.path.join(OUT, 'c2.json'), 'w'), default=str, indent=1)


if __name__ == '__main__':
    main()
