#!/usr/bin/env python3
"""LA-20 cycle 3: what is the surviving order made of?
 (a) amounts: does a training ranking predict held-out pair order beyond the amounts written on
     the test tablet (pairs with equal/missing amounts; logistic combination)?
 (b) sites: fit on Hagia Triada lists, predict other sites, and the reverse.
 (c) cyclic triads among items with a majority verdict, vs within-list shuffles.
 (d) Bayesian Bradley-Terry by random-walk Metropolis: posterior rank intervals.
usage: la20_c3.py TYPES NREP"""
import sys, json, time, math
from la20_common import *

types, NREP = sys.argv[1], int(sys.argv[2])
docs = load_la()


def qlists(T, site=None):
    out = []
    for d in docs:
        if site == 'HT' and d['site'] != 'Haghia Triada': continue
        if site == 'nonHT' and d['site'] == 'Haghia Triada': continue
        it = d['items'][T]
        if len(it) >= 2: out.append(it)
    return out


def amount_test(ql, seed=0, folds=5):
    """pairs (a before b) in held-out lists: accuracy of (i) training BT, (ii) larger-amount-first,
    (iii) BT on pairs whose amounts are equal or missing."""
    rng = random.Random(seed); idx = list(range(len(ql))); rng.shuffle(idx)
    r = Counter()
    for f in range(folds):
        te = set(idx[f::folds])
        sc = fit_bt([[x for x, _ in ql[i]] for i in range(len(ql)) if i not in te])
        for i in te:
            o = ql[i]
            for p in range(len(o)):
                for q in range(p + 1, len(o)):
                    (a, qa), (b, qb) = o[p], o[q]
                    kn = a in sc and b in sc
                    qq = qa is not None and qb is not None and qa != qb
                    if qq: r['q_n'] += 1; r['q_c'] += qa > qb
                    if kn:
                        d = sc[a] - sc[b]; w = 1 if d > 0 else (0.5 if d == 0 else 0)
                        r['bt_n'] += 1; r['bt_c'] += w
                        if not qq: r['btnq_n'] += 1; r['btnq_c'] += w
                        else:
                            r['both_n'] += 1; r['both_c'] += w
                            r['btq_agree'] += (d > 0) == (qa > qb)
                            if (d > 0) != (qa > qb): r['dis_n'] += 1; r['dis_bt'] += w
    return dict(r)


def cross_site(T, tr_site, te_site):
    tr = [[x for x, _ in o] for o in qlists(T, tr_site)]
    ts = [[x for x, _ in o] for o in qlists(T, te_site)]
    return tr, ts


def cyclic_triads(ords):
    net = Counter()
    for o in ords:
        for p in range(len(o)):
            for q in range(p + 1, len(o)):
                net[(o[p], o[q])] += 1
    dom = {}
    for (a, b), v in net.items():
        w = net.get((b, a), 0)
        if v > w: dom[(a, b)] = 1
    succ = defaultdict(set)
    for a, b in dom: succ[a].add(b)
    items = sorted(succ.keys() | {b for s in succ.values() for b in s})
    rel = set(dom)
    tri = cyc = 0
    pairs_by = defaultdict(set)
    for a, b in rel: pairs_by[a].add(b); pairs_by[b].add(a)
    for a in items:
        for b in pairs_by[a]:
            if b <= a: continue
            for c in pairs_by[a] & pairs_by[b]:
                if c <= b: continue
                tri += 1
                if ((a, b) in rel and (b, c) in rel and (c, a) in rel) or ((b, a) in rel and (c, b) in rel and (a, c) in rel): cyc += 1
    return tri, cyc


def mcmc_bt(ords, steps=200000, lam=1.0, seed=0):
    items = sorted({x for o in ords for x in o}); index = {x: i for i, x in enumerate(items)}
    pt = PairTable(ords, index); n = len(items)
    def lp(s): return -(pt.w * np.logaddexp(0, -(s[pt.a] - s[pt.b]))).sum() - 0.5 * lam * (s ** 2).sum()
    rng = np.random.default_rng(seed)
    s = fit_bt_pairs(pt, pt.w, n, lam); cur = lp(s)
    ranks = []; accn = 0
    for t in range(steps):
        j = rng.integers(n); prop = s.copy(); prop[j] += rng.normal(0, 0.8)
        new = lp(prop)
        if math.log(rng.random()) < new - cur: s, cur = prop, new; accn += 1
        if t % 200 == 0 and t > steps // 5: ranks.append(np.argsort(np.argsort(-s)))
    ranks = np.array(ranks)
    return items, ranks, accn / steps


if __name__ == '__main__':
    res = {}
    for T in types:
        ql = qlists(T)
        ords = [[x for x, _ in o] for o in ql]
        # (a) amounts
        real = amount_test(ql)
        nul = [amount_test([random.Random(r).sample(o, len(o)) for o in ql], seed=r) for r in range(NREP)]
        res[T + '_amount'] = {'real': real, 'null': nul}
        def fr(d, k): return d.get(k + '_c', 0) / max(d.get(k + '_n', 0), 1)
        print(T, 'amount: q-first acc %.3f (n %d); BT acc %.3f (n %d); BT on pairs w/o amount order %.3f (n %d) null %.3f+-%.3f; BT where it disagrees with amounts %.3f (n %d)' % (
            fr(real, 'q'), real.get('q_n', 0), fr(real, 'bt'), real.get('bt_n', 0), fr(real, 'btnq'), real.get('btnq_n', 0),
            np.mean([fr(x, 'btnq') for x in nul]), np.std([fr(x, 'btnq') for x in nul]),
            real.get('dis_bt', 0) / max(real.get('dis_n', 0), 1), real.get('dis_n', 0)), flush=True)
        # (b) cross-site
        for a, b in [('HT', 'nonHT'), ('nonHT', 'HT')]:
            tr, ts = cross_site(T, a, b)
            e = eval_pairs(fit_bt(tr), ts, cooc_set(tr))
            ne = [eval_pairs(fit_bt(shuffle_within(tr, random.Random(r))), ts) for r in range(NREP)]
            na = np.array([x[1] / x[0] if x[0] else np.nan for x in ne])
            res[T + '_site_' + a] = (e, float(np.nanmean(na)), float(np.nanstd(na)), float(np.mean(na >= e[1] / max(e[0], 1))))
            print(T, 'train', a, 'test', b, 'n %d acc %.3f; null %.3f+-%.3f P %.3f' % (e[0], e[1] / max(e[0], 1), res[T + '_site_' + a][1], res[T + '_site_' + a][2], res[T + '_site_' + a][3]), flush=True)
        # (c) cyclic triads
        tri, cyc = cyclic_triads(ords)
        nc = [cyclic_triads(shuffle_within(ords, random.Random(r))) for r in range(NREP)]
        nfrac = np.array([c / t if t else np.nan for t, c in nc])
        res[T + '_triads'] = (tri, cyc, float(np.nanmean(nfrac)), float(np.mean(nfrac <= (cyc / tri if tri else 1))))
        print(T, 'triads %d cyclic %d (%.3f); null cyclic share %.3f, P(null <= real) %.3f' % (tri, cyc, cyc / max(tri, 1), res[T + '_triads'][2], res[T + '_triads'][3]), flush=True)
        # (d) MCMC
        items, ranks, ar = mcmc_bt(ords, steps=150000 if T == 'L' else 60000)
        cnt = Counter(x for o in ords for x in o)
        med = np.median(ranks, 0); lo = np.quantile(ranks, 0.05, 0); hi = np.quantile(ranks, 0.95, 0)
        rec = sorted([i for i, x in enumerate(items) if cnt[x] >= (2 if T == 'L' else 4)], key=lambda i: med[i])
        res[T + '_mcmc'] = [(items[i], cnt[items[i]], float(med[i]), float(lo[i]), float(hi[i])) for i in rec]
        print(T, 'MCMC accept %.2f; posterior median rank (90%% interval) of n items %d:' % (ar, len(items)))
        for x in res[T + '_mcmc'][:12] + [('...',)] + res[T + '_mcmc'][-8:]: print('   ', x)
        json.dump(res, open(os.path.join(CK, 'c3_%s.json' % types), 'w'), default=str)
