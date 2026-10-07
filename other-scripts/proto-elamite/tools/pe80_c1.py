"""pe80 cycle 1: NUMBERS THAT ARE NOT AMOUNTS (label / recurrence search).

Random and exhaustive (scope filter x recurrence relation) hypotheses; score = exact hits
between consecutive in-scope values inside a tablet. Null A: values permuted within tablet
inside the scope. Null B: rank-preserving resample (values drawn from the scope pool of the
same half, put in the real rank order), which kills 'sorted lists look like steps'.
Train on half A, replicate on half B.
usage: python3 pe80_c1.py RUN [seed]   RUN in pe | plant | shuf | ur3
"""
import sys, json, random, os, math
import numpy as np
from scipy.stats import poisson
import pe80_common as pc

RELS = ['+1', '-1', '+2', '-2', '+5', '+10', '-10', 'x2', 'x1/2', 'x3', 'AP', 'SUM', 'IDX']
NPERM = 200


def pairs_hits(V, seg, rel):
    """V: (R, n) int array (row = one ordering), seg: (n,) segment ids. returns (R,) hits."""
    same1 = seg[1:] == seg[:-1]
    a, b = V[:, :-1], V[:, 1:]
    if rel == 'IDX':
        # value equals 1-based position within segment
        pos = np.zeros(len(seg), int)
        c = 0
        for i in range(len(seg)):
            c = c + 1 if i > 0 and seg[i] == seg[i - 1] else 1
            pos[i] = c
        return ((V == pos[None, :]) & (pos[None, :] >= 1)).sum(1) - 0  # counts every matching line
    if rel in ('AP', 'SUM'):
        same2 = same1[1:] & same1[:-1]
        x, y, z = V[:, :-2], V[:, 1:-1], V[:, 2:]
        if rel == 'AP':
            h = (y - x == z - y) & (y != x)
        else:
            h = (z == x + y)
        return (h & same2[None, :]).sum(1)
    d = {'+1': 1, '-1': -1, '+2': 2, '-2': -2, '+5': 5, '+10': 10, '-10': -10}
    if rel in d:
        h = (b - a == d[rel])
    elif rel == 'x2':
        h = (b == 2 * a)
    elif rel == 'x1/2':
        h = (2 * b == a)
    elif rel == 'x3':
        h = (b == 3 * a)
    return (h & same1[None, :]).sum(1)


def scope(tabs, filt):
    vals, seg = [], []
    for i, t in enumerate(tabs):
        vv = [l['v'] for l in t['lines'] if l['v'] is not None and filt(l)]
        if len(vv) >= 2:
            vals += vv
            seg += [i] * len(vv)
    return np.array(vals, int), np.array(seg, int)


def perm_within(vals, seg, R, rng):
    key = seg[None, :] * 1.0 + rng.random((R, len(seg)))
    idx = np.argsort(key, axis=1)
    return vals[idx]


def rank_resample(vals, seg, R, rng):
    out = np.empty((R, len(vals)), int)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], len(seg)]
    for r in range(R):
        draw = vals[rng.integers(0, len(vals), len(vals))]
        for s, e in zip(starts, ends):
            real = vals[s:e]
            d = np.sort(draw[s:e])
            rk = np.argsort(np.argsort(real, kind='stable'), kind='stable')
            out[r, s:e] = d[rk]
    return out


def make_hyps(tabs, rng, nrand=1500):
    from collections import Counter
    cnt = {m: Counter() for m in ('in', 'first', 'last')}
    for t in tabs:
        for l in t['lines']:
            if l['v'] is None:
                continue
            for x in set(l['tok']):
                cnt['in'][x] += 1
            if l['first']:
                cnt['first'][l['first']] += 1
            if l['last']:
                cnt['last'][l['last']] += 1
    H = [('ALL', None)]
    for m in cnt:
        for x, c in cnt[m].most_common(80):
            if c >= 15:
                H.append((m, x))
    pool = [x for x, c in cnt['in'].most_common(120) if c >= 15]
    for _ in range(nrand):
        k = rng.integers(2, 4)
        H.append(('any', tuple(sorted(rng.choice(pool, k, replace=False)))))
    return list(dict.fromkeys(H))


def filt_of(h):
    m, x = h
    if m == 'ALL':
        return lambda l: True
    if m == 'in':
        return lambda l: x in l['tok']
    if m == 'first':
        return lambda l: l['first'] == x
    if m == 'last':
        return lambda l: l['last'] == x
    if m == 'any':
        s = set(x)
        return lambda l: bool(s & set(l['tok']))


def eval_h(tabs, h, rels, rng, nullB=False):
    vals, seg = scope(tabs, filt_of(h))
    res = {}
    if len(vals) < 6:
        return res
    P = perm_within(vals, seg, NPERM, rng)
    B = rank_resample(vals, seg, 100, rng) if nullB else None
    for rel in rels:
        H = int(pairs_hits(vals[None, :], seg, rel)[0])
        nh = pairs_hits(P, seg, rel)
        mu, sd = float(nh.mean()), float(nh.std())
        pA = float(poisson.sf(H - 1, max(mu, 1e-3))) if H > mu else 1.0
        pemp = float((1 + (nh >= H).sum()) / (NPERM + 1))
        r = dict(H=H, mu=mu, sd=sd, pA=pA, pAemp=pemp, n=int(len(vals)))
        if nullB:
            nb = pairs_hits(B, seg, rel)
            r['muB'] = float(nb.mean())
            r['pB'] = float((1 + (nb >= H).sum()) / 101)
        res[rel] = r
    return res


def plant(tabs, rng, n_tabs=int(os.environ.get("PE80_NPLANT", 40))):
    pool = [s for t in tabs for l in t['lines'] for s in l['tok']]
    idx = rng.choice(len(tabs), n_tabs, replace=False)
    for i in idx:
        t = tabs[i]
        k = int(rng.integers(3, 6))
        pos = int(rng.integers(0, len(t['lines']) + 1))
        new = []
        for j in range(k):
            other = pool[rng.integers(len(pool))]
            new.append(dict(tok=['MPLANT', other], first='MPLANT', last=other, v=j + 1, surf='obverse'))
        t['lines'][pos:pos] = new
    return tabs


def shuffle_values(tabs, rng):
    for t in tabs:
        v = [l['v'] for l in t['lines']]
        rng.shuffle(v)
        for l, x in zip(t['lines'], v):
            l['v'] = x
    return tabs


def main():
    run = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = np.random.default_rng(1000 + seed)
    prng = random.Random(seed)
    if run == 'ur3':
        U = pc.build_ur3()
        U = [t for t in U if sum(l['v'] is not None for l in t['lines']) >= 2]
        prng.shuffle(U)
        tabs = [dict(id=t['id'], lines=t['lines']) for t in U[:1585]]
        tabs, mp = pc.opaque(tabs, seed)
        truth = {mp.get('u4'), mp.get('kam')}
    else:
        tabs = pc.load_pe()
        truth = {'MPLANT'}
    order = list(range(len(tabs)))
    prng.shuffle(order)
    A = [tabs[i] for i in order[: len(order) // 2]]
    Bh = [tabs[i] for i in order[len(order) // 2:]]
    if run == 'plant':
        A = plant(A, rng)
        Bh = plant(Bh, rng)
    if run == 'shuf':
        A = shuffle_values(A, prng)
        Bh = shuffle_values(Bh, prng)
    hyps = make_hyps(A, rng)
    ntest = len(hyps) * len(RELS)
    surv = []
    for h in hyps:
        r = eval_h(A, h, RELS, rng)
        for rel, x in r.items():
            if x['H'] >= 4 and x['pA'] < 0.05 / ntest and x['pAemp'] <= 1 / (NPERM + 1) + 1e-9:
                surv.append((h, rel, x))
    # null B on survivors, then replication on half B
    final = []
    for h, rel, x in surv:
        rb = eval_h(A, h, [rel], rng, nullB=True)[rel]
        x['pB'] = rb['pB']; x['muB'] = rb['muB']
        if rb['pB'] > 0.01:
            final.append(dict(h=h, rel=rel, train=x, stage='killed_by_rankB'))
            continue
        rr = eval_h(Bh, h, [rel], rng, nullB=True).get(rel)
        rep = rr is not None and rr['H'] > rr['mu'] and rr['pAemp'] < 0.01 and rr['pB'] < 0.01
        final.append(dict(h=h, rel=rel, train=x, test=rr, stage='replicated' if rep else 'not_replicated'))
    def is_truth(h):
        m, x = h
        xs = set(x) if isinstance(x, tuple) else {x}
        return bool(xs & truth)
    out = dict(run=run, seed=seed, n_hyp=len(hyps), n_tests=ntest, n_surv_A=len(surv),
               n_rankB_pass=sum(f['stage'] != 'killed_by_rankB' for f in final),
               n_repl=sum(f['stage'] == 'replicated' for f in final),
               n_repl_truth=sum(f['stage'] == 'replicated' and is_truth(f['h']) for f in final),
               final=[dict(f, h=[f['h'][0], f['h'][1] if not isinstance(f['h'][1], tuple) else list(f['h'][1])], truth=is_truth(f['h'])) for f in final])
    fn = os.path.join(pc.CK, 'c1_%s_%d.json' % (run, seed))
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    print(run, seed, {k: v for k, v in out.items() if k != 'final'})
    for f in final:
        if f['stage'] == 'replicated':
            print('  REPL', f['h'], f['rel'], f['train']['H'], round(f['train']['mu'], 1), (f['test']['H'], round(f['test']['mu'], 1)), is_truth(f['h']))


if __name__ == '__main__':
    main()
