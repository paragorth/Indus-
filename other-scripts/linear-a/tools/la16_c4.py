#!/usr/bin/env python3
"""LA-16 cycle 4: massive random guessing with held-out re-test.
Each split: attributed documents are divided at random into TRAIN and TEST halves (stratified by site).
10,000 random hypotheses = random sets of 5 sign pairs (drawn from sign pairs with >= 1 train word pair).
Each hypothesis is scored on TRAIN by the summed hand-contrast z (cycle-1 statistic, 300 hand permutations).
The top 1 % survive and are re-scored on TEST, counting only TEST word pairs never seen in TRAIN (new words,
new tokens). Statistic: survivors' mean TEST z minus all hypotheses' mean TEST z. 40 splits.
Null: the same pipeline with TRAIN hands permuted (so selection is blind). LB (KN+PY) is the positive control.
"""
import sys, os, json, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la16_common import *

say = say_to(os.path.join(OUT, 'c4_report.txt'))
NH, KSET, NPERM = 10000, 5, 300


def zstat(docs, strat, rnd, permute=False):
    E = Engine(docs, strat=strat)
    if permute: E.lab0 = E.perm(np.random.default_rng(rnd.randrange(10 ** 6)))
    real = E.stat(E.lab0); N = E.null(NPERM, rnd.randrange(10 ** 6))
    z = (real - N.mean(0)) / (N.std(0) + 1e-9)
    return E, z


def heldout_z(E, excl, rnd):
    """z per sign pair on TEST, using only word pairs not in excl (set of (w1,w2))."""
    keep = np.array([(E.T[i], E.T[j]) not in excl for i, j, x, y in E.P])
    def st(lab):
        return np.bincount(E.pk[keep], weights=E.cross(lab)[keep], minlength=len(E.keys))
    real = st(E.lab0); g = np.random.default_rng(rnd.randrange(10 ** 6))
    N = np.array([st(E.perm(g)) for _ in range(NPERM)])
    n = np.bincount(E.pk[keep], minlength=len(E.keys))
    z = (real - N.mean(0)) / (N.std(0) + 1e-9); z[n == 0] = np.nan
    return {k: z[i] for i, k in enumerate(E.keys) if n[i] > 0}


def split_once(docs, strat, rnd, permute):
    att = [d for d in docs if d.get('hand')]
    by = collections.defaultdict(list)
    for d in att: by[d['site']].append(d)
    tr, te = [], []
    for v in by.values():
        v = v[:]; rnd.shuffle(v); h = len(v) // 2; tr += v[:h]; te += v[h:]
    Etr, ztr = zstat(tr, strat, rnd, permute)
    trpairs = {(Etr.T[i], Etr.T[j]) for i, j, x, y in Etr.P}
    Ete = Engine(te, strat=strat)
    zte = heldout_z(Ete, trpairs, rnd)
    keys = [k for k in Etr.keys if k in zte]
    if len(keys) < 3 * KSET: return None
    ktr = {k: ztr[i] for i, k in enumerate(Etr.keys)}
    H = [rnd.sample(keys, KSET) for _ in range(NH)]
    s_tr = np.array([sum(ktr[k] for k in h) for h in H])
    s_te = np.array([np.mean([zte[k] for k in h]) for h in H])
    top = np.argsort(-s_tr)[:NH // 100]
    # also: single best train pairs
    best = sorted(keys, key=lambda k: -ktr[k])[:5]
    return float(s_te[top].mean() - s_te.mean()), len(keys), [(k, round(float(ktr[k]), 2), round(float(zte[k]), 2)) for k in best]


def run(tag, docs, strat, nsplit, rnd):
    out = {}
    for mode in ('real', 'permuted'):
        vals = []; bests = collections.Counter()
        for s in range(nsplit):
            r = split_once(docs, strat, rnd, mode == 'permuted')
            if r is None: continue
            vals.append(r[0])
            for k, a, b in r[2]: bests[k] += 1
        v = np.array(vals)
        say(f'   {tag} {mode}: splits {len(v)}, survivors held-out z gain mean {v.mean():.3f} sd {v.std():.3f} '
            f'(share > 0: {(v > 0).mean():.2f})')
        out[mode] = v.tolist()
        if mode == 'real': say('     most often top-5 on TRAIN: ' + ', '.join(f'{a}~{b} x{n}' for (a, b), n in bests.most_common(10)))
    a, b = np.array(out['real']), np.array(out['permuted'])
    rnd2 = np.random.default_rng(0); allv = np.concatenate([a, b]); d0 = a.mean() - b.mean()
    perm = [(lambda p: p[:len(a)].mean() - p[len(a):].mean())(rnd2.permutation(allv)) for _ in range(5000)]
    say(f'   {tag}: real - permuted = {d0:.3f}, P {(np.array(perm) >= d0).mean():.4f}')
    out['diff'] = d0
    return out


def main():
    rnd = random.Random(44)
    res = {}
    lb = lb_corpus(('KN', 'PY'))
    for d in lb: d['ss'] = d['site'] + ':' + (d.get('support') or '')
    say('== LB KN+PY (positive control)')
    res['LB'] = run('LB', lb, 'ss', int(os.environ.get('NSLB', 20)), rnd)
    la = la_corpus()
    say('== LA hands')
    res['LA'] = run('LA', la, 'site', int(os.environ.get('NSLA', 60)), rnd)
    json.dump(res, open(os.path.join(OUT, 'c4.json'), 'w'), default=str, indent=1)


if __name__ == '__main__':
    main()
