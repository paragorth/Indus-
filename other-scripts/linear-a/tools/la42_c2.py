#!/usr/bin/env python3
"""LA-42 cycle 2: leave-one-out on Linear A. Each LB-valued LA sign is hidden in turn (full count, thinned to 10 and
to 5 tokens) and placed by the weights frozen on Linear B (cycle 1). Controls: the same with LA's values permuted
among the valued signs (R3: wrong values, grid occupancy kept; 20 permutations); grid-structure-only model (PRIOR).
Second model: ALL + blind la21 rows, fitted on LA itself by nested leave-one-sign-out (and the same on permuted values)."""
import os, sys, pickle, collections
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la42_common as G

NPERM = 20
R21 = G.la21('LA')


def job(args):
    tag, words, V, h, keep = args
    c = G.make_case(words, V, h, keep=keep, seed=sum(map(ord, h)) + (keep or 0), r21=R21)
    if c is not None:
        c['tag'] = tag; c['keep'] = keep or 0
    return c


def nested(cases, use):
    hits = []
    for i in range(len(cases)):
        m = G.fit(cases[:i] + cases[i + 1:], use)
        hits.append(G.evaluate(m, [cases[i]]))
    return {k: float(np.mean([x[k] for x in hits])) for k in hits[0] if k != 'n'} | dict(n=len(hits))


def main():
    W = [r['w'] for r in G.la_units()]
    V = G.known_values(W)
    cnt = collections.Counter(s for w in W for s in w)
    jobs = []
    for h in sorted(V):
        for keep in (None, 10, 5):
            if keep is None or cnt[h] > keep:
                jobs.append(('LA', W, V, h, keep))
    rng = np.random.default_rng(42)
    signs = sorted(V)
    for p in range(NPERM):
        perm = rng.permutation(len(signs))
        Vp = {s: V[signs[perm[i]]] for i, s in enumerate(signs)}
        for h in signs:
            jobs.append((f'P{p}', W, Vp, h, None))
            if cnt[h] > 10:
                jobs.append((f'P{p}', W, Vp, h, 10))
    print(len(jobs), 'cases', flush=True)
    with Pool(2) as pool:
        cases = [c for c in pool.imap(job, jobs, chunksize=4) if c is not None]
    pickle.dump(cases, open(os.path.join(G.CK, 'c2_cases.pkl'), 'wb'))
    lines = []
    models = {nm: pickle.load(open(os.path.join(G.CK, f'c1_model_{nm}.pkl'), 'rb')) for nm in ('PRIOR', 'PHON', 'SIM', 'ALL', 'ALLnoE')}
    for nm, m in models.items():
        for keep in (0, 10, 5):
            e = G.evaluate(m, [c for c in cases if c['tag'] == 'LA' and c['keep'] == keep])
            line = f"LA {nm} (LB-frozen) keep={keep or 'full'}: {G.fmt_eval(e)}"
            if keep in (0, 10):
                pe = [G.evaluate(m, [c for c in cases if c['tag'] == f'P{p}' and c['keep'] == keep]) for p in range(NPERM)]
                for k in ('top1', 'top3', 'row1', 'col1', 'lp'):
                    arr = np.array([x[k] for x in pe])
                    line += f" | perm {k} mean {arr.mean():.3f} max {arr.max():.3f} P {(1 + (arr >= e[k]).sum()) / (NPERM + 1):.3f}"
            lines.append(line); print(line, flush=True)
    # per-sign detail (ALL, full)
    m = models['ALL']
    for c in [c for c in cases if c['tag'] == 'LA' and c['keep'] == 0]:
        p = G.predict(m, c); o = np.argsort(-p)
        lines.append(f"  {c['sign']:5s} n={c['ntok']:3d} true {c['cands'][c['truth']]} rank {int(np.where(o == c['truth'])[0][0]) + 1} "
                     f"top3 {[(''.join(c['cands'][i]) or '-', round(float(p[i]), 2)) for i in o[:3]]}")
    # nested LOO on LA with la21 rows
    for use_nm, use in (('ALL', models['ALL']['use']), ('ALL+r21', models['ALL']['use'] + ['r21']), ('PRIOR+r21', models['PRIOR']['use'] + ['r21'])):
        e = nested([c for c in cases if c['tag'] == 'LA' and c['keep'] == 0], use)
        pe = [nested([c for c in cases if c['tag'] == f'P{p}' and c['keep'] == 0], use) for p in range(min(NPERM, 10))]
        line = f"LA nested-LOO fit on LA, {use_nm}, full: {G.fmt_eval(e)}"
        for k in ('top1', 'top3', 'row1', 'col1'):
            arr = np.array([x[k] for x in pe])
            line += f" | perm {k} mean {arr.mean():.3f} max {arr.max():.3f}"
        lines.append(line); print(line, flush=True)
    open(os.path.join(G.CK, 'c2_report.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
