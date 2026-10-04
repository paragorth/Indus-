"""pe32 cycle 4: SPLIT-TABLET PREDICTION (the held-out pool of cycle 2 is too thin to test anything).
Repeat over random 50/50 tablet splits: the class is re-derived on half A only (counted signs before a standard
M288 line, count known); sign features are computed from A tablets only (predecessor lines of M288 entries
removed, M288 successors never used); the top-20 non-class signs by affinity to the A class are frozen; then on
half B the M288 entries after those signs are scored (standard = 60 x, or 60 x / 120 x).  Control: for every
split, 100 seeds of random signs (frequency-matched from the other A-side M288 predecessors) go through the same
pipeline; their hits on B are the null.  A second control asks whether the A class itself predicts B (members
of class A that precede M288 on B), against the same seeds."""
import os, sys, json, time
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe32_common import *  # noqa
from pe32_cycle2 import affinity, FAM2

K = 20
NSPL = int(os.environ.get('NSPL', 10))
NS = int(os.environ.get('NS', 100))
t0 = time.time()


def hits(lst, B, mult):
    s = set(lst)
    ev = [e for e in B if e['pfin'] in s and e['x']]
    return len(ev), sum(is_std(e, mult) for e in ev)


if __name__ == '__main__':
    rng = np.random.default_rng(44)
    T = table()
    E = m288_events(T)
    ids = sorted({t['id'] for t in T})
    out = []
    for sp in range(NSPL):
        A = set(rng.choice(ids, len(ids) // 2, replace=False).tolist())
        TA = [t for t in T if t['id'] in A]
        EA = [e for e in E if e['tid'] in A]
        EB = [e for e in E if e['tid'] not in A and e['x']]
        clsA = sorted({e['pfin'] for e in EA if e['x'] and is_std(e)})
        F = sign_features(TA, exclude={(e['tid'], e['line'] - 1) for e in EA})
        univ = sorted(s for s, n in F['fin_n'].items() if n >= 2 and s not in ('x', '-'))
        ix = {s: i for i, s in enumerate(univ)}
        M = sim_matrix(F, univ)
        seed = [s for s in clsA if s in ix]
        cand = set(univ) - set(clsA) - {e['pfin'] for e in EA}
        aff = affinity(M, ix, seed, univ)
        order = [univ[i] for i in np.argsort(-aff)]
        P = [s for s in order if s in cand][:K]
        Bn = [e for e in EB if e['pfin'] not in clsA]
        r = {'split': sp, 'clsA': clsA, 'P': P, 'P_h1': hits(P, Bn, (1,)), 'P_h12': hits(P, Bn, (1, 2)),
             'cls_h1': hits(clsA, EB, (1,)), 'cls_h12': hits(clsA, EB, (1, 2)),
             'base_B': (len(Bn), sum(is_std(e) for e in Bn)),
             'P_hit_signs': [(e['pfin'], e['tid'], e['x'], e['y'], is_std(e)) for e in Bn if e['pfin'] in set(P)]}
        pool = [s for s in sorted({e['pfin'] for e in EA} - set(clsA)) if s in ix]
        N1, N12 = [], []
        for k in range(NS):
            sd = freq_matched(rng, seed, pool, F['fin_n']) if len(pool) >= len(seed) else list(rng.choice(pool, len(seed)))
            cd = set(univ) - set(sd) - {e['pfin'] for e in EA}
            a = affinity(M, ix, sd, univ)
            Pr = [univ[i] for i in np.argsort(-a) if univ[i] in cd][:K]
            Bs = [e for e in EB if e['pfin'] not in set(sd)]
            N1.append(hits(Pr, Bs, (1,))); N12.append(hits(Pr, Bs, (1, 2)))
        r['null_h1'] = np.array(N1).tolist(); r['null_h12'] = np.array(N12).tolist()
        print(sp, 'clsA', len(clsA), 'P', r['P_h1'], r['P_h12'], 'null mean', np.mean([x[1] for x in N1]),
              np.mean([x[1] for x in N12]), 'class->B', r['cls_h1'], 'baseB', r['base_B'], round(time.time() - t0),
              flush=True)
        out.append(r)
    sP1 = sum(r['P_h1'][1] for r in out); sP12 = sum(r['P_h12'][1] for r in out)
    nP = sum(r['P_h1'][0] for r in out)
    N1 = np.array([[r['null_h1'][k][1] for r in out] for k in range(NS)]).sum(1)
    N12 = np.array([[r['null_h12'][k][1] for r in out] for k in range(NS)]).sum(1)
    NE = np.array([[r['null_h1'][k][0] for r in out] for k in range(NS)]).sum(1)
    summ = {'splits': len(out), 'P_events': nP, 'P_hits_std': sP1, 'P_hits_std_or_double': sP12,
            'null_events_mean': float(NE.mean()), 'null_std_mean': float(N1.mean()),
            'p_std': float((1 + (N1 >= sP1).sum()) / (1 + NS)), 'null_12_mean': float(N12.mean()),
            'p_12': float((1 + (N12 >= sP12).sum()) / (1 + NS)),
            'class_to_B': [sum(r['cls_h1'][0] for r in out), sum(r['cls_h1'][1] for r in out)],
            'base_B': [sum(r['base_B'][0] for r in out), sum(r['base_B'][1] for r in out)],
            'P_hits_by_sign': dict(Counter(h[0] for r in out for h in r['P_hit_signs'] if h[4])),
            'P_events_by_sign': dict(Counter(h[0] for r in out for h in r['P_hit_signs'])),
            'P_hits_C14': sum(1 for r in out for h in r['P_hit_signs'] if h[4] and h[0] in C14),
            'P_freq': dict(Counter(s for r in out for s in r['P']).most_common(25))}
    print(json.dumps(summ))
    json.dump({'summary': summ, 'splits': out}, open(os.path.join(CK, 'cycle4.json'), 'w'), default=str)
    print('done', round(time.time() - t0))
