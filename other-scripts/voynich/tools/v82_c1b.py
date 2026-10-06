"""v82 cycle 1b: CONFIRM the near-kills. The final stage left no setting passing the pre-registered kill in all 3 seeds
(single-seed band misses on doubling / nbr_ratio / line_first / hapax, and one setting at the +0.03 edge). Here the
closest settings, and the strong harmony settings with the agreement strength scaled down (x0.5, x0.7, x0.85) so the
gain lands near the Voynich, are re-run with 10 fresh seeds each (ZL-fitted, 20 re-rolls, same-section and same-page
nulls, lag ratio). Per seed: KILL if min gain >= 0.030 with z >= 3 and every statistic in band."""
import os, sys, json, copy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v82_c1 as C1, v82_lib as K
from multiprocessing import Pool

if __name__ == '__main__':
    F = C1.load('c1_final.jsonl')
    by = {}
    for r in F:
        if r['which'] != 'ZL': continue
        by.setdefault(json.dumps(r['P'], sort_keys=True), []).append(r)
    def score_set(rs):
        return sum(min(r['g']) >= 0.03 and min(r['z']) >= 3 and r['ok'] for r in rs)
    ranked = sorted(by.values(), key=lambda rs: (score_set(rs), -abs(np.mean([min(r['g']) for r in rs]) - 0.05)), reverse=True)
    cands = []
    for rs in ranked[:4]:
        cands.append(('near', rs[0]['P']))
    strong = [rs for rs in by.values() if np.mean([min(r['g']) for r in rs]) > 0.1 and sum(r['ok'] for r in rs) >= 2]
    strong.sort(key=lambda rs: -sum(r['ok'] for r in rs))
    for rs in strong[:2]:
        for f in (0.5, 0.7, 0.85):
            P = copy.deepcopy(rs[0]['P']); P['agr_lam'] *= f; P['mood_beta'] *= f; cands.append(('scaled%.2f' % f, P))
    jobs = [(P, 9100 + s, 20, True, 'ZL') for tag, P in cands for s in range(10)]
    with Pool(2, initializer=C1.init) as Pl:
        out = list(Pl.imap(C1.score, jobs))
    for i, r in enumerate(out): r['tag'] = cands[i // 10][0]; r['cand'] = i // 10
    C1.jl('c1b.jsonl', out)
