"""pe6 cycle 3: leave-one-statistic-out predictive check.
For every statistic (and every statistic group), refit the ABC posterior WITHOUT it, simulate new corpora at
posterior draws, and ask where the held-out observed value falls in that posterior predictive distribution.
A statistic the economy cannot predict from the others is a part of PE writing the fake Susa does not explain.
Also: role assignment when the whole number-system group is left out (do roles rest on numerals alone?).
Output: data/pe6_cycle3.json (checkpointed per statistic).
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe6_common import load_real, stats, STATS, GROUPS, PEDATA, sign_features
from pe6_bank import load_bank
from pe6_pipe import fit, post_sims, train_assign, grade_table

OUT = os.path.join(PEDATA, 'pe6_cycle3.json')


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    bank = load_bank(); ok = np.isfinite(bank[3]).all(1); bank = tuple(b[ok] for b in bank)
    R = load_real(); so = stats(R)
    K = len(STATS)
    items = [('stat', k, [k]) for k in range(K)] + \
            [('group', g, [k for k, s in enumerate(STATS) if s.startswith(g + '_')]) for g in GROUPS]
    for kind, key, drop in items:
        name = f'{kind}:{STATS[key] if kind == "stat" else key}'
        if name in out:
            continue
        use = [k for k in range(K) if k not in drop]
        adj, idx, d = fit(so, bank, frac=0.01, use=use)
        if kind == 'stat':
            PS = bank[3][idx]          # rejection-ABC posterior predictive straight from the bank (200 runs)
        else:
            res = post_sims(adj, 60, seed=500 + len(out), want_stats=True)
            PS = np.array([r[2] for r in res])
        row = {}
        for k in drop:
            q = float((PS[:, k] < so[k]).mean() + 0.5 * (PS[:, k] == so[k]).mean())
            row[STATS[k]] = (float(so[k]), float(np.median(PS[:, k])), float(np.percentile(PS[:, k], 5)),
                             float(np.percentile(PS[:, k], 95)), q)
        if kind == 'group' and key in ('N', 'E', 'H'):
            Fobs = sign_features(R, 15)
            assign, cal, per = train_assign(res, Fobs, seed=7)
            rows = grade_table(assign)
            row['_roles'] = [(s, r, round(p, 3), g) for s, r, p, sd, g in rows]
        out[name] = row
        print(name, {k: (round(v[0], 3), round(v[1], 3), round(v[4], 3)) for k, v in row.items() if k[0] != '_'},
              flush=True)
        json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
