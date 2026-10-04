"""pe6 cycle 2: posterior role assignment of real signs, with controls.
 REAL      : ABC posterior -> 200 posterior corpora -> random-forest role classifier -> real signs.
 SHUFFLE-G : all sign tokens permuted over the corpus; whole pipeline rerun (roles must NOT be recoverable).
 SHUFFLE-L : sign order permuted within each line (position destroyed, line contents kept).
 PLANTED   : corpora written by known simulated economies (posterior median, and two other economies);
             whole pipeline rerun on them; confident assignments scored against the truth.
 HALVES    : real corpus split into two random halves of tablets; pipeline on each (half-size sims).
Output: data/pe6_cycle2.json (checkpointed per arm).
"""
import os, sys, json, random
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe6_common import (load_real, stats, ROLES, shuffle_global, shuffle_within_line, simulate, from_unbounded,
                        draw_prior, PEDATA, PNAMES)
from pe6_bank import load_bank
from pe6_pipe import pipeline, grade_table

OUT = os.path.join(PEDATA, 'pe6_cycle2.json')


def summarize(name, P, truth=None):
    rows = grade_table(P['assign'])
    cnt = Counter((r[1], r[4]) for r in rows)
    roles_conf = Counter(r[1] for r in rows if r[4] in 'AB')
    modal = Counter(r[1] for r in rows).most_common(1)[0][0]
    d = dict(n_signs=len(rows), grades={f'{a}|{b}': c for (a, b), c in cnt.items()},
             conf_by_role=dict(roles_conf), modal_role=modal,
             conf_nonmodal=sum(1 for r in rows if r[4] in 'AB' and r[1] != modal),
             conf_nonsyl=sum(1 for r in rows if r[4] in 'AB' and r[1] != 'SYL'),
             dist=(float(P['d'].min()), float(np.median(P['d']))), cal={str(k): v for k, v in P['cal'].items()},
             per_role_cal=P['per_role'], post={k: P['post'][k] for k in PNAMES},
             rows=[(s, r, round(p, 3), round(sd, 3), g) for s, r, p, sd, g in rows])
    if truth is not None:
        sc = {}
        for g in ('A', 'B', 'C', '-'):
            sel = [(truth[s], r) for s, r, p, sd, gg in rows if gg == g]
            sc[g] = (len(sel), float(np.mean([a == b for a, b in sel])) if sel else None)
        conf = [(truth[s], r) for s, r, p, sd, gg in rows if gg in 'AB']
        byrole = {}
        for rl in ROLES:
            tp = sum(1 for t, r in conf if r == rl and t == rl); fp = sum(1 for t, r in conf if r == rl and t != rl)
            nt = sum(1 for s, *_ in rows if truth[s] == rl)
            byrole[rl] = dict(true_signs=nt, conf_hits=tp, conf_false=fp)
        d['planted'] = dict(by_grade=sc, by_role=byrole)
    print(name, json.dumps({k: d[k] for k in ('n_signs', 'conf_by_role', 'conf_nonsyl', 'dist')}), flush=True)
    if truth is not None:
        print('   planted', json.dumps(d['planted']['by_grade']), flush=True)
    return d


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    bank = load_bank(); ok = np.isfinite(bank[3]).all(1); bank = tuple(b[ok] for b in bank)
    R = load_real()
    arms = ['real', 'shuf_global', 'shuf_line', 'planted_med', 'planted_alt1', 'planted_alt2', 'half_a', 'half_b']
    for arm in arms:
        if arm in out:
            continue
        truth = None
        if arm == 'real':
            C = R
        elif arm == 'shuf_global':
            C = shuffle_global(R, 3)
        elif arm == 'shuf_line':
            C = shuffle_within_line(R, 4)
        elif arm.startswith('planted'):
            if arm == 'planted_med':
                adj = np.load(os.path.join(PEDATA, 'pe6_post_real.npy'))
                th = from_unbounded(np.median(adj, 0)); seed = 999
            else:
                # another economy: a posterior draw far from the median (alt1) / a prior draw near the data (alt2)
                adj = np.load(os.path.join(PEDATA, 'pe6_post_real.npy'))
                if arm == 'planted_alt1':
                    dd = ((adj - np.median(adj, 0)) ** 2).sum(1); th = from_unbounded(adj[np.argmax(dd)]); seed = 998
                else:
                    th = from_unbounded(bank[1][np.argsort(((bank[3] - stats(R)) ** 2 /
                                         (bank[3].std(0) ** 2 + 1e-9)).sum(1))[50]]); seed = 997
            C, role = simulate(th, seed)
            truth = role
            out.setdefault('planted_theta', {})[arm] = th
        else:
            rs = random.Random(21); ids = list(range(len(R))); rs.shuffle(ids)
            h = set(ids[:len(R) // 2])
            C = [t for i, t in enumerate(R) if (i in h) == (arm == 'half_a')]
        P = pipeline(C, bank, nsim=200, seed=arms.index(arm) + 1, min_n=15 if not arm.startswith('half') else 8)
        out[arm] = summarize(arm, P, truth)
        json.dump(out, open(OUT, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
