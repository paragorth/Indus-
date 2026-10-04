"""pe6 cycle 4: calibrated grading of role assignments.
 NULL  : 3 independent global sign shuffles, whole pipeline each; per role, the null distribution of the
         probability that role gets on ANY shuffled sign (not only arg-max) -> 99th percentile and max.
 REAL  : full role-probability vectors for real signs, real halves (re-run with full vectors), and
         within-line shuffle (position destroyed) as a 'what does the assignment rest on' check.
 DIAG  : in posterior simulations, which roles do header-dominant signs (hdr1 >= 0.6) have?  (opener vs name)
Grades: A = top-role probability above the null MAX for that role AND same top role in both halves;
        B = above the null 99th percentile and same top role in at least one half; C = top role >= 0.5 only.
Output: data/pe6_cycle4.json
"""
import os, sys, json, random
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe6_common import load_real, ROLES, shuffle_global, shuffle_within_line, PEDATA, FEATS
from pe6_bank import load_bank
from pe6_pipe import pipeline, post_sims

OUT = os.path.join(PEDATA, 'pe6_cycle4.json')


def full(P):
    return {s: [float(x) for x in m] for s, (m, sd) in P['assign'].items()}


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    bank = load_bank(); ok = np.isfinite(bank[3]).all(1); bank = tuple(b[ok] for b in bank)
    R = load_real()
    rs = random.Random(21); ids = list(range(len(R))); rs.shuffle(ids); h = set(ids[:len(R) // 2])
    arms = {'real': lambda: R, 'null1': lambda: shuffle_global(R, 101), 'null2': lambda: shuffle_global(R, 102),
            'null3': lambda: shuffle_global(R, 103), 'line': lambda: shuffle_within_line(R, 104),
            'half_a': lambda: [t for i, t in enumerate(R) if i in h],
            'half_b': lambda: [t for i, t in enumerate(R) if i not in h]}
    for k, (arm, f) in enumerate(arms.items()):
        if arm in out:
            continue
        C = f()
        P = pipeline(C, bank, nsim=200, seed=40 + k, min_n=8 if arm.startswith('half') else 15)
        out[arm] = full(P)
        print(arm, 'done', flush=True)
        json.dump(out, open(OUT, 'w'))
    if 'diag' not in out:
        adj = np.load(os.path.join(PEDATA, 'pe6_post_real.npy'))
        res = post_sims(adj, 60, seed=77)
        i1 = FEATS.index('hdr1')
        c = Counter(); c2 = Counter()
        for X, y, _ in res:
            for x, r in zip(X, y):
                if x[i1] >= 0.6:
                    c[r] += 1
                if r == 'OPEN':
                    c2['hdr1>=0.6' if x[i1] >= 0.6 else 'hdr1<0.6'] += 1
        out['diag'] = dict(header_dominant_roles=dict(c), open_signs=dict(c2))
        json.dump(out, open(OUT, 'w'))
    # grading
    null = {r: [] for r in ROLES}
    for a in ('null1', 'null2', 'null3'):
        for s, v in out[a].items():
            for j, r in enumerate(ROLES):
                null[r].append(v[j])
    thr = {r: (float(np.percentile(null[r], 99)), float(max(null[r]))) for r in ROLES}
    top = lambda v: (ROLES[int(np.argmax(v))], float(max(v)))
    grades = []
    for s, v in out['real'].items():
        r, p = top(v)
        ha = top(out['half_a'][s])[0] if s in out['half_a'] else None
        hb = top(out['half_b'][s])[0] if s in out['half_b'] else None
        ln = top(out['line'][s]) if s in out['line'] else None
        nh = (ha == r) + (hb == r)
        if p > thr[r][1] and nh == 2:
            g = 'A'
        elif p > thr[r][0] and nh >= 1:
            g = 'B'
        elif p >= 0.5:
            g = 'C'
        else:
            g = '-'
        grades.append((s, r, round(p, 3), g, ha, hb, ln[0] if ln else None, round(ln[1], 3) if ln else None))
    # same grading applied to null1 against null2+null3 (false-positive rate of the grading itself)
    null23 = {r: [] for r in ROLES}
    for a in ('null2', 'null3'):
        for s, v in out[a].items():
            for j, r in enumerate(ROLES):
                null23[r].append(v[j])
    thr23 = {r: (float(np.percentile(null23[r], 99)), float(max(null23[r]))) for r in ROLES}
    fp = Counter()
    for s, v in out['null1'].items():
        r, p = top(v)
        if r != 'SYL' and p > thr23[r][0]:
            fp['above99'] += 1
        if r != 'SYL' and p > thr23[r][1]:
            fp['abovemax'] += 1
    out['thr'] = thr; out['grades'] = grades; out['null1_fp'] = dict(fp)
    json.dump(out, open(OUT, 'w'), indent=1)
    gc = Counter((g[1], g[3]) for g in grades)
    print('thresholds', thr); print('grades', dict(gc)); print('null fp', dict(fp)); print(out['diag'])
    for g in sorted(grades, key=lambda x: (x[3], -x[2])):
        if g[3] in 'AB' or (g[3] == 'C' and g[1] != 'SYL'):
            print(g)


if __name__ == '__main__':
    main()
