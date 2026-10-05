#!/usr/bin/env python3
"""LA-56 cycle 1: thousands of random inclusion rules on written totals; held-out re-test; nulls and planted.

Usage: python3 la56_c1.py CORPUS JOB [n] [seed]
  CORPUS: LA | UR3 | LB
  JOB: real | N1 (totals permuted) | N2 (entries dealt across tablets) | N3 (noise on balanced totals)
       | PEXCL / PNEG (a random word feature planted as excluded / subtracted in every section that has it)
Writes data/la56_ckpt/c1_<CORPUS>_<JOB>_<seed>.json
"""
import sys, json, os, random, copy, time
import numpy as np
import la56_common as C

N_SPLITS = 12
N_PAIRS = 1500


def load(name):
    return {'LA': C.load_la, 'LAs': lambda: C.load_la(True), 'UR3': C.load_ur3, 'LB': C.load_lb}[name]()


def base_diff_int(s):
    return sum(it['v'] for it in s['items'] if it['role'] == 'main') - s['total'][0]


def exact_total(s, skip=lambda it: False, sign=lambda it: 1):
    v = 0; l = {}
    for it in s['items']:
        if it['role'] != 'main' or skip(it): continue
        g = sign(it)
        v += g * it['v']
        for L, c in it['l'].items(): l[L] = l.get(L, 0) + g * c
    return v, {L: c for L, c in l.items() if c}


def make(secs, job, rng):
    S = copy.deepcopy(secs)
    info = {}
    if job == 'N1':
        T = [s['total'] for s in S]; rng.shuffle(T)
        for s, t in zip(S, T): s['total'] = t
    elif job == 'N2':
        pool = [it for s in S for it in s['items'] if it['role'] == 'main']
        rng.shuffle(pool); i = 0
        for s in S:
            n = sum(it['role'] == 'main' for it in s['items'])
            s['items'] = pool[i:i + n] + [it for it in s['items'] if it['role'] != 'main']; i += n
    elif job == 'N3':
        diffs = [base_diff_int(s) for s in secs]
        nz = [d for d in diffs if d != 0] or [1]
        frac_bad = np.mean([d != 0 for d in diffs])
        for s in S:
            v, l = exact_total(s)
            if v < 0: v = 0
            if rng.random() < frac_bad:
                v = max(0, v + rng.choice([-1, 1]) * abs(rng.choice(nz)))
            s['total'] = (v, l)
    elif job in ('PEXCL', 'PNEG'):
        fs = {}
        for k, s in enumerate(S):
            for it in s['items']:
                if it['role'] != 'main': continue
                for f in it['f']:
                    if f.startswith('w=') and f != 'w=NONE' and f not in ('w=KU-RO', 'w=PO-TO-KU-RO'): fs.setdefault(f, set()).add(k)
        cand = sorted(f for f, ks in fs.items() if len(ks) >= 2)
        f = rng.choice(cand)
        for k in fs[f]:
            s = S[k]
            if job == 'PEXCL':
                v, l = exact_total(s, skip=lambda it: f in it['f'])
            else:
                v, l = exact_total(s, sign=lambda it: -1 if f in it['f'] else 1)
            if v < 0 or any(c < 0 for c in l.values()):
                continue
            s['total'] = (v, l)
        info = {'planted': f, 'op': 'EXCL' if job == 'PEXCL' else 'NEG', 'secs': sorted(fs[f])}
    return S, info


def run(name, job, seed):
    rng = random.Random(seed * 7919 + sum(map(ord, job)))
    secs = load(name)
    S, info = make(secs, job, rng) if job != 'real' else (secs, {})
    eng = C.Engine(S, D=600, seed=seed)
    singles, pairs = C.candidate_rules(S, n_pairs=N_PAIRS, seed=seed)
    rules = singles + pairs
    t = time.time()
    rows = C.eval_rows(eng, rules)
    allidx = list(range(len(S)))
    g_full = C.gains(eng, rows, allidx)
    order = np.argsort(-g_full)
    srng = np.random.default_rng(seed)
    held = []; rep = np.zeros(len(rules))
    for sp in range(N_SPLITS):
        a, b = C.split_tabs(S, random.Random(seed * 100 + sp))
        ga = C.gains(eng, rows, a); gb = C.gains(eng, rows, b)
        ia = int(np.argmax(ga))
        held.append(float(gb[ia]))
        # reverse direction too
        ib = int(np.argmax(gb)); held.append(float(ga[ib]))
        rep += ((ga > 1.0) & (gb > 1.0))
    out = {'corpus': name, 'job': job, 'seed': seed, 'info': info, 'n_secs': len(S), 'n_rules': len(rules),
           'base_ll': float(eng.base_ll()), 'n_close': int((eng.base.mean(1) > 0.5).sum()),
           'best_full': float(g_full[order[0]]),
           'top': [(C.rule_str(rules[i]), float(g_full[i]), int(rep[i]), sorted(rows[i].keys())) for i in order[:40]],
           'held_mean': float(np.mean(held)), 'held': held,
           'n_rep': int((rep >= N_SPLITS / 2).sum()),
           'rep_rules': [(C.rule_str(rules[i]), int(rep[i]), float(g_full[i])) for i in np.argsort(-rep)[:30] if rep[i] > 0],
           'secs': [s['id'] for s in S], 'sec_close': eng.base.mean(1).tolist(), 'time': time.time() - t}
    if info:
        target = (info['op'], info['planted'])
        ri = [i for i, r in enumerate(rules) if len(r) == 1 and r[0] == target]
        if ri:
            i = ri[0]
            out['plant_rank'] = int((g_full > g_full[i]).sum()) + 1
            out['plant_gain'] = float(g_full[i]); out['plant_rep'] = int(rep[i])
            # rank among singles only
            gs = g_full[:len(singles)]
            out['plant_rank_singles'] = int((gs > g_full[i]).sum()) + 1
    fn = os.path.join(C.CK, 'c1_%s_%s_%d.json' % (name, job, seed))
    json.dump(out, open(fn, 'w'))
    print(name, job, seed, 'best %.2f held %.2f nrep %d' % (out['best_full'], out['held_mean'], out['n_rep']),
          'plant rank %s' % out.get('plant_rank'), '%.0fs' % out['time'], flush=True)


if __name__ == '__main__':
    name, job = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    s0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    for sd in range(s0, s0 + n):
        if os.path.exists(os.path.join(C.CK, 'c1_%s_%s_%d.json' % (name, job, sd))): continue
        run(name, job, sd)
