#!/usr/bin/env python3
"""X-2 cycle 1: calibrate the cross-script alignment on Linear B <-> Ur III (known commodities).

Conditions (A = Linear B DAMOS, B = Ur III CDLI administrative):
  full : LB bootstrap (2,478 docs)   vs Ur III random 2,478 docs
  pe   : LB random 1,510             vs Ur III random 1,510   (Proto-Elamite size)
  lape : LB random 352               vs Ur III random 1,510   (Linear A x Proto-Elamite sizes)
  la   : LB random 352               vs Ur III random 352     (Linear A size)
Each condition: REPS real replicates and REPS shuffled replicates (commodity labels and
designation tokens permuted across entries on both sides). Per replicate and method:
MRR of the gold Ur III target for each gold LB commodity, P@1, and a 200x target-label
permutation P. Words: total / deficit / delivery / 'from' gold.
Checkpoint: data/x2/c1_runs.jsonl (one line per task; reruns skip finished tasks). 2 workers.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
import x2_common as X

REPS = int(os.environ.get('X2_REPS', 50))
CK = os.path.join(X.DX, 'c1_runs.jsonl')
COND = {'full': ('boot', 2478), 'pe': (1510, 1510), 'lape': (352, 1510), 'la': (352, 352)}
G = json.load(open(X.GOLD_FILE))
_LB = _UR = None


def task(args):
    global _LB, _UR
    cond, rep, shuf = args
    if _LB is None:
        _LB = X.load('LB'); _UR = X.load('UR3')
    rng = np.random.default_rng(1000 * rep + 7 * (cond == 'pe') + 13 * (cond == 'lape') + 17 * (cond == 'la') + 500000 * shuf)
    na, nb = COND[cond]
    A = X.boot(_LB, rng) if na == 'boot' else [_LB[i] for i in rng.choice(len(_LB), na, replace=False)]
    B = [_UR[i] for i in rng.choice(len(_UR), nb, replace=False)]
    if shuf:
        A = X.shuffle_all(A, rng); B = X.shuffle_all(B, rng)
    itA, itB, out = X.run_pair(A, B, rng)
    res = {'cond': cond, 'rep': rep, 'shuf': shuf, 'nA_c': sum(i.startswith('c:') for i in itA),
           'nB_c': sum(i.startswith('c:') for i in itB), 'm': {}}
    for m, (S, obj) in out.items():
        r = {'obj': obj, 'c': X.gold_eval(itA, itB, S, G['gold'], rng, 200),
             'w': X.gold_eval(itA, itB, S, G['gold_words'], rng, 200, pre='w:')}
        # top-1 target for each gold source (for reading off)
        ia = {it: i for i, it in enumerate(itA)}
        cb = [j for j, it in enumerate(itB) if it.startswith('c:')]
        r['top'] = {s: itB[cb[int(np.argmax(S[ia['c:' + s], cb]))]][2:] for s in G['gold'] if 'c:' + s in ia}
        res['m'][m] = r
    return res


def main():
    done = set()
    if os.path.exists(CK):
        for l in open(CK):
            d = json.loads(l); done.add((d['cond'], d['rep'], d['shuf']))
    tasks = [(c, r, s) for r in range(REPS) for c in COND for s in (0, 1) if (c, r, s) not in done]
    print('tasks to run', len(tasks), file=sys.stderr)
    t0 = time.time()
    with Pool(2) as P, open(CK, 'a') as f:
        for i, res in enumerate(P.imap_unordered(task, tasks)):
            f.write(json.dumps(res) + '\n'); f.flush()
            if i % 20 == 0:
                print(i, round(time.time() - t0), file=sys.stderr)


def summarise():
    rows = X.jsonl(CK)
    out = {}
    for cond in COND:
        for m in X.METHODS:
            for lev in ('c', 'w'):
                real = [r['m'][m][lev] for r in rows if r['cond'] == cond and not r['shuf'] and r['m'][m][lev]]
                sh = [r['m'][m][lev] for r in rows if r['cond'] == cond and r['shuf'] and r['m'][m][lev]]
                if not real:
                    continue
                mr = np.array([x['mrr'] for x in real]); ms = np.array([x['mrr'] for x in sh]) if sh else np.array([np.nan])
                out['%s|%s|%s' % (cond, m, lev)] = {
                    'reps': len(real), 'n_src': float(np.mean([x['n'] for x in real])),
                    'mrr_real': float(mr.mean()), 'mrr_perm_null': float(np.mean([x['null_mrr'] for x in real])),
                    'frac_p05': float(np.mean([x['p'] < 0.05 for x in real])),
                    'p1_real': float(np.mean([x['p1'] / x['n'] for x in real])),
                    'mrr_shuf': float(np.nanmean(ms)), 'mrr_shuf95': float(np.nanpercentile(ms, 95)),
                    'frac_real_gt_shuf95': float(np.mean(mr > np.nanpercentile(ms, 95))),
                    'p_real_vs_shuf_mean': float((1 + (ms >= mr.mean()).sum()) / (1 + len(ms)))}
    # per-source top-1 hit rate (real, seed)
    hits = {}
    for cond in COND:
        for m in ('seed', 'prof', 'freq'):
            hh = {}
            for r in rows:
                if r['cond'] != cond or r['shuf']:
                    continue
                for s, t in r['m'][m]['top'].items():
                    hh.setdefault(s, []).append(t)
            hits['%s|%s' % (cond, m)] = {s: {'hit': float(np.mean([x in G['gold'][s] for x in v])),
                                             'modal': max(set(v), key=v.count), 'modal_share': v.count(max(set(v), key=v.count)) / len(v)}
                                        for s, v in hh.items()}
    json.dump({'summary': out, 'hits': hits}, open(os.path.join(X.DX, 'c1_summary.json'), 'w'), indent=1)
    for k, v in out.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    for k in ('full|seed', 'pe|seed', 'lape|seed', 'la|seed'):
        print(k, {s: (round(v['hit'], 2), v['modal'], round(v['modal_share'], 2)) for s, v in hits[k].items()})


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'sum':
        summarise()
    else:
        main(); summarise()
