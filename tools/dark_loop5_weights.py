"""Loop 5 cycle 4b: do seal texts encode Indus weight-series values {1,2,4,8,16,32,64,160,200,320,640}?
Anneal integer values (1..VMAX) for the 60 commonest signs so that the sum (or positional base-b number)
of a text lands in the series as often as possible on train sites; score the hit rate on held-out sites.
Null: the same annealing on a sign-shuffled corpus (lengths kept) and on 3 random weight-sets of the same size/range.
Baseline: hit rate of the stroke-count assignment."""
import sys, json, math, random, collections, time
sys.path.insert(0, 'tools')
from multiprocessing import Pool
import numpy as np
from dark_loop5_engine import *

WEIGHTS = [1, 2, 4, 8, 16, 32, 64, 160, 200, 320, 640]
RESTARTS = 30; STEPS = 4000; VMAX = 12
STROKES = {int(k): v for k, v in json.load(open('data/derived/dark/loop5_strokes.json')).items()}
objs = load_rows(); SIGNS = top_signs([o['seq_raw'] for o in objs])
S = [o for o in objs if o['type'].startswith('SEAL')]

def hit_rate(vals, wset):
    v = np.rint(vals).astype(int)
    return float(np.isin(v, list(wset)).mean())

def anneal_hits(prob, wset, seed, vmax=VMAX, steps=STEPS):
    rng = random.Random(seed); K = prob.K
    assign = [rng.randint(1, vmax) for _ in range(K)]
    cur = prob.values(assign); sc = hit_rate(cur, wset); best = (sc, list(assign))
    for it in range(steps):
        T = 0.02 * (1 - it / steps) + 1e-4
        k = rng.randrange(K); old = assign[k]; new = rng.randint(1, vmax)
        if new == old: continue
        nv = cur + prob.M[:, k] * (new - old); ns = hit_rate(nv, wset)
        if ns >= sc or rng.random() < math.exp((ns - sc) / T):
            assign[k] = new; cur = nv; sc = ns
            if sc > best[0]: best = (sc, list(assign))
    return best

def job(args):
    rule, kind, seed, wset = args
    rng = np.random.default_rng(seed)
    seqs_tr = [o['seq_raw'] for o in S if o['site'] in TRAIN_SITES]
    seqs_te = [o['seq_raw'] for o in S if o['site'] not in TRAIN_SITES]
    if kind == 'shuffle':
        pool = [a for s in seqs_tr + seqs_te for a in s if a not in NUMERALS and a not in EXCL]
        rng.shuffle(pool); it = iter(pool)
        resh = lambda s: [a if (a in NUMERALS or a in EXCL) else next(it) for a in s]
        seqs_tr = [resh(s) for s in seqs_tr]; seqs_te = [resh(s) for s in seqs_te]
    ptr = Problem(seqs_tr, SIGNS, rule, 10); pte = Problem(seqs_te, SIGNS, rule, 10)
    runs = []
    for r in range(RESTARTS):
        sc, a = anneal_hits(ptr, wset, seed * 100 + r)
        runs.append((sc, hit_rate(pte.values(a), wset), a))
    runs.sort(key=lambda z: -z[0]); top = runs[:10]
    stab = np.array([collections.Counter(a[k] for _, _, a in top).most_common(1)[0][1] / 10 for k in range(len(SIGNS))])
    # chance: random assignments
    chance = np.mean([hit_rate(pte.values([random.Random(s).randint(1, VMAX) for _ in range(ptr.K)]), wset) for s in range(50)])
    return rule, kind, seed, dict(train=runs[0][0], test=runs[0][1], test_max=max(z[1] for z in runs), chance_test=float(chance),
                                  stable_frac=float((stab > 0.8).mean()), stable=[SIGNS[k] for k in range(len(SIGNS)) if stab[k] > 0.8], assign=runs[0][2])

if __name__ == '__main__':
    rng = np.random.default_rng(7)
    jobs = [('sum', 'real', 1, WEIGHTS), ('pos', 'real', 1, WEIGHTS), ('sum', 'shuffle', 2, WEIGHTS), ('pos', 'shuffle', 2, WEIGHTS)]
    for p in range(3):   # random weight sets: 11 values in the same range, log-uniform
        rw = sorted(set(int(round(math.exp(x))) for x in rng.uniform(0, math.log(640), 11)))
        jobs.append(('sum', f'randset{p}:{rw}', 40 + p, rw))
    t0 = time.time()
    with Pool(4) as P: res = P.map(job, jobs, chunksize=1)
    lines = [f'# loop5 cycle 4b: weight-series hit rate. seals n_train={sum(o["site"] in TRAIN_SITES for o in S)} n_test={sum(o["site"] not in TRAIN_SITES for o in S)}; series {WEIGHTS}; restarts {RESTARTS}, steps {STEPS}, values 1..{VMAX}; wall {time.time()-t0:.0f}s']
    for rule in ('sum', 'pos'):
        seqs = [o['seq_raw'] for o in S if o['site'] not in TRAIN_SITES]
        p = Problem(seqs, SIGNS, rule, 10)
        lines.append(f'FIXED stroke-count [{rule}] held-out hit rate = {hit_rate(p.values([STROKES.get(a,1) for a in SIGNS]+[1]), WEIGHTS):.3f}')
    for rule, kind, seed, r in res:
        lines.append(f'{rule:4s} {kind:40s}: train hit={r["train"]:.3f} test(best-train)={r["test"]:.3f} test_max={r["test_max"]:.3f} chance(random values)={r["chance_test"]:.3f} stable>80%={r["stable_frac"]:.2f} {r["stable"][:8]}')
    r = [x for x in res if x[1] == 'real' and x[0] == 'sum'][0][3]
    lines.append('best sum assignment: ' + ' '.join(f'W{s}={v}' for s, v in zip(SIGNS, r['assign'])))
    open('data/derived/dark/loop5_cycle4b.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
