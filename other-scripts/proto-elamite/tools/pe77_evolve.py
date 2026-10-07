"""pe77 cycle 2: evolve forgers.  A population of forger specs is mutated and selected to MINIMISE the full
detector's held-out AUC (2 seeds per evaluation).  The final elite (top 8) is re-run on 12 fresh seeds each;
unforgeability is then measured against the strongest forgers that search could find.
Output: data/pe77_ckpt/evo_<corpus>.json and evo_<corpus>_elite.jsonl (same format as pe77_run)."""
import sys, os, json, random, math, copy, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pe77_common as pc
import pe77_run as runner

OPS = ['entry_swap', 'num_swap', 'num_within', 'sign_swap', 'string_swap', 'header_swap', 'drop_dup', 'num_jitter']


def mutate(spec, rng):
    s = copy.deepcopy(spec)
    if s['base'] == 'scratch':
        if rng.random() < 0.3:
            return pc.random_forger(rng)
        s['cond'] = min(1, max(0, s['cond'] + rng.gauss(0, 0.2)))
        if rng.random() < 0.3: s['order'] = rng.choice([0, 1, 2])
        if rng.random() < 0.3: s['strings'] = not s['strings']
        if rng.random() < 0.3: s['total'] = not s['total']
        return s
    r = rng.random()
    ops = s['ops']
    if r < 0.2 and len(ops) < 4:
        o = rng.choice([o for o in OPS if o not in ops]); ops[o] = math.exp(rng.uniform(math.log(0.05), 0))
    elif r < 0.35 and len(ops) > 1:
        del ops[rng.choice(list(ops))]
    else:
        o = rng.choice(list(ops)); ops[o] = min(1.0, max(0.01, ops[o] * math.exp(rng.gauss(0, 0.6))))
    if rng.random() < 0.2: s['cond'] = min(1, max(0, s['cond'] + rng.gauss(0, 0.3)))
    if rng.random() < 0.15: s['fix_total'] = not s['fix_total']
    return s


def ev(a):
    spec, seeds = a
    As = []
    for sd in seeds:
        names, A, d, z = pc.run_forger(runner.G['T'], runner.G['maps'], spec, sd)
        As.append(A)
    return float(np.mean(As))


def elite_job(a):
    k, spec, seed, shuf = a
    names, A, d, z = pc.run_forger(runner.G['T'], runner.G['maps'], spec, seed, shuffle_labels=shuf)
    return {'k': k, 'seed': seed, 'spec': spec, 'A': A,
            'f': {n: [None if np.isnan(x) else round(float(x), 4), None if np.isnan(zz) else round(float(zz), 3)]
                  for n, x, zz in zip(names, d, z)}}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus'); ap.add_argument('--pop', type=int, default=16); ap.add_argument('--gen', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--elite', type=int, default=8)
    ap.add_argument('--reps', type=int, default=12)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tag = 'evo_%s_%d' % (a.corpus, a.seed)
    with Pool(2, initializer=runner.init, initargs=(a.corpus, 0)) as P:
        pop = [pc.random_forger(rng) for _ in range(a.pop * 2)]
        hist = []
        for g in range(a.gen):
            seeds = [rng.randrange(1 << 30) for _ in range(2)]
            fit = P.map(ev, [(s, seeds) for s in pop], chunksize=1)
            order = np.argsort(fit)
            hist.append({'gen': g, 'best': float(fit[order[0]]), 'median': float(np.median(fit)),
                         'best_spec': pop[order[0]]})
            print(tag, g, round(fit[order[0]], 3), round(float(np.median(fit)), 3), pop[order[0]], flush=True)
            keep = [pop[i] for i in order[:a.pop // 2]]
            pop = keep + [mutate(rng.choice(keep), rng) for _ in range(a.pop - len(keep))]
        # final selection on fresh seeds
        seeds = [rng.randrange(1 << 30) for _ in range(2)]
        fit = P.map(ev, [(s, seeds) for s in pop], chunksize=1)
        order = np.argsort(fit)
        elite = [pop[i] for i in order[:a.elite]]
        json.dump({'hist': hist, 'elite': elite, 'elite_fit': [float(fit[i]) for i in order[:a.elite]]},
                  open(os.path.join(pc.CKPT, tag + '.json'), 'w'), indent=1)
        jobs = []
        for e, spec in enumerate(elite):
            for r in range(a.reps):
                jobs.append((e * 100 + r, spec, 900000 + a.seed * 1000 + e * 100 + r, False))
        with open(os.path.join(pc.CKPT, tag + '_elite.jsonl'), 'w') as out:
            for res in P.imap_unordered(elite_job, jobs, chunksize=1):
                out.write(json.dumps(res) + '\n'); out.flush()
        jobs = [(e * 100 + r, spec, 950000 + a.seed * 1000 + e * 100 + r, True) for e, spec in enumerate(elite) for r in range(4)]
        with open(os.path.join(pc.CKPT, tag + '_elite_shuf.jsonl'), 'w') as out:
            for res in P.imap_unordered(elite_job, jobs, chunksize=1):
                out.write(json.dumps(res) + '\n'); out.flush()
    print('done', tag)
