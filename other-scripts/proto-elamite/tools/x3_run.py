#!/usr/bin/env python3
"""X-3 driver: transfer matrix. usage: x3_run.py TAG SLICE SEEDS XLIST YLIST [EXTRA_PAIRS]
  TAG     output name (results -> $X3_SCR/res_<TAG>.jsonl, checkpointed per job)
  SLICE   Y fine-tune slice in tokens
  SEEDS   comma list of seeds (pretraining seed = slice seed = fine-tune seed)
  XLIST / YLIST comma lists of corpus ids; all ordered pairs X != Y are run
  EXTRA_PAIRS  comma list of X>Y pairs added (controls such as LBa>LBb, PLa>PLb)
One job = (X, condition, seed): pretrain once, fine-tune on every Y. Scratch jobs = (Y, seed).
Label mode is run only when >= 20% of Y's test tokens carry a label that X also has.
2 worker processes, 1 thread each.
"""
import os, sys, json, random
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x3_common as X

CONDS = ['real', 'relab', 'band', 'shuf', 'markov']


def ydata(y, seed, slice_tok):
    C = X.corpora()
    te, va, tr = X.split(C[y])
    ytr = X.sample_blocks(tr, slice_tok, random.Random(seed * 101 + 7))
    return ytr, va, te


def overlap(xrowmap, te):
    n = sum(len(t) for t in te)
    return sum(1 for t in te for s in t if s in xrowmap) / n


def job(args):
    tag, kind, a, seed, ys, slice_tok = args
    out = []
    if kind == 'scratch':
        ytr, va, te = ydata(a, seed, slice_tok)
        r = X.finetune(None, X.y_map({}, [], ytr, 'rank'), ytr, va, te, seed)
        out.append(dict(tag=tag, x=None, y=a, cond='scratch', mode='-', seed=seed, slice=slice_tok, **r))
        return out
    xname, cond = a
    sd, rowmap, xo = X.pretrain(xname, cond, seed)
    for y in ys:
        ytr, va, te = ydata(y, seed, slice_tok)
        ov = overlap(rowmap, te)
        modes = ['rank'] + (['label'] if ov >= 0.2 else [])
        for mode in modes:
            r = X.finetune(sd, X.y_map(rowmap, xo, ytr, mode), ytr, va, te, seed)
            out.append(dict(tag=tag, x=xname, y=y, cond=cond, mode=mode, seed=seed, slice=slice_tok,
                            overlap=round(ov, 3), **r))
    return out


def main():
    tag, slice_tok = sys.argv[1], int(sys.argv[2])
    seeds = [int(s) for s in sys.argv[3].split(',')]
    xs, ys = sys.argv[4].split(','), sys.argv[5].split(',')
    extra = [p.split('>') for p in sys.argv[6].split(',')] if len(sys.argv) > 6 and sys.argv[6] else []
    conds = os.environ.get('X3_CONDS', ','.join(CONDS)).split(',')
    path = os.path.join(X.SCR, f'res_{tag}.jsonl')
    done = set()
    if os.path.exists(path):
        for L in open(path):
            d = json.loads(L); done.add(d['jobkey'])
    targets = {}
    for x in xs:
        targets[x] = [y for y in ys if y != x]
    for x, y in extra:
        targets.setdefault(x, []).append(y)
    jobs = []
    for seed in seeds:
        for y in sorted(set(ys) | {y for _, y in extra}):
            jobs.append((tag, 'scratch', y, seed, None, slice_tok))
        for x, yl in targets.items():
            for c in conds:
                jobs.append((tag, 'pre', (x, c), seed, yl, slice_tok))
    key = lambda j: f"{j[1]}|{j[2]}|{j[3]}"
    jobs = [j for j in jobs if key(j) not in done]
    print(len(jobs), 'jobs to run', flush=True)
    with Pool(2) as pool, open(path, 'a') as f:
        for j, res in zip(jobs, pool.imap(job, jobs)):
            for r in res:
                r['jobkey'] = key(j)
                f.write(json.dumps(r) + '\n')
            f.flush()
            print('done', key(j), flush=True)


if __name__ == '__main__':
    main()
