#!/usr/bin/env python3
"""R-3 runner: draw worlds (prior or a supplied proposal), evolve their scripts, score the panel.
usage: r3_run.py TAG N [SEED0] [PROPOSAL.json]
Writes data/r3_ckpt/sims_TAG.jsonl (resumable). At most 2 worker processes."""
import sys, os, json, random, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3_lib import *

PROP = None
V2 = os.environ.get('R3_V2') == '1'


def job(i):
    seed = SEED0 + i
    r = random.Random(seed * 7919 + 3)
    if PROP is None:
        th = draw_prior(r, PRIOR2 if V2 else None)
    else:
        # proposal: pick an accepted theta and perturb it (Gaussian, within prior bounds)
        base = PROP['thetas'][r.randrange(len(PROP['thetas']))]
        th = {}
        for k, (lo, hi, kind) in (PRIOR2 if V2 else PRIOR).items():
            v = base[k] + r.gauss(0, PROP['sd'][k])
            v = min(hi, max(lo, v))
            th[k] = int(round(v)) if kind == 'i' else v
    try:
        s = simulate(th, seed)
    except Exception as e:
        return None
    if s is None:
        return None
    return {'i': i, 'seed': seed, 'th': th, 's': s}


def main():
    global SEED0, PROP
    tag = sys.argv[1]; n = int(sys.argv[2])
    SEED0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000000
    if len(sys.argv) > 4:
        PROP = json.load(open(sys.argv[4]))
    out = os.path.join(CK, 'sims_%s.jsonl' % tag)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                done.add(json.loads(l)['i'])
            except Exception:
                pass
    todo = [i for i in range(n) if i not in done]
    t = time.time()
    with Pool(2, initializer=_init, initargs=(SEED0, PROP)) as P, open(out, 'a') as f:
        for k, res in enumerate(P.imap_unordered(job, todo, chunksize=8)):
            if res:
                f.write(json.dumps(res) + '\n')
            if k % 500 == 0:
                f.flush()
                print(tag, k, len(todo), round(time.time() - t), flush=True)


def _init(s0, prop):
    global SEED0, PROP
    SEED0 = s0; PROP = prop


if __name__ == '__main__':
    main()
