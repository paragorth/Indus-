"""v87 simulation bank: sim i is fully determined by seed (base + i). Usage: v87_bank.py NAME START COUNT [prior]
prior: 'base' (cycle 1 encoder prior) or 'wide' (cycle 3: alternative encoder prior)."""
import sys, os, json, random, time
from multiprocessing import Pool
import numpy as np
import v87_lib as L

S = L.load_sources(); SIDS = sorted(S)
FEATS = L.TRAIN + L.HELD


def wide_scheme(rng):
    sc = L.random_scheme(rng)
    # alternative prior: more verbose letter codes, more homophony, more nulls, more positional tables
    sc['unit'] = rng.choice(['letter', 'chunk', 'word', 'nomen'])
    sc['homo'] = rng.choice([1, 2, 3, 4, 6])
    sc['nulls'] = rng.choice([0.0, 0.1, 0.25])
    sc['linefirst'] = rng.random() < 0.6
    sc['conc'] = rng.choice([0.05, 0.1, 0.3, 1.0, 5.0])
    sc['trunc'] = rng.choice([None, 2, 3, 4])
    return sc


def one(args):
    seed, prior = args
    rng = random.Random(seed)
    sid = rng.choice(SIDS)
    sc = (wide_scheme if prior == 'wide' else L.random_scheme)(rng)
    ents, lines, n = L.simulate(S[sid], sc, rng)
    if n < L.NTOK * 0.9:
        return None
    f = L.feats(lines); p = L.plain_props(ents, S[sid])
    return dict(seed=seed, sid=sid, unit=sc['unit'], props=[p[k] for k in L.PROPS], feats=[float(f[k]) for k in FEATS])


if __name__ == '__main__':
    name, start, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    prior = sys.argv[4] if len(sys.argv) > 4 else 'base'
    out = os.path.join(L.CK, 'bank_%s.jsonl' % name)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            done.add(json.loads(l)['seed'])
    todo = [(s, prior) for s in range(start, start + count) if s not in done]
    t = time.time()
    with Pool(2) as pool, open(out, 'a') as fo:
        for i, r in enumerate(pool.imap_unordered(one, todo, chunksize=8)):
            if r: fo.write(json.dumps(r) + '\n')
            if i % 500 == 0:
                fo.flush(); print(i, len(todo), round(time.time() - t), flush=True)
