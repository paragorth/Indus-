"""pe77 driver: run N random forgers on a corpus (PE, PC, W1 planted world with hidden herd, W0 without),
optionally with shuffled real/forged labels.  Output: data/pe77_ckpt/<tag>.jsonl (one forger per line)."""
import sys, os, json, random, argparse
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pe77_common as pc

G = {}


def init(corpus, wseed):
    if corpus in ('PE', 'PC'):
        G['T'], G['maps'] = pc.load_corpus(corpus)
    else:
        G['T'], G['maps'] = pc.planted_world(seed=wseed, hidden=(corpus == 'W1'))


def job(a):
    k, seed, shuf = a
    rng = random.Random(seed * 7919 + 13)
    spec = pc.random_forger(rng)
    names, A, d, z = pc.run_forger(G['T'], G['maps'], spec, seed, shuffle_labels=shuf)
    return {'k': k, 'seed': seed, 'spec': spec, 'A': A,
            'f': {n: [None if np.isnan(x) else round(float(x), 4), None if np.isnan(zz) else round(float(zz), 3)]
                  for n, x, zz in zip(names, d, z)}}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus'); ap.add_argument('n', type=int); ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--shuf', action='store_true'); ap.add_argument('--tag'); ap.add_argument('--wseed', type=int, default=0)
    a = ap.parse_args()
    tag = a.tag or '%s_%s%d' % (a.corpus, 'shuf_' if a.shuf else '', a.seed0)
    fn = os.path.join(pc.CKPT, tag + '.jsonl')
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)['k'] for l in open(fn)}
    todo = [(k, a.seed0 + k, a.shuf) for k in range(a.n) if k not in done]
    with Pool(2, initializer=init, initargs=(a.corpus, a.wseed)) as P, open(fn, 'a') as out:
        for r in P.imap_unordered(job, todo, chunksize=2):
            out.write(json.dumps(r) + '\n'); out.flush()
    print('done', tag, len(done) + len(todo))
