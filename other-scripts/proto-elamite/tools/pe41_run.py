"""pe41 runner: simulate populations, align each invented system to fixed targets, write votes.
usage: python3 pe41_run.py TAG SEED0 N [workers]
targets: PE, PE shuffled (3 shuffles), PC, PE half A / half B (tablet split), 12 held-out invented systems.
"""
import sys, os, json, gzip, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe41_lib import *

CK = os.path.join(DATA, 'pe41_ckpt'); os.makedirs(CK, exist_ok=True)
NHELD = 12
TG = None


def build_targets():
    pe, _ = load_real('PE'); pc, _ = load_real('PC')
    rng = np.random.default_rng(41)
    tg = {}
    def add(name, tabs, maxs=300):
        k, G, A = features(tabs, maxsigns=maxs); tg[name] = (k, rankq(G), A, None)
    add('PE', pe)
    for i in range(3):
        add('SHUF%d' % i, shuffle_corpus(pe, np.random.default_rng(100 + i)))
    add('PC', pc)
    perm = rng.permutation(len(pe)); h = len(pe) // 2
    add('PEA', [pe[i] for i in perm[:h]]); add('PEB', [pe[i] for i in perm[h:]])
    for j in range(NHELD):
        s = 900000 + j
        P = Population(sample_params(np.random.default_rng(s)), s)
        tabs, lab = P.write_corpus()
        k, G, A = features(tabs, maxsigns=300)
        tg['HELD%d' % j] = (k, rankq(G), A, [lab.get(x, -1) for x in k])
    return tg


def init():
    global TG
    TG = build_targets()


def one(seed):
    try:
        rng = np.random.default_rng(seed)
        p = sample_params(rng)
        P = Population(p, seed)
        tabs, lab = P.write_corpus()
        k, G, A = features(tabs, maxsigns=400)
        if len(k) < 10:
            return None
        R = rankq(G)
        labs = np.array([lab.get(x, -1) for x in k])
        out = dict(seed=seed, p=p, L=P.L, pol=P.pol, stats=corpus_stats(tabs).tolist(), nsig=len(k),
                   labshare=np.bincount(labs[labs >= 0], minlength=len(LABELS)).tolist())
        res = {}
        for name, (kt, Gt, At, _) in TG.items():
            m, c = align(Gt, At, R, A)
            v = [int(labs[j]) if j >= 0 else -1 for j in m]
            res[name] = dict(cost=c, v=v)
        out['res'] = res
        return out
    except Exception as e:
        return dict(seed=seed, err=repr(e))


if __name__ == '__main__':
    tag, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    w = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    init()
    meta = {name: dict(signs=[str(x) for x in kt], truth=tr) for name, (kt, _, _, tr) in TG.items()}
    json.dump(meta, open(os.path.join(CK, 'targets_meta.json'), 'w'))
    fn = os.path.join(CK, '%s.jsonl.gz' % tag)
    t0 = time.time(); done = 0
    with gzip.open(fn, 'at') as f, Pool(w, initializer=init) as pool:
        for r in pool.imap_unordered(one, range(s0, s0 + n), chunksize=4):
            if r is None:
                continue
            f.write(json.dumps(r) + '\n'); done += 1
            if done % 50 == 0:
                f.flush(); print(done, round(time.time() - t0), flush=True)
    print('done', done, round(time.time() - t0))
