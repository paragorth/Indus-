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
NHELD = 12 if os.environ.get("PE41_MODE", "c1") == "c1" else 8
TG = None


MODE = os.environ.get('PE41_MODE', 'c1')


def norm(G):
    return rankq(G) if MODE == 'c1' else prep(G)


SHR = 0.0 if MODE == 'c1' else 5.0


def build_targets():
    pe, _ = load_real('PE'); pc, _ = load_real('PC')
    rng = np.random.default_rng(41)
    tg = {}
    def add(name, tabs, maxs=300):
        k, G, A = features(tabs, maxsigns=maxs, shrink=SHR); tg[name] = (k, norm(G), A, None)
    def halves(tag, tabs, seed):
        perm = np.random.default_rng(seed).permutation(len(tabs)); h = len(tabs) // 2
        add(tag + 'A', [tabs[i] for i in perm[:h]]); add(tag + 'B', [tabs[i] for i in perm[h:]])
    add('PE', pe)
    sh = [shuffle_corpus(pe, np.random.default_rng(100 + i)) for i in range(3)]
    for i in range(3):
        add('SHUF%d' % i, sh[i])
    add('PC', pc)
    if MODE == 'c1':
        perm = rng.permutation(len(pe)); h = len(pe) // 2
        add('PEA', [pe[i] for i in perm[:h]]); add('PEB', [pe[i] for i in perm[h:]])
    else:
        halves('PE', pe, 41); halves('SHUF0', sh[0], 42); halves('PC', pc, 43)
        halves('PCSH', shuffle_corpus(pc, np.random.default_rng(200)), 44)
    for j in range(NHELD):
        s = 900000 + j
        P = Population(sample_params(np.random.default_rng(s)), s)
        tabs, lab = P.write_corpus()
        k, G, A = features(tabs, maxsigns=300, shrink=SHR)
        tg['HELD%d' % j] = (k, norm(G), A, [lab.get(x, -1) for x in k])
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
        k, G, A = features(tabs, maxsigns=400, shrink=SHR)
        if len(k) < 10:
            return None
        R = norm(G)
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
    json.dump(meta, open(os.path.join(CK, 'targets_meta%s.json' % ('' if MODE == 'c1' else '_' + MODE)), 'w'))
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
