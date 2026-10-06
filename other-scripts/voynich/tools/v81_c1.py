"""v81 cycle 1: 3,000 random onset definitions x 14 corpora, selection stage (leaf half 0, page fold 0 -> fold 1).
Checkpoint: data/v81_ckpt/c1_sel_<chunk>.pkl"""
import os, sys, random, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v81_lib as L
import warnings; warnings.filterwarnings('ignore')
from multiprocessing import Pool

NDEF = int(os.environ.get('NDEF', 3000)); CH = 100
CC = None


def init():
    global CC
    CC = L.pload('comp.pkl')


def work(ci):
    fn = os.path.join(L.CK, 'c1_sel_%03d.pkl' % ci)
    if os.path.exists(fn): return ci
    rng = random.Random(8100 + ci)
    out = []
    for j in range(CH):
        d = L.random_def(rng)
        r = {}
        for k, c in CC.items():
            o = L.onset(c, d); tr, te = L.masks(c, 'sel'); r[k] = L.stream_feats(c, o, tr, te)
        out.append((d, r))
    pickle.dump(out, open(fn, 'wb'))
    return ci


if __name__ == '__main__':
    if L.pload('comp.pkl') is None:
        C = L.pload('corpora.pkl') or L.build()
        L.psave('comp.pkl', {k: L.Comp(v) for k, v in C.items()})
    t = time.time()
    with Pool(2, initializer=init) as P:
        for ci in P.imap_unordered(work, range(NDEF // CH)):
            print('chunk', ci, round(time.time() - t), flush=True)
