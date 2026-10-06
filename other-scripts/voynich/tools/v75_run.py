"""v75 driver: thousands of random model types, LOSO on replicate A, survivors re-tested on replicate B,
then the Voynich readout. Usage: python3 v75_run.py TAG N_MODELS [with_gen 1/0] [feature-exclude,comma]"""
import os, sys, pickle, random, json, time
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_lib as X
import v75_cls as C

TAG = sys.argv[1] if len(sys.argv) > 1 else 'c1'
NM = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
WG = (sys.argv[3] != '0') if len(sys.argv) > 3 else True
EXCL = set(sys.argv[4].split(',')) if len(sys.argv) > 4 and sys.argv[4] else set()
SYSEXCL = set(sys.argv[5].split(',')) if len(sys.argv) > 5 and sys.argv[5] else set()

G = {}


def init(rep):
    feats, rows, F = C.load(rep)
    keep = [i for i, r in enumerate(rows) if r['base'] not in SYSEXCL]
    G['feats'] = feats; G['rows'] = [rows[i] for i in keep]; G['F'] = F[keep]


def ev(m):
    L = C.loso(m, G['rows'], G['F'], with_gen=WG)
    return m, C.summarize(L)


def ev_perm(args):
    m, seed = args
    rows = G['rows']; rng = random.Random(seed)
    bases = sorted({r['base'] for r in rows if r['kind'] not in ('?', 'GEN')})
    kinds = [next(r['kind'] for r in rows if r['base'] == b and r['role'] == 'real') for b in bases]
    rng.shuffle(kinds); km = dict(zip(bases, kinds))
    R2 = [dict(r, kind=km[r['base']]) if r['kind'] not in ('?', 'GEN') else r for r in rows]
    L = C.loso(m, R2, G['F'], with_gen=WG)
    return C.summarize(L)['kind_acc']


if __name__ == '__main__':
    t0 = time.time()
    feats = C.load('A')[0]
    rng = random.Random(75)
    allowed = [f for f in feats if f not in EXCL]
    models = []
    while len(models) < NM:
        m = C.random_model(rng, feats)
        if any(feats[i] in EXCL or any(feats[i].endswith(':' + e) for e in EXCL) for i in m['sub']): continue
        models.append(m)
    with Pool(2, initializer=init, initargs=('A',)) as P:
        resA = P.map(ev, models, chunksize=20)
    print('A done %.0fs' % (time.time() - t0), flush=True)
    ka = np.array([s['kind_acc'] for _, s in resA]); gg = np.array([s['gen_as_gen'] for _, s in resA])
    thr = np.quantile(ka, 0.9)
    surv = [i for i in range(len(resA)) if ka[i] >= thr and (not WG or gg[i] >= 0.8)]
    if len(surv) < 20: surv = list(np.argsort(-ka)[:50])
    # null: best models re-run with kind labels permuted across systems
    top = list(np.argsort(-ka)[:20])
    with Pool(2, initializer=init, initargs=('A',)) as P:
        nul = P.map(ev_perm, [(models[i], 1000 + j) for i in top for j in range(5)])
    print('perm done %.0fs' % (time.time() - t0), flush=True)
    with Pool(2, initializer=init, initargs=('B',)) as P:
        resB = P.map(ev, [models[i] for i in surv], chunksize=5)
    print('B done %.0fs' % (time.time() - t0), flush=True)
    out = dict(tag=TAG, with_gen=WG, excl=sorted(EXCL), sysexcl=sorted(SYSEXCL), models=models,
               A=[s for _, s in resA], surv=surv, B=[s for _, s in resB], perm=nul, top=top)
    # readout
    RO = {}
    for rep in ('A', 'B'):
        feats_, rows, F = C.load(rep)
        keep = [i for i, r in enumerate(rows) if r['base'] not in SYSEXCL]
        rows = [rows[i] for i in keep]; F = F[keep]
        agg = defaultdict(Counter); rk = defaultdict(Counter)
        for i in surv:
            o, rank = C.readout(models[i], rows, F, with_gen=WG)
            for k, c in o.items(): agg[k].update(c)
            for k, lst in rank.items():
                for order in lst:
                    nonl = [c for c in order if c not in ('GEN', 'LANG')]
                    rk[k]['best_desig:' + nonl[0]] += 1
                    rk[k]['lang_before_desig'] += int(order.index('LANG') < order.index(nonl[0]))
        RO[rep] = dict(agg={str(k): dict(v) for k, v in agg.items()}, rank={str(k): dict(v) for k, v in rk.items()})
    out['readout'] = RO
    out['featfile'] = os.environ.get('V75_FEATS', 'feats')
    pickle.dump(out, open(os.path.join(X.CK, 'run_%s.pkl' % TAG), 'wb'))
    print('saved %.0fs' % (time.time() - t0), flush=True)
