#!/usr/bin/env python3
"""LA-14 runner. A task = (corpus, forger, seed, role, specs); each spec = (feature groups, classifier).
role 'real': half-B real documents vs forgeries trained on half A.
role 'null': two independent forgery draws from the same forger (should give AUC 0.5).
Results are appended to data/la14/<tag>.jsonl (checkpoint: finished task keys are skipped).
Usage: python3 la14_run.py <tag> <plan>   (plans defined in PLANS below)
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '1'
import json, random, sys, time, warnings
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings('ignore')
import la14_common as C

FDIR = os.path.join(C.OUT, 'forg'); os.makedirs(FDIR, exist_ok=True)


def get_forg(cname, fname, seed, A, n, salt=0):
    p = os.path.join(FDIR, '%s_%s_%d_%d.json' % (cname, fname, seed, salt))
    if os.path.exists(p): return json.load(open(p))
    f, g = C.forge(fname, A, n, seed + 10000 * salt)
    json.dump([f, g], open(p, 'w'))
    return f, g


def task(args):
    cname, fname, seed, role, specs = args
    t0 = time.time()
    docs = C.corpus(cname)
    A, B = C.split(docs, seed)
    fake, fg = get_forg(cname, fname, seed, A, len(B))
    if role == 'real':
        real = C.real_for(fname, B); rg = ['r%d' % i for i in range(len(real))]
    else:
        real, rg = get_forg(cname, fname, seed, A, len(B), salt=1)
        rg = ['x' + g for g in rg]
        if fname in C.CopyEdit.TYPES:  # copies of the same sources: keep a source in one fold
            rg = [g[1:] for g in rg]
    groups = rg + list(fg)
    X, y, names = C.build_X(real, fake, C.ALLG)
    pref = np.array([n.split(':')[0] for n in names])
    out = []
    for gs, clf in specs:
        cols = np.where(np.isin(pref, gs))[0]
        if len(cols) == 0: continue
        Xs = X[:, cols]
        want = clf == 'LR' and len(gs) == 1
        r = C.cv_auc(Xs, y, clf, seed, coefs=want, groups=groups)
        rec = {'c': cname, 'f': fname, 's': seed, 'role': role, 'g': list(gs), 'clf': clf, 'nreal': len(real),
               'nfake': len(fake)}
        if want:
            auc, co = r; rec['auc'] = auc
            if co is not None:
                o = np.argsort(co); nm = [names[i] for i in cols]
                rec['neg'] = [(nm[i], round(float(co[i]), 3)) for i in o[:12]]
                rec['pos'] = [(nm[i], round(float(co[i]), 3)) for i in o[::-1][:12]]
        else: rec['auc'] = r
        out.append(rec)
    return (cname, fname, seed, role), out, time.time() - t0


SINGLE = [[g] for g in C.ALLG]


def plan_c1():
    """Cycle 1: every forger on every corpus, each single feature group and all groups, LR, 8 seeds."""
    specs = [(g, 'LR') for g in SINGLE] + [(C.ALLG, 'LR')]
    T = []
    for c in ['LA', 'PLA', 'LB', 'FW_MK2', 'FW_FLAT', 'FW_NEUR']:
        for f in C.GEN_FORGERS + C.CopyEdit.TYPES:
            for s in range(6):
                T.append((c, f, s, 'real', specs))
                if f in C.GEN_FORGERS: T.append((c, f, s, 'null', specs))
    return T


def plan_c2():
    """Cycle 2: thousands of random pairings (random group subsets x classifier x seed)."""
    rng = random.Random(14)
    T = []
    forgers = ['MK1', 'MK2', 'MK3', 'WMK2', 'FLAT', 'NEUR', 'COPY0', 'E_num', 'E_word', 'E_splice', 'E_numshuf',
               'E_wordshuf', 'E_swapline', 'E_first', 'E_last']
    for c in ['LA', 'LB', 'PLA', 'FW_FLAT', 'FW_MK2', 'FW_NEUR']:
        for f in forgers:
            for s in range(8, 20):
                specs = []
                for _ in range(12):
                    k = rng.choice([1, 2, 3, 4, 6])
                    specs.append((sorted(rng.sample(C.ALLG, k)), rng.choice(['LR'] * 5 + ['LR10'] * 2 + ['RF'] * 3 + ['MLP', 'HGB'])))
                T.append((c, f, s, 'real', specs))
    return T


PLANS = {'c1': plan_c1, 'c2': plan_c2}

if __name__ == '__main__':
    tag, plan = sys.argv[1], sys.argv[2]
    path = os.path.join(C.OUT, tag + '.jsonl'); donep = os.path.join(C.OUT, tag + '.done')
    done = set(open(donep).read().split('\n')) if os.path.exists(donep) else set()
    T = [t for t in PLANS[plan]() if '|'.join(map(str, t[:4])) not in done]
    # world corpora must exist before forking
    for c in sorted({t[0] for t in T}): C.corpus(c)
    print('tasks', len(T), flush=True)
    t0 = time.time()
    with Pool(2) as P, open(path, 'a') as fo, open(donep, 'a') as fd:
        for i, (key, recs, dt) in enumerate(P.imap_unordered(task, T, chunksize=1)):
            for r in recs: fo.write(json.dumps(r) + '\n')
            fo.flush(); fd.write('|'.join(map(str, key)) + '\n'); fd.flush()
            if i % 20 == 0: print(i, key, round(dt, 1), round(time.time() - t0), flush=True)
