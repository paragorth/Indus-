#!/usr/bin/env python3
"""PE-61 cycle 1 (arm A, all seven administrations): the la57 cycle-2 SURVIVOR specs per role (random trained
taggers that kept held-out AUC >= 0.6 on every held-out known system) are refitted on all eligible known systems
and applied to Proto-Elamite (value-free form, LA-size draws), its S1/S2/S3 shuffles and the la57 planted
administration.  NULL: the same specs refitted on role labels permuted inside each training corpus (NNULL reps).
Output: per role, per model, per corpus key: {sign: mean within-draw rank of the sign's occurrences}.
Usage: pe61_c1.py [NNULL] [tag] [roles]"""
import os, sys, json, time, pickle, warnings, collections
import numpy as np
from multiprocessing import Pool
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe61_common as P
warnings.filterwarnings('ignore')
L = P.L
NNULL = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TAG = sys.argv[2] if len(sys.argv) > 2 else 'c1'
ROLES = sys.argv[3].split(',') if len(sys.argv) > 3 else ['COM', 'PER', 'PLA', 'TRA', 'HDR', 'UNI']
OUT = os.path.join(P.CK, TAG); os.makedirs(OUT, exist_ok=True)
G = {}


def per_sign(blocks, m, sp):
    acc, cnt = collections.defaultdict(float), collections.Counter()
    for b in blocks:
        r = rankdata(P.score(m, sp, b['X'])) / len(b['X'])
        for t, v in zip(b['types'], r):
            acc[t] += v; cnt[t] += 1
    return {t: (acc[t] / cnt[t], cnt[t]) for t in acc}


def job(a):
    role, si, sp, rep = a
    el = G['elig'][role]
    m = P.fit(sp, [G['Xk'][k] for k in el], [G['Y'][role][k][rep] for k in el])
    if m is None:
        return role, si, rep, None
    res = {'REAL': per_sign(G['PE']['REAL'], m, sp)}
    if rep == 0:
        for key, bl in G['PE'].items():
            if key != 'REAL':
                res[key] = per_sign(bl, m, sp)
    pl = []
    for j in range(3):
        f = G['F'][('PLANT', j)]
        rows = [i for i, l in enumerate(f['labs']) if l is not None]
        if any(f['labs'][i] == role for i in rows):
            pl.append(L.auc(P.score(m, sp, f['X'][rows]), np.array([f['labs'][i] == role for i in rows])))
    res['PLANT'] = pl
    return role, si, rep, res


def init(g):
    G.update(g)


def main():
    t0 = time.time()
    F, sysd, Xk, elig = P.known()
    rng = np.random.default_rng(P.seed('pe61-' + TAG))
    Y = {}
    for role in ROLES:
        Y[role] = {k: L.label_sets(sysd[k], role, NNULL, rng)[1] for k in elig[role]}
    PE = pickle.load(open(os.path.join(P.CK, 'pe_feats.pkl'), 'rb'))
    tasks = []
    for role in ROLES:
        S = json.load(open(os.path.join(L.CK, 'c2_' + role, 'summary.json')))[role]
        for si, (f, fam, par) in enumerate(S['surv_specs']):
            for rep in range(NNULL + 1):
                tasks.append((role, si, (np.array(f), fam, par), rep))
    print('tasks', len(tasks), flush=True)
    g = dict(Xk=Xk, Y=Y, elig=elig, PE=PE, F={k: F[k] for k in F if k[0] == 'PLANT'})
    R = {}
    with Pool(2, initializer=init, initargs=(g,)) as pool:
        for n, (role, si, rep, res) in enumerate(pool.imap_unordered(job, tasks, chunksize=2)):
            R[(role, si, rep)] = res
            if n % 100 == 0:
                print('  %d/%d %.0fs' % (n, len(tasks), time.time() - t0), flush=True)
    pickle.dump(dict(R=R, elig=elig), open(os.path.join(OUT, 'votes.pkl'), 'wb'))
    print('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
