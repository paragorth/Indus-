#!/usr/bin/env python3
"""PE-61 cycle 3: A PARLIAMENT OF BUREAUCRACIES VOTES ON PROTO-ELAMITE.
Every known administration trains its OWN experts (NSPEC random specs fitted on that system alone, la57_c3 style);
Linear A sits as an extra voter for COMMODITY in its structural form (logogram tokens vs syllabic words, la57
LA_ADM), and the khipus vote on TOTAL (top cords).  Each voter scores Proto-Elamite signs; a NULL voter set is
trained on labels permuted inside its own corpus.  Per role:
  - each voter's agreement with the pe59 frozen classes, real vs null voter vs PE shuffles (S1/S2/S3);
  - inter-voter agreement on PE (mean Spearman across voters over PE signs) vs shuffles;
  - parliament verdict: signs put in the top decile by a majority of voters, with proto-cuneiform's vote shown
    separately (leakage gauge) and the majority recomputed without it.
Controls: every voter on the la57 planted administration; non-PC voters on proto-cuneiform re-laid out in PE order.
Usage: pe61_c3.py [NSPEC] [NNULL] [tag]"""
import os, sys, json, time, pickle, random, warnings, collections
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe61_common as P
from pe61_c1 import per_sign
from pe61_c2 import pc_blocks
warnings.filterwarnings('ignore')
L = P.L
NSPEC = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NNULL = int(sys.argv[2]) if len(sys.argv) > 2 else 2
TAG = sys.argv[3] if len(sys.argv) > 3 else 'c3'
ROLES = ['COM', 'PER', 'PLA', 'TRA', 'TOT', 'HDR', 'UNI']
SHK = ['S%d_%d' % (s, j) for s in (1, 2, 3) for j in range(6)]
OUT = os.path.join(P.CK, TAG); os.makedirs(OUT, exist_ok=True)
G = {}


def job(a):
    role, src, rep = a
    X, y = G['X'][src], G['Y'][(role, src)][rep]
    rng = np.random.default_rng(P.seed('pe61c3-%s-%s' % (role, src)))   # same specs for real and null
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    cnts = {}
    pl, pc = [], []
    for _ in range(NSPEC):
        sp = P.spec(rng)
        m = P.make(sp)
        m.fit(X[:, sp[0]], y)
        if not hasattr(m, 'predict_proba') or len(getattr(m, 'classes_', [0, 1])) < 2:
            continue
        keys = ['REAL'] + (SHK if rep == 0 else [])
        for k in keys:
            ps = per_sign(G['PE'][k], m, sp)
            for t, (v, c) in ps.items():
                acc[k][t] += v / NSPEC; cnts.setdefault(k, {})[t] = c
        for j in range(3):
            f = G['F'][('PLANT', j)]
            rows = [i for i, l in enumerate(f['labs']) if l is not None]
            if any(f['labs'][i] == role for i in rows):
                pl.append(L.auc(P.score(m, sp, f['X'][rows]), np.array([f['labs'][i] == role for i in rows])))
        for b in G['PCPE'].get(role, []):
            pc.append(L.auc(P.score(m, sp, b['X']), b['y']))
    return role, src, rep, dict(tabs={k: {t: (v, cnts[k][t]) for t, v in d.items()} for k, d in acc.items()},
                                plant=pl, pcpe=pc)


def init(g):
    G.update(g)


def main():
    t0 = time.time()
    F, sysd, Xk, elig = P.known()
    rng = np.random.default_rng(P.seed('pe61-' + TAG))
    PE = pickle.load(open(os.path.join(P.CK, 'pe_feats.pkl'), 'rb'))
    g = dict(X=dict(Xk), Y={}, PE={k: PE[k] for k in ['REAL'] + SHK}, F={k: F[k] for k in F if k[0] == 'PLANT'}, PCPE={})
    voters = {}
    for role in ROLES:
        voters[role] = list(elig[role])
        for k in elig[role]:
            g['Y'][(role, k)] = L.label_sets(sysd[k], role, NNULL, rng)[1]
        if 'PC' in elig[role]:
            g['PCPE'][role] = pc_blocks(role, True)
    # Linear A as a COM voter in its structural form: logogram tokens vs syllabic words (admin documents)
    la = F[('LA_ADM', 0)]
    g['X']['LA'] = la['X']
    y = np.array([t.startswith('L:') for t in la['types']])
    g['Y'][('COM', 'LA')] = np.array([y] + [rng.permutation(y) for _ in range(NNULL)])
    voters['COM'].append('LA')
    if 'KH' not in voters['TOT']:
        voters['TOT'].append('KH')
    tasks = [(r, s, rep) for r in ROLES for s in voters[r] for rep in range(NNULL + 1)]
    print('tasks', len(tasks), voters, flush=True)
    V = {}
    with Pool(2, initializer=init, initargs=(g,)) as pool:
        for n, (r, s, rep, res) in enumerate(pool.imap_unordered(job, tasks)):
            V[(r, s, rep)] = res
            print('  %d/%d %s %s %d %.0fs' % (n, len(tasks), r, s, rep, time.time() - t0), flush=True)
    pickle.dump(dict(V=V, voters=voters), open(os.path.join(OUT, 'votes.pkl'), 'wb'))
    print('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
