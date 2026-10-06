#!/usr/bin/env python3
"""PE-61 cycle 2 (arm B, PROTO-CUNEIFORM NEVER SEEN): fresh random trained taggers are selected by leave-one-system-
out over the known systems EXCEPT proto-cuneiform (survivor = held-out AUC >= THR on every non-PC held-out system),
refitted on all non-PC systems, then
  (i)  scored on proto-cuneiform (its conventional role labels; the hardest held-out control: PE's neighbour),
  (ii) scored on proto-cuneiform re-laid out in the PE order (numeral after the signs) = layout-transfer control,
  (iii) scored on the la57 planted administration,
  (iv) read out on Proto-Elamite and its S1/S2/S3 shuffles.
NULL: role labels permuted inside each training corpus (NNULL reps, same specs) for selection, PC and PE.
Usage: pe61_c2.py [NSPEC] [NNULL] [roles] [tag]"""
import os, sys, json, time, pickle, random, warnings, collections
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe61_common as P
from pe61_c1 import per_sign
warnings.filterwarnings('ignore')
L = P.L
NSPEC = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NNULL = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ROLES = sys.argv[3].split(',') if len(sys.argv) > 3 else ['COM', 'TRA']
TAG = sys.argv[4] if len(sys.argv) > 4 else 'c2'
THR = 0.6
OUT = os.path.join(P.CK, TAG); os.makedirs(OUT, exist_ok=True)
G = {}


def sel_job(a):
    role, si, sp, rep = a
    el = G['el'][role]
    res = {}
    for h in el:
        tr = [k for k in el if k != h]
        m = P.fit(sp, [G['Xk'][k] for k in tr], [G['Y'][role][k][rep] for k in tr])
        res[h] = 0.5 if m is None else L.auc(P.score(m, sp, G['Xk'][h]), G['Y'][role][h][0])
    return role, si, rep, res


def read_job(a):
    role, si, sp, rep = a
    el = G['el'][role]
    m = P.fit(sp, [G['Xk'][k] for k in el], [G['Y'][role][k][rep] for k in el])
    if m is None:
        return role, si, rep, None
    res = {'REAL': per_sign(G['PE']['REAL'], m, sp)}
    for name in ('PC', 'PCPE'):
        res[name] = [L.auc(P.score(m, sp, b['X']), b['y']) for b in G[name]]
    pl = []
    for j in range(3):
        f = G['F'][('PLANT', j)]
        rows = [i for i, l in enumerate(f['labs']) if l is not None]
        if any(f['labs'][i] == role for i in rows):
            pl.append(L.auc(P.score(m, sp, f['X'][rows]), np.array([f['labs'][i] == role for i in rows])))
    res['PLANT'] = pl
    if rep == 0:
        for key, bl in G['PE'].items():
            if key != 'REAL':
                res[key] = per_sign(bl, m, sp)
    return role, si, rep, res


def init(g):
    G.update(g)


def pc_blocks(role, layout):
    """4 LA-size draws of proto-cuneiform (la57 draw seeds) in its own or the PE layout, labelled rows only."""
    docs = P.pc_pe_layout() if layout else L.pc_docs()
    out = []
    for j in range(4):
        d = L.draw(docs, P.LA_SIZE, random.Random(L.seed('la57-draw-PC-%d' % j)))
        X, types, labs, _ = L.features(d, L.pc_truth(d))
        rows = [i for i, l in enumerate(labs) if l is not None]
        out.append(dict(X=X[rows], y=np.array([labs[i] == role for i in rows])))
    return out


def main():
    t0 = time.time()
    F, sysd, Xk, elig = P.known()
    rng = np.random.default_rng(P.seed('pe61-' + TAG))
    el = {r: [k for k in elig[r] if k != 'PC'] for r in ROLES}
    Y = {r: {k: L.label_sets(sysd[k], r, NNULL, rng)[1] for k in el[r]} for r in ROLES}
    specs = {r: [P.spec(rng) for _ in range(NSPEC)] for r in ROLES}
    g = dict(Xk=Xk, Y=Y, el=el)
    S = {}
    tasks = [(r, si, specs[r][si], rep) for r in ROLES for si in range(NSPEC) for rep in range(NNULL + 1)]
    print('selection tasks', len(tasks), el, flush=True)
    with Pool(2, initializer=init, initargs=(g,)) as pool:
        for n, (r, si, rep, res) in enumerate(pool.imap_unordered(sel_job, tasks, chunksize=4)):
            S[(r, si, rep)] = res
            if n % 200 == 0:
                print('  sel %d/%d %.0fs' % (n, len(tasks), time.time() - t0), flush=True)
    surv, summ = {}, {}
    for r in ROLES:
        sv = lambda rep: [si for si in range(NSPEC) if all(S[(r, si, rep)][h] >= THR for h in el[r])]
        surv[r] = sv(0)
        summ[r] = dict(el=el[r], n_surv=len(surv[r]), null=[len(sv(rep)) for rep in range(1, NNULL + 1)])
        print(r, 'non-PC survivors', summ[r], flush=True)
    json.dump(summ, open(os.path.join(OUT, 'selection.json'), 'w'))
    PE = pickle.load(open(os.path.join(P.CK, 'pe_feats.pkl'), 'rb'))
    R = {}
    for r in ROLES:
        g2 = dict(g, PE=PE, F={k: F[k] for k in F if k[0] == 'PLANT'}, PC=pc_blocks(r, False), PCPE=pc_blocks(r, True))
        print(r, 'PC rows', [int(b['y'].sum()) for b in g2['PC']], flush=True)
        tasks = [(r, si, specs[r][si], rep) for si in surv[r] for rep in range(NNULL + 1)]
        with Pool(2, initializer=init, initargs=(g2,)) as pool:
            for r_, si, rep, res in pool.imap_unordered(read_job, tasks, chunksize=1):
                R[(r_, si, rep)] = res
        print(r, 'readout done %.0fs' % (time.time() - t0), flush=True)
    pickle.dump(dict(R=R, elig=el, selection=summ), open(os.path.join(OUT, 'votes.pkl'), 'wb'))
    print('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
