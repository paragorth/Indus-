"""v38 cycle 4: visual twins. If text names or describes the plant only when two drawings show
the SAME kind of plant, the signal sits in the tail of the visual-similarity distribution, not in
the Mantel average. Statistic: mean confound-residualised text similarity over (a) each page's
visual nearest neighbour (pages >= 3 binding positions apart), (b) the top 1%, 2%, 5% most
visually similar pairs. Null: page permutation of the visual matrix within strata (2000).
Run on Voynich, on the Gerard control and on a planted Voynich text (10%).
Writes data/v38_ckpt/c4.json and loops/v38_cycle4.txt.
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *

NPERM = int(os.environ.get('NPERM', 2000))


def tail_stats(Vm, Tres_mat, order, qs=(0.01, 0.02, 0.05), mingap=3):
    n = Vm.shape[0]
    far = np.abs(order[:, None] - order[None, :]) >= mingap
    V = np.where(far, Vm, -np.inf)
    np.fill_diagonal(V, -np.inf)
    nn = V.argmax(1)
    s_nn = float(np.mean(Tres_mat[np.arange(n), nn]))
    iu = np.triu_indices(n, 1)
    v = V[iu]; t = Tres_mat[iu]
    ok = np.isfinite(v)
    v, t = v[ok], t[ok]
    o = np.argsort(-v)
    out = [s_nn] + [float(t[o[:max(3, int(q * len(v)))]].mean()) for q in qs]
    return np.array(out)


def run(V, T, conf, order, strata, rng, nperm):
    n = len(order)
    part = Partial(list(conf.values()), n)
    res = {}
    for m, Tm in T.items():
        r = zs(part.res(upper(Tm)))
        R = np.zeros((n, n)); R[np.triu_indices(n, 1)] = r; R = R + R.T
        res[m] = R
    out = {}
    for f, Vm in V.items():
        a = zs(part.res(upper(Vm)))
        A = np.zeros((n, n)); A[np.triu_indices(n, 1)] = a; A = A + A.T
        for m, R in res.items():
            obs = tail_stats(A, R, order)
            null = np.array([tail_stats(A[np.ix_(p, p)], R, order) for p in (perm(n, rng, strata) for _ in range(nperm))])
            z = (obs - null.mean(0)) / null.std(0)
            pv = (1 + (null >= obs).sum(0)) / (1 + nperm)
            out['%s|%s' % (f, m)] = dict(obs=obs.tolist(), z=z.tolist(), p=pv.tolist())
    Z = np.array([v['z'] for v in out.values()])
    return out, Z.mean(0).tolist()


if __name__ == '__main__':
    rng = np.random.default_rng(3839)
    from v38_cycle1 import voynich_setup
    pages, keys, words, vis, conf, strata = voynich_setup()
    order = np.array([p['order'] for p in pages], float)
    fams = ['dino', 'r18', 'shape', 'colour', 'hog', 'edge']
    mets = ['tfidf', 'rare', 'tri']
    V = vis_sims(vis, keys, fams)
    T = {m: M for m, M in text_sims(text_profiles(words)).items() if m in mets}
    rows, store = [], {}
    lab = ['NN', 'top1%', 'top2%', 'top5%']
    vo, vz = run(V, T, conf, order, strata, rng, NPERM)
    store['voynich'] = vo
    best = sorted(vo.items(), key=lambda kv: -max(kv[1]['z']))[:4]
    rows.append(('V-38.13', 'TWINS, Voynich: residual text similarity of each page\'s visual nearest neighbour (>= 3 pages apart) and of the top 1/2/5%% most similar drawing pairs; 6 families x 3 text metrics; within-stratum permutation (%d)' % NPERM,
                 'mean z over 18 cells: ' + ', '.join('%s %+.2f' % (l, z) for l, z in zip(lab, vz)) + '; strongest cells: ' +
                 '; '.join('%s %s' % (k, '/'.join('%+.1f' % x for x in v['z'])) for k, v in best),
                 ''))
    # planted
    allw = Counter(w for ws in words for w in ws)
    pool = [w for w, c in allw.items() if 5 <= c <= 40]
    Z = np.array([vis[k]['dino'] for k in keys], float)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pw = plant_text(words, U[:, :10] * S[:10], 0.10, rng, pool)
    Tp = {m: M for m, M in text_sims(text_profiles(pw)).items() if m in mets}
    po, pz = run(V, Tp, conf, order, strata, rng, NPERM // 4)
    store['planted10'] = po
    rows.append(('V-38.14', 'TWINS, PLANTED control (10% of tokens from a DINO-driven vocabulary)', 'mean z over 18 cells: ' + ', '.join('%s %+.2f' % (l, z) for l, z in zip(lab, pz)) +
                 '; dino cells: ' + '; '.join('%s %s' % (k, '/'.join('%+.1f' % x for x in v['z'])) for k, v in po.items() if k.startswith('dino')), ''))
    # Gerard
    from v38_cycle2 import gerard_setup
    gk, gw, gvis, gconf = gerard_setup()
    gorder = np.array([int(k) for k in gk], float) / 8.
    GV = vis_sims(gvis, gk, [f for f in fams if f != 'colour'])
    GT = {m: M for m, M in text_sims(text_profiles(gw)).items() if m in mets}
    go, gz = run(GV, GT, gconf, gorder, None, rng, NPERM // 2)
    store['gerard'] = go
    best = sorted(go.items(), key=lambda kv: -max(kv[1]['z']))[:4]
    rows.append(('V-38.15', 'TWINS, Gerard control (%d pages; 5 families x 3 metrics)' % len(gk), 'mean z: ' + ', '.join('%s %+.2f' % (l, z) for l, z in zip(lab, gz)) + '; strongest: ' +
                 '; '.join('%s %s' % (k, '/'.join('%+.1f' % x for x in v['z'])) for k, v in best), ''))
    json.dump(dict(store=store, vz=vz, pz=pz, gz=gz), open(os.path.join(CK, 'c4.json'), 'w'))
    with open(os.path.join(LOOPS, 'v38_cycle4.txt'), 'w') as fh:
        fh.write('# v38 cycle 4 - visual twins (tail test)\n| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            fh.write('| %s | %s | %s | %s |\n' % r)
    print(open(os.path.join(LOOPS, 'v38_cycle4.txt')).read())
