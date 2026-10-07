"""pe78 cycle 2: the simulator-trained reader on Proto-Elamite.

1. PE keys read on the whole corpus; share read as POISSON (no-biology model must lose).
2. Split-half consistency: P(LIVING) of each key on tablet half A vs half B (10 splits), Spearman; null = the
   same after values are permuted across keys within tablet.
3. Planted PE world: random half of the PE keys get simulated LIVING herd values on their real group structure,
   the rest ALLOC values; can the reader find which (20 reps)?
4. Pre-registered structural groups (fixed before reading, from earlier loops, no meanings): HERD run
   (pe20: M362, M367, M346, M006), GRAIN office (pe25: M010 M106 M002 M243 M075 M081 M265 M266 M296 M112),
   M288 (pe69 per-head allotment line).  Expectation if the reader worked: HERD -> LIVING, M288 -> ALLOC.
   Null: random key sets of the same size.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe78_common as C
from pe78_cycle1 import Reader, auc, real_sigs

HERD = {'M362', 'M367', 'M346', 'M006'}
GRAIN = {'M010', 'M106', 'M002', 'M243', 'M075', 'M081', 'M265', 'M266', 'M296', 'M112'}
MING = 4


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def read(R, E, rng):
    S = real_sigs(E, rng, MING)
    keys = sorted(S)
    if not keys:
        return {}, S
    P = R.post(np.array([C.vec(S[k]) for k in keys]))
    return {k: P[i] for i, k in enumerate(keys)}, S


def halves(R, E, seed):
    rng = np.random.default_rng(seed)
    tabs = sorted({e[0] for e in E})
    A = set(rng.choice(tabs, len(tabs) // 2, replace=False).tolist())
    PA, _ = read(R, [e for e in E if e[0] in A], rng)
    PB, _ = read(R, [e for e in E if e[0] not in A], rng)
    ks = sorted(set(PA) & set(PB))
    if len(ks) < 5:
        return float('nan'), len(ks)
    return spearman([PA[k][0] for k in ks], [PB[k][0] for k in ks]), len(ks)


def plant(E, seed):
    rng = np.random.default_rng(seed)
    G = C.groups(E)
    keys = [k for k, L in G.items() if len(L) >= MING]
    lab = {k: ('LIVING' if rng.random() < 0.5 else 'ALLOC') for k in keys}
    newvals = {}
    for k in keys:
        sizes = [len(g) for _, g in G[k]]
        vals, _ = C.SIMS[lab[k]](sizes, rng)
        for (t, g), v in zip(G[k], vals):
            newvals[(k, t)] = list(v)
    out = []
    for e in E:
        nv = newvals.get((e[1], e[0]))
        if nv:
            out.append((e[0], e[1], float(nv.pop())) + tuple(e[3:]))
        elif (e[1], e[0]) in newvals:
            continue
        else:
            out.append(e)
    return out, lab


if __name__ == '__main__':
    t0 = time.time()
    d = np.load(os.path.join(C.CK, 'sims_c1.npz'), allow_pickle=True)
    R = Reader(d['X'], d['y'])
    E = C.pe_entries()
    rows = []
    P, S = read(R, E, np.random.default_rng(3))
    keys = sorted(P)
    pois = np.mean([P[k].argmax() == 3 for k in keys])
    calls = {k: C.KINDS[int(P[k].argmax())] for k in keys}
    cnt = {kk: sum(v == kk for v in calls.values()) for kk in C.KINDS}
    print('PE keys', len(keys), cnt, 'POISSON share %.2f' % pois)
    for k in keys:
        print('  %-18s ng%3d b %.2f P %s' % (k, S[k]['ng'], S[k]['b'], np.round(P[k], 2)))
    # split halves
    real = [halves(R, E, s) for s in range(10)]
    shuf = [halves(R, C.shuffle_within_tablet(E, np.random.default_rng(500 + s)), s) for s in range(10)]
    rr = np.nanmean([r[0] for r in real]); rs = np.nanmean([r[0] for r in shuf])
    print('split-half rho real %.3f (keys %.1f) shuffled %.3f' % (rr, np.mean([r[1] for r in real]), rs))
    # planted
    pl = []
    for s in range(20):
        Ep, lab = plant(E, 900 + s)
        Pp, _ = read(R, Ep, np.random.default_rng(s))
        ks = [k for k in Pp if k in lab]
        pl.append(auc([Pp[k][0] for k in ks if lab[k] == 'LIVING'], [Pp[k][0] for k in ks if lab[k] == 'ALLOC']))
    pl = np.array(pl)
    print('planted PE world AUC %.3f +- %.3f' % (np.nanmean(pl), np.nanstd(pl)))
    # pre-registered groups vs random sets
    pl_ = np.array([P[k][0] for k in keys]); pa_ = np.array([P[k][1] for k in keys])
    rng = np.random.default_rng(11)
    res = {}
    for gname, gset, col in (('HERD', HERD, 0), ('GRAIN', GRAIN, 0), ('M288', {'M288'}, 1)):
        idx = [i for i, k in enumerate(keys) if k in gset]
        if not idx:
            res[gname] = None
            continue
        v = (pl_ if col == 0 else pa_)[idx].mean()
        null = np.array([(pl_ if col == 0 else pa_)[rng.choice(len(keys), len(idx), replace=False)].mean() for _ in range(5000)])
        p = float((null >= v).mean())
        res[gname] = dict(n=len(idx), keys=[keys[i] for i in idx], v=float(v), p=p)
        print(gname, res[gname])
    json.dump(dict(calls=calls, P={k: P[k].tolist() for k in keys}, S=S, real=real, shuf=shuf, planted=pl.tolist(),
                   groups=res), open(os.path.join(C.CK, 'c2_res.json'), 'w'))
    g = lambda n: ('%s: mean %s %.2f, p %.3f (n %d)' % (n, 'P(ALLOC)' if n == 'M288' else 'P(LIVING)', res[n]['v'], res[n]['p'], res[n]['n'])) if res.get(n) else '%s: absent' % n
    rows.append('| PE-78.2.1 | The reader on PE: %d keys with >= %d tablets carrying >= 2 entries; no-biology POISSON must lose | calls %s; POISSON share %.2f | %s |' % (
        len(keys), MING, cnt, pois, 'Poisson loses (every account scales like b ~ 2)' if pois < 0.1 else 'Poisson does not lose'))
    rows.append('| PE-78.2.2 | Split-half stability of P(LIVING) per key (10 random tablet halves); kill = values permuted across keys within tablet | Spearman real %.2f vs shuffled %.2f (keys per split %.1f) | %s |' % (
        rr, rs, np.mean([r[1] for r in real]), 'key-specific and stable' if rr - rs > 0.2 else 'not beyond the within-tablet shuffle: the reading reflects tablet value sets, not keys'))
    rows.append('| PE-78.2.3 | Planted PE world: random half of the PE keys rewritten with simulated LIVING herds (random biology) on their real group structure, the rest with ALLOC rate x heads; 20 reps | recovery AUC %.2f +- %.2f | %s |' % (
        np.nanmean(pl), np.nanstd(pl), 'power at PE size' if np.nanmean(pl) > 0.7 else 'little power at PE size'))
    rows.append('| PE-78.2.4 | Pre-registered structural groups vs 5,000 random key sets of the same size | %s; %s; %s | %s |' % (
        g('HERD'), g('GRAIN'), g('M288'), 'see cycle verdict'))
    open(os.path.join(C.CK, 'c2_rows.txt'), 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))
    print('done %.0fs' % (time.time() - t0))
