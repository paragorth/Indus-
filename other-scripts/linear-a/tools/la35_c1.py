"""LA-35 cycle 1: does the ending of a word depend on the number after it, within stem families?
Stats per (definition x class scheme): G_cond (any ending-number dependence within families) and
CMH (the same ending tracks the same number class across >= 2 stems = an agreement morpheme).
Nulls: N1 numbers shuffled within (tablet, commodity); N2 within (site, commodity).
Correction: max-z over all definitions x schemes x stats (z from the null of the same family).
Controls: LB at LA size (subsamples by tablet); planted rule in LA; LA with shuffled numbers."""
import json, sys, random, time
import numpy as np
from la35_common import *

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 400
DEFS = definitions(30)


def run(E, nperm, seed, nulls=('N1', 'N2')):
    rng = np.random.default_rng(seed)
    designs = [(d[0], Design(E, d[1], d[2], d[3])) for d in DEFS]
    vals = np.array([e['v'] for e in E])
    def allstats(v):
        out = []
        for sk, f in SCHEMES.items():
            K = 3 if sk == 'K3' else 2
            cls = np.array([f(x) for x in v])
            for nm, Dz in designs:
                g, c, n = Dz.stats(cls, K)
                out.append((g, c))
        return np.array(out)  # (ncell, 2)
    real = allstats(vals)
    res = {'n_entries': len(E), 'cells': [(sk, nm) for sk in SCHEMES for nm, _ in designs]}
    for nl in nulls:
        key = (lambda e: (e['doc'], e['com'])) if nl == 'N1' else (lambda e: (e['site'], e['com']))
        groups = strata(E, key)
        sims = np.array([allstats(permute_vals(vals, groups, rng)) for _ in range(nperm)])
        mu, sd = sims.mean(0), sims.std(0) + 1e-9
        zr = (real - mu) / sd
        zs = (sims - mu) / sd
        # family-wise: max z over cells for each stat
        out = {}
        for k, st in enumerate(('G', 'CMH')):
            mx = zs[:, :, k].max(1)
            pcell = ((sims[:, :, k] >= real[None, :, k]).sum(0) + 1) / (nperm + 1)
            best = int(np.argmax(zr[:, k]))
            out[st] = dict(maxz=float(zr[:, k].max()), p_fw=float(((mx >= zr[:, k].max()).sum() + 1) / (nperm + 1)),
                           best=res['cells'][best], p_best=float(pcell[best]),
                           n_cells_p05=int((pcell < 0.05).sum()), n_cells=len(pcell),
                           main=[float(zr[i, k]) for i, c in enumerate(res['cells']) if c[1] in ('pre2_E1', 'pre2_R', 'pre1_E1')])
        res[nl] = out
    return res


def plant(E, p, seed):
    """With prob p, each entry whose word has >= 2 signs gets a final sign X if 1 else Y."""
    rng = random.Random(seed)
    fin = Counter(e['word'][-1] for e in E if len(e['word']) >= 2)
    X, Y = rng.sample([s for s, _ in fin.most_common(15)], 2)
    out = []
    for e in E:
        e = dict(e)
        if len(e['word']) >= 2 and rng.random() < p:
            e['word'] = e['word'][:-1] + ((X,) if e['v'] == 1 else (Y,))
        out.append(e)
    return out, (X, Y)


def lb_sub(LB, n_target, seed):
    rng = random.Random(seed)
    docs = sorted({e['doc'] for e in LB}); rng.shuffle(docs)
    by = defaultdict(list)
    for e in LB: by[e['doc']].append(e)
    out = []
    for d in docs:
        if len(out) >= n_target: break
        out += by[d]
    return out


if __name__ == '__main__':
    t0 = time.time()
    LA = la_entries(); LB = lb_entries()
    rep = {}
    rep['LA'] = run(LA, NP, 1)
    print('LA', json.dumps(rep['LA']['N1']), json.dumps(rep['LA']['N2']), flush=True)
    rng = np.random.default_rng(7)
    # shuffled-number LA (numbers permuted across the whole corpus): a negative control
    sh = [dict(e) for e in LA]; vv = [e['v'] for e in sh]; rng.shuffle(vv)
    for e, v in zip(sh, vv): e['v'] = v
    rep['LA_shuf'] = run(sh, NP // 2, 2)
    print('LAshuf', json.dumps(rep['LA_shuf']['N1']), flush=True)
    rep['plant'] = []
    for p in (0.15, 0.3):
        for s in range(3):
            P, xy = plant(LA, p, 100 + s)
            r = run(P, NP // 2, 3 + s, nulls=('N1',)); r['p'] = p; r['xy'] = xy
            rep['plant'].append(r)
            print('plant', p, xy, r['N1']['CMH']['maxz'], r['N1']['CMH']['p_fw'], r['N1']['G']['p_fw'], flush=True)
    rep['LB'] = []
    for s in range(6):
        S = lb_sub(LB, len(LA), s)
        r = run(S, NP // 2, 20 + s); rep['LB'].append(r)
        print('LBsub', s, len(S), json.dumps(r['N1']['CMH']), json.dumps(r['N2']['CMH']), r['N1']['G']['p_fw'], r['N2']['G']['p_fw'], flush=True)
    rep['LB_full'] = run(LB, NP // 2, 40)
    print('LBfull', json.dumps(rep['LB_full']['N1']), json.dumps(rep['LB_full']['N2']), flush=True)
    json.dump(rep, open(os.path.join(CK, 'c1%s.json' % ('_notot' if EXCL_TOT else '')), 'w'), indent=1)
    print('done %.0fs' % (time.time() - t0))
