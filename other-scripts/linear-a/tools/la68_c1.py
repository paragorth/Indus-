#!/usr/bin/env python3
"""la68 cycle 1: random abbreviation-rule search, sealing single signs -> tablet word counts.

For LA and LB (calibration): score 165 named + NR random rules on all units; nulls
  N1 receipt documents shuffled across sites (counts per site kept)
  N2 tablet words replaced by frequency-matched random word types (pool: all word types of the script)
Statistics: max over rules (family-wise), held-out (best rule on fold A scored on fold B, both ways).
Planted control: hidden rule writes a fraction f of each LA site's sealing tokens (clumped like the real ones).
"""
import sys, json, os, collections, math, time, copy
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la68_common as C

NR = int(os.environ.get('NR', 3000))
NNULL = int(os.environ.get('NNULL', 150))
SEED = int(os.environ.get('SEED', 68))

CFG = {
    'LA': dict(load=C.load_la, units=['Haghia Triada', 'Khania', 'Knossos', 'Phaistos'],
               foldA=['Haghia Triada'], foldB=['Khania', 'Knossos', 'Phaistos']),
    'LB': dict(load=C.load_lb, units=['KN', 'PY', 'TH', 'MY'], foldA=['KN'], foldB=['PY', 'TH', 'MY']),
}


def make_rules(seed):
    rng = np.random.default_rng(seed)
    R = list(C.named_rules().items())
    for i in range(NR):
        r = C.random_rule(rng)
        R.append((C.rule_name(r), r))
    return R


def sweep(E, R, abbrs=None):
    """scores [nrules, nunits]"""
    out = np.zeros((len(R), len(E.names)))
    for i, (_, r) in enumerate(R):
        for j, s in enumerate(E.names):
            t = E.predicted(s, r)
            out[i, j] = E.score_unit(s, t, None if abbrs is None else abbrs[s])
    return out


def stats(M, names, cfg):
    ia = [names.index(s) for s in cfg['foldA']]
    ib = [names.index(s) for s in cfg['foldB']]
    allsc = M.mean(1)
    A = M[:, ia].mean(1)
    B = M[:, ib].mean(1)
    return dict(max_all=float(allsc.max()), best=int(allsc.argmax()),
                ho_AB=float(B[A.argmax()]), ho_BA=float(A[B.argmax()]),
                bestA=int(A.argmax()), bestB=int(B.argmax()))


def null_job(args):
    script, kind, k, seed = args
    cfg = CFG[script]
    U = cfg['load']()
    rng = np.random.default_rng(seed * 1000 + k + (0 if kind == 'N1' else 500000))
    names = cfg['units']
    if kind == 'N1':
        docs = [(s, d) for s in names for d in U[s]['abbr_docs']]
        labels = [s for s, _ in docs]
        rng.shuffle(labels)
        for s in names:
            U[s]['abbr'] = collections.Counter()
        for (_, d), s in zip(docs, labels):
            U[s]['abbr'].update(d)
    elif kind == 'N3':
        for s in names:
            for w in U[s]['words']:
                sg = list(w['signs'])
                rng.shuffle(sg)
                w['signs'] = tuple(sg)
    else:
        pool = collections.Counter()
        for s in U:
            for w in U[s]['words']:
                pool[w['signs']] += 1
        allw = all_word_types(script)
        band = lambda f: 0 if f <= 1 else 1 if f == 2 else 2 if f <= 4 else 3 if f <= 8 else 4
        byband = collections.defaultdict(list)
        for w, f in allw.items():
            byband[band(f)].append(w)
        for s in names:
            types = sorted({w['signs'] for w in U[s]['words']})
            mp = {}
            for t in types:
                cand = byband[band(allw.get(t, 1))]
                mp[t] = cand[rng.integers(len(cand))]
            for w in U[s]['words']:
                w['signs'] = mp[w['signs']]
    E = C.Engine(U, names)
    R = make_rules(seed)
    M = sweep(E, R)
    st = stats(M, names, cfg)
    st['named'] = {nm: float(M[i].mean()) for i, (nm, _) in enumerate(R) if '|' in nm and nm.count('/') == 0}
    return (script, kind, k, st)


_AW = {}


def all_word_types(script):
    if script in _AW:
        return _AW[script]
    c = collections.Counter()
    if script == 'LA':
        d = json.load(open(os.path.join(C.DATA, 'corpus.json')))
        for r in d:
            for t in r['tokens']:
                if t['t'] == 'word' and len(t['s']) >= 2:
                    c[tuple(t['s'])] += 1
    else:
        U = C.load_lb()
        for s in U:
            for w in U[s]['words']:
                c[w['signs']] += 1
    _AW[script] = c
    return c


def canon(o, l):
    o = int(o)
    if o < 6:
        return min(o, l - 1)
    if o == 6:
        return l - 1
    if o == 7:
        return l - 2
    return C.OPT_NAMES[o]


def plant_job(args):
    f, k, seed, R_sel = args
    cfg = CFG['LA']
    U = cfg['load']()
    names = cfg['units']
    rng = np.random.default_rng(seed * 7919 + k + int(f * 1000))
    E0 = C.Engine(U, names)
    Rall = make_rules(seed)
    # hidden rule: half of plants a named uniform-position rule, half a random rule
    if k % 2 == 0:
        o = [0, 6, 1, 8, 7][(k // 2) % 5]
        pm = np.full(C.LMAX + 1, o, dtype=np.int32)
        hidden = (pm, C.FILTERS[rng.integers(5)], C.WEIGHTS[rng.integers(3)])
    else:
        hidden = C.random_rule(rng)
    abbrs = {}
    for s in names:
        u = U[s]
        N = int(sum(u['abbr'].values()))
        ntypes = len(u['abbr'])
        D = max(3, 2 * ntypes)
        t = E0.predicted(s, hidden)
        pt = t / t.sum()
        realpool = [x for x, c in u['abbr'].items() for _ in range(c)]
        reps = np.array(sorted(u['abbr'].values()))
        draws = []
        for _ in range(D):
            if rng.random() < f:
                draws.append(E0.signs[rng.choice(E0.V, p=pt)])
            else:
                draws.append(realpool[rng.integers(len(realpool))])
        w = rng.choice(reps, size=D).astype(float)
        w = np.maximum(1, np.round(w / w.sum() * N)).astype(int)
        a = np.zeros(E0.V)
        for x, c in zip(draws, w):
            a[E0.idx[x]] += c
        abbrs[s] = a
    M = sweep(E0, Rall, abbrs)
    st = stats(M, names, cfg)
    best = Rall[st['best']][1]
    fam_h = [canon(hidden[0][l], l) for l in (2, 3, 4)]
    fam_b = [canon(best[0][l], l) for l in (2, 3, 4)]
    match = sum(a == b for a, b in zip(fam_h, fam_b))
    return dict(f=f, k=k, hidden=C.rule_name(hidden), best_rule=C.rule_name(best), match_len234=match, **st)


def main():
    t0 = time.time()
    res = {}
    for script in ('LA', 'LB'):
        cfg = CFG[script]
        U = cfg['load']()
        E = C.Engine(U, cfg['units'])
        R = make_rules(SEED)
        M = sweep(E, R)
        st = stats(M, cfg['units'], cfg)
        order = np.argsort(-M.mean(1))
        top = [(R[i][0], round(float(M[i].mean()), 4), [round(float(x), 3) for x in M[i]]) for i in order[:15]]
        named = {nm: [round(float(M[i].mean()), 4)] + [round(float(x), 3) for x in M[i]]
                 for i, (nm, _) in enumerate(R) if nm.endswith('|all|token')}
        fam = collections.Counter(C.rule_family(R[i][1]) for i in order[:100])
        named_all = {nm: float(M[i].mean()) for i, (nm, _) in enumerate(R) if '|' in nm and nm.count('/') == 0}
        res[script] = dict(named_all=named_all, real=st, top=top, named=named, fam_top100=fam,
                           bestA=R[st['bestA']][0], bestB=R[st['bestB']][0], nrules=len(R))
        print(script, st, R[st['best']][0], fam.most_common(5), flush=True)
    jobs = [(sc, kind, k, SEED) for sc in ('LA', 'LB') for kind in ('N1', 'N2', 'N3') for k in range(NNULL)]
    with Pool(2) as P:
        nulls = P.map(null_job, jobs, chunksize=4)
    for sc in ('LA', 'LB'):
        for kind in ('N1', 'N2', 'N3'):
            arr = [n[3] for n in nulls if n[0] == sc and n[1] == kind]
            r = res[sc]['real']
            d = {}
            for key in ('max_all', 'ho_AB', 'ho_BA'):
                v = np.array([a[key] for a in arr])
                d[key] = dict(null_mean=float(v.mean()), null_sd=float(v.std()),
                              P=float((1 + (v >= r[key]).sum()) / (1 + len(v))))
            fams = {}
            for nm, v in res[sc]['named_all'].items():
                nv = np.array([a['named'][nm] for a in arr])
                fams[nm] = dict(real=v, null_mean=float(nv.mean()), z=float((v - nv.mean()) / (nv.std() + 1e-9)),
                                P=float((1 + (nv >= v).sum()) / (1 + len(nv))))
            d['named'] = fams
            res[sc][kind] = d
            print(sc, kind, {k: v for k, v in d.items() if k != 'named'}, flush=True)
            for nm in sorted(d['named'], key=lambda x: -d['named'][x]['z'])[:6]:
                print('   ', nm, d['named'][nm], flush=True)
    pj = [(f, k, SEED, None) for f in (1.0, 0.5, 0.25) for k in range(20)]
    with Pool(2) as P:
        pl = P.map(plant_job, pj)
    n1 = [n[3] for n in nulls if n[0] == 'LA' and n[1] == 'N1']
    q95 = {key: float(np.quantile([a[key] for a in n1], 0.95)) for key in ('max_all', 'ho_AB', 'ho_BA')}
    summ = {}
    for f in (1.0, 0.5, 0.25):
        rows = [p for p in pl if p['f'] == f]
        summ[f] = dict(match3=float(np.mean([p['match_len234'] == 3 for p in rows])),
                       match2=float(np.mean([p['match_len234'] >= 2 for p in rows])),
                       sig_all=float(np.mean([p['max_all'] > q95['max_all'] for p in rows])),
                       sig_hoAB=float(np.mean([p['ho_AB'] > q95['ho_AB'] for p in rows])),
                       sig_hoBA=float(np.mean([p['ho_BA'] > q95['ho_BA'] for p in rows])))
        print('plant', f, summ[f], flush=True)
    res['plant'] = dict(summary={str(k): v for k, v in summ.items()}, rows=pl, q95_N1=q95)
    res['secs'] = time.time() - t0
    json.dump(res, open(os.path.join(C.CK, f'c1_seed{SEED}.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
