"""pe87 cycle 1: checkerboards (competitive exclusion) between sign forms, curveball null within strata."""
import sys, os, json, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe87_common as K

OUT = os.path.join(K.CK, 'c1.json')
NN = int(os.environ.get('PE87_NN', 300))


def ck(name, fn):
    p = os.path.join(K.CK, name + '.json')
    if os.path.exists(p):
        return json.load(open(p))
    r = fn()
    json.dump(r, open(p, 'w'))
    return r


def split_scores(tabs, seed, nn=NN, rename=None, randomise=False):
    """seg_z and ctx for one subset of tablets. Returns dict form->index, z, ctx, counts."""
    rng = np.random.default_rng(seed)
    forms, R, st = K.incidence(tabs, min_tab=2, rename=rename)
    if randomise:
        K.curveball(R, st, rng, 30 * len(R))
    z, obs, mu = K.seg_z(R, st, len(forms), rng, n_null=nn)
    cnt = np.array([sum(1 for s in R if i in s) for i in range(len(forms))])
    S = K.ctx_matrix(K.contexts(tabs, forms, rename=rename))
    return {'forms': forms, 'z': z, 'ctx': S, 'cnt': cnt}


def ctx_ref(D, base, rng, n=20000):
    F = len(D['forms'])
    a = rng.integers(0, F, n); b = rng.integers(0, F, n)
    m = np.array([base(D['forms'][i]) != base(D['forms'][j]) for i, j in zip(a, b)])
    v = D['ctx'][a[m], b[m]]
    return float(v.mean()), float(v.std()), float(np.quantile(v, 0.75))


def calibrate(D, base, rng, min_tab=6):
    f = D['forms']; F = len(f)
    ok = [i for i in range(F) if D['cnt'][i] >= min_tab]
    pos = [(i, j) for i, j in itertools.combinations(ok, 2)
           if base(f[i]) == base(f[j]) and not f[i].startswith('|') and not f[j].startswith('|')]
    lf = np.log(D['cnt'] + 1)
    neg = []
    for i, j in pos:
        for _ in range(20):
            a = ok[rng.integers(len(ok))]; b = ok[rng.integers(len(ok))]
            tries = 0
            while (base(f[a]) == base(f[b]) or abs(lf[a] - lf[i]) > 0.35 or abs(lf[b] - lf[j]) > 0.35) and tries < 400:
                a = ok[rng.integers(len(ok))]; b = ok[rng.integers(len(ok))]; tries += 1
            if tries < 400:
                neg.append((a, b))
    mu, sd, _ = ctx_ref(D, base, rng)
    def sc(pairs):
        s = np.array([D['z'][i, j] for i, j in pairs]); c = np.array([(D['ctx'][i, j] - mu) / sd for i, j in pairs])
        return s, c, s + c
    P = sc(pos); N = sc(neg)
    return {'n_pos': len(pos), 'n_neg': len(neg), 'auc_seg': K.auc(P[0], N[0]), 'auc_ctx': K.auc(P[1], N[1]),
            'auc_comb': K.auc(P[2], N[2]), 'pos_seg_mean': float(P[0].mean()) if len(pos) else None,
            'neg_seg_mean': float(N[0].mean()) if len(neg) else None,
            'top_pos': sorted([(f[i], f[j], round(float(D['z'][i, j]), 2)) for i, j in pos], key=lambda x: -x[2])[:12]}


def pipeline(TR, TE, rng, n_guild=6000, top=50, min_tab=6):
    """Exhaustive diff-base pairs + random guilds on train; re-test top on test."""
    b = K.pe_base
    f = TR['forms']; ok = [i for i in range(len(f)) if TR['cnt'][i] >= min_tab]
    mu, sd, _ = ctx_ref(TR, b, rng)
    tmu, tsd, tq75 = ctx_ref(TE, b, rng)
    tix = {x: i for i, x in enumerate(TE['forms'])}
    okA = np.array(ok)
    Z = TR['z'][np.ix_(okA, okA)]; C = (TR['ctx'][np.ix_(okA, okA)] - mu) / sd
    S = Z + C
    bases = np.array([b(f[i]) for i in ok])
    same = bases[:, None] == bases[None, :]
    iu = np.triu_indices(len(ok), 1)
    sv = S[iu]; valid = ~same[iu]
    order = np.argsort(-np.where(valid, sv, -1e9))[:top]
    pairs = [(ok[iu[0][k]], ok[iu[1][k]]) for k in order]
    n_pairs = int(valid.sum())

    def test_guild(names):
        if not all(x in tix for x in names):
            return None
        ii = [tix[x] for x in names]
        pz = [TE['z'][a, c] for a, c in itertools.combinations(ii, 2)]
        pc = [TE['ctx'][a, c] for a, c in itertools.combinations(ii, 2)]
        return float(np.mean(pz)), float(np.mean(pc))

    res_pairs = []
    for i, j in pairs:
        t = test_guild([f[i], f[j]])
        surv = bool(t and t[0] >= 1.64 and t[1] > tq75)
        res_pairs.append({'g': [f[i], f[j]], 'train_seg': float(TR['z'][i, j]), 'train_ctx': float(TR['ctx'][i, j]),
                          'test': t, 'surv': surv})
    # random guilds of 3-4
    gl = []
    for _ in range(n_guild):
        k = int(rng.integers(3, 5))
        g = list(rng.choice(len(ok), k, replace=False))
        if len(set(bases[g])) < k:
            continue
        sub = [(a, c) for a, c in itertools.combinations(g, 2)]
        sc = float(np.mean([S[a, c] for a, c in sub]))
        gl.append((sc, [f[ok[x]] for x in g]))
    gl.sort(key=lambda x: -x[0])
    res_g = []
    for sc, names in gl[:top]:
        t = test_guild(names)
        res_g.append({'g': names, 'train_score': sc, 'test': t, 'surv': bool(t and t[0] >= 1.64 and t[1] > tq75)})
    return {'n_pairs_scored': n_pairs, 'n_guilds_scored': len(gl), 'pairs': res_pairs, 'guilds': res_g,
            'surv_pairs': sum(r['surv'] for r in res_pairs), 'surv_guilds': sum(r['surv'] for r in res_g),
            'test_ctx_q75': tq75}


def tabs_split():
    tabs, T = K.load_pe()
    tr = [x for x in tabs if x[1].endswith('|M26')]
    te = [x for x in tabs if not x[1].endswith('|M26')]
    return tabs, tr, te


def shuffled_run(seed):
    tabs, tr, te = tabs_split()
    TR = split_scores(tr, seed, nn=150, randomise=True)
    TE = split_scores(te, seed + 1, nn=150, randomise=True)
    r = pipeline(TR, TE, np.random.default_rng(seed + 2))
    return {'seed': seed, 'surv_pairs': r['surv_pairs'], 'surv_guilds': r['surv_guilds']}


def planted_run(seed):
    tabs, tr, te = tabs_split()
    rng = np.random.default_rng(seed)
    cnt = {}
    for tid, st, lines in tabs:
        for f in set(x for fs, _, _ in lines for x in fs):
            cnt[f] = cnt.get(f, 0) + 1
    cand = sorted(f for f, c in cnt.items() if c >= 20 and not f.startswith('|'))
    chosen = list(rng.choice(cand, 6, replace=False))
    rename = {}
    for k, f in enumerate(chosen):
        tl = [tid for tid, st, lines in tabs if any(f in fs for fs, _, _ in lines)]
        half = rng.choice(tl, len(tl) // 2, replace=False)
        for tid in half:
            rename[(tid, f)] = f'PLANT{k}'
    TR = split_scores(tr, seed, nn=150, rename=rename)
    TE = split_scores(te, seed + 1, nn=150, rename=rename)
    # planted forms must count as a different base
    r = pipeline(TR, TE, np.random.default_rng(seed + 2))
    pp = {frozenset([f, f'PLANT{k}']) for k, f in enumerate(chosen)}
    found = [x for x in r['pairs'] if frozenset(x['g']) in pp]
    return {'seed': seed, 'chosen': chosen, 'in_top50': len(found), 'surv_planted': sum(x['surv'] for x in found),
            'surv_pairs': r['surv_pairs']}


def main():
    res = {}
    rng = np.random.default_rng(87001)
    tabs, tr, te = tabs_split()
    # calibration on full corpora
    Dpe = split_scores(tabs, 87010)
    res['calib_pe'] = calibrate(Dpe, K.pe_base, rng)
    print('calib PE', res['calib_pe'], flush=True)
    Dpc = split_scores(K.load_pc(), 87011)
    res['calib_pc'] = calibrate(Dpc, K.pc_base, rng)
    print('calib PC', res['calib_pc'], flush=True)
    json.dump(res, open(OUT, 'w'), indent=1, default=float)
    # real pipeline
    TR = split_scores(tr, 87020); TE = split_scores(te, 87021)
    res['real'] = pipeline(TR, TE, np.random.default_rng(87022))
    print('real surv', res['real']['surv_pairs'], res['real']['surv_guilds'], flush=True)
    json.dump(res, open(OUT, 'w'), indent=1, default=float)
    with Pool(2) as P:
        res['shuffled'] = P.map(shuffled_run, [87100 + 10 * k for k in range(20)])
        print('shuffled', [(x['surv_pairs'], x['surv_guilds']) for x in res['shuffled']], flush=True)
        json.dump(res, open(OUT, 'w'), indent=1, default=float)
        res['planted'] = P.map(planted_run, [87500 + 10 * k for k in range(6)])
        print('planted', res['planted'], flush=True)
    json.dump(res, open(OUT, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
