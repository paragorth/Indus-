"""LA-80 cycle 3: CLAY LOTS AS A FILING KEY. Do tablets made alike (outline, size, clay tone) carry related text?
Pairwise: random weighted shape distances vs text similarity, within deposit (HT) / within site (test).
select on HT only -> freeze -> test on non-HT within-site pairs."""
import json, os, sys, collections, hashlib
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la80_common as L, la80_text as TX
from la80_c1 import build, SHAPES
FEATS = SHAPES + ['tone']
NM = int(os.environ.get('LA80_NM', 5000))

def table(ids, shape_fn, group_key):
    B = build(ids, shape_fn)
    shp = json.load(open(shape_fn))
    keep = [k for k in B['tabs'] if k in B['TS']]
    rows = []
    for k in keep:
        i = B['tabs'].index(k)
        recs = [shp[d['id']] for d in B['T'][k] if shp.get(d['id'], {}).get('ok')]
        f = [np.nanmean([r.get(s, np.nan) for r in recs]) for s in FEATS]
        toks = [t for d in B['T'][k] for t in d['tokens']]
        sig = collections.Counter(s for t in toks if t['t'] == 'word' for s in t['s'])
        sig.update(t.get('v', '').split('+')[0] for t in toks if t['t'] == 'logo')
        words = set('-'.join(t['s']) for t in toks if t['t'] == 'word' and len(t['s']) > 1) | set(t.get('v', '') for t in toks if t['t'] == 'logo')
        rows.append(dict(id=k, grp=group_key(B, i), f=f, cov=B['cov'][i], sig=sig, words=words))
    return rows

def pairs(rows, min_grp=6):
    g = collections.defaultdict(list)
    for i, r in enumerate(rows): g[r['grp']].append(i)
    P, G = [], []
    for k, ix in g.items():
        if k == '?' or len(ix) < min_grp: continue
        for a in range(len(ix)):
            for b in range(a + 1, len(ix)):
                P.append((ix[a], ix[b])); G.append(k)
    return np.array(P), np.array(G)

def zfeat(rows):
    F = np.array([r['f'] for r in rows], float)
    # standardise within group, fill missing with 0
    out = np.zeros_like(F)
    for gname in set(r['grp'] for r in rows):
        m = np.array([r['grp'] == gname for r in rows])
        sub = F[m]; mu = np.nanmean(sub, 0); sd = np.nanstd(sub, 0) + 1e-9
        out[m] = np.nan_to_num((sub - mu) / sd)
    return out

def textsims(rows, P):
    keys = sorted(set(k for r in rows for k in r['sig']))
    V = np.array([[r['sig'].get(k, 0) for k in keys] for r in rows], float)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    cos = np.sum(V[P[:, 0]] * V[P[:, 1]], 1)
    jac = np.array([len(rows[a]['words'] & rows[b]['words']) / max(1, len(rows[a]['words'] | rows[b]['words'])) for a, b in P])
    return {'cos': cos, 'jac': jac}

def paircov(rows, P, G):
    C = np.array([r['cov'] for r in rows])
    X = np.c_[np.abs(C[P[:, 0], 0] - C[P[:, 1], 0]), np.abs(C[P[:, 0], 1] - C[P[:, 1], 1]), C[P[:, 0], 2] + C[P[:, 1], 2],
              C[P[:, 0], 0] + C[P[:, 1], 0]]
    ug = sorted(set(G))
    D = np.array([[g == u for u in ug[1:]] for g in G], float).reshape(len(G), -1)
    return np.c_[np.ones(len(P)), X, D]

def resid_cols(Y, X):
    """rank-transform each column of Y, residualise on X, unit-normalise."""
    R = np.apply_along_axis(rankdata, 0, Y)
    R = R - R.mean(0)
    Q, _ = np.linalg.qr(X)
    R = R - Q @ (Q.T @ R)
    return R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)

def metric_r(Z, P, W, Xp, ts_res, G, groups):
    Dabs = np.abs(Z[P[:, 0]] - Z[P[:, 1]])         # pairs x feats
    Dm = Dabs @ W                                  # pairs x metrics
    Rd = resid_cols(Dm, Xp)
    pooled = ts_res @ Rd                            # metrics
    per = []
    for g in groups:
        m = G == g
        a = Rd[m] - Rd[m].mean(0); b = ts_res[m] - ts_res[m].mean()
        per.append((b @ a) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b) + 1e-12))
    return pooled, np.array(per)

def select():
    rng = np.random.default_rng(803)
    ht_ids = json.load(open(os.path.join(L.CK, 'ht_ids.json')))
    rows = table(ht_ids, os.path.join(L.CK, 'shape_ht.json'), lambda B, i: B['dep'][i])
    P, G = pairs(rows); Z = zfeat(rows); Xp = paircov(rows, P, G)
    groups = sorted(set(G))
    TS = textsims(rows, P)
    W = (rng.random((len(FEATS), NM)) < 0.3) * np.exp(rng.normal(0, 0.7, (len(FEATS), NM)))
    W[:, W.sum(0) == 0] = 1.0
    res = {}
    for tk in ('cos', 'jac'):
        tsr = resid_cols(TS[tk][:, None], Xp)[:, 0]
        pooled, per = metric_r(Z, P, W, Xp, tsr, G, groups)
        nulls = []
        for s in range(20):
            Zs = Z.copy()
            for gname in set(r['grp'] for r in rows):
                ix = np.array([i for i, r in enumerate(rows) if r['grp'] == gname]); Zs[ix] = Z[rng.permutation(ix)]
            nulls.append(metric_r(Zs, P, W, Xp, tsr, G, groups))
        thr = np.percentile(np.concatenate([n[0] for n in nulls]), 1)     # 1st pct of null pooled r
        def surv(pl, pr): return np.where((pl <= thr) & ((pr < 0).sum(0) >= len(groups) - 1))[0]
        S_real = surv(pooled, per); S_null = [surv(*n) for n in nulls]
        # planted: shapes nudged toward the text's first principal component (0.5 sd on 3 features)
        keys = sorted(set(k for r in rows for k in r['sig']))
        V = np.array([[r['sig'].get(k, 0) for k in keys] for r in rows], float); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        u = np.linalg.svd(V - V.mean(0), full_matrices=False)[0][:, 0]; u = (u - u.mean()) / u.std()
        pl_found = []
        for w in range(5):
            Zp = Z.copy(); fs = rng.choice(len(FEATS), 3, replace=False); Zp[:, fs] += 0.5 * u[:, None]
            pp, pq = metric_r(Zp, P, W, Xp, tsr, G, groups); pl_found.append(len(surv(pp, pq)))
        res[tk] = dict(real=len(S_real), null=[len(s) for s in S_null], planted=pl_found, thr=float(thr),
                       best=int(np.argmin(pooled)), best_r=float(pooled.min()), null_min=[float(n[0].min()) for n in nulls],
                       surv=S_real.tolist(), null_surv=[s.tolist() for s in S_null], pooled_surv=[float(pooled[i]) for i in S_real])
        print(tk, 'pairs', len(P), 'real', len(S_real), 'null', [len(s) for s in S_null], 'planted', pl_found,
              'best r %.3f null-min mean %.3f' % (pooled.min(), np.mean(res[tk]['null_min'])))
    frozen = dict(plan='LA-80 c3. For each text similarity (cos signs, jac words): frozen surviving metrics scored on non-HT within-site pairs '
                       '(KH, ZA, PH, others with >= 6 tablets), same covariates + site. Primary: share of survivors with r < 0 vs null-run survivor sets; '
                       'pre-specified single test: the best HT metric, within-site permutation p (2,000). Kill: share <= 0.60 or best metric p > 0.05.',
                  feats=FEATS, W=W.tolist(), res=res, n_pairs=len(P))
    fn = os.path.join(L.DATA, 'la80_frozen_c3.json'); json.dump(frozen, open(fn, 'w'))
    h = hashlib.sha256(open(fn, 'rb').read()).hexdigest(); open(fn.replace('.json', '.sha256'), 'w').write(h + '  la80_frozen_c3.json\n')
    print('sha256', h)

def test():
    fz = os.path.join(L.DATA, 'la80_frozen_c3.json')
    assert open(fz.replace('.json', '.sha256')).read().split()[0] == hashlib.sha256(open(fz, 'rb').read()).hexdigest()
    fr = json.load(open(fz)); W = np.array(fr['W'])
    ids = json.load(open(os.path.join(L.CK, 'test_ids.json')))
    rows = table(ids, os.path.join(L.CK, 'shape_test.json'), lambda B, i: B['site'][i])
    P, G = pairs(rows); Z = zfeat(rows); Xp = paircov(rows, P, G); groups = sorted(set(G))
    TS = textsims(rows, P); rng = np.random.default_rng(9); out = {'pairs': len(P), 'groups': dict(collections.Counter(G))}
    for tk in ('cos', 'jac'):
        R = fr['res'][tk]; tsr = resid_cols(TS[tk][:, None], Xp)[:, 0]
        pooled, per = metric_r(Z, P, W, Xp, tsr, G, groups)
        def sh(ix): return (float(np.mean(pooled[ix] < 0)), float(np.mean(pooled[ix]))) if len(ix) else (None, None)
        real = sh(np.array(R['surv'], int)); nulls = [sh(np.array(s, int)) for s in R['null_surv']]
        b = R['best']; obs = pooled[b]; perm = []
        for k in range(2000):
            Zs = Z.copy()
            for gname in groups:
                ix = np.array([i for i, r in enumerate(rows) if r['grp'] == gname]); Zs[ix] = Z[rng.permutation(ix)]
            Dm = (np.abs(Zs[P[:, 0]] - Zs[P[:, 1]]) @ W[:, [b]])
            perm.append(float(tsr @ resid_cols(Dm, Xp)[:, 0]))
        p = (1 + np.sum(np.array(perm) <= obs)) / 2001
        # power: plant text PC1 into 3 features at 0.5 sd, as in selection
        keys = sorted(set(k for r in rows for k in r['sig']))
        V = np.array([[r['sig'].get(k, 0) for k in keys] for r in rows], float); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        u = np.linalg.svd(V - V.mean(0), full_matrices=False)[0][:, 0]; u = (u - u.mean()) / u.std()
        Zp = Z.copy(); Zp[:, :] += 0.0
        fs = np.argsort(-W[:, b])[:3]; Zp[:, fs] += 0.5 * u[:, None]
        pp, _ = metric_r(Zp, P, W, Xp, tsr, G, groups)
        out[tk] = dict(real=real, null=nulls, best=b, best_r_test=float(obs), best_p=float(p), best_r_ht=R['best_r'],
                       planted_best_r=float(pp[b]), all_metrics_share_neg=float(np.mean(pooled < 0)), per_site_best=[float(x) for x in per[:, b]], sites=groups)
        nm = [n[0] for n in nulls if n[0] is not None]
        print(tk, 'pairs', len(P), 'real', real, 'null share mean %.3f max %.3f' % (np.mean(nm), np.max(nm)) if nm else '',
              'best HT r %.3f -> test r %.4f p %.3f (planted %.3f)' % (R['best_r'], obs, p, pp[b]), 'all metrics share<0 %.3f' % np.mean(pooled < 0),
              dict(zip(groups, np.round(per[:, b], 3))))
    json.dump(out, open(os.path.join(L.CK, 'c3_test.json'), 'w'), indent=1, default=float)

if __name__ == '__main__':
    {'select': select, 'test': test}[sys.argv[1]]()
