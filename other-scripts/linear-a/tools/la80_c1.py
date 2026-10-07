"""LA-80 cycle 1: PROPHESY THE CLAY. Text -> tablet outline, selected on Hagia Triada only.
Stage 'select': random hypotheses, shuffles, plants, freeze predictions (no non-HT shape is read).
Stage 'test' : read non-HT shapes (measured after the freeze) and score the frozen survivors."""
import json, os, sys, collections, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la80_common as L
import la80_text as TX
from scipy.stats import rankdata

SHAPES = ['aspect', 'rectness', 'solidity', 'corner_mean', 'corner_min', 'corner_sd', 'taper', 'endw', 'waist',
          'rough', 'fd2', 'fd3', 'fd4', 'fd5', 'fd6', 'fd8', 'long_cm', 'area_cm2', 'tone_sd']
NH = int(os.environ.get('LA80_NH', 20000))

def resid(y, X):
    y = rankdata(y); y = (y - y.mean()) / (y.std() + 1e-12)
    A = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    return r / (np.linalg.norm(r) + 1e-12)

def tablet_shapes(shp, T):
    out = {}
    for k, docs in T.items():
        recs = [shp.get(d['id']) for d in docs]
        recs = [r for r in recs if r and r.get('ok')]
        if recs:
            out[k] = {s: float(np.mean([r[s] for r in recs if s in r])) for s in SHAPES if any(s in r for r in recs)}
    return out

def build(ids_face, shape_file, extra_cov=None):
    T = TX.load_tablets(ids_face)
    shp = json.load(open(shape_file)) if shape_file else {}
    tabs, cov, F, S, G, first = TX.features(T)
    TS = tablet_shapes(shp, T)
    dep = [T[k][0]['findspot'] or '?' for k in tabs]
    site = [T[k][0]['site'] for k in tabs]
    return dict(T=T, tabs=tabs, cov=cov, F=F, S=S, G=G, first=first, TS=TS, dep=dep, site=site)

def shape_matrix(B, keep):
    M = np.full((len(keep), len(SHAPES)), np.nan)
    for i, k in enumerate(keep):
        for j, s in enumerate(SHAPES):
            M[i, j] = B['TS'][k].get(s, np.nan)
    return M

def score_pool(pool, B, keep_idx, M, X, groups, min_pos=5, TV=None):
    """partial Spearman per hypothesis; per-group signs."""
    Rs = []
    for j in range(M.shape[1]):
        col = M[:, j]; ok = ~np.isnan(col)
        Rs.append((ok, resid(col[ok], X[ok]) if ok.sum() > 10 else None))
    out = []
    ug = [g for g in sorted(set(groups)) if g != '?']
    gidx = {g: np.array([x == g for x in groups]) for g in ug}
    for ii, (h, j) in enumerate(pool):
        t = TV[ii] if TV is not None else TX.eval_text_feature(h, B['F'], B['S'], B['G'], B['first'])[keep_idx]
        ok, rs = Rs[j]
        if rs is None: out.append(None); continue
        tt = t[ok]
        if len(set(tt)) < 2 or (set(tt) <= {0, 1} and min(tt.sum(), len(tt) - tt.sum()) < min_pos):
            out.append(None); continue
        rt = resid(tt, X[ok]); r = float(rt @ rs)
        gs = []
        for g in ug:
            m = gidx[g][ok]
            if m.sum() >= 6 and len(set(tt[m])) > 1:
                a = rt[m] - rt[m].mean(); b = rs[m] - rs[m].mean()
                den = np.linalg.norm(a) * np.linalg.norm(b)
                gs.append(float(a @ b / den) if den > 0 else 0.0)
        out.append((r, gs, int(ok.sum())))
    return out

def survivors(scores, zthr=2.5, need=3):
    surv = []
    for i, s in enumerate(scores):
        if s is None: continue
        r, gs, n = s
        z = r * np.sqrt(max(n - 4, 1))
        if abs(z) >= zthr and sum(np.sign(g) == np.sign(r) for g in gs) >= need and len(gs) >= 3:
            surv.append(i)
    return surv

def main_select():
    rng = np.random.default_rng(80)
    ht_ids = json.load(open(os.path.join(L.CK, 'ht_ids.json')))
    B = build(ht_ids, os.path.join(L.CK, 'shape_ht.json'))
    keep = [k for k in B['tabs'] if k in B['TS']]
    keep_idx = np.array([B['tabs'].index(k) for k in keep])
    M = shape_matrix(B, keep); X = B['cov'][keep_idx]
    groups = [B['dep'][i] for i in keep_idx]
    cnt = collections.Counter()
    for s in B['S']: cnt.update(s)
    sign_pool = sorted([s for s, c in cnt.items() if c >= 8])
    lc = collections.Counter()
    for g in B['G']: lc.update(g)
    logo_pool = sorted([s for s, c in lc.items() if c >= 5 and s])
    Fn = sorted(B['F'])
    pool = [(TX.random_text_feature(rng, Fn, sign_pool, logo_pool), int(rng.integers(len(SHAPES)))) for _ in range(NH)]
    log = {'n_tablets': len(keep), 'n_faces': len(ht_ids), 'groups': collections.Counter(groups), 'sign_pool': len(sign_pool), 'logo_pool': logo_pool}
    TV = [TX.eval_text_feature(h, B['F'], B['S'], B['G'], B['first'])[keep_idx] for h, j in pool]
    sc = score_pool(pool, B, keep_idx, M, X, groups, TV=TV)
    S_real = survivors(sc)
    log['n_valid'] = sum(s is not None for s in sc); log['real_surv'] = len(S_real)
    # shuffle control: shapes permuted among tablets within line-count tertiles
    tert = np.digitize(X[:, 1], np.quantile(X[:, 1], [1 / 3, 2 / 3]))
    shuf = []; shuf_sets = []
    for sh in range(20):
        perm = np.arange(len(keep))
        for t in set(tert):
            ix = np.where(tert == t)[0]; perm[ix] = rng.permutation(ix)
        sc2 = score_pool(pool, B, keep_idx, M[perm], X, groups, TV=TV)
        s2 = survivors(sc2); shuf.append(len(s2)); shuf_sets.append(s2)
    log['shuffle_surv'] = shuf
    # planted control: 3 links per world, effect added to the shape rank
    planted = []
    for w in range(10):
        Mp = M.copy(); links = []
        while len(links) < 3:
            i = int(rng.integers(len(pool)))
            if sc[i] is None: continue
            h, j = pool[i]
            t = TX.eval_text_feature(h, B['F'], B['S'], B['G'], B['first'])[keep_idx]
            if t.std() == 0: continue
            col = Mp[:, j]; ok = ~np.isnan(col)
            zt = (t - t.mean()) / t.std()
            Mp[ok, j] = col[ok] + 0.35 * np.nanstd(col) * zt[ok]
            links.append(i)
        sc3 = score_pool(pool, B, keep_idx, Mp, X, groups, TV=TV)
        s3 = set(survivors(sc3))
        planted.append(sum(l in s3 for l in links))
    log['planted_found_of_3'] = planted
    return B, pool, sc, S_real, shuf_sets, log, keep

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'select':
        B, pool, sc, S_real, shuf_sets, log, keep = main_select()
        def hj(i):
            (h, j) = pool[i]; return {'h': [h[0], list(h[1]) if isinstance(h[1], tuple) else h[1]], 'shape': SHAPES[j], 'r_ht': sc[i][0], 'groups': sc[i][1]}
        frozen = {'plan': 'LA-80 c1. Score each survivor on non-HT tablets (Tablet + Lames, all sites but HT, KH5 excluded): '
                          'partial Spearman with covariates log signs, log lines, edge share and site dummies. Primary: share of survivors whose '
                          'test sign equals the HT sign, against (a) the same share for survivors of the 20 shuffled-HT runs, (b) 2,000 random '
                          'valid pairs (sign of their HT r). Kill: primary share <= 0.60 or not above the 95th pct of shuffled-run survivors.',
                  'survivors': [hj(i) for i in S_real],
                  'shuffle_survivors': [[hj(i) for i in s] for s in shuf_sets],
                  'random_ref': [hj(i) for i in np.random.default_rng(81).choice([i for i, s in enumerate(sc) if s is not None], min(2000, sum(s is not None for s in sc)), replace=False)],
                  'log': {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in log.items()}}
        fn = os.path.join(L.DATA, 'la80_frozen_c1.json')
        json.dump(frozen, open(fn, 'w'), indent=0, default=float)
        h = hashlib.sha256(open(fn, 'rb').read()).hexdigest()
        open(fn.replace('.json', '.sha256'), 'w').write(h + '  la80_frozen_c1.json\n')
        print(json.dumps(frozen['log'], default=float)); print('survivors', len(S_real), 'sha256', h)
