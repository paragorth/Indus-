"""pe85 cycle 1: use-model tournament on the squared header end (single worker)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe85_common as C
import pe83_common as P

FZ = json.load(open(os.path.join(C.CK, 'frozen_c1.json')))
ATTR = ['tag', 'sealed', 'indoss', 'h157', 'h327', 'hlen', 'm288', 'cap', 'm153', 'outpost', 'rev', 'nl', 'la', 'tw', 'asp',
        'n_entries', 'numonly', 'dx', 'pubr']


def setup(intact):
    R = C.table(intact)
    # within-volume publication rank (archive batch order)
    for v in set(r['vol'] for r in R):
        idx = [i for i, r in enumerate(R) if r['vol'] == v]
        pubs = np.array([R[i]['pub'] for i in idx], float)
        rk = np.argsort(np.argsort(np.nan_to_num(pubs, nan=np.nanmedian(pubs) if np.isfinite(pubs).any() else 0)))
        for i, k in zip(idx, rk):
            R[i]['pubr'] = k / max(1, len(idx) - 1)
    hd = C.col(R, 'hd')
    extra = [] if intact else [C.col(R, 'l1ok')]
    Z = np.column_stack([P.design(R, extra), hd])
    A = {a: C.zs(C.col(R, a)) for a in ATTR}
    for a in ATTR:
        A['hd*' + a] = C.zs(A[a] * (hd - hd.mean()))
    vol = np.array([r['vol'] for r in R])
    strata = np.array([f'{v}_{s}' for v, s in zip(np.where(np.isin(vol, ['MDP 26', 'MDP 17', 'MDP 06']), vol, 'oth'), P.strata(Z))])
    return R, hd, Z, A, vol, strata


def pred(model, A):
    v = np.zeros(len(next(iter(A.values()))))
    for a, s in model.get('main', {}).items():
        v += s * A[a]
    for a, s in model.get('inter', {}).items():
        v += s * A['hd*' + a]
    return v


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(b[ok]) == 0:
        return 0.0
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def named(y, A, strata, rng, mask, B=2000):
    PR = {nm: pred(m, A) for nm, m in FZ['models'].items() if np.any(pred(m, A))}
    r0 = {nm: corr(y[mask], p[mask]) for nm, p in PR.items()}
    cnt = {nm: 0 for nm in PR}
    for _ in range(B):
        yp = C.perm_within(y, strata, rng)
        for nm, p in PR.items():
            cnt[nm] += corr(yp[mask], p[mask]) >= r0[nm]
    return {nm: dict(r=round(r0[nm], 3), p_one_sided=round((cnt[nm] + 1) / (B + 1), 4)) for nm in PR}


def random_models(rng, K=5000):
    keys = list(ATTR) + ['hd*' + a for a in ATTR]
    M = []
    for _ in range(K):
        k = rng.integers(1, 5); ks = rng.choice(keys, k, replace=False)
        M.append({str(a): int(rng.choice([-1, 1])) for a in ks})
    return M


def mat(M, A):
    keys = list(A)
    W = np.zeros((len(M), len(keys)))
    for i, m in enumerate(M):
        for a, s in m.items():
            W[i, keys.index(a)] = s
    X = np.column_stack([A[k] for k in keys])
    return W, X


def screen(y, W, X, tr, te, top=50):
    """select top models on tr, return their held-out r on te and how many replicate (r>0, one-sided p<.05 normal approx)."""
    def rs(mask):
        Pm = X[mask] @ W.T
        Pm = (Pm - Pm.mean(0)) / (Pm.std(0) + 1e-12)
        yy = y[mask]; yy = (yy - yy.mean()) / yy.std()
        return yy @ Pm / mask.sum()
    rtr = rs(tr); rte = rs(te)
    sel = np.argsort(-rtr)[:top]
    crit = 1.645 / np.sqrt(te.sum())
    return rtr, rte, sel, int((rte[sel] > crit).sum())


def main():
    rng = np.random.default_rng(85)
    out = {'frozen_sha': C.sha(FZ)}
    for intact in (True, False):
        tag = 'intact' if intact else 'full'
        R, hd, Z, A, vol, strata = setup(intact)
        y = C.resid(C.col(R, 'cf_top'), Z)
        yb = C.resid(C.col(R, 'cf_bot'), Z)
        tr = vol == 'MDP 26'; te = ~tr
        res = {'n': len(R), 'n_train': int(tr.sum()), 'n_test': int(te.sum())}
        res['header_r_top'] = P.partial(C.col(R, 'cf_top'), hd, Z[:, :-1])['r']
        for half, m in (('all', np.ones(len(R), bool)), ('train', tr), ('test', te)):
            res['named_' + half] = named(y, A, strata, rng, m, B=1000 if half == 'all' else 600)
        res['named_bottom_all'] = {nm: dict(r=round(corr(yb, pred({'main': mm['bottom']}, A)), 3)) for nm, mm in FZ['models'].items() if mm['bottom']}
        # single-attribute table (top and bottom) for the record
        res['single'] = {a: [round(corr(y, A[a]), 3), round(corr(yb, A[a]), 3)] for a in A}
        # massive random guessing
        M = random_models(rng); W, X = mat(M, A)
        rtr, rte, sel, nrep = screen(y, W, X, tr, te)
        rtr2, rte2, sel2, nrep2 = screen(y, W, X, te, tr)
        res['random'] = dict(K=len(M), survivors_26_to_rest=nrep, survivors_rest_to_26=nrep2,
                             best_train=[(M[i], round(float(rtr[i]), 3), round(float(rte[i]), 3)) for i in sel[:8]],
                             best_rev=[(M[i], round(float(rtr2[i]), 3), round(float(rte2[i]), 3)) for i in sel2[:8]])
        # named models' percentile among random models of the same size (all tablets)
        rall = W @ (X.T @ ((y - y.mean()) / y.std())) / len(y)
        Pn = X @ W.T; sd = Pn.std(0) + 1e-12
        rall = ((y - y.mean()) / y.std()) @ ((Pn - Pn.mean(0)) / sd) / len(y)
        res['named_percentile_vs_random'] = {}
        for nm, m in FZ['models'].items():
            p = pred(m, A)
            if np.any(p):
                r = corr(y, p); res['named_percentile_vs_random'][nm] = round(float((rall < r).mean()), 3)
        # shuffled control: whole pipeline on y permuted within strata
        nulls = []
        for b in range(100):
            yp = C.perm_within(y, strata, rng)
            nulls.append(screen(yp, W, X, tr, te)[3] + screen(yp, W, X, te, tr)[3])
        res['shuffle_survivors'] = dict(real=nrep + nrep2, null_mean=round(float(np.mean(nulls)), 2), null_q95=float(np.quantile(nulls, .95)),
                                        p=round((np.sum(np.array(nulls) >= nrep + nrep2) + 1) / 101, 3))
        # planted worlds
        pw = {}
        for wn, add in (('spine', 0.25 * A['tag'] + 0.15 * A['indoss']), ('grip', -0.25 * A['hd*la'])):
            yy = C.resid(C.col(R, 'cf_top'), Z)
            yy = (yy - yy.mean()) / yy.std() + add
            nm_r = {nm: round(corr(yy, pred(m, A)), 3) for nm, m in FZ['models'].items() if np.any(pred(m, A))}
            rtrp, rtep, selp, nrp = screen(yy, W, X, tr, te)
            pw[wn] = dict(named=nm_r, winner=max(nm_r, key=nm_r.get), random_survivors=nrp,
                          top=[(M[i], round(float(rtep[i]), 3)) for i in selp[:3]])
        res['planted'] = pw
        out[tag] = res
        json.dump(out, open(os.path.join(C.CK, 'c1.json'), 'w'), indent=1)
        print(tag, json.dumps({k: v for k, v in res.items() if k != 'single'}, indent=0)[:6000], flush=True)
    print(json.dumps({k: out['full']['single'][k] for k in out['full']['single']}, indent=0))


if __name__ == '__main__':
    main()
