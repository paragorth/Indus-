"""la75 cycle 1: massive random search for per-document number fingerprints (counted vs made-up numbers).
Discovery half / held-out half of documents; stratum-permutation null (site x commodity x magnitude bin);
planted worlds (rounding estimators, digit taboo); Linear B KN/PY controls."""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
from la75_common import *
import scipy.sparse as sp

NF = int(os.environ.get('NF', 6000)); NP = int(os.environ.get('NP', 1000)); NPH = int(os.environ.get('NPH', 2000))


class FastPanel(Panel):
    def __init__(self, rows):
        super().__init__(rows)
        self.M = sp.csr_matrix((np.ones(len(rows)), (self.g, np.arange(len(rows)))), shape=(len(self.docs), len(rows)))

    def stat_R(self, R):
        G = self.M @ R; Q = self.M @ (R ** 2)
        return (G ** 2 - Q).sum(0) / self.den / ((R ** 2).mean(0) + 1e-12)


def pvals(P, F, nperm, rng):
    R = P.resid(F).astype(np.float64)
    obs = P.stat_R(R); ge = np.ones(F.shape[1]); nulls = []
    for _ in range(nperm):
        s = P.stat_R(R[P.perm(rng)]); ge += s >= obs - 1e-12; nulls.append(s)
    return obs, ge / (nperm + 1), np.array(nulls)


def bh(p, q=0.1):
    o = np.argsort(p); n = len(p); ok = p[o] <= q * (np.arange(1, n + 1) / n)
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    return set(o[:k].tolist())


def subset(rows, docs):
    return [r for r in rows if r['doc'] in docs]


def dedupe(F):
    _, first = np.unique(F.T, axis=0, return_index=True)
    return np.sort(first)


def maxT(P, F, nperm, rng):
    R = P.resid(F).astype(np.float64); obs = P.stat_R(R)
    nulls = np.array([P.stat_R(R[P.perm(rng)]) for _ in range(nperm)])
    mu, sd = nulls.mean(0), nulls.std(0) + 1e-12
    z = (obs - mu) / sd; zn = (nulls - mu) / sd
    p = (1 + (nulls >= obs - 1e-12).sum(0)) / (nperm + 1)
    mx = zn.max(1); fwer = (1 + (mx[:, None] >= z[None, :]).sum(0)) / (nperm + 1)
    n01 = int((p <= 0.01).sum())
    sub = nulls[:300]; null_n01 = np.array([((sub >= sub[i]).mean(0) <= 0.01).sum() for i in range(len(sub))])
    return obs, z, p, fwer, n01, null_n01


def pipeline(rows, feats, rng, nperm=NP, nperm_h=NPH, label='', topk=10):
    docs = sorted(set(r['doc'] for r in rows)); site = {r['doc']: r['site'] for r in rows}
    F = fmatrix(rows, feats); P = FastPanel(rows)
    var = (P.resid(F) ** 2).sum(0); keep = np.where(var > 2)[0]
    keep = keep[dedupe(F[:, keep])]
    obs, z, p, fwer, n01, null_n01 = maxT(P, F[:, keep], nperm, rng)
    P_glob = (1 + (null_n01 >= n01).sum()) / (1 + len(null_n01))
    order = np.argsort(-z)
    out = dict(label=label, n_rows=len(rows), n_docs=len(docs), n_feat=len(keep), n01=n01,
               null_n01_mean=float(null_n01.mean()), null_n01_95=float(np.percentile(null_n01, 95)), P_glob=float(P_glob),
               n_fwer=int((fwer <= 0.05).sum()),
               top=[dict(f=feats[keep[i]], r=float(obs[i]), z=float(z[i]), p=float(p[i]), fwer=float(fwer[i])) for i in order[:15]])
    # 2-fold document split: select top-k on one half, test on the other (Bonferroni over k)
    D1 = set()
    for s in sorted(set(site.values())):
        ds = [d for d in docs if site[d] == s]; rng.shuffle(ds); D1 |= set(ds[: (len(ds) + 1) // 2])
    D2 = set(docs) - D1; folds = []
    for A, B in ((D1, D2), (D2, D1)):
        ra, rb = subset(rows, A), subset(rows, B)
        Fa, Fb = fmatrix(ra, feats)[:, keep], fmatrix(rb, feats)[:, keep]
        Pa, Pb = FastPanel(ra), FastPanel(rb)
        ok = np.where(((Pa.resid(Fa) ** 2).sum(0) > 1) & ((Pb.resid(Fb) ** 2).sum(0) > 1))[0]
        _, pa, _ = pvals(Pa, Fa[:, ok], nperm // 2, rng)
        sel = ok[np.argsort(pa)[:topk]]
        ob, pb, _ = pvals(Pb, Fb[:, sel], nperm_h, rng)
        folds.append(dict(n_pass=int((pb <= 0.05 / topk).sum()), min_p=float(pb.min()),
                          sel=[(feats[keep[i]], float(x)) for i, x in zip(sel, pb)]))
    out['folds'] = folds; out['n_pass_h'] = sum(f['n_pass'] for f in folds)
    return out


def plant(rows, rng, kind, frac_docs=0.25, pr=0.7):
    P = Panel(rows); perm = P.perm(rng)
    new = [dict(rows[i]) for i in range(len(rows))]
    for k, r in enumerate(new):           # shuffled-number base world: numbers moved within stratum
        src = rows[perm[k]]; r['v'] = src['v']; r['frac'] = src['frac']
    docs = sorted(set(r['doc'] for r in new)); pl = set(rng.choice(docs, int(len(docs) * frac_docs), replace=False).tolist())
    for r in new:
        if r['doc'] in pl:
            if kind == 'round5' and r['v'] >= 6 and rng.random() < pr:
                r['v'] = int(5 * round(r['v'] / 5))
            if kind == 'taboo7' and r['v'] % 8 == 7:
                r['v'] += 1
            if kind == 'evenfrac' and r['frac'] and rng.random() < 0.8:
                r['frac'] = ('J',)
        r['mb'] = magbin(r['v']); r['stratum'] = (r['site'], r['tg'], r['mb'])
    return new, pl


if __name__ == '__main__':
    t0 = time.time(); rng = np.random.default_rng(75001)
    feats = gen_features(rng, NF) + [('div', 5), ('div', 10), ('mod', 8, (7,)), ('frac', (0,))]
    res = {}
    rows, _ = la_entries(); rows = prepare(rows)
    res['LA_read'] = pipeline(rows, feats, rng, label='LA read-only'); print('LA', time.time() - t0, flush=True)
    rows2, _ = la_entries(read_only=False); rows2 = prepare(rows2)
    res['LA_all'] = pipeline(rows2, feats, rng, label='LA read+damaged'); print('LA all', time.time() - t0, flush=True)
    for kind, fd, pr in (('round5', .25, .7), ('round5', .25, 1.0), ('round5', .5, 1.0), ('taboo7', .5, 1.0), ('evenfrac', .5, .8)):
        for s in range(2):
            prw, pl = plant(rows, np.random.default_rng(7500 + s), kind, fd, pr)
            o = pipeline(prw, feats, rng, nperm=400, nperm_h=1000, label=f'plant {kind} docs{fd} p{pr} s{s}')
            o['planted'] = sorted(pl); res[f'plant_{kind}_{fd}_{pr}_{s}'] = o; print(kind, s, time.time() - t0, flush=True)
    for s in range(2):   # pure shuffled world: false-positive control
        pr, _ = plant(rows, np.random.default_rng(7600 + s), 'none')
        res[f'shuffle_{s}'] = pipeline(pr, feats, rng, nperm=400, nperm_h=1000, label=f'shuffled s{s}')
    for p in ('KN', 'PY'):
        b = prepare(lb_entries(p)); res['LB_' + p] = pipeline(b, feats, rng, nperm=400, nperm_h=1000, label='LB ' + p)
        print(p, time.time() - t0, flush=True)
    pickle.dump(dict(res=res, feats=feats), open(os.path.join(CK, 'c1.pkl'), 'wb'))
    for k, o in res.items():
        print(k, {x: o[x] for x in ('n_rows', 'n_docs', 'n_feat', 'n01', 'null_n01_mean', 'null_n01_95', 'P_glob', 'n_fwer', 'n_pass_h')})
        for s in o['top'][:5]:
            print('   ', s)
        for f in o['folds']:
            print('    fold', f['n_pass'], round(f['min_p'], 4), f['sel'][:3])
