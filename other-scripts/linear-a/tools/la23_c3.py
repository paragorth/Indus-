"""LA-23 cycle 3: massive random guessing of the Hagia Triada population (ABC).

Tens of thousands of random populations (size N, capture heterogeneity sigma, locality to the
Villa vs Casa archives, homonym rate, spelling-variant rate) are written onto the real tablet
frame (each tablet keeps its number of person-like slots and its archive). Summaries: names seen,
captures, names on 1 / 2 / 3+ tablets, names in Villa, Casa, both. Rejection ABC (nearest 1 %)
gives a posterior for N and posterior-predictive counts (new names on one more tablet; distinct
names in a k-times larger archive).
Controls: (i) pseudo-observed planted populations from the prior (coverage, error);
(ii) Linear B KN and PY: LA-sized draws with the series-derived name labels, posterior-predictive
distinct names in the FULL site archive vs the true count; (iii) HT names shuffled across tablets.
Usage: la23_c3.py [ht|lb|planted] [set=strict|clf]
"""
import sys, json
from la23_common import *

WHAT = sys.argv[1] if len(sys.argv) > 1 else 'ht'
SET = sys.argv[2] if len(sys.argv) > 2 else 'strict'
os.environ['LA23_SET'] = SET
import la23_c2 as C2

rng = np.random.default_rng(seed('la23c3' + WHAT + SET))
NSIM = int(os.environ.get('LA23_NSIM', '30000'))


def gen(theta, slots, grp, rng, extra=0):
    """theta = (N, sigma, local, hom, var). slots: int array per tablet; grp: 0/1 archive per tablet.
    Returns per-tablet arrays of name ids (list of np arrays)."""
    N, sigma, local, hom, var = theta
    N = int(N)
    w = np.exp(rng.normal(0, sigma, N))
    pA = (slots[grp == 0].sum() + 1) / (slots.sum() + 2)
    home = (rng.random(N) >= pA).astype(int)
    name = np.arange(N)
    m = rng.random(N) < hom
    name[m] = rng.integers(0, N, m.sum())
    vform = rng.random(N) < (var * 3)  # persons who have a variant form at all
    pools = []
    for g in (0, 1):
        idx = np.where(home == g)[0]
        if len(idx) == 0: idx = np.arange(N)
        c = np.cumsum(w[idx]); pools.append((idx, c / c[-1]))
    c = np.cumsum(w); pall = (np.arange(N), c / c[-1])
    out = []
    for t in range(len(slots)):
        k = int(slots[t])
        if k == 0:
            out.append(np.zeros(0, int)); continue
        nloc = rng.binomial(k, local)
        pi, pc = pools[grp[t]]
        a = pi[np.minimum(np.searchsorted(pc, rng.random(nloc)), len(pi) - 1)]
        b = np.minimum(np.searchsorted(pall[1], rng.random(k - nloc)), N - 1)
        ppl = np.concatenate([a, b])
        nm = name[ppl] + N * (vform[ppl] & (rng.random(len(ppl)) < 0.33))
        out.append(np.unique(nm))
    return out


def summ(tabs, grp):
    cnt = collections.Counter()
    A, B = set(), set()
    for t, g in zip(tabs, grp):
        for x in t:
            cnt[x] += 1
        (A if g == 0 else B).update(t.tolist() if hasattr(t, 'tolist') else t)
    v = np.array(list(cnt.values())) if cnt else np.zeros(0)
    S = len(v); n = int(v.sum())
    return np.array([S, n, (v == 1).sum(), (v == 2).sum(), (v >= 3).sum(), len(A), len(B), len(A & B)], float)


SNAMES = ['S', 'n', 'q1', 'q2', 'q3p', 'A', 'B', 'AB']


def tf(x):
    return np.log1p(x)


def prior(rng):
    return (math.exp(rng.uniform(math.log(30), math.log(20000))), rng.uniform(0, 2), rng.uniform(0, 0.95),
            rng.uniform(0, 0.3), rng.uniform(0, 0.2))


def reference(slots, grp, nsim, rng, tag):
    path = os.path.join(CK, 'c3_ref_%s.json' % tag)
    if os.path.exists(path):
        d = json.load(open(path))
        return np.array(d['th']), np.array(d['ss'])
    th, ss = [], []
    for i in range(nsim):
        t = prior(rng)
        th.append(t); ss.append(summ(gen(t, slots, grp, rng), grp))
        if i % 5000 == 4999: print(tag, 'sims', i + 1, flush=True)
    json.dump(dict(th=np.array(th).tolist(), ss=np.array(ss).tolist()), open(path, 'w'))
    return np.array(th), np.array(ss)


def abc(th, ss, obs, frac=0.01):
    X = tf(ss); o = tf(obs)
    sc = np.median(np.abs(X - np.median(X, 0)), 0) + 1e-9
    d = np.sqrt((((X - o) / sc) ** 2).sum(1))
    k = max(50, int(frac * len(d)))
    idx = np.argsort(d)[:k]
    # local-linear regression adjustment on log N (Beaumont et al. 2002)
    h = d[idx].max() + 1e-12
    wts = 1 - (d[idx] / h) ** 2
    Z = np.column_stack([np.ones(k), (X[idx] - o) / sc])
    y = np.log(th[idx, 0])
    W = np.diag(wts)
    beta = np.linalg.lstsq(Z.T @ W @ Z + 1e-6 * np.eye(Z.shape[1]), Z.T @ W @ y, rcond=None)[0]
    yadj = y - ((X[idx] - o) / sc) @ beta[1:]
    return idx, np.exp(yadj), wts, float(np.median(d[idx])), d


def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, c, x[o]))


def post_summary(th, idx, Nadj, wts):
    return dict(N_rej_med=float(np.median(th[idx, 0])), N_rej_lo=float(np.percentile(th[idx, 0], 5)),
                N_rej_hi=float(np.percentile(th[idx, 0], 95)),
                N_adj_med=wq(Nadj, wts, 0.5), N_adj_lo=wq(Nadj, wts, 0.05), N_adj_hi=wq(Nadj, wts, 0.95),
                sigma_med=float(np.median(th[idx, 1])), local_med=float(np.median(th[idx, 2])),
                hom_med=float(np.median(th[idx, 3])), var_med=float(np.median(th[idx, 4])))


def ht_frame(shuffle=False):
    S = C2.strict_rule(C2.OCC) if SET == 'strict' else {w for w, (p, _) in C2.P.items() if p >= 0.5}
    tabs = [sorted(w for w in t if w in S) for t in C2.TAB]
    grp = np.array([0 if f == 'Villa Magazine' else 1 for f in C2.DOCF])  # Villa (+ unknown) vs Casa
    grp = np.array([0 if (f == 'Villa Magazine' or f is None) else 1 for f in C2.DOCF])
    if shuffle:
        allw = [w for t in tabs for w in t]; rng.shuffle(allw)
        it = iter(allw); tabs = [sorted(set(next(it) for _ in t)) for t in tabs]
    slots = np.array([len(t) for t in tabs])
    return tabs, slots, grp


def run_ht():
    tabs, slots, grp = ht_frame()
    obs = summ([np.array(t) for t in tabs], grp)
    th, ss = reference(slots, grp, NSIM, rng, 'ht_' + SET)
    idx, Nadj, wts, dmed, d = abc(th, ss, obs)
    res = dict(obs=dict(zip(SNAMES, obs.tolist())), post=post_summary(th, idx, Nadj, wts), dmed=dmed,
               slots_total=int(slots.sum()), tablets_with=int((slots > 0).sum()))
    # posterior predictive: one new tablet with slots drawn from the non-empty slot distribution
    newn, full2, full5 = [], [], []
    nz = slots[slots > 0]
    for j in rng.choice(idx, 400):
        t = th[j]
        k = int(rng.choice(nz)); g = int(rng.integers(0, 2))
        sl = np.concatenate([slots, [k]]); gg = np.concatenate([grp, [g]])
        tb = gen(t, sl, gg, rng)
        seen = set(x for tt in tb[:-1] for x in tt.tolist())
        newn.append(len(set(tb[-1].tolist()) - seen) / max(1, len(tb[-1])) * k)
        # archive 5x larger (same structure)
        tb5 = gen(t, np.tile(slots, 5), np.tile(grp, 5), rng)
        full5.append(len(set(x for tt in tb5 for x in tt.tolist())))
    res['pp_new_names_one_tablet'] = dict(med=float(np.median(newn)), lo=float(np.percentile(newn, 5)), hi=float(np.percentile(newn, 95)),
                                          mean=float(np.mean(newn)), mean_slots=float(nz.mean()))
    res['pp_new_frac'] = float(np.sum(newn) / (len(newn) * nz.mean()))
    res['pp_distinct_5x'] = dict(med=float(np.median(full5)), lo=float(np.percentile(full5, 5)), hi=float(np.percentile(full5, 95)))
    # shuffle control
    stabs, sslots, sgrp = ht_frame(shuffle=True)
    sobs = summ([np.array(t) for t in stabs], sgrp)
    sidx, sN, sw, sd, _ = abc(th, ss, sobs)
    res['shuffled'] = dict(obs=dict(zip(SNAMES, sobs.tolist())), post=post_summary(th, sidx, sN, sw), dmed=sd)
    # how typical is the real fit? distance of obs vs distances of pseudo-observed sims to the reference
    pod = []
    for j in rng.choice(len(th), 200, replace=False):
        o2 = ss[j]; X = tf(ss); sc = np.median(np.abs(X - np.median(X, 0)), 0) + 1e-9
        dd = np.sqrt((((X - tf(o2)) / sc) ** 2).sum(1)); dd[j] = np.inf
        pod.append(np.median(np.sort(dd)[:max(50, int(0.01 * len(dd)))]))
    res['fit_pctile_real'] = float((np.array(pod) < dmed).mean())
    res['fit_pctile_shuffled'] = float((np.array(pod) < sd).mean())
    print(json.dumps(res, indent=1), flush=True)
    return res


def run_planted(npod=60):
    tabs, slots, grp = ht_frame()
    th, ss = reference(slots, grp, NSIM, rng, 'ht_' + SET)
    cov, err, rows = [], [], []
    for j in rng.choice(len(th), npod, replace=False):
        mask = np.ones(len(th), bool); mask[j] = False
        idx, Nadj, wts, _, _ = abc(th[mask], ss[mask], ss[j])
        p = post_summary(th[mask], idx, Nadj, wts)
        N = th[j, 0]
        cov.append(p['N_adj_lo'] <= N <= p['N_adj_hi']); err.append(math.log(p['N_adj_med'] / N))
        rows.append((N, p['N_adj_med'], p['N_adj_lo'], p['N_adj_hi'], th[j, 1], th[j, 2], th[j, 3]))
    rows = np.array(rows)
    res = dict(coverage90=float(np.mean(cov)), median_abs_log_err=float(np.median(np.abs(err))),
               bias_log=float(np.median(err)), rows=rows.tolist())
    # by N band
    for lo, hi in ((30, 300), (300, 1000), (1000, 3000), (3000, 20000)):
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if m.sum():
            res['band_%d_%d' % (lo, hi)] = dict(n=int(m.sum()), cov=float(np.mean(np.array(cov)[m])),
                                                 mae=float(np.median(np.abs(np.array(err)[m]))))
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1), flush=True)
    return res


def run_lb(ndraw=3):
    LB = lb_docs(); L = lb_name_label(); target = len(C2.OCC)
    out = {}
    for site in ('KN', 'PY'):
        docs = [d for d in LB if d['site'] == site and any(k == 'W' for ln in d['lines'] for k, _ in ln)]
        fslots = np.array([len(set(v for ln in d['lines'] for k, v in ln if k == 'W' and v in L)) for d in docs])
        truth = len(set(v for d in docs for ln in d['lines'] for k, v in ln if k == 'W' and v in L))
        hands = collections.Counter(d['scribe'] for d in docs).most_common(1)[0][0]
        fgrp = np.array([0 if d['scribe'] == hands else 1 for d in docs])
        res = []
        for r in range(ndraw):
            idx = rng.permutation(len(docs)); sel, n = [], 0
            for i in idx:
                d = docs[i]; sel.append(i); n += sum(1 for ln in d['lines'] for t, _ in ln if t == 'W')
                if n >= target: break
            sel = np.array(sel)
            tabs = [np.array(sorted(set(v for ln in docs[i]['lines'] for k, v in ln if k == 'W' and v in L))) for i in sel]
            slots = fslots[sel]; grp = fgrp[sel]
            obs = summ(tabs, grp)
            th, ss = reference(slots, grp, max(6000, NSIM // 3), rng, 'lb_%s_%d' % (site, r))
            ii, Nadj, wts, dmed, _ = abc(th, ss, obs)
            p = post_summary(th, ii, Nadj, wts)
            # posterior predictive of distinct names in the full site archive
            pp = []
            for j in rng.choice(ii, 150):
                tb = gen(th[j], fslots, fgrp, rng)
                pp.append(len(set(x for t in tb for x in t.tolist())))
            p['pp_full_archive'] = dict(med=float(np.median(pp)), lo=float(np.percentile(pp, 5)), hi=float(np.percentile(pp, 95)))
            p['truth_full_archive'] = truth; p['obs'] = dict(zip(SNAMES, obs.tolist()))
            p['covered'] = p['pp_full_archive']['lo'] <= truth <= p['pp_full_archive']['hi']
            print(site, r, json.dumps(p), flush=True)
            res.append(p)
        out[site] = res
    return out


if __name__ == '__main__':
    fn = os.path.join(CK, 'c3_%s_%s.json' % (WHAT, SET))
    r = {'ht': run_ht, 'planted': run_planted, 'lb': run_lb}[WHAT]()
    json.dump(r, open(fn, 'w'), indent=1)
