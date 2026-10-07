"""la73 cycle 2: fit the Hagia Triada archive (rd; read-only as check). Controls: token-shuffled HT
(5 shuffles), prior-only posterior, posterior predictive on held-out statistics, held-out sites
(Khania, Zakros, Phaistos) predicted by HT-fitted worlds written at each site's tablet sizes."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la73_abc as A, la73_common as C, la73_prep as P
import numpy as np

C.set_version(2)
KEYS = ['b', 'logTau', 'logH', 'logR', 'R_eff', 'logNp', 'sig', 'phi', 'logK', 'loc', 'kappa', 'logm', 'log_persons_ever',
        'log_surv_frac', 'log_tablets_ever', 'log_distinct_persons_seen']

def fmt(o):
    return {k: [round(x, 3) for x in o[k]] for k in KEYS}

def width_ratio(o, th, d):
    Y = np.hstack([th, np.log10(np.maximum(d[:, 1:], 1e-9)), d[:, :1]])
    names = C.PNAMES + ['log_' + n for n in A.DER[1:]] + ['R_eff']
    out = {}
    for k in KEYS:
        j = names.index(k); q = np.quantile(Y[:, j], [0.1, 0.5, 0.9])
        out[k] = dict(width_ratio=round((o[k][2] - o[k][1]) / (q[2] - q[0]), 3),
                      shift_in_prior_sd=round((o[k][0] - q[1]) / Y[:, j].std(), 3))
    return out

def ppc(o, th, lengths, s_obs, rng, n=300):
    idx = o['_idx']; w = o['_w'] / o['_w'].sum()
    pick = rng.choice(idx, n, p=w); S = []
    for i in pick:
        dd, _ = C.simulate(th[i], lengths, rng); S.append(C.stats(dd, rng))
    S = np.array(S); out = {}
    for k in C.HELD + C.TRAIN:
        j = C.SNAMES.index(k); out[k] = round(float(np.mean(S[:, j] < s_obs[j]) + 0.5 * np.mean(S[:, j] == s_obs[j])), 3)
    return out, S

if __name__ == '__main__':
    rng = np.random.default_rng(7302); res = {}
    # (2a) recalibrate the bursty simulator on Ur III dated archives before touching HT
    import la73_c1 as c1
    from scipy.stats import spearmanr
    dr = c1.ur3_draws(np.random.default_rng(73)) + c1.ur3_draws(np.random.default_rng(74))
    pool = [np.array([len(x) for x in g['docs']]) for g in dr]
    uth, us, ud = A.table('v2_ur3', 40000, pool, seed=22)
    w = np.log10([g['win'] for g in dr]); site = np.array([g['site'] for g in dr]); cal = {}
    for use_name, use in (('train', C.TRAIN), ('all', C.SNAMES)):
        est = []; sh = []
        for g in dr:
            est.append(A.abc(C.stats(C.to_int_docs(g['docs'])), uth, us, ud, use=use)['R_eff'][0])
            flat = [x for d0 in g['docs'] for x in d0]; rng.shuffle(flat); it = iter(flat)
            sh.append(A.abc(C.stats(C.to_int_docs([[next(it) for _ in d0] for d0 in g['docs']])), uth, us, ud, use=use)['R_eff'][0])
        est = np.array(est); sh = np.array(sh)
        per = {s0: [float(x) for x in spearmanr(w[site == s0], est[site == s0])] for s0 in set(site)}
        cal[use_name] = dict(rho=[float(x) for x in spearmanr(w, est)], rho_shuf=[float(x) for x in spearmanr(w, sh)], per_site=per,
                             by_window={int(k): round(float(np.mean(est[w == np.log10(k)])), 3) for k in (3, 12, 36, 120, 480)})
        print('ur3 v2', use_name, cal[use_name], flush=True)
    res['ur3_v2'] = cal
    # planted v2
    pth, ps, pdd = A.table('v2_planted', 300, L_HT := np.array([len(x['w']) for x in P.la_docs('Haghia Triada')]), seed=98, chunks=6)
    D = P.la_docs('Haghia Triada'); L = np.array([len(x['w']) for x in D])
    th, s, d = A.table('v2_ht', 60000, L, seed=21)
    pr = []
    for i in range(len(ps)):
        o = A.abc(ps[i], th, s, d); pr.append((pdd[i, 0], o['R_eff'][0], np.log10(pdd[i, 1]), o['log_persons_ever'][0]))
    pr = np.array(pr)
    res['planted_v2'] = dict(R_eff_r=float(np.corrcoef(pr[:, 0], pr[:, 1])[0, 1]), persons_r=float(np.corrcoef(pr[:, 2], pr[:, 3])[0, 1]))
    print('planted v2', res['planted_v2'], flush=True)
    so = C.stats(C.to_int_docs([x['w'] for x in D]))
    res['ht_stats'] = dict(zip(C.SNAMES, so.round(4)))
    o = A.abc(so, th, s, d); res['ht_rd'] = fmt(o); res['ht_vs_prior'] = width_ratio(o, th, d)
    print('HT rd', res['ht_rd'], flush=True); print('vs prior', res['ht_vs_prior'], flush=True)
    # held-out statistics: posterior vs prior predictive
    pp, _ = ppc(o, th, L, so, rng)
    prior_o = dict(_idx=np.arange(len(th)), _w=np.ones(len(th)))
    pr, _ = ppc(prior_o, th, L, so, rng)
    res['ppc_post'] = pp; res['ppc_prior'] = pr
    print('ppc post', pp, '\nppc prior', pr, flush=True)
    # read-only check (own table: different lengths)
    Dr = P.la_docs('Haghia Triada', 'read'); Lr = np.array([len(x['w']) for x in Dr])
    thr, sr, dr = A.table('v2_ht_read', 30000, Lr, seed=23)
    sor = C.stats(C.to_int_docs([x['w'] for x in Dr]))
    res['ht_read'] = fmt(A.abc(sor, thr, sr, dr)); print('HT read', res['ht_read'], flush=True)
    # shuffled HT (tokens across tablets, lengths kept)
    flat = [w for x in D for w in x['w']]; sh = []
    for r in range(5):
        rng.shuffle(flat); it = iter(flat)
        docs = [[next(it) for _ in x['w']] for x in D]
        sh.append(fmt(A.abc(C.stats(C.to_int_docs(docs)), th, s, d)))
        print('shuffle', r, {k: sh[-1][k] for k in ('R_eff', 'logNp', 'phi', 'loc', 'logK')}, flush=True)
    res['ht_shuffled'] = sh
    # held-out sites: HT posterior written at each site's sizes vs prior predictive
    hs = {}
    for site in ('Khania', 'Zakros', 'Phaistos'):
        Ds = P.la_docs(site); Ls = np.array([len(x['w']) for x in Ds])
        ss = C.stats(C.to_int_docs([x['w'] for x in Ds]))
        a, Sa = ppc(o, th, Ls, ss, rng, 300); b, Sb = ppc(prior_o, th, Ls, ss, rng, 300)
        # predictive log density score on the 5 robust stats via per-stat KDE-free z
        def score(S):
            z = []
            for k in ('ttr', 'f1', 'f2', 'linked', 'rep_in'):
                j = C.SNAMES.index(k); z.append(abs(ss[j] - S[:, j].mean()) / (S[:, j].std() + 1e-9))
            return float(np.mean(z))
        tb = [len(Ds), int(Ls.sum())]
        sth, sst, sd = A.table('v2_site_' + site[:2], 15000, Ls, seed=40 + len(site))
        own = fmt(A.abc(ss, sth, sst, sd))
        hs[site] = dict(n=tb, ppc_post=a, ppc_prior=b, meanz_post=round(score(Sa), 3), meanz_prior=round(score(Sb), 3),
                        own_fit={k: own[k] for k in ('R_eff', 'logNp', 'phi', 'loc')})
        print(site, hs[site], flush=True)
    res['sites'] = hs
    json.dump(res, open(os.path.join(A.CK, 'c2.json'), 'w'), indent=1, default=float)
