"""pe48 cycle 2: the trade-network model on Proto-Elamite.

Real search: N_REAL random theta, 5-fold site-stratified CV. Nulls: NNULL site-label shuffles and NNULL
frequency resamples (each tablet's tokens redrawn from the corpus token distribution, length and site
kept), N_NULL_TH theta each. Plant: PLANT in 30% of Yahya tablets and 0.3% of Susa, ROUTE in 20% of
Yahya and Malyan tablets, same search. Survivors (top 10% by outpost gain) -> travel posterior.
Syntax test: does a sign's travel LLR depend on its slot (SOLO numbered entry / STRING of 3+ signs /
HEAD line), within frequency bins? Permutation null: slot labels permuted within frequency bins.
"""
import sys, json, os, random
import numpy as np
from multiprocessing import Pool
from collections import Counter
import pe48_lib as L

N_REAL = int(os.environ.get('N_REAL', 2000))
N_NULL_TH = int(os.environ.get('N_NULL_TH', 200))
NNULL = int(os.environ.get('NNULL', 8))


def docs_variant(kind, k):
    docs = L.pe_docs()
    rng = random.Random(500 + k)
    if kind == 'real':
        return docs
    if kind == 'plant':
        docs = L.plant(docs, {'Yahya': 0.3, 'Susa': 0.003}, 'PLANT', rng)
        return L.plant(docs, {'Yahya': 0.2, 'Malyan': 0.2}, 'ROUTE', rng)
    if kind == 'siteshuf':
        lab = [d['site'] for d in docs]; rng.shuffle(lab)
        return [dict(d, site=s) for d, s in zip(docs, lab)]
    if kind == 'freqres':
        pool = [t for d in docs for t in d['toks']]
        return [dict(d, toks=[rng.choice(pool) for _ in d['toks']]) for d in docs]


def job(args):
    kind, k, n, seed = args
    fn = os.path.join(L.CK, 'c2_%s_%d_%d.json' % (kind, k, seed))
    if os.path.exists(fn):
        return json.load(open(fn))
    dat = L.Data(docs_variant(kind, k), 'PE', max_vocab=2000)
    res = L.search(dat, n, seed=seed)
    out = dict(kind=kind, k=k, seed=seed, res=res)
    json.dump(out, open(fn, 'w'))
    return out


if __name__ == '__main__':
    jobs = [('real', 0, N_REAL // 4, s) for s in range(4)]
    jobs += [('plant', 0, 400, 11)]
    jobs += [('siteshuf', k, N_NULL_TH, 100 + k) for k in range(NNULL)]
    jobs += [('freqres', k, N_NULL_TH, 200 + k) for k in range(NNULL)]
    with Pool(2) as p:
        outs = p.map(job, jobs, chunksize=1)
    # NB: real folds differ by seed (4 fold draws); compare per-seed best-of-N_NULL_TH with nulls
    summ = {}
    real = [r for o in outs if o['kind'] == 'real' for r in o['res']]
    real_first = [max(r[2] for r in o['res'][:N_NULL_TH]) for o in outs if o['kind'] == 'real']
    real_first_all = [max(r[1] for r in o['res'][:N_NULL_TH]) for o in outs if o['kind'] == 'real']
    for kind in ('siteshuf', 'freqres'):
        summ[kind] = dict(best_out=[max(r[2] for r in o['res']) for o in outs if o['kind'] == kind],
                          best_all=[max(r[1] for r in o['res']) for o in outs if o['kind'] == kind])
    summ['real'] = dict(best_out_per_seed=real_first, best_all_per_seed=real_first_all,
                        best_out=max(r[2] for r in real), best_all=max(r[1] for r in real),
                        frac_theta_out_gain_pos=float(np.mean([r[2] > 0 for r in real])))
    # survivors
    real.sort(key=lambda r: -r[2])
    surv = [r[0] for r in real[:len(real) // 10]]
    json.dump(surv, open(os.path.join(L.CK, 'c2_survivors.json'), 'w'))
    summ['surv_params'] = dict(
        L_med=float(np.median([t['L'] for t in surv])), eps_med=float(np.median([t['eps'] for t in surv])),
        b_med=float(np.median([t['b'] for t in surv])), pen_med=float(np.median([t['pen'] for t in surv])),
        route_frac=float(np.mean([t['metric'] == 'route' for t in surv])),
        all_L_med=float(np.median([r[0]['L'] for r in real])), all_route_frac=float(np.mean([r[0]['metric'] == 'route' for r in real])))
    dat = L.Data(L.pe_docs(), 'PE', max_vocab=2000)
    P, llr = L.travel_posterior(dat, surv)
    # plant recovery with survivors of the plant search
    po = [o for o in outs if o['kind'] == 'plant'][0]['res']; po.sort(key=lambda r: -r[2])
    pdat = L.Data(docs_variant('plant', 0), 'PE', max_vocab=2000)
    pP, _ = L.travel_posterior(pdat, [r[0] for r in po[:40]])
    vi = {t: i for i, t in enumerate(pdat.vocab)}
    summ['plant'] = dict(PLANT_home=dict(zip(['T'] + pdat.sites, map(float, pP[vi['PLANT']]))),
                         ROUTE_home=dict(zip(['T'] + pdat.sites, map(float, pP[vi['ROUTE']]))),
                         plant_best_out=po[0][2])
    # sign table
    hubdf = dat.Y[dat.site == 0].mean(0)
    roles = Counter(); tot = Counter()
    for d in dat.docs:
        for s, r in d['roles']:
            roles[(s, r)] += 1; tot[s] += 1
    vi = {t: i for i, t in enumerate(dat.vocab)}
    table = []
    for s, t in enumerate(dat.vocab):
        nout = int(dat.Y[dat.outpost, s].sum())
        table.append(dict(sign=t, hub_df=float(hubdf[s]), out_docs=nout, llr=float(llr[s]),
                          P=dict(zip(['T'] + dat.sites, map(float, P[s]))),
                          solo=roles[(t, 'SOLO')] / tot[t], string=roles[(t, 'STRING')] / tot[t],
                          head=roles[(t, 'HEAD')] / tot[t], n=tot[t]))
    json.dump(table, open(os.path.join(L.CK, 'c2_signs.json'), 'w'))
    # syntax test
    sol = np.array([x['solo'] for x in table]); stg = np.array([x['string'] for x in table])
    hd = np.array([x['head'] for x in table]); n = np.array([x['n'] for x in table])
    ok = n >= 10
    isS = ok & (sol >= 0.5); isG = ok & (stg >= 0.5); isH = ok & (hd >= 0.3)
    obs = L.strat_auc(llr, hubdf, isS, isG)
    rng = np.random.default_rng(0)
    # permutation within frequency quintiles
    q = np.quantile(hubdf[ok], np.linspace(0, 1, 6)); b = np.clip(np.searchsorted(q, hubdf, side='right') - 1, 0, 4)
    nulls = []
    lab = np.where(isS, 1, np.where(isG, 2, 0))
    for _ in range(2000):
        l2 = lab.copy()
        for k in range(5):
            idx = np.where(ok & (b == k) & (lab > 0))[0]
            l2[idx] = rng.permutation(l2[idx])
        nulls.append(L.strat_auc(llr, hubdf, l2 == 1, l2 == 2))
    nulls = np.array([x for x in nulls if x == x])
    summ['syntax'] = dict(n_solo=int(isS.sum()), n_string=int(isG.sum()), n_head=int(isH.sum()),
                          sauc_solo_vs_string=obs, p_two_sided=float(np.mean(np.abs(nulls - 0.5) >= abs(obs - 0.5))),
                          sauc_head_vs_string=L.strat_auc(llr, hubdf, isH, isG),
                          mean_PT_solo=float(P[isS, 0].mean()), mean_PT_string=float(P[isG, 0].mean()),
                          mean_PT_head=float(P[isH, 0].mean()))
    json.dump(summ, open(os.path.join(L.CK, 'c2_summary.json'), 'w'), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
