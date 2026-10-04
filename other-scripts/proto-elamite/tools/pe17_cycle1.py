"""pe17 cycle 1: blind plateau-likeness ranking of Susa tablets, with controls, then FREEZE.

Written and run BEFORE the hXRF provenance table (Yeganeh et al. 2025) was downloaded or read.
Controls:
  N  shuffled site labels (labels permuted within length bins), whole pipeline incl. selection
  P1 hidden real imports: 10 Yahya + 6 Malyan tablets relabelled 'Susa' and withheld from training
  P2 synthetic fingerprint: 3 random rare features switched on in half the plateau tablets and in 15 Susa tablets
Selection: top 10% of hypotheses by mean leave-site-out AUC (Malyan, Yahya); re-test on 'Other' plateau sites.
"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe17_common import load, build, length_bin, auc, sha, DATA, CKPT
from pe17_engine import ensemble, group_of

H_REAL = int(os.environ.get('H_REAL', 3000))
H_CTRL = int(os.environ.get('H_CTRL', 600))
N_NULL = int(os.environ.get('N_NULL', 20))

tabs = load()
X, names, fam = build(tabs)
site = np.array([t['site'] for t in tabs])
grp = np.array([group_of(s) for s in site])
lbin = np.array([length_bin(t) for t in tabs])
ids = [t['id'] for t in tabs]


def summarise(E, grp_eval, extra_pos=None):
    s = E['surv']
    V = E['VAL']
    r = dict(val_all=np.nanmean(V, 0).tolist(), val_surv=np.nanmean(V[s], 0).tolist())
    score = np.nanmean(E['SS'][s], 0)
    if extra_pos is not None:
        sus = np.where(grp_eval == 'Susa')[0]
        pos = np.intersect1d(sus, extra_pos)
        neg = np.setdiff1d(sus, extra_pos)
        r['plant_auc'] = auc(score[pos], score[neg])
        r['plant_auc_allhyp'] = auc(np.nanmean(E['SS'], 0)[pos], np.nanmean(E['SS'], 0)[neg])
    return r, score


def job(args):
    kind, rep = args
    rng = np.random.default_rng(1000 + rep)
    if kind == 'null':
        g = grp.copy()
        for b in range(6):
            ix = np.where(lbin == b)[0]
            g[ix] = g[rng.permutation(ix)]
        E = ensemble(X, fam, g, lbin, H_CTRL, 50 + rep)
        return kind, rep, summarise(E, g)[0]
    if kind == 'P1':
        g = grp.copy()
        y = rng.choice(np.where(grp == 'Yahya')[0], 10, replace=False)
        m = rng.choice(np.where(grp == 'Malyan')[0], 6, replace=False)
        hid = np.r_[y, m]
        g[hid] = 'Susa'
        E = ensemble(X, fam, g, lbin, H_CTRL, 70 + rep)
        return kind, rep, summarise(E, g, hid)[0]
    if kind == 'P2':
        Xp = X.copy()
        rare = np.where((fam != 'FMT') & (X.sum(0) >= 5) & (X.sum(0) <= 30))[0]
        f3 = rng.choice(rare, 3, replace=False)
        plat = np.where(grp != 'Susa')[0]
        pl = rng.choice(plat, len(plat) // 2, replace=False)
        sus = np.where((grp == 'Susa') & (lbin >= 2))[0]
        planted = rng.choice(sus, 15, replace=False)
        for f in f3:
            Xp[np.r_[pl, planted], f] = 1.0
        E = ensemble(Xp, fam, grp, lbin, H_CTRL, 90 + rep)
        r = summarise(E, grp, planted)[0]
        r['features'] = [names[f] for f in f3]
        return kind, rep, r


if __name__ == '__main__':
    t0 = time.time()
    print('tablets', len(tabs), 'features', X.shape[1], {g: int((grp == g).sum()) for g in set(grp)}, flush=True)
    jobs = [('real', 0)] + [('P1', r) for r in range(5)] + [('P2', r) for r in range(5)] + [('null', r) for r in range(N_NULL)]
    res = {}
    with Pool(2) as pool:
        Er = pool.apply_async(ensemble, (X, fam, grp, lbin, H_REAL, 17))
        ctrl = pool.map(job, jobs[1:], chunksize=1)
        E = Er.get()
    for kind, rep, r in ctrl:
        res.setdefault(kind, []).append(r)
        print(kind, rep, {k: (np.round(v, 3).tolist() if not isinstance(v, list) or not isinstance(v[0], str) else v) for k, v in r.items()}, flush=True)
    real, score = summarise(E, grp)
    res['real'] = real
    np.savez_compressed(os.path.join(CKPT, 'cycle1_real.npz'), VAL=E['VAL'], SS=E['SS'], PL=E['PL'], surv=E['surv'])
    # family carriers: survivors' family membership vs all
    fams = ['SIGN', 'VAR', 'NUM', 'FMT', 'BIG']
    fam_rate = {f: [float(np.mean([f in h['fams'] for h in E['hyps']])),
                    float(np.mean([f in E['hyps'][i]['fams'] for i in E['surv']]))] for f in fams}
    kind_rate = {k: [float(np.mean([h['kind'] == k for h in E['hyps']])),
                     float(np.mean([E['hyps'][i]['kind'] == k for i in E['surv']]))] for k in ['cent', 'nb', 'lr2', 'lr1']}
    # feature weight: how often each feature appears in survivors vs all
    cnt_all = np.zeros(X.shape[1]); cnt_s = np.zeros(X.shape[1])
    for i, h in enumerate(E['hyps']):
        cnt_all[h['cols']] += 1
        if i in set(E['surv'].tolist()):
            cnt_s[h['cols']] += 1
    enr = (cnt_s + 1) / (cnt_all * len(E['surv']) / len(E['hyps']) + 1)
    topf = [(names[j], round(float(enr[j]), 2), int(cnt_s[j])) for j in np.argsort(-enr)[:25] if cnt_s[j] >= 5]
    # plateau tablets: Susa-likeness (low plateau score out of their group) from survivors
    plat_sc = np.nanmean(E['PL'][E['surv']], 0)
    # Malyan vs Yahya: leave-one-out origin prediction for plateau tablets (random centroid / NB hypotheses)
    my = np.where(np.isin(grp, ['Malyan', 'Yahya']))[0]
    rng2 = np.random.default_rng(171)
    from pe17_engine import draw_hyp, fit_score
    MY = []
    yv = (grp[my] == 'Yahya').astype(float)
    accs = []
    for k in range(2000):
        h = draw_hyp(rng2, fam)
        kind = 'cent' if h['kind'] in ('cent', 'lr2') else 'nb'
        Xh = X[np.ix_(my, h['cols'])]
        sc = np.zeros(len(my))
        for i in range(len(my)):
            tr = np.setdiff1d(np.arange(len(my)), [i])
            Y = tr[yv[tr] == 1]; M = tr[yv[tr] == 0]
            sc[i] = fit_score(kind, 1.0, Xh[Y], Xh[M], np.ones(len(M)), Xh[[i]])[0]
        MY.append(sc); accs.append(auc(sc[yv == 1], sc[yv == 0]))
    MY = np.array(MY); accs = np.array(accs)
    s_my = np.argsort(-accs)[:200]
    yscore = MY[s_my].mean(0)
    res['malyan_vs_yahya'] = dict(auc_all=float(np.nanmean(accs)), auc_top200_insample=float(np.nanmean(accs[s_my])),
                                  auc_ensemble_all=float(auc(MY.mean(0)[yv == 1], MY.mean(0)[yv == 0])))
    yahya_like = dict(zip([ids[i] for i in my], np.round(MY.mean(0), 4).tolist()))
    # frozen ranking
    sus = np.where(grp == 'Susa')[0]
    order = sus[np.argsort(-score[sus])]
    # length-residualised percentile: rank within length bin
    resid = np.full(len(X), np.nan)
    for b in range(6):
        ix = sus[lbin[sus] == b]
        if len(ix):
            resid[ix] = (np.argsort(np.argsort(score[ix])) + 0.5) / len(ix)
    frozen = dict(
        note='pe17 blind ranking of Susa tablets by plateau-likeness; frozen before the hXRF table was read',
        primary='plateau_score (higher = more plateau-like); secondary = within_length_bin_percentile',
        susa=[dict(id=ids[i], des=tabs[i]['des'], rank=r + 1, plateau_score=round(float(score[i]), 4),
                   within_length_bin_percentile=round(float(resid[i]), 4), lines=len(tabs[i]['lines']))
              for r, i in enumerate(order)],
        plateau=[dict(id=ids[i], des=tabs[i]['des'], site=site[i], plateau_score_out_of_group=round(float(plat_sc[i]), 4),
                      yahya_vs_malyan_loo=yahya_like.get(ids[i]))
                 for i in np.where(grp != 'Susa')[0]])
    h = sha(frozen)
    frozen['sha256_of_content_without_this_field'] = h
    json.dump(frozen, open(os.path.join(DATA, 'pe17_frozen_ranking.json'), 'w'), indent=1)
    out = dict(res=res, fam_rate=fam_rate, kind_rate=kind_rate, top_features=topf, sha256=h,
               n_features=int(X.shape[1]), H_REAL=H_REAL, H_CTRL=H_CTRL, secs=time.time() - t0)
    json.dump(out, open(os.path.join(DATA, 'pe17_cycle1.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ['fam_rate', 'kind_rate', 'top_features', 'sha256']}, default=float))
    print('REAL', real)
    print('top 20 Susa:', [(tabs[i]['id'], tabs[i]['des'], round(float(score[i]), 2)) for i in order[:20]])
    print('secs', time.time() - t0)
