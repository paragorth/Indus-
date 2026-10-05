"""pe48 cycle 1: calibrate the trade-network model on Ur III provinces and Linear B sites.

For each control corpus and draw: a PE-shaped sample (hub 1,502 docs, outposts 27/22/12/11/1/1/1) or a
BALANCED sample (hub 1,502, outposts 200 each); plant PLANT (30% of docs at outpost 1, 0.3% at hub) and
ROUTE (20% of docs at outposts 1 and 2); random theta search with 5-fold site-stratified CV; the same
search on a site-label-shuffled copy (null). Survivors = top 10% theta by outpost gain.
Read-outs: best outpost gain real vs null; travel LLR AUC gold T (commodities, titles) vs gold N
(personal names) and M (month names), raw and frequency-stratified; frequency-only AUC; planted recovery.
"""
import sys, json, os, random
import numpy as np
from multiprocessing import Pool
import pe48_lib as L

N_THETA = int(os.environ.get('N_THETA', 200))

JOBS = []
for corpus, hub, outs in (('UR3', 'Umma', ['Puzriš-Dagan', 'Girsu', 'Ur', 'Nippur', 'Garšana', 'Irisagrig']),
                          ('UR3', 'Girsu', ['Umma', 'Puzriš-Dagan', 'Nippur', 'Ur', 'Garšana', 'Irisagrig']),
                          ('LB', 'KN', ['PY', 'TH', 'MY', 'TI', 'KH'])):
    for shape in ('PE', 'BAL'):
        for draw in range(3 if shape == 'PE' else 1):
            JOBS.append((corpus, hub, tuple(outs), shape, draw))


def run(job):
    corpus, hub, outs, shape, draw = job
    fn = os.path.join(L.CK, 'c1_%s_%s_%s_%d.json' % (corpus, hub, shape, draw))
    if os.path.exists(fn):
        return json.load(open(fn))
    src = L.ur3_docs() if corpus == 'UR3' else L.lb_docs()
    rng = random.Random(1000 + draw)
    sh = L.PE_SHAPE if shape == 'PE' else [200] * len(outs)
    docs = L.pe_shaped(src['docs'], hub, list(outs), rng, shape=sh)
    docs = L.plant(docs, {outs[0]: 0.3, hub: 0.003}, 'PLANT', rng)
    docs = L.plant(docs, {outs[0]: 0.2, outs[1]: 0.2}, 'ROUTE', rng)
    dat = L.Data(docs, corpus)
    res = L.search(dat, N_THETA, seed=draw)
    # null: site labels shuffled among docs
    lab = [d['site'] for d in docs]; rng.shuffle(lab)
    nd = [dict(d, site=s) for d, s in zip(docs, lab)]
    ndat = L.Data(nd, corpus)
    nres = L.search(ndat, N_THETA, seed=draw + 77)
    res.sort(key=lambda r: -r[2])
    surv = [r[0] for r in res[:max(5, N_THETA // 10)]]
    P, llr = L.travel_posterior(dat, surv)
    gold = src['gold']
    vi = {t: i for i, t in enumerate(dat.vocab)}
    hubdf = dat.Y[dat.site == 0].mean(0)
    isT = np.array([gold.get(t) == 'T' for t in dat.vocab])
    isN = np.array([gold.get(t) == 'N' for t in dat.vocab])
    isM = np.array([gold.get(t) == 'M' for t in dat.vocab])
    # outpost-observed signs only (a sign never seen off the hub cannot show travel)
    seen_out = dat.Y[dat.outpost].any(0)
    out = dict(job=job, n_vocab=len(dat.vocab), nT=int(isT.sum()), nN=int(isN.sum()), nM=int(isM.sum()),
               best_out=res[0][2], best_all=max(r[1] for r in res),
               null_best_out=max(r[2] for r in nres), null_best_all=max(r[1] for r in nres),
               top_theta=res[:5],
               auc_TN=L.auc(llr[isT], llr[isN]), auc_TM=L.auc(llr[isT], llr[isM]),
               sauc_TN=L.strat_auc(llr, hubdf, isT, isN), sauc_TM=L.strat_auc(llr, hubdf, isT, isM),
               fauc_TN=L.auc(hubdf[isT], hubdf[isN]),
               pT_T=float(P[isT, 0].mean()) if isT.any() else None,
               pT_N=float(P[isN, 0].mean()) if isN.any() else None,
               pT_M=float(P[isM, 0].mean()) if isM.any() else None,
               plant=dict(P_home_out1=float(P[vi['PLANT'], 1 + dat.sites.index(outs[0])]) if 'PLANT' in vi else None,
                          P_T=float(P[vi['PLANT'], 0]) if 'PLANT' in vi else None,
                          route_P_home_out12=float(P[vi['ROUTE'], [1 + dat.sites.index(o) for o in outs[:2]]].sum()) if 'ROUTE' in vi else None),
               homed_out_signs=[(dat.vocab[s], dat.sites[int(P[s, 1:].argmax())], float(P[s, 1:].max()), gold.get(dat.vocab[s]))
                                for s in np.argsort(-P[:, 2:].max(1))[:15]],
               # among signs seen at an outpost:
               sauc_TN_seen=L.strat_auc(llr, hubdf, isT & seen_out, isN & seen_out))
    json.dump(out, open(fn, 'w'), default=float)
    return out


if __name__ == '__main__':
    with Pool(2) as p:
        outs = p.map(run, JOBS, chunksize=1)
    for o in outs:
        print(json.dumps({k: o[k] for k in ('job', 'n_vocab', 'nT', 'nN', 'nM', 'best_out', 'null_best_out',
                                             'auc_TN', 'sauc_TN', 'sauc_TM', 'fauc_TN', 'sauc_TN_seen',
                                             'pT_T', 'pT_N', 'pT_M', 'plant')}, default=float))
    json.dump(outs, open(os.path.join(L.CK, 'cycle1_all.json'), 'w'), default=float)
