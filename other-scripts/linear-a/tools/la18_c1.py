#!/usr/bin/env python3
"""LA-18 cycle 1: pairwise anti-co-occurrence of form-related word pairs.
usage: la18_c1.py lb | la     (two workers)
"""
import sys, json, random, collections, time
import numpy as np
from la18_common import *

V_LB = {}  # LB sign values used ONLY to label the positive-control truth set


def cv(s):
    s = s.rstrip('0123456789')
    if s in ('A', 'E', 'I', 'O', 'U'): return '', s
    return s[:-1], s[-1]


EUS = {'U', 'WE', 'WO', 'WI'}
SUFTRUE = {'JO', 'I', 'DE', 'QE', 'SI', 'O', 'PI'}


def lb_truth(a, b, fam):
    """True if the LB pair is a known inflectional / clitic alternation of one lemma (Greek morphology)."""
    if fam == 'SUF1':
        add = b[-1]
        if add in SUFTRUE: return True
        return False
    if fam == 'FIN':
        x, y = a[-1], b[-1]
        if x in EUS and y in EUS: return True
        (cx, vx), (cy, vy) = cv(x), cv(y)
        if cx == cy and {vx, vy} <= {'O', 'A', 'I', 'E'} and len({vx, vy}) == 2: return True
        return False
    return False


def analyse(S, site, nsamp, seed, label, pairs=None, truthf=None, extra_rnd=1):
    df = collections.Counter(w for s in S for w in s)
    types = sorted(df)
    P = pairs if pairs is not None else form_pairs(types)
    P = {k: v for k, v in P.items() if df.get(k[0], 0) and df.get(k[1], 0)}
    R = random_pairs(types, df, P, extra_rnd, seed + 7)
    allp = dict(P); allp.update(R)
    keys = list(allp)
    t0 = time.time()
    obs, N = null_matrix(S, site, keys, nsamp, seed)
    p_obs, p_null = lower_p(obs, N)
    E = N.mean(0); p0 = (N == 0).mean(0)
    fam = np.array([allp[k] for k in keys])
    res = dict(label=label, ndocs=len(S), ntypes=len(types), nsamp=nsamp, secs=round(time.time() - t0, 1), fam={})
    for f in ['SUF1', 'PRE1', 'FIN', 'INI', 'RND', 'PLANT']:
        idx = np.where(fam == f)[0]
        if len(idx): res['fam'][f] = family_excess(obs, N, idx)
    form = np.isin(fam, ['SUF1', 'PRE1', 'FIN', 'INI'])
    if truthf:
        tr = np.array([bool(truthf(k[0], k[1], allp[k])) and form[i] for i, k in enumerate(keys)])
        res['fam']['TRUE'] = family_excess(obs, N, np.where(tr)[0])
        res['fam']['FORM_OTHER'] = family_excess(obs, N, np.where(form & ~tr)[0])
        # testable = null P(O=0) < 0.5 (some power)
        tm = form & (p0 < 0.5)
        if tr[tm].sum() and (~tr[tm]).sum():
            a, b = p_obs[tm & tr], p_obs[tm & ~tr]
            auc = (np.sum(a[:, None] < b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (len(a) * len(b))
            res['auc_true_lower_p'] = round(float(auc), 3); res['n_testable_true'] = int(tr[tm].sum())
            res['n_testable_other'] = int((~tr[tm]).sum())
    # pair-level: testable form pairs; FWER by max-stat (min p) over them, BH
    tm = form & (p0 < 0.5)
    ix = np.where(tm)[0]
    fw, mn = fwer(p_obs, p_null, tm)
    q = bh(p_obs[ix])
    order = np.argsort(p_obs[ix])
    top = []
    for r in order[:25]:
        j = ix[r]; k = keys[j]
        top.append(dict(a='-'.join(k[0]), b='-'.join(k[1]), fam=allp[k], dfa=df[k[0]], dfb=df[k[1]], O=int(obs[j]),
                        E=round(float(E[j]), 2), p=round(float(p_obs[j]), 4), fwer=round(float(fw[r]), 3),
                        q=round(float(q[r]), 3), truth=(bool(truthf(k[0], k[1], allp[k])) if truthf else None)))
    res['n_testable'] = int(len(ix)); res['n_fwer05'] = int((fw < 0.05).sum()); res['n_q10'] = int((q < 0.10).sum())
    res['min_achievable_p'] = round(float(np.min(p0[ix])) if len(ix) else 1, 4)
    res['top'] = top
    # attraction too (the opposite tail), for reference
    res['n_attract_q10'] = int((bh(1 - p_obs[ix] + 1 / (nsamp + 1)) < 0.10).sum())
    return res, dict(keys=keys, obs=obs, N=N, fam=allp)


def plant(S, site, nplant, seed, minDf=3):
    """split nplant words (df >= minDf) into two forms: each document uses ONE form (coin flip)."""
    r = random.Random(seed)
    df = collections.Counter(w for s in S for w in s)
    cand = [w for w, c in df.items() if c >= minDf and w not in ()]
    ch = r.sample(cand, min(nplant, len(cand)))
    S2 = []
    for s in S:
        s = set(s)
        for w in ch:
            if w in s and r.random() < 0.5:
                s.discard(w); s.add(w + ('#P',))
        S2.append(frozenset(s))
    return S2, [(w, w + ('#P',)) for w in ch]


def main():
    job = sys.argv[1]
    out = {}
    if job == 'lb':
        docs = lb_corpus()
        S, site, _ = units(docs)
        res, _ = analyse(S, site, 1000, 1, 'LB_full', truthf=lb_truth)
        out['LB_full'] = res; print(json.dumps({k: v for k, v in res.items() if k != 'top'}), flush=True)
        dump(out, 'c1_lb.json')
        # LA-sized LB subsamples (245 docs, sites drawn proportionally)
        subs = []
        for rep in range(10):
            r = random.Random(100 + rep); idx = r.sample(range(len(S)), 245)
            res, _ = analyse([S[i] for i in idx], [site[i] for i in idx], 500, 200 + rep, 'LB_245_%d' % rep, truthf=lb_truth)
            subs.append(res); print(rep, json.dumps(res['fam'].get('TRUE')), res['n_fwer05'], flush=True)
        out['LB_245'] = subs
        dump(out, 'c1_lb.json')
    else:
        docs = la_corpus()
        S, site, _ = units(docs)
        res, raw = analyse(S, site, 2000, 1, 'LA')
        out['LA'] = res; print(json.dumps({k: v for k, v in res.items() if k != 'top'}), flush=True)
        dump(out, 'c1_la.json')
        neg = []
        for rep in range(5):
            S2 = doc_shuffle(S, site, 300 + rep)
            r2, _ = analyse(S2, site, 1000, 400 + rep, 'LAshuf_%d' % rep)
            neg.append({k: v for k, v in r2.items() if k != 'top'}); print('neg', rep, r2['fam'], r2['n_fwer05'], flush=True)
        out['LA_shuffled'] = neg
        pl = []
        for rep in range(6):
            S2, planted = plant(S, site, 8, 500 + rep)
            df = collections.Counter(w for s in S2 for w in s)
            P = form_pairs(df)
            for a, b in planted: P[(a, b)] = 'PLANT'
            r2, raw2 = analyse(S2, site, 1000, 600 + rep, 'LAplant_%d' % rep, pairs=P)
            keys = raw2['keys']; obs, N = raw2['obs'], raw2['N']
            p_obs, p_null = lower_p(obs, N)
            p0 = (N == 0).mean(0)
            fam = np.array([raw2['fam'][k] for k in keys])
            tm = np.isin(fam, ['SUF1', 'PRE1', 'FIN', 'INI', 'PLANT']) & (p0 < 0.5)
            fw, _ = fwer(p_obs, p_null, tm); q = bh(p_obs[tm])
            kk = [keys[j] for j in np.where(tm)[0]]
            rec = []
            for a, b in planted:
                k = (a, b)
                if k in kk:
                    i = kk.index(k); rec.append(dict(w='-'.join(a), testable=True, O=int(obs[keys.index(k)]),
                                                     p=round(float(p_obs[keys.index(k)]), 4), fwer=round(float(fw[i]), 3), q=round(float(q[i]), 3)))
                else:
                    rec.append(dict(w='-'.join(a), testable=False))
            pl.append(dict(rep=rep, plant=r2['fam'].get('PLANT'), rec=rec, n_fwer05=r2['n_fwer05']))
            print('plant', rep, r2['fam'].get('PLANT'), sum(1 for x in rec if x.get('fwer', 1) < .05),
                  sum(1 for x in rec if x.get('q', 1) < .1), flush=True)
        out['LA_planted'] = pl
        dump(out, 'c1_la.json')


if __name__ == '__main__':
    main()
