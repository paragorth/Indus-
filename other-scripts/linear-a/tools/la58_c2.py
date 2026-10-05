#!/usr/bin/env python3
"""LA-58 cycle 2.
A. Held-out sites: rejection ABC on every statistic EXCEPT site h's own (and its pairs); the accepted worlds
   predict h's 13-statistic profile. Error vs prior-predictive (random worlds) and vs no-institution worlds.
   Same on shuffled LA, Linear B and Ur III.
B. Institutional words: worlds accepted for the real corpus are re-run to documents; a classifier learns
   which words are titles (office words) from word-level features; applied to real words.
   Controls: Ur III (titles ugula, nu-banda3, szabra, sanga, ensi2) and Linear B (known titles),
   shuffled-site LA.
"""
import sys, os, json, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la58_common import *
from la58_run import corpus, nk_of, load_bank
from sklearn.ensemble import RandomForestClassifier
from la58_fit import auc

PS = 13


def site_cols(K, h):
    cols = list(range(h * PS, (h + 1) * PS))
    p = K * PS
    for k in range(K):
        for l in range(k + 1, K):
            if k == h or l == h:
                cols.append(p)
            p += 1
    return cols


def zscale(S):
    med = np.median(S, 0); mad = np.median(np.abs(S - med), 0) * 1.4826
    mad[mad < 1e-3] = np.std(S, 0)[mad < 1e-3] + 1e-3
    return med, mad


def heldout(S, T, real, K, sites, nacc=500, rs=0):
    S = np.nan_to_num(S, nan=-2); real = np.nan_to_num(real, nan=-2)
    med, mad = zscale(S)
    Z = (S - med) / mad; zr = (real - med) / mad
    rng = np.random.RandomState(rs)
    out = {}
    for h in sites:
        own = list(range(h * PS, (h + 1) * PS))
        ex = np.ones(S.shape[1], bool); ex[site_cols(K, h)] = False
        d = np.sqrt(((Z[:, ex] - zr[ex]) ** 2).sum(1))
        acc = np.argsort(d)[:nacc]
        nul = np.where(T[:, 0] == 5)[0]; accn = nul[np.argsort(d[nul])[:nacc]]
        rnd = rng.choice(len(S), nacc, replace=False)
        err = lambda ix: float(np.mean(np.abs(np.median(Z[ix][:, own], 0) - zr[own])))
        out[h] = dict(post=err(acc), prior=err(rnd), null=err(accn),
                      types={TYPES[t]: round(float((T[acc, 0] == t).mean()), 3) for t in range(6)},
                      centre=[round(float((T[acc, 2 + k] == 1).mean()), 3) for k in range(K)])
    return out


def full_reject(S, T, real, nacc=500):
    S = np.nan_to_num(S, nan=-2); real = np.nan_to_num(real, nan=-2)
    med, mad = zscale(S)
    d = np.sqrt((((S - med) / mad - (real - med) / mad) ** 2).sum(1))
    return np.argsort(d)[:nacc], d


# ----------------------------------------------------------------- word features
def word_feats(docs, K, min_tok=3):
    tok = collections.Counter(); st = collections.defaultdict(collections.Counter)
    dt = collections.defaultdict(lambda: np.zeros(4)); first = collections.Counter(); nd = collections.Counter()
    withnum = collections.Counter(); single = collections.Counter()
    big = collections.Counter(d['site'] for d in docs).most_common(1)[0][0]
    for d in docs:
        seen = set()
        for i, w in enumerate(d['words']):
            tok[w] += 1; st[w][d['site']] += 1
            if w in seen:
                continue
            seen.add(w); nd[w] += 1; dt[w][d['dtype']] += 1
            if i == 0: first[w] += 1
            if d['nums']: withnum[w] += 1
            if len(d['words']) == 1: single[w] += 1
    words, F = [], []
    for w, c in tok.items():
        if c < min_tok:
            continue
        n = nd[w]; sc = np.array(list(st[w].values()), float); p = sc / sc.sum()
        F.append([np.log(c), len(st[w]), -(p * np.log(p)).sum(), st[w][big] / c, *(dt[w] / n), first[w] / n,
                  withnum[w] / n, single[w] / n, c / n])
        words.append(w)
    return words, np.array(F)


def title_model(S, T, real, K, nk, tabonly, nworld=24, sd=0):
    acc, _ = full_reject(S, T, real, nacc=300)
    rng = np.random.RandomState(sd)
    X, Y = [], []
    for j, i in enumerate(rng.choice(acc, nworld, replace=False)):
        if tabonly:
            lib().la58_set_tabonly(1)
        docs, th = sim_docs(K, nk, seed('la58-c2w-%d-%d' % (sd, j)), force_theta=T[i])
        cls = {}
        for d in docs:
            for w, c in zip(d['words'], d['wclass']):
                cls[w] = c
        ws, F = word_feats(docs, K)
        if not len(ws):
            continue
        X.append(F); Y.append(np.array([cls[w] == 2 for w in ws]))
    lib().la58_set_tabonly(0)
    X = np.vstack(X); Y = np.concatenate(Y)
    n = len(X); idx = rng.permutation(n); te = idx[:n // 5]; tr = idx[n // 5:]
    rf = RandomForestClassifier(200, min_samples_leaf=3, n_jobs=2, random_state=sd).fit(X[tr], Y[tr])
    return rf, float(auc(rf.predict_proba(X[te])[:, 1], Y[te])), float(Y.mean())


LB_TITLES = set('e-qe-ta ko-re-te po-ro-ko-re-te da-mo-ko-ro qa-si-re-u ra-wa-ke-ta wa-na-ka te-re-ta ka-ra-wi-po-ro '
                'i-je-re-u i-je-re-ja du-ma mo-ro-pa2 ko-re-te-re me-ri-du-ma-te pa2-si-re-u ki-ri-te-wi-ja '
                'ke-ro-si-ja po-ro-du-ma-te wa-na-ka-te-ro ra-wa-ke-si-jo e-qe-si-jo'.split())
UR_TITLES = set('ugula nu-banda3 szabra sanga ensi2 lugal dub-sar sukkal kuruszda kurušda'.split())


def main():
    res = {}
    for c in ['LA', 'LB', 'UR'] + ['LASHUF%d' % i for i in range(5)]:
        bank = 'LA' if c.startswith('LA') else c
        S, T, meta = load_bank(bank)
        docs, K, ab, tabonly = corpus(c); nk = nk_of(docs, K)
        real = stats(docs, K)
        sites = [k for k in range(K) if nk[k] >= 30]
        res[c + '_heldout'] = {ab[h]: v for h, v in heldout(S, T, real, K, sites).items()}
        acc, d = full_reject(S, T, real)
        res[c + '_reject'] = dict(dist=float(d[acc].mean()), types={TYPES[t]: round(float((T[acc, 0] == t).mean()), 3) for t in range(6)},
                                  centre={ab[k]: round(float((T[acc, 2 + k] == 1).mean()), 3) for k in range(K)},
                                  dependent={ab[k]: round(float((T[acc, 2 + k] == 2).mean()), 3) for k in range(K)},
                                  share=float(np.median(T[acc, 2 + 2 * K + 18])), ptrav=float(np.median(T[acc, 2 + 2 * K + 15])),
                                  pm=float(np.median(T[acc, 2 + 2 * K + 19])), pseal=float(np.median(T[acc, 2 + 2 * K + 14])))
        if c in ('LA', 'LB', 'UR', 'LASHUF0'):
            rf, a_sim, base = title_model(S, T, real, K, nk, tabonly)
            ws, F = word_feats(docs, K)
            p = rf.predict_proba(F)[:, 1]
            order = np.argsort(-p)
            r = dict(sim_auc=a_sim, base=base, top=[[ws[i], round(float(p[i]), 3)] for i in order[:25]], n=len(ws))
            truth = LB_TITLES if c == 'LB' else UR_TITLES if c == 'UR' else None
            if truth:
                y = np.array([w in truth for w in ws])
                r['truth_n'] = int(y.sum()); r['truth_auc'] = float(auc(p, y)) if y.any() else None
                # null: random word sets of the same size and frequency band
                rng = np.random.RandomState(1); lf = F[:, 0]; nulls = []
                for _ in range(1000):
                    yy = np.zeros(len(ws), bool)
                    for i in np.where(y)[0]:
                        cand = np.where(np.abs(lf - lf[i]) < 0.35)[0]; yy[rng.choice(cand)] = True
                    nulls.append(auc(p, yy))
                r['truth_p'] = float((np.array(nulls) >= r['truth_auc']).mean())
            res[c + '_titles'] = r
        print(c, json.dumps(res[c + '_reject']), flush=True)
    jdump(res, 'c2_results.json')


if __name__ == '__main__':
    main()
