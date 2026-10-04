"""v31 cycle 2: massive random guessing. Thousands of random classifiers (random feature subsets of 3-12
features x random model family: logistic, k-nearest-neighbours, shrinkage LDA, small random forest), each scored on
half the training corpora (grouped 4-fold CV by corpus); survivors are re-tested on the OTHER half (corpora never
seen in selection). Survivors then vote on the Voynich (ZL3b; IT2a as a held-out transcription).
Null: the whole pipeline re-run with corpus labels permuted (same selection threshold).
Positive control: survivors must class the held-out-half MAGIC corpora as MAGIC.
"""
import os, sys, json, random, time
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_lib as L, v31_cycle1 as C1
from multiprocessing import Pool

N = 100
FN = 'v31_cycle2.txt'
NH = int(os.environ.get('V31_NH', 3000))
NNULL = int(os.environ.get('V31_NNULL', 6))


def model(kind, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    if kind == 'lr': return LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced')
    if kind == 'knn': return KNeighborsClassifier(7)
    if kind == 'lda': return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    return RandomForestClassifier(60, max_depth=6, class_weight='balanced', random_state=seed, n_jobs=1)


def corp_level(pred_p, classes, corp):
    out = {}
    for c in set(corp):
        out[c] = classes[int(np.argmax(pred_p[corp == c].mean(0)))]
    return out


def bal(pred, truth):
    per = defaultdict(list)
    for c, p in pred.items(): per[truth[c]].append(p == truth[c])
    return float(np.mean([np.mean(v) for v in per.values()]))


def fitpred(kind, seed, Xa, ya, Xb):
    mu = Xa.mean(0); sd = Xa.std(0); sd[sd == 0] = 1
    m = model(kind, seed).fit((Xa - mu) / sd, ya)
    return m.predict_proba((Xb - mu) / sd), list(m.classes_)


def hypothesis(args):
    h, cols, kind, seed, D = args
    X, y, corp, half, truth, V, VI = D['X'], D['y'], D['corp'], D['half'], D['truth'], D['V'], D['VI']
    Xc = X[:, cols]
    # selection score: grouped 4-fold CV inside half A
    A = half == 0
    ca = sorted(set(corp[A])); rng = np.random.RandomState(seed); fold = dict(zip(ca, rng.permutation(len(ca)) % 4))
    pred = {}
    for f in range(4):
        te = A & np.array([fold.get(c, -1) == f for c in corp]); tr = A & ~te
        if len(set(y[tr])) < 5: return None
        p, cl = fitpred(kind, seed, Xc[tr], y[tr], Xc[te])
        pred.update(corp_level(p, cl, corp[te]))
    sA = bal(pred, truth)
    # re-test on half B (trained on all of A)
    B = half == 1
    p, cl = fitpred(kind, seed, Xc[A], y[A], Xc[B])
    pb = corp_level(p, cl, corp[B]); sB = bal(pb, truth)
    magicB = [pb[c] == 'MAGIC' for c in pb if truth[c] == 'MAGIC']
    # Voynich vote: trained on all training corpora
    T = np.isin(y, L.CLASSES)
    pv, cl = fitpred(kind, seed, Xc[T], y[T], np.vstack([V[:, cols], VI[:, cols]]))
    nv = len(V)
    vz = dict(zip(cl, pv[:nv].mean(0))); vi = dict(zip(cl, pv[nv:].mean(0)))
    return {'h': h, 'cols': cols, 'kind': kind, 'sA': sA, 'sB': sB, 'magicB': float(np.mean(magicB)) if magicB else None,
            'vz': vz, 'vi': vi}


def run(D, keys, nh, seed0, tag):
    rng = random.Random(seed0)
    jobs = []
    for h in range(nh):
        k = rng.randint(3, 12)
        cols = sorted(rng.sample(range(len(keys)), k))
        kind = rng.choice(['lr', 'lr', 'knn', 'lda', 'rf'])
        jobs.append((h, cols, kind, seed0 * 100000 + h, D))
    with Pool(2) as p:
        R = [r for r in p.imap_unordered(hypothesis, jobs, chunksize=20) if r]
    return R


def main():
    Rr, keys, X, corp, cls = C1.load(N)
    tr = np.isin(cls, L.CLASSES)
    Xt, yt, ct = X[tr], cls[tr], corp[tr]
    corpora = sorted(set(ct)); truth = {c: yt[ct == c][0] for c in corpora}
    # stratified halves by class
    rng = np.random.RandomState(7); half_of = {}
    for k in L.CLASSES:
        cs = [c for c in corpora if truth[c] == k]; rng.shuffle(cs)
        for i, c in enumerate(cs): half_of[c] = i % 2
    half = np.array([half_of[c] for c in ct])
    D = {'X': Xt, 'y': yt, 'corp': ct, 'half': half, 'truth': truth, 'V': X[corp == 'V_ZL'], 'VI': X[corp == 'V_IT']}
    t0 = time.time()
    R = run(D, keys, NH, 1, 'real')
    print('real hypotheses', len(R), 'time', round(time.time() - t0), flush=True)
    sA = np.array([r['sA'] for r in R])
    thr = float(np.percentile(sA, 90))
    surv = [r for r in R if r['sA'] >= thr and r['sB'] >= 0.5]
    print('threshold', thr, 'survivors', len(surv), flush=True)
    def summarize(S):
        vz = Counter(max(r['vz'], key=r['vz'].get) for r in S)
        vi = Counter(max(r['vi'], key=r['vi'].get) for r in S)
        mp = {k: float(np.mean([r['vz'].get(k, 0) for r in S])) for k in L.CLASSES}
        return vz, vi, mp
    vz, vi, mp = summarize(surv)
    mg = [r['magicB'] for r in surv if r['magicB'] is not None]
    print('Voynich ZL votes', vz, 'IT votes', vi, 'mean posterior', mp, 'magic recall on half B', np.mean(mg), flush=True)
    # feature frequency among survivors vs all (which features carry the verdict)
    fc = Counter(keys[c] for r in surv for c in r['cols']); fa = Counter(keys[c] for r in R for c in r['cols'])
    enrich = sorted(((fc[k] / max(1, len(surv))) / (fa[k] / len(R)), k) for k in keys)[::-1]
    # survivors that vote MAGIC / LANG for the Voynich: what features do they use?
    byvote = defaultdict(Counter)
    for r in surv:
        byvote[max(r['vz'], key=r['vz'].get)].update(keys[c] for c in r['cols'])
    # null: permuted corpus labels
    nullshare = []; nulln = []
    for it in range(NNULL):
        prng = np.random.RandomState(100 + it)
        perm = dict(zip(corpora, prng.permutation([truth[c] for c in corpora])))
        Dn = dict(D); Dn['y'] = np.array([perm[c] for c in ct]); Dn['truth'] = perm
        Rn = run(Dn, keys, max(300, NH // 5), 1000 + it, 'null')
        sn = [r for r in Rn if r['sA'] >= thr and r['sB'] >= 0.5]
        nulln.append(len(sn) / len(Rn))
        if sn:
            c = Counter(max(r['vz'], key=r['vz'].get) for r in sn)
            nullshare.append(c.most_common(1)[0][1] / len(sn))
        print('null', it, 'survivor rate', round(nulln[-1], 4), 'top vote share', nullshare[-1] if sn else None, flush=True)
    top, topn = vz.most_common(1)[0]
    share = topn / len(surv)
    L.save('cycle2.json', {'keys': keys, 'thr': thr, 'n': len(R), 'surv': surv, 'vz': vz, 'vi': vi, 'mp': mp,
                           'enrich': enrich, 'byvote': {k: dict(v) for k, v in byvote.items()}, 'nullrate': nulln,
                           'nullshare': nullshare})
    L.row(FN, 'V-31.2a', f'{len(R)} random hypotheses (3-12 random features x logistic/kNN/LDA/forest), selected on half A of the training corpora '
          f'(grouped 4-fold, top 10% = bal. acc >= {thr:.2f}), re-tested on unseen half B (>= 0.50 kept); null = same pipeline, corpus labels permuted ({NNULL} runs)',
          f'{len(surv)} survivors ({len(surv) / len(R):.1%}); null survivor rate {np.mean(nulln):.2%} (max {max(nulln):.2%}); survivors classify held-out MAGIC corpora as MAGIC in {np.mean(mg):.0%} of cases',
          'random guessing finds real class structure; magic-word texture is learnable' if np.mean(mg) > 0.6 and len(surv) / len(R) > 3 * max(nulln + [0.001]) else 'weak')
    L.row(FN, 'V-31.2b', 'Survivors vote on the Voynich (ZL3b; IT2a = held-out transcription); null = top-class vote share among null survivors',
          f'ZL votes {dict(vz)}; IT votes {dict(vi)}; mean posteriors {", ".join(f"{k} {v:.2f}" for k, v in mp.items())}; top class {top} share {share:.2f} vs null shares {[round(x, 2) for x in nullshare]}',
          f'Voynich = {top} by random-subset vote' + (' (MAGIC never/rarely chosen)' if vz.get('MAGIC', 0) / len(surv) < 0.05 else ''))
    L.row(FN, 'V-31.2c', 'Which features carry the survivors (enrichment among survivors vs all hypotheses); features used by survivors that vote each class for the Voynich',
          'enriched: ' + ', '.join(f'{k} {e:.1f}x' for e, k in enrich[:8]) + '; per-vote top features: ' +
          '; '.join(f"{v}: " + ', '.join(k for k, _ in c.most_common(4)) for v, c in byvote.items()), 'descriptive')


if __name__ == '__main__':
    main()
