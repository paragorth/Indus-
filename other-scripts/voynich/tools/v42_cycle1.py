"""v42 cycle 1: where does a long, declared-meaningless, one-hand book (Codex Seraphinianus) fall in the v31
five-class texture space (LANG / MAGIC / INVENT / GIBB / GEN), and where does the Voynich fall next to it?

Classifiers never see the Voynich or the CS. Three training worlds: clean (v31 features), n09 and n18 (every
training corpus passed through the OCR-like channel; n18 matches the CS transliteration's validation error).
Controls: LOCO (each training corpus held out must be classed correctly), label-permutation null, random-subset
survivors (grouped by corpus, selected on half A, re-tested on half B), alphabet-sensitive features dropped.
"""
import os, sys, json, random
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v42_lib as L
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

FN = 'v42_cycle1.txt'
CLASSES = ['LANG', 'MAGIC', 'INVENT', 'GIBB', 'GEN']
ALPHA = {'wl_mean', 'h1', 'alph_eff', 'zlib', 'wl_H'}


def load():
    R = L.load('feats.json')
    keys = sorted(R[0]['F'])
    X = np.array([[r['F'][k] for k in keys] for r in R], float); X[~np.isfinite(X)] = 0
    corp = np.array([r['corpus'] for r in R]); cls = np.array([r['cls'] for r in R]); var = np.array([r['var'] for r in R])
    return keys, X, corp, cls, var


def mk(kind):
    if kind == 'lr': return LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced')
    if kind == 'lda': return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    return KNeighborsClassifier(7)


def fit(X, y, kind='lr'):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    m = mk(kind).fit((X - mu) / sd, y)
    return (lambda Z: m.predict_proba((Z - mu) / sd)), list(m.classes_)


def loco(X, y, corp):
    per = defaultdict(list); out = {}
    for c in sorted(set(corp)):
        te = corp == c
        P, cl = fit(X[~te], y[~te]); p = P(X[te]).mean(0)
        pr = cl[int(p.argmax())]; per[y[te][0]].append(pr == y[te][0]); out[c] = pr
    return float(np.mean([np.mean(v) for v in per.values()])), {k: f'{sum(v)}/{len(v)}' for k, v in per.items()}, out


def bal(pred, truth):
    per = defaultdict(list)
    for c, p in pred.items(): per[truth[c]].append(p == truth[c])
    return float(np.mean([np.mean(v) for v in per.values()]))


def survivors(X, y, corp, tests, nh=1000, seed=0, permute=False, thr=0.6):
    """random classifiers; select on half A (grouped 4-fold), re-test on half B; survivors vote on tests."""
    rng = np.random.RandomState(seed)
    cs = sorted(set(corp)); truth = {c: y[corp == c][0] for c in cs}
    if permute:
        lab = rng.permutation([truth[c] for c in cs]); truth = dict(zip(cs, lab)); y = np.array([truth[c] for c in corp])
    # stratified halves by class
    half = {}
    for k in set(truth.values()):
        m = [c for c in cs if truth[c] == k]; rng.shuffle(m)
        for i, c in enumerate(m): half[c] = i % 2
    hv = np.array([half[c] for c in corp])
    votes = {t: Counter() for t in tests}; nsurv = 0
    for h in range(nh):
        k = rng.randint(4, 13); cols = rng.choice(X.shape[1], k, replace=False); kind = ['lr', 'lda', 'knn'][rng.randint(3)]
        Xc = X[:, cols]
        A = hv == 0; ca = sorted(set(corp[A])); fold = dict(zip(ca, rng.permutation(len(ca)) % 4)); pred = {}
        fa = np.array([fold.get(c, -1) for c in corp])
        for f in range(4):
            te = A & (fa == f); tr = A & ~te & (fa >= 0)
            P, cl = fit(Xc[tr], y[tr], kind); pp = P(Xc[te])
            for c in set(corp[te]): pred[c] = cl[int(pp[corp[te] == c].mean(0).argmax())]
        if bal(pred, truth) < thr: continue
        P, cl = fit(Xc[A], y[A], kind); pp = P(Xc[~A])
        predB = {c: cl[int(pp[corp[~A] == c].mean(0).argmax())] for c in set(corp[~A])}
        if bal(predB, truth) < thr: continue
        nsurv += 1
        P, cl = fit(Xc, y, kind)
        for t, Xt in tests.items(): votes[t][cl[int(P(Xt[:, cols]).mean(0).argmax())]] += 1
    return nsurv, votes


def main():
    keys, X, corp, cls, var = load()
    keep = np.array([k not in ALPHA for k in keys])
    res = {}
    TESTS = [('S_CS', 'raw'), ('S_CS', 'col'), ('S_CS', 'sb'), ('S_CS1', 'raw'), ('S_CS2', 'raw'), ('S_CS1', 'col'), ('S_CS2', 'col'),
             ('V_ZL', 'clean'), ('V_IT', 'clean'), ('V_ZL', 'col'), ('V_ZL', 'n09'), ('V_ZL', 'n18'), ('V_IT', 'n18')]
    tests = {f'{c}:{v}': X[(corp == c) & (var == v)] for c, v in TESTS}
    tests = {k: v for k, v in tests.items() if len(v)}
    for world in ('clean', 'n09', 'n18'):
        for fs, cols in (('all', np.ones(len(keys), bool)), ('noalpha', keep)):
            tr = (var == world) & np.isin(cls, CLASSES)
            Xt, yt, ct = X[tr][:, cols], cls[tr], corp[tr]
            ba, per, lo = loco(Xt, yt, ct)
            P, cl = fit(Xt, yt)
            post = {t: dict(zip(cl, P(v[:, cols]).mean(0).round(3))) for t, v in tests.items()}
            # GEN-ness of each training GEN corpus under LOCO, for reference
            print(world, fs, 'LOCO', round(ba, 3), per, flush=True)
            for t in post: print('   ', t, post[t])
            res[f'{world}|{fs}'] = {'loco': ba, 'per': per, 'post': post, 'loco_pred': lo}
    # permutation null for the CS GEN posterior (n18 world, all features)
    tr = (var == 'n18') & np.isin(cls, CLASSES); Xt, yt, ct = X[tr], cls[tr], corp[tr]
    cs_ = sorted(set(ct)); lab = {c: yt[ct == c][0] for c in cs_}; rng = np.random.RandomState(1)
    obs = {t: res['n18|all']['post'][t].get('GEN', 0) for t in ('S_CS:raw', 'V_ZL:n18')}
    null = defaultdict(list)
    for it in range(100):
        mp = dict(zip(cs_, rng.permutation([lab[c] for c in cs_]))); yp = np.array([mp[c] for c in ct])
        P, cl = fit(Xt, yp)
        for t in obs: null[t].append(float(P(tests[t]).mean(0)[cl.index('GEN')]))
    pnull = {t: (1 + sum(v >= obs[t] for v in null[t])) / 101 for t in obs}
    print('perm null', {t: (obs[t], np.mean(null[t]), pnull[t]) for t in obs})
    # random survivors in the n18 world and clean world; null with permuted labels
    surv = {}
    for world in ('n18', 'clean'):
        tr = (var == world) & np.isin(cls, CLASSES)
        n, v = survivors(X[tr], cls[tr], corp[tr], tests, nh=int(os.environ.get('NH', 800)), seed=7)
        nn, _ = survivors(X[tr], cls[tr], corp[tr], {}, nh=300, seed=8, permute=True)
        surv[world] = (n, {t: dict(c) for t, c in v.items()}, nn)
        print('survivors', world, n, 'null', nn, {t: dict(c) for t, c in v.items()}, flush=True)
    # distances in the n18 world (standardised on training)
    tr = (var == 'n18') & np.isin(cls, CLASSES)
    mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1; Z = (X - mu) / sd
    cen = {}
    for c in set(corp[tr]): cen[c] = Z[tr & (corp == c)].mean(0)
    for t, v in tests.items(): cen[t] = ((v - mu) / sd).mean(0)
    def near(t, k=8):
        d = sorted((float(np.linalg.norm(cen[t] - cen[c])), c) for c in cen if c != t and not c.startswith(t.split(':')[0]))
        return [(c, round(x, 2)) for x, c in d[:k]]
    nears = {t: near(t) for t in ('S_CS:raw', 'S_CS:col', 'V_ZL:n18', 'V_ZL:clean', 'V_IT:n18')}
    for t, v in nears.items(): print('near', t, v)
    # spread: distance between CS volumes vs distance Voynich-CS
    dvc = float(np.linalg.norm(cen['V_ZL:n18'] - cen['S_CS:raw']))
    d12 = float(np.linalg.norm(cen['S_CS1:raw'] - cen['S_CS2:raw']))
    dzi = float(np.linalg.norm(cen['V_ZL:n18'] - cen['V_IT:n18']))
    cls_c = {k: np.mean([cen[c] for c in set(corp[tr]) if cls[tr][corp[tr] == c][0] == k], 0) for k in CLASSES}
    dcls = {k: round(float(np.linalg.norm(cen['V_ZL:n18'] - v)), 2) for k, v in cls_c.items()}
    dcls_cs = {k: round(float(np.linalg.norm(cen['S_CS:raw'] - v)), 2) for k, v in cls_c.items()}
    print('dist V-CS', dvc, 'CS1-CS2', d12, 'ZL-IT', dzi, 'V->class', dcls, 'CS->class', dcls_cs)
    L.save('cycle1.json', {'res': res, 'pnull': pnull, 'obs': obs, 'surv': surv, 'near': nears, 'dvc': dvc, 'd12': d12,
                           'dcls': dcls, 'dcls_cs': dcls_cs})
    # ---------------- rows
    f = lambda d: ', '.join(f'{k} {v:.2f}' for k, v in sorted(d.items(), key=lambda x: -x[1]) if v >= 0.05)
    for w in ('clean', 'n09', 'n18'):
        r = res[f'{w}|all']; r2 = res[f'{w}|noalpha']
        L.row(FN, f'V-42.1.{ {"clean": 1, "n09": 2, "n18": 3}[w] }',
              f'Control: v31 5-class logistic (32 features) trained on the {w} world ({"no added noise" if w == "clean" else "every training corpus through the OCR-like channel at " + w[1:] + "% char error"}); LOCO = every training corpus held out',
              f'LOCO balanced acc {r["loco"]:.2f} {r["per"]}; without 5 alphabet-sensitive features {r2["loco"]:.2f}',
              'held-out members classed correctly well above chance (0.20)' if r['loco'] > 0.5 else 'world too noisy to classify')
    i = 4
    for t in ('S_CS:raw', 'S_CS:col', 'S_CS:sb', 'S_CS1:raw', 'S_CS2:raw'):
        L.row(FN, f'V-42.1.{i}', f'Codex Seraphinianus ({t}) classified, never trained on: posteriors in clean / n18 worlds (all features; no-alphabet features)',
              f'clean: {f(res["clean|all"]["post"][t])}; n18: {f(res["n18|all"]["post"][t])}; n18 no-alpha: {f(res["n18|noalpha"]["post"][t])}', 'see verdict'); i += 1
    for t in ('V_ZL:clean', 'V_ZL:n18', 'V_IT:n18', 'V_ZL:col'):
        if t not in tests: continue
        L.row(FN, f'V-42.1.{i}', f'Voynich ({t}) classified, never trained on: clean / n18 worlds',
              f'clean: {f(res["clean|all"]["post"][t])}; n18: {f(res["n18|all"]["post"][t])}', 'see verdict'); i += 1
    L.row(FN, f'V-42.1.{i}', 'Null: corpus labels permuted 100x (n18 world); GEN posterior of CS and noise-matched Voynich',
          f'CS GEN {obs["S_CS:raw"]:.2f} (null mean {np.mean(null["S_CS:raw"]):.2f}, p {pnull["S_CS:raw"]:.3f}); Voynich n18 GEN {obs["V_ZL:n18"]:.2f} (null {np.mean(null["V_ZL:n18"]):.2f}, p {pnull["V_ZL:n18"]:.3f})',
          'label structure, not chance' if max(pnull.values()) < 0.05 else 'not above the label-permutation null'); i += 1
    for w in ('n18', 'clean'):
        n, v, nn = surv[w]
        sh = lambda t: ', '.join(f'{k} {c / max(1, n):.0%}' for k, c in sorted(v.get(t, {}).items(), key=lambda x: -x[1]))
        L.row(FN, f'V-42.1.{i}', f'Random-subset survivors ({w} world): 800 random classifiers (4-12 features; logistic, shrinkage LDA, 7-NN), selected on half the corpora, re-tested on the other half; null 300 with permuted labels',
              f'survivors {n} (null {nn}); CS raw: {sh("S_CS:raw")}; CS col: {sh("S_CS:col")}; Voynich ZL clean: {sh("V_ZL:clean")}; ZL n18: {sh("V_ZL:n18")}; IT n18: {sh("V_IT:n18")}',
              'see verdict'); i += 1
    L.row(FN, f'V-42.1.{i}', 'Nearest training corpora (n18 world, standardised centroid distance); class-centroid distances',
          f'CS raw: {nears["S_CS:raw"][:5]}; Voynich n18: {nears["V_ZL:n18"][:5]}; d(Voynich,CS) {dvc:.2f}, d(CS V1,V2) {d12:.2f}, d(ZL,IT) {dzi:.2f}; Voynich->class {dcls}; CS->class {dcls_cs}',
          'descriptive')


if __name__ == '__main__':
    main()
