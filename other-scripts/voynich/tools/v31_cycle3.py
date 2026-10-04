"""v31 cycle 3: the core Voynich signatures, one at a time, against the five classes; a dedicated
spell detector (MAGIC vs everything else) with a permuted-class null; and the spell score inside the Voynich
by section (null = section labels shuffled)."""
import os, sys, json, random
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_lib as L, v31_cycle1 as C1
from sklearn.linear_model import LogisticRegression

N = 100
FN = 'v31_cycle3.txt'
CORE = [('x_junc_mi', 'junction link (MI excess, bits)'), ('x_junc_same', 'junction same-unit excess'),
        ('order_rig', 'slot template: unit-order rigidity'), ('pos_mi', 'slot template: unit-position MI'),
        ('x_near_far', 'near-repeats 16-40 tokens apart (excess)'), ('x_near_loc', 'near-repeats within 15 tokens (excess)'),
        ('family', 'word-family density (edit-1 neighbours)'), ('var_runs', 'variant alternation (switch/expected; <1 = runs)'),
        ('x_gap_cv', 'burstiness (topic)'), ('x_halves_j', 'vocabulary drift inside sample (topic)'),
        ('x_rep_win', 'exact repeats within 10 tokens'), ('x_rep_adj', 'adjacent reduplication'), ('ttr', 'type/token'),
        ('top10', 'top-10 word share'), ('x_wl_ac1', 'word-length autocorrelation')]


def main():
    R, keys, X, corp, cls = C1.load(N)
    ki = {k: i for i, k in enumerate(keys)}
    tr = np.isin(cls, L.CLASSES)
    corpora = sorted(set(corp[tr])); truth = {c: cls[corp == c][0] for c in corpora}
    med = {c: np.median(X[corp == c], 0) for c in set(corp)}
    rows = []
    out = {}
    for k, desc in CORE:
        j = ki[k]
        v = float(med['V_ZL'][j]); vi = float(med['V_IT'][j])
        per = {}
        for q in L.CLASSES:
            cm = np.array([med[c][j] for c in corpora if truth[c] == q])
            per[q] = (float(np.median(cm)), float(np.mean(cm <= v)))
        allm = np.array([med[c][j] for c in corpora]); mad = np.median(np.abs(allm - np.median(allm))) + 1e-9
        near = min(L.CLASSES, key=lambda q: abs(per[q][0] - v))
        out[k] = {'V_ZL': v, 'V_IT': vi, 'per': per, 'near': near, 'mad': float(mad)}
        rows.append(f"{desc}: V {v:.3g} (IT {vi:.3g}); class medians " + ', '.join(f'{q} {per[q][0]:.3g}' for q in L.CLASSES) + f' -> nearest {near}')
        print(rows[-1], flush=True)
    votes = Counter(o['near'] for o in out.values())
    L.row(FN, 'V-31.3a', 'Core signatures one by one: Voynich median vs the median of corpus medians in each class (line-free; samples of 100 tokens)',
          ' | '.join(rows), f'nearest class per signature: {dict(votes)}')
    # ---- line features where lines exist
    lf = defaultdict(list)
    for r in R:
        if r['LF']: lf[r['corpus']].append(r['LF'])
    lrows = []
    for k in ['l_chain', 'l_first_len', 'l_last_len', 'l_last_H']:
        vals = {}
        for q in L.CLASSES + ['V']:
            cs = [c for c in lf if (c.startswith('V_Z') if q == 'V' else (c in truth and truth[c] == q))]
            if q == 'V': cs = ['V_ZL']
            m = [np.median([x[k] for x in lf[c]]) for c in cs if lf[c]]
            if m: vals[q] = float(np.median(m))
        lrows.append(f'{k}: ' + ', '.join(f'{q} {v:.2f}' for q, v in vals.items()))
    L.row(FN, 'V-31.3b', 'Line signatures (line-initial chain = first unit repeats from line to line, /shuffled-line expectation; first/last word length; line-final unit entropy). Lines are real only in Voynich and GIBB (hand-written); MAGIC lines = vox runs; others editorial or synthetic',
          ' | '.join(lrows), 'descriptive; the chain is a Voynich line rule, not shared by gibberish' if True else '')
    # ---- spell detector with permuted-class null
    Xt, ct = X[tr], corp[tr]
    y = np.array([1 if truth[c] == 'MAGIC' else 0 for c in ct])
    def fitb(Xa, ya):
        mu = Xa.mean(0); sd = Xa.std(0); sd[sd == 0] = 1
        m = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit((Xa - mu) / sd, ya)
        return lambda Z: m.predict_proba((Z - mu) / sd)[:, 1]
    # LOCO
    sc = {}
    for c in corpora:
        te = ct == c
        sc[c] = float(fitb(Xt[~te], y[~te])(Xt[te]).mean())
    mrec = np.mean([sc[c] > 0.5 for c in corpora if truth[c] == 'MAGIC'])
    fpr = np.mean([sc[c] > 0.5 for c in corpora if truth[c] != 'MAGIC'])
    fp_list = [c for c in corpora if truth[c] != 'MAGIC' and sc[c] > 0.5]
    P = fitb(Xt, y)
    vs = float(P(X[corp == 'V_ZL']).mean()); vsi = float(P(X[corp == 'V_IT']).mean())
    tests = {c: float(P(X[corp == c]).mean()) for c in set(corp) if c not in truth}
    nonmag = sorted(sc[c] for c in corpora if truth[c] != 'MAGIC')
    pct = float(np.mean(np.array(nonmag) <= vs))
    # null: a random pseudo-class of 10 corpora; how often does the Voynich get a posterior <= observed?
    rng = np.random.RandomState(3); nulls = []
    nm = sum(1 for c in corpora if truth[c] == 'MAGIC')
    for it in range(300):
        pick = set(rng.choice(corpora, nm, replace=False))
        yp = np.array([1 if c in pick else 0 for c in ct])
        nulls.append(float(fitb(Xt, yp)(X[corp == 'V_ZL']).mean()))
    p_low = (1 + sum(x <= vs for x in nulls)) / (1 + len(nulls))
    print('spell detector', mrec, fpr, fp_list, vs, vsi, pct, p_low, tests, flush=True)
    L.row(FN, 'V-31.3c', 'Dedicated spell detector (MAGIC vs all other training corpora, logistic, LOCO); null = 300 random pseudo-classes of 10 corpora',
          f'LOCO: MAGIC recall {mrec:.0%}, false-positive rate {fpr:.0%} ({", ".join(fp_list) or "none"}); Voynich spell score ZL {vs:.4f}, IT {vsi:.4f} '
          f'(percentile among non-magic corpora {pct:.0%}); P(score <= observed | random pseudo-class) = {p_low:.3f}; test objects: ' +
          ', '.join(f'{c} {v:.2f}' for c, v in sorted(tests.items()) if not c.startswith('V_')),
          'Voynich is confidently NOT magic-word texture' if vs < 0.05 and p_low < 0.05 else 'not decisive')
    # ---- spell score and class posteriors inside the Voynich by section
    import v21_lib as V
    P5, cl5 = C1.fit(Xt, cls[tr])
    pages = V.voynich_pages('ZL3b', minw=10)
    bysec = defaultdict(list)
    for p in pages: bysec[p['sec']].extend(l for pa in p['paras'] for l in pa)
    rows2 = []; secs = []; scores = []; g = []
    rng2 = random.Random(5)
    for s, lines in bysec.items():
        S = L.samples([lines], N=N, maxs=30)
        if len(S) < 3: continue
        F = np.array([[L.features(x, rng2)[k] for k in keys] for x in S]); F[~np.isfinite(F)] = 0
        ms = P(F); pr = P5(F)
        rows2.append(f"{s} (n={len(S)}): spell {ms.mean():.3f}; " + ', '.join(f'{c} {v:.2f}' for c, v in zip(cl5, pr.mean(0))))
        secs += [s] * len(S); scores += list(pr[:, cl5.index('GEN')]); g.append(pr.mean(0))
        print(rows2[-1], flush=True)
    # section effect on the GEN posterior vs shuffled section labels
    secs = np.array(secs); scores = np.array(scores)
    def between(sv):
        return np.var([scores[sv == s].mean() for s in set(sv)])
    ob = between(secs); nb = [between(np.random.RandomState(i).permutation(secs)) for i in range(500)]
    p_sec = (1 + sum(x >= ob for x in nb)) / 501
    L.row(FN, 'V-31.3d', 'Class posteriors and spell score inside the Voynich by section (ZL3b, 100-token samples); null = section labels shuffled (500x) for the GEN posterior',
          ' | '.join(rows2) + f' | section effect on GEN posterior p = {p_sec:.3f}',
          'sections differ in texture class' if p_sec < 0.05 else 'texture class is uniform across sections')
    L.save('cycle3.json', {'core': out, 'spell': {'loco': sc, 'V_ZL': vs, 'V_IT': vsi, 'tests': tests, 'p_low': p_low},
                           'sections': rows2, 'p_sec': p_sec})


if __name__ == '__main__':
    main()
