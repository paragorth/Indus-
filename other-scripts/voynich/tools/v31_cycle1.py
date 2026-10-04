"""v31 cycle 1: where does the Voynich fall among LANG / MAGIC / INVENT / GIBB / GEN?

Line-free feature battery (v31_lib), samples of N tokens. Multinomial logistic regression trained WITHOUT the
Voynich. Validation: leave-one-corpus-out (LOCO; a corpus's samples are never in its own training set).
Positive control: every MAGIC corpus, held out, must be classed MAGIC. Null: class labels permuted across
training corpora (corpus-level, class sizes kept); the Voynich verdict's confidence is compared with the same
statistic under the null.
"""
import os, sys, json, random
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_lib as L
from sklearn.linear_model import LogisticRegression

N = int(os.environ.get('V31_N', 100))
FN = 'v31_cycle1.txt'
TRAIN = L.CLASSES


def load(N=N):
    R = L.load(f'feats_N{N}.json')
    keys = sorted(R[0]['F'].keys())
    X = np.array([[r['F'][k] for k in keys] for r in R], float)
    X[~np.isfinite(X)] = 0
    corp = np.array([r['corpus'] for r in R]); cls = np.array([r['cls'] for r in R])
    return R, keys, X, corp, cls


def fit(X, y, C=0.5):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    m = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit((X - mu) / sd, y)
    return lambda Z: m.predict_proba((Z - mu) / sd), list(m.classes_)


def loco(X, y, corp, cols=None, C=0.5):
    """corpus-level LOCO predictions: {corpus: (true, predicted, mean proba dict)}"""
    if cols is not None: X = X[:, cols]
    out = {}
    for c in sorted(set(corp)):
        te = corp == c; tr = ~te
        P, cl = fit(X[tr], y[tr], C)
        p = P(X[te]).mean(0)
        out[c] = (y[te][0], cl[int(np.argmax(p))], dict(zip(cl, p)))
    return out


def bal_acc(res):
    per = defaultdict(list)
    for c, (t, p, _) in res.items(): per[t].append(t == p)
    return float(np.mean([np.mean(v) for v in per.values()])), {k: (int(sum(v)), len(v)) for k, v in per.items()}


def main():
    R, keys, X, corp, cls = load()
    tr = np.isin(cls, TRAIN)
    Xt, yt, ct = X[tr], cls[tr], corp[tr]
    print('training samples', tr.sum(), 'corpora', len(set(ct)), Counter(yt))
    # ---- LOCO
    res = loco(Xt, yt, ct)
    ba, per = bal_acc(res)
    conf = Counter((t, p) for t, p, _ in res.values())
    print('LOCO balanced corpus accuracy', round(ba, 3), per)
    print('confusion', dict(conf))
    mis = {c: (t, p) for c, (t, p, _) in res.items() if t != p}
    print('misclassified', mis)
    magic = {c: (p, round(pr['MAGIC'], 2)) for c, (t, p, pr) in res.items() if t == 'MAGIC'}
    print('MAGIC held out ->', magic)
    # ---- fit all, classify test objects
    P, cl = fit(Xt, yt)
    tests = sorted(set(corp[~tr]))
    T = {}
    for c in tests:
        m = corp == c
        pr = P(X[m]); share = Counter(np.array(cl)[pr.argmax(1)])
        T[c] = {'n': int(m.sum()), 'mean_p': dict(zip(cl, pr.mean(0).round(3))), 'votes': dict(share)}
        print(c, T[c])
    # ---- label-permutation null for the Voynich verdict (and LOCO accuracy)
    rng = np.random.RandomState(0)
    corpora = sorted(set(ct)); lab = {c: yt[ct == c][0] for c in corpora}
    labs = [lab[c] for c in corpora]
    vz = corp == 'V_ZL'
    obs_p = max(T['V_ZL']['mean_p'].values())
    nullp, nullba = [], []
    for it in range(200):
        perm = rng.permutation(labs); mp = dict(zip(corpora, perm))
        yp = np.array([mp[c] for c in ct])
        Pn, cln = fit(Xt, yp)
        nullp.append(float(Pn(X[vz]).mean(0).max()))
        if it < 30:
            nullba.append(bal_acc(loco(Xt, yp, ct))[0])
    p_conf = (1 + sum(v >= obs_p for v in nullp)) / (1 + len(nullp))
    p_ba = (1 + sum(v >= ba for v in nullba)) / (1 + len(nullba))
    print('Voynich max mean posterior', obs_p, 'null mean', np.mean(nullp), '95%', np.percentile(nullp, 95), 'p', p_conf)
    print('LOCO bal acc null mean', np.mean(nullba), 'max', max(nullba), 'p', p_ba)
    # ---- nearest corpora (standardised centroid distance, training scale)
    mu = Xt.mean(0); sd = Xt.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    cen = {c: Z[corp == c].mean(0) for c in set(corp)}
    near = {}
    for v in ['V_ZL', 'V_IT', 'V_ZL_A', 'V_ZL_B']:
        d = sorted(((float(np.linalg.norm(cen[v] - cen[c])), c) for c in corpora))
        near[v] = d[:10]
        print(v, 'nearest', [(c, round(x, 2)) for x, c in d[:10]])
    L.save('cycle1.json', {'keys': keys, 'loco': res, 'bal_acc': ba, 'per': per, 'tests': T, 'nullp': nullp,
                           'nullba': nullba, 'p_conf': p_conf, 'p_ba': p_ba, 'near': near})
    # ---- rows
    mg = sum(1 for v in magic.values() if v[0] == 'MAGIC')
    L.row(FN, 'V-31.1a', f'LOCO multinomial logistic, {len(keys)} line-free features, N={N}-token samples, {len(corpora)} training corpora '
          f'({dict(Counter(lab.values()))}); null = corpus labels permuted (30 LOCO runs)',
          f'balanced corpus accuracy {ba:.2f} (null mean {np.mean(nullba):.2f}, max {max(nullba):.2f}); per class {per}',
          'battery separates the five classes' if p_ba < 0.05 else 'battery does NOT separate the classes')
    L.row(FN, 'V-31.1b', 'Positive control: each MAGIC corpus held out (LOCO), must be classed MAGIC',
          f'{mg}/{len(magic)} classed MAGIC: ' + ', '.join(f'{c}->{p}({q})' for c, (p, q) in magic.items()),
          'control passes' if mg >= 0.6 * len(magic) else 'control FAILS: magic words are not recognised as a class')
    vz = T['V_ZL']; vi = T['V_IT']
    L.row(FN, 'V-31.1c', 'Voynich (ZL3b; IT2a) samples classified by the model trained on all training corpora; null = 200 label permutations',
          f"ZL mean posterior {vz['mean_p']}, votes {vz['votes']}; IT {vi['mean_p']}, votes {vi['votes']}; max posterior {obs_p:.2f} vs null 95% {np.percentile(nullp, 95):.2f} (p={p_conf:.3f})",
          'see cycle summary')
    others = ', '.join(f"{c}: {max(T[c]['mean_p'], key=T[c]['mean_p'].get)} ({max(T[c]['mean_p'].values()):.2f})" for c in tests if not c.startswith('V_'))
    L.row(FN, 'V-31.1d', 'Other test objects (Steganographia conjurations = covert cipher; Helene Smith Martian; Lingua Ignota list; v21 forgers trained on the Voynich)',
          others, 'descriptive')
    L.row(FN, 'V-31.1e', 'Nearest training corpora to the Voynich (standardised centroid distance)',
          '; '.join(f"{v}: " + ', '.join(c for _, c in near[v][:5]) for v in near), 'descriptive')


if __name__ == '__main__':
    main()
