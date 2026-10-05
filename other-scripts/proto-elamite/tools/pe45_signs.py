"""pe45 cycles 2-3: read sign classes and kin-group tablets off the surviving grammars.
Simulation-trained classifier: corpora simulated from the ABC survivors (true sign classes known:
free / god-like / lineage marker) train a per-sign classifier on corpus-internal features; it is
then applied to the real corpus and to its nulls.
Controls: held-out posterior simulations (planted truth); Ur III opaque names (true god signs);
nulls: names resampled across tablets, signs shuffled within names.
Kin tablets: score = sum over signs shared by >= 2 names of P(marker) * share; AUC on planted kin
tablets; Ur III 'A dumu B' tablets; PE tablets vs the resampled null.
usage: python3 pe45_signs.py PE|UR3 n_train  -> data/pe45_ckpt/signs_<which>.json"""
import os, sys, json, random, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe45_common import simulate, sign_features, encode, load_corpora, PNAMES, CK, FNAMES  # noqa
from pe45_abc import load_bank, null_within, null_across  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    # ties: average ranks
    from scipy.stats import rankdata
    r = rankdata(allv)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def kin_scores(C, pm):
    """pm: dict sign -> P(marker).  Returns per-tablet score (None for tablets with < 3 names)."""
    out = []
    for t in C:
        u = list(dict.fromkeys(t))
        if len(u) < 3:
            out.append(None)
            continue
        cc = collections.Counter(e for n in u for e in set(n))
        s = sum(pm.get(e, 0.0) * (c - 1) / (len(u) - 1) for e, c in cc.items() if c >= 2)
        out.append(s)
    return out


def thetas(which, k=200):
    R = json.load(open(os.path.join(CK, 'abc.json')))
    P, _ = load_bank(which)
    acc = R[which]['acc'][:k]
    return [dict(zip(PNAMES, P[i])) for i in acc]


def train(which, ntrain, C, rng):
    sizes = [len(t) for t in C]
    lens = [len(n) for t in C for n in t]
    TH = thetas(which)
    X, Y, sims = [], [], []
    for r in range(ntrain + 40):
        th = TH[rng.randrange(len(TH))]
        Cs, cls, kin = simulate(th, sizes, lens, rng)
        keys, F = sign_features(Cs)
        y = np.array([cls[e] for e in keys])
        if r < ntrain:
            X.append(F)
            Y.append(y)
        else:
            sims.append((Cs, keys, F, y, kin))
    X, Y = np.concatenate(X), np.concatenate(Y)
    # balance classes by weight
    w = np.ones(len(Y))
    for c in range(3):
        if (Y == c).sum():
            w[Y == c] = len(Y) / (3 * (Y == c).sum())
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
    clf.fit(X, Y, sample_weight=w)
    # planted (held-out posterior sims) evaluation
    ev = {'auc_god': [], 'auc_mark': [], 'auc_kin': [], 'n_god': [], 'n_mark': []}
    for Cs, keys, F, y, kin in sims:
        pr = clf.predict_proba(F)
        cl = list(clf.classes_)
        for c, nm in ((1, 'god'), (2, 'mark')):
            if c in cl and (y == c).sum() and (y != c).sum():
                p = pr[:, cl.index(c)]
                ev['auc_' + nm].append(auc(p[y == c], p[y != c]))
                ev['n_' + nm].append(int((y == c).sum()))
        if 2 in cl:
            pm = dict(zip(keys, pr[:, cl.index(2)]))
            ks = kin_scores(Cs, pm)
            pos = [s for s, k in zip(ks, kin) if s is not None and k]
            neg = [s for s, k in zip(ks, kin) if s is not None and not k]
            if pos and neg:
                ev['auc_kin'].append(auc(pos, neg))
    return clf, ev


def apply(clf, C):
    keys, F = sign_features(C)
    pr = clf.predict_proba(F)
    cl = list(clf.classes_)
    col = lambda c: pr[:, cl.index(c)] if c in cl else np.zeros(len(keys))
    return keys, F, col(0), col(1), col(2)


def main(which, ntrain):
    D = load_corpora()
    rng = random.Random(4545)
    C, inv = encode([t['names'] for t in D[which]])
    clf, ev = train(which, ntrain, C, rng)
    out = {'planted': {k: [float(np.nanmedian(v)) if v else None, len(v)] for k, v in ev.items()}}
    keys, F, pf, pg, pm = apply(clf, C)
    freq = collections.Counter(e for t in C for n in t for e in n)
    out['signs'] = [dict(sign=inv[e], freq=freq[e], p_free=float(a), p_god=float(b), p_mark=float(c),
                         feat=dict(zip(FNAMES, map(float, f)))) for e, f, a, b, c in zip(keys, F, pf, pg, pm)]
    # nulls: per-sign P(marker)/P(god) under resampled corpora
    nul = {'across': collections.defaultdict(list), 'within': collections.defaultdict(list)}
    nul_g = {'across': collections.defaultdict(list), 'within': collections.defaultdict(list)}
    kin_null_max = []
    for r in range(20):
        for nm, f in (('across', null_across), ('within', null_within)):
            Cn = f(C, rng)
            k2, _, _, g2, m2 = apply(clf, Cn)
            for e, a, b in zip(k2, g2, m2):
                nul[nm][inv[e]].append(float(b))
                nul_g[nm][inv[e]].append(float(a))
            if nm == 'across':
                ks = [s for s in kin_scores(Cn, dict(zip(k2, m2))) if s is not None]
                kin_null_max.append(ks)
    for s in out['signs']:
        for nm in ('across', 'within'):
            v = nul[nm].get(s['sign'], [])
            s['pm_null_' + nm] = float(np.mean(v)) if v else None
            s['pm_null_' + nm + '_p'] = float(np.mean([x >= s['p_mark'] for x in v])) if v else None
            g = nul_g[nm].get(s['sign'], [])
            s['pg_null_' + nm] = float(np.mean(g)) if g else None
            s['pg_null_' + nm + '_p'] = float(np.mean([x >= s['p_god'] for x in g])) if g else None
    # kin tablets
    pmd = dict(zip(keys, pm))
    ks = kin_scores(C, pmd)
    flat_null = np.array([x for v in kin_null_max for x in v])
    thr = float(np.percentile(flat_null, 99)) if len(flat_null) else None
    tabs = []
    for t, s, raw in zip(C, ks, D[which]):
        if s is None:
            continue
        tabs.append(dict(id=raw['id'], n=len(set(t)), score=float(s), p_null=float((flat_null >= s).mean()),
                         kin=raw.get('kin')))
    out['kin_thr99'] = thr
    out['kin_n_over'] = int(sum(1 for x in tabs if x['score'] > thr))
    out['kin_n_tested'] = len(tabs)
    out['kin_null_over_mean'] = float(np.mean([sum(1 for x in v if x > thr) for v in kin_null_max]))
    out['tablets'] = sorted(tabs, key=lambda x: -x['score'])
    if which == 'UR3':
        gods = set(D['UR3_gods'])
        sg = [s for s in out['signs']]
        isg = [s['sign'] in gods for s in sg]
        out['ur3_god_auc'] = auc([s['p_god'] for s, g in zip(sg, isg) if g], [s['p_god'] for s, g in zip(sg, isg) if not g])
        out['ur3_god_auc_freq'] = auc([s['freq'] for s, g in zip(sg, isg) if g], [s['freq'] for s, g in zip(sg, isg) if not g])
        out['ur3_god_auc_pgnot_free'] = auc([1 - s['p_free'] for s, g in zip(sg, isg) if g], [1 - s['p_free'] for s, g in zip(sg, isg) if not g])
        out['ur3_n_gods_tested'] = int(sum(isg))
        # frequency-matched: within freq strata
        fq = np.array([math_log(s['freq']) for s in sg])
        out['ur3_god_auc_freqmatched'] = strat_auc([s['p_god'] for s in sg], isg, fq)
        out['ur3_kin_auc'] = auc([x['score'] for x in tabs if x['kin']], [x['score'] for x in tabs if not x['kin']])
        out['ur3_kin_n'] = sum(1 for x in tabs if x['kin'])
    json.dump(out, open(os.path.join(CK, 'signs_%s.json' % which), 'w'))
    print(json.dumps({k: v for k, v in out.items() if k not in ('signs', 'tablets')}, indent=1))


def math_log(x):
    return float(np.log(x))


def strat_auc(score, lab, fq, nb=4):
    score, lab = np.asarray(score), np.asarray(lab)
    qs = np.quantile(fq, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(qs, fq, side='right') - 1, 0, nb - 1)
    num = den = 0.0
    for k in range(nb):
        m = b == k
        p, n = score[m & lab], score[m & ~lab]
        if len(p) and len(n):
            a = auc(p, n)
            num += a * len(p) * len(n)
            den += len(p) * len(n)
    return num / den if den else float('nan')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
