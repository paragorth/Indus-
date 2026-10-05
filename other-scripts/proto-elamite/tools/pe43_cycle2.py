"""pe43 cycle 2: THE TREE TELLS TIME. Children are born after their parents.

2A birth-order test (PC only, real periods): parent-child form pairs (variant -> base, compound -> parts);
   asymmetry 'child new in Uruk III while parent already in IV' vs 'parent new in III while child in IV',
   against period labels shuffled among tablets.
2B massive random seriation: 4,000 random orders per family of hypotheses
   R = random projections + 0-3 reciprocal-averaging steps;
   T = tree 'terminus post quem': random birth ages along the phylogeny (child = parent + Exp), tablet date =
       latest birth among its forms.
   fitness = compactness of the FIT forms (70%); held-out = compactness of the other 30% of forms.
   Orientation by the phylogeny (child first-appears later than parent) using FIT pairs, tested on HELD-OUT pairs.
   Nulls: token-shuffled corpus; variant labels shuffled among signs (pairs randomised).
   Truth: PC Uruk IV < III (AUC); PLANT true t (Spearman).
usage: python3 pe43_cycle2.py PLANT|PC|PE
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr
from pe43_common import *
from pe43_cycle1 import shuffle_tokens


def birth_order_pc(T, rng, nperm=500):
    forms = Counter(f for t in T for l in t['lines'] for f in l['forms'])
    pairs = [(p, f) for f in forms for p in parents(f) if p in forms]
    per = np.array([t['period'] == 'Uruk IV' for t in T])
    tabs = defaultdict(set)
    for i, t in enumerate(T):
        for l in t['lines']:
            for f in l['forms']:
                tabs[f].add(i)
    def stat(isIV):
        inIV = {f: any(isIV[i] for i in s) for f, s in tabs.items()}
        a = sum(1 for p, c in pairs if inIV[p] and not inIV[c])      # child born after
        b = sum(1 for p, c in pairs if inIV[c] and not inIV[p])      # child before parent
        return a, b
    a, b = stat(per)
    nul = [stat(rng.permutation(per)) for _ in range(nperm)]
    r = (a - b) / max(1, a + b)
    rn = np.array([(x - y) / max(1, x + y) for x, y in nul])
    return {'pairs': len(pairs), 'child_later': a, 'child_earlier': b, 'asym': r,
            'null_asym': [float(rn.mean()), float(rn.std())], 'z': float((r - rn.mean()) / rn.std()),
            'p_greater': float((rn >= r).mean()), 'p_less': float((rn <= r).mean())}


def build(T, rng, min_tab=3):
    forms = Counter()
    for t in T:
        for f in {f for l in t['lines'] for f in l['forms']}:
            forms[f] += 1
    F = sorted(f for f in forms if forms[f] >= min_tab)
    perm = rng.permutation(len(F))
    fit = set(F[i] for i in perm[:int(0.7 * len(F))])
    fidx = {f: i for i, f in enumerate(F)}
    rows = []
    keep_t = []
    for ti, t in enumerate(T):
        s = sorted({fidx[f] for l in t['lines'] for f in l['forms'] if f in fidx})
        if sum(F[i] in fit for i in s) >= 2:
            rows.append(s); keep_t.append(ti)
    X = np.zeros((len(rows), len(F)), bool)
    for r, s in enumerate(rows):
        X[r, s] = True
    fitmask = np.array([f in fit for f in F])
    return F, fidx, X, fitmask, keep_t


def compact(ranks01, X, mask):
    """mean normalised spread of the tablet ranks of each form in mask (lower = more compact)."""
    cols = np.nonzero(mask & (X.sum(0) >= 2))[0]
    out = []
    for c in cols:
        r = ranks01[X[:, c]]
        out.append(r.std() / 0.288675)
    return float(np.mean(out))


def to_ranks(s, rng):
    s = s + 1e-9 * rng.standard_normal(len(s))
    r = np.empty(len(s)); r[np.argsort(s)] = np.arange(len(s))
    return r / max(1, len(s) - 1)


def gen_R(X, fitmask, rng):
    Xf = X[:, fitmask].astype(float)
    rs = Xf.sum(1); cs = np.maximum(Xf.sum(0), 1)
    w = rng.standard_normal(Xf.shape[1])
    s = Xf @ w / rs
    for _ in range(rng.integers(0, 4)):
        w = Xf.T @ s / cs; w = (w - w.mean()) / (w.std() + 1e-12)
        s = Xf @ w / rs
    return s


def gen_T(X, fitmask, F, fidx, rng):
    # birth ages along the phylogeny of ALL forms (fit forms decide dates)
    b = {}
    scale = rng.uniform(0.1, 1.0)
    def birth(f):
        if f in b:
            return b[f]
        ps = parents(f)
        base_age = max([birth(p) for p in ps], default=0.0) if ps else rng.uniform(0, 0.3)
        b[f] = base_age + (rng.exponential(scale) if ps else 0.0)
        return b[f]
    ages = np.array([birth(f) for f in F])
    ages = ages + 0.05 * rng.standard_normal(len(F))
    A = np.where(X[:, fitmask], ages[fitmask][None, :], -np.inf)
    s = A.max(1)
    # tie-break by mean age
    s = s + 1e-3 * np.where(X[:, fitmask], ages[fitmask][None, :], 0).sum(1) / np.maximum(1, X[:, fitmask].sum(1))
    return s


def orientation(ranks01, X, F, fidx, pairs):
    """share of (parent, child) pairs whose child first-appears (10th pct rank) later than the parent."""
    if not pairs:
        return np.nan
    q = {}
    def fa(f):
        if f not in q:
            r = ranks01[X[:, fidx[f]]]
            q[f] = np.quantile(r, 0.1) if len(r) else np.nan
        return q[f]
    v = [fa(c) > fa(p) for p, c in pairs if not np.isnan(fa(c)) and not np.isnan(fa(p)) and fa(c) != fa(p)]
    return float(np.mean(v)) if v else np.nan, len(v)


def run_corpus(T, rng, truth=None, period=None, nh=4000, label=''):
    F, fidx, X, fitmask, keep_t = build(T, rng)
    Fset = set(F)
    pairs = [(p, f) for f in F for p in parents(f) if p in Fset]
    rng.shuffle(pairs)
    pf, pt = pairs[:len(pairs) // 2], pairs[len(pairs) // 2:]
    res = {'n_tab': int(X.shape[0]), 'n_forms': len(F), 'n_pairs': len(pairs)}
    for fam in ['R', 'T']:
        H = []
        for h in range(nh):
            s = gen_R(X, fitmask, rng) if fam == 'R' else gen_T(X, fitmask, F, fidx, rng)
            r = to_ranks(s, rng)
            H.append((compact(r, X, fitmask), r))
        H.sort(key=lambda x: x[0])
        top = H[:max(20, nh // 50)]
        o = {'fit_best': H[0][0], 'fit_med': float(np.median([h[0] for h in H]))}
        o['heldout_top'] = float(np.mean([compact(r, X, ~fitmask) for _, r in top]))
        o['heldout_all_med'] = float(np.median([compact(r, X, ~fitmask) for _, r in H[::20]]))
        ors = []
        for _, r in top:
            of, nf = orientation(r, X, F, fidx, pf)
            # orient by fit pairs
            if of < 0.5:
                r = 1 - r
            ot, nt = orientation(r, X, F, fidx, pt)
            entry = {'or_fit': max(of, 1 - of), 'or_test': ot}
            if truth is not None:
                tt = np.array([truth[i] for i in keep_t])
                entry['rho_true'] = float(spearmanr(r, tt)[0])
            if period is not None:
                pp = np.array([period[i] for i in keep_t])
                # AUC: III later
                from scipy.stats import mannwhitneyu
                a = r[pp == 'Uruk III']; b = r[pp == 'Uruk IV']
                entry['auc_III_later'] = float(mannwhitneyu(a, b).statistic / (len(a) * len(b)))
            ors.append(entry)
        for k in ors[0]:
            v = np.array([e[k] for e in ors], float)
            o[k] = [float(np.nanmean(v)), float(np.nanstd(v))]
        o['n_test_pairs'] = len(pt)
        res[fam] = o
        print(label, fam, json.dumps(o), flush=True)
    return res


def main(name):
    rng = np.random.default_rng(4302)
    R = {}
    truth = period = None
    if name == 'PE':
        T = load_pe()
    elif name == 'PC':
        T = load_pc(); period = [t['period'] for t in T]
        R['birth_order'] = birth_order_pc(T, rng)
        print('PC birth order', R['birth_order'], flush=True)
    else:
        T, tr = make_plant(seed=2); truth = [t['t'] for t in T]
    R['real'] = run_corpus(T, rng, truth, period, label=name + ' real')
    R['tokshuf'] = run_corpus(shuffle_tokens(T, 7), rng, truth, period, nh=1500, label=name + ' tokshuf')
    # phylogeny shuffled: derived simple forms renamed onto random roots (variant labels shuffled)
    forms = Counter(f for t in T for l in t['lines'] for f in l['forms'])
    rts = sorted({root(f) for f in forms if not f.startswith('|')})
    vm = {}
    for f in forms:
        if not f.startswith('|') and is_derived(f):
            suf = f[len(root(f)):]
            vm[f] = rts[rng.integers(len(rts))] + suf
    T2 = [dict(t, lines=[dict(l, forms=[vm.get(f, f) for f in l['forms']]) for l in t['lines']]) for t in T]
    R['vshuf'] = run_corpus(T2, rng, truth, period, nh=1500, label=name + ' vshuf')
    json.dump(R, open(os.path.join(CK, f'c2_{name}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
